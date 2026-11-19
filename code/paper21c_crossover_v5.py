#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "pyarrow", "torch", "transformers"]
# ///
"""
Paper 21c — the complete 2x2. One script, four cells, one overnight.

  SMOKE TEST FIRST (2 minutes, do this before committing the night):
    uv run paper21c_crossover_v1.py --two-site ~/projects/paper9/two_site.py \
        --workdir ~/projects/paper21c_x --smoke

  THE REAL RUN:
    uv run paper21c_crossover_v1.py --two-site ~/projects/paper9/two_site.py \
        --workdir ~/projects/paper21c_x

  REPORT ONLY (any time, reads what is on disk):
    uv run paper21c_crossover_v1.py --workdir ~/projects/paper21c_x --report-only

THE FOUR CELLS
--------------
  1. train on mimic3 (MLM)          -> test on mimic3 and mimic4
  2. train on mimic4 (MLM)          -> test on mimic4 and mimic3
  3. fine-tune on mimic3 (contrast) -> test on mimic3 and mimic4
  4. fine-tune on mimic4 (contrast) -> test on mimic4 and mimic3

Each era is measured under both training conditions, so each era is its own
control and the era effect -- documentation style, de-identification, baseline
difficulty -- cancels in the within-era difference. The quantity of interest per
objective is

    d(e) = mean over queries q in era e of
             [ RR(q, e, trained on e) - RR(q, e, trained on the other) ]

computed on PAIRED per-query reciprocal ranks, because the same 1,000 queries are
scored under both conditions. Both d positive means each era retains its own.
Reported alongside it, and a different question:

    b(e) = mean over q of [ RR(q, e, trained on e) - RR(q, e, unadapted) ]

d can be positive while b is negative -- training helping relatively while
hurting absolutely.

WHAT IS ALREADY KNOWN (cell 1, 25,000 steps, 16 Nov)
----------------------------------------------------
      steps        mimic3        mimic4
          0        0.1642        0.2243
       1000        0.1132        0.1175
       3000        0.1127        0.1108
      10000        0.1085        0.1033
      25000        0.1146        0.1188

The collapse is complete by 1,000 steps and the curve is flat to 25,000 while MLM
loss halves again. That licenses 3,000 steps for the MLM cells here rather than
25,000. It says nothing about locality: cell 1 alone has no control condition,
which is what cell 2 supplies.

Cell 1 is rerun here even though it has been run, for two reasons: that run saved
only bootstrap summaries and discarded the per-query ranks, so the paired
analysis is not recoverable from it; and running all four cells through one code
path removes any question about comparability.

BUILT FOR AN UNATTENDED NIGHT
-----------------------------
  * Results are written after EVERY checkpoint, not at the end. If the machine
    dies at 3am you keep everything that finished.
  * Any cell that raises is logged and the run continues to the next cell.
  * Re-running skips cells already complete on disk, so a crash is resumed by
    issuing the same command again.
  * Checkpoints are evaluated immediately and then deleted, so disk stays at one
    checkpoint at a time (~440 MB) rather than ten.
  * Every tensor has a fixed width. Variable widths are what fragmented the MPS
    allocator in Paper 21a and turned a 38-minute projection into 169 minutes.
  * The evaluation pools are built once and reused by every cell, and each cell
    records a hash of its query list. The paired analysis refuses to run if two
    cells disagree, rather than silently comparing different queries.

THE ONE KNOWN THREAT TO VALIDITY, NOT FIXED BY DEFAULT
------------------------------------------------------
MIMIC-III (~2001-2012) and MIMIC-IV (~2008-2019) are both Beth Israel Deaconess
and overlap by roughly four years, so notes used for training in one cell may sit
in the test pool of another. That contaminates the control condition and biases
the within-era difference toward zero.

--mimic4-min-year YYYY drops MIMIC-IV admissions at or before YYYY to remove the
overlap. It is OFF by default because it needs a year column that I have not been
able to verify in your cohort.csv; when enabled it looks for one of
anchor_year_group / anchor_year / admit_year / admityear / real_year_lower and,
if none is present, WARNS AND PROCEEDS WITHOUT RESTRICTING rather than failing
the night. Check the warning in the log before relying on it. If the column is
missing, this is a limitation to report rather than a defect in the run.

NOT A CROSSOVER TRIAL, AND BETTER THAN ONE
------------------------------------------
The clinical crossover model is x_ij = a + D*T_ij + b*I(j=2) + xi_i + e_ij with a
carryover term for the previous period's treatment. Here there are no periods:
each cell trains a fresh model from the same checkpoint and scores static
documents, so no treatment can precede another on the same unit. Carryover is
structurally impossible, there is no sequence to randomise and no washout
question. In an AB/BA trial the leftover degree of freedom is shared between
carryover and treatment-by-period, so neither is identifiable; with no carryover
that degree of freedom is free and the era-by-training interaction -- which is
the locality effect -- is cleanly identified.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
TOP_K = 10
N_BOOT = 2000
WINDOW = 510          # Paper 20 convention
STRIDE = 384
MIN_CHARS = 400       # two_site.py eligibility
MIN_TARGET = 100      # Paper 20
MODEL = "sentence-transformers/all-mpnet-base-v2"
HEAD_SOURCE = "microsoft/mpnet-base"
MNRL_SCALE = 20.0     # sentence-transformers MultipleNegativesRankingLoss default
TARGET_CAP = 1024     # tokens retained per training target
YEAR_COLS = ("year_group", "real_year_lower", "anchor_year_group", "anchor_year",
             "admit_year", "admityear")
# MIMIC-III keeps de-identification placeholders in the note text ("[**Hospital1
# 18**]", "[**8-14**]"); MIMIC-IV strips them to blanks. That is an era-specific
# surface feature only MIMIC-III training can learn, and it is a candidate
# explanation for the within-era asymmetry. --scrub-deid removes them.
DEID_RE = re.compile(r"\[\*\*.*?\*\*\]")
WS_RE = re.compile(r"\s{2,}")
DEFAULT_CELLS = "mlm:mimic3,mlm:mimic4,contrastive:mimic3,contrastive:mimic4"
UNIFORM_TOL = 0.02        # |loss - log(batch)| within 2% of log(batch) = uniform
COLLAPSE_PATIENCE = 50    # consecutive steps sitting on log(batch)
COLLAPSE_ARM_AFTER = 100  # never test before this step, nor before warmup ends
MIXED_NAME = "mixed"   # train_on value meaning "both eras pooled", for the
                       # retraining case: a site holding old and new notes at once


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------ imports ---


def load_two_site(path: Path):
    """Import the published extractor rather than reimplementing its rules.

    Query construction is the part of a known-item design that silently decides
    the answer, and two_site.py's extractor is on its third rewrite. Importing it
    is what puts the baseline on the same scale as the published mpnet value.
    """
    if not path.exists():
        raise SystemExit(
            f"two_site.py not found at {path}\n"
            "  pass --two-site <path>; it is the source of the query convention"
        )
    spec = importlib.util.spec_from_file_location("two_site", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["two_site"] = mod
    spec.loader.exec_module(mod)
    for fn in ("sentences", "build_df", "narrative_query", "is_prose"):
        if not hasattr(mod, fn):
            raise SystemExit(f"{path} has no {fn}(); is this the right file?")
    return mod


# ------------------------------------------------------------------- cohort ---


def build_pairs(ts, texts: list[str]) -> list[tuple[str, str]]:
    """(target, query) pairs, query sentences removed from the target.

    Unchanged from the script that ran successfully on 16 Nov. Removal operates
    on the sentence list rather than by string replace, because narrative_query
    returns scrubbed, reflowed text that is not a substring of the original.
    """
    df = ts.build_df(texts)
    out = []
    for t in texts:
        q = ts.narrative_query(t, df)
        if not q:
            continue
        sents = ts.sentences(t)
        qset = set(q.split(" . "))
        keep, removed = [], 0
        for s in sents:
            if removed < 2 and (s in q or q.startswith(s) or s in qset):
                removed += 1
                continue
            keep.append(s)
        target = " ".join(keep)
        if len(target) < MIN_TARGET:
            continue
        out.append((target, q))
    return out


def _restrict_mimic4(notes: pd.DataFrame, cohort: pd.DataFrame, min_year: int) -> pd.DataFrame:
    """Drop MIMIC-IV admissions at or before min_year, if a year column exists.

    Defensive by design: a missing column warns and proceeds. Failing an
    overnight run over an unverified column name would be the worse outcome.
    """
    col = next((c for c in YEAR_COLS if c in cohort.columns), None)
    if col is None:
        log(f"  WARNING: --mimic4-min-year {min_year} requested but none of "
            f"{YEAR_COLS} is in cohort.csv")
        log("           proceeding WITHOUT the restriction; the 2008-2012 era "
            "overlap remains a limitation")
        return notes
    yr = cohort[["hadm_id", col]].drop_duplicates(subset="hadm_id", keep="first").copy()
    # anchor_year_group arrives as a string like "2011 - 2013"; take its lower bound.
    yr["_y"] = (yr[col].astype(str).str.extract(r"(\d{4})")[0].astype(float))
    before = len(notes)
    notes = notes.merge(yr[["hadm_id", "_y"]], on="hadm_id", how="left")
    if len(notes) != before:
        raise SystemExit(
            f"the year merge changed the note count ({before:,} -> {len(notes):,}); "
            "hadm_id is not unique in cohort.csv even after de-duplication"
        )
    drop = (notes["corpus"] == "mimic4") & (notes["_y"] <= min_year)
    notes = notes[~drop.fillna(False)].drop(columns=["_y"])
    log(f"  --mimic4-min-year {min_year} via '{col}': dropped "
        f"{before - len(notes):,} of {before:,} notes")
    return notes


def _split_within(notes: pd.DataFrame, cohort: pd.DataFrame, within: str,
                  split_year: int) -> pd.DataFrame:
    """Restrict to one corpus and replace 'corpus' with two period labels.

    This is the test that separates time from corpus identity. The MIMIC-III
    versus MIMIC-IV contrast confounds them: MIMIC-III-trained models beat
    MIMIC-IV-trained models on BOTH eras, which is equally consistent with
    "older documentation transfers forward" and with "MIMIC-III is simply the
    better training corpus". Splitting a single database by period holds the
    de-identification regime, the EHR generation and the corpus identity fixed,
    so only time varies. If the asymmetry survives here it is temporal; if it
    disappears, it was corpus quality.

    Eras are named <corpus>_early (year lower bound <= split_year) and
    <corpus>_late, so --eras and --cells take the same form as usual.
    """
    col = next((c for c in YEAR_COLS if c in cohort.columns), None)
    if col is None:
        raise SystemExit(
            f"--within {within} needs a year column; none of {YEAR_COLS} is in "
            f"cohort.csv, which has {sorted(cohort.columns)}"
        )
    keep = notes["corpus"] == within
    if not keep.any():
        raise SystemExit(
            f"--within {within}: no notes with corpus == {within!r}; "
            f"present: {sorted(notes['corpus'].unique())}"
        )
    notes = notes[keep]
    yr = cohort[["hadm_id", col]].drop_duplicates(subset="hadm_id", keep="first").copy()
    yr["_y"] = yr[col].astype(str).str.extract(r"(\d{4})")[0].astype(float)
    n_before = len(notes)
    notes = notes.merge(yr[["hadm_id", "_y"]], on="hadm_id", how="left")
    if len(notes) != n_before:
        raise SystemExit(f"the year merge changed the note count "
                         f"({n_before:,} -> {len(notes):,})")
    missing = int(notes["_y"].isna().sum())
    if missing:
        log(f"  --within {within}: dropping {missing:,} notes with no year value")
        notes = notes[notes["_y"].notna()]
    early, late = f"{within}_early", f"{within}_late"
    notes = notes.assign(
        corpus=np.where(notes["_y"] <= split_year, early, late)).drop(columns=["_y"])
    counts = notes["corpus"].value_counts().to_dict()
    log(f"  --within {within} --split-year {split_year} via '{col}': "
        f"{early} {counts.get(early, 0):,} notes, {late} {counts.get(late, 0):,} notes")
    for e in (early, late):
        if counts.get(e, 0) == 0:
            raise SystemExit(f"--split-year {split_year} leaves {e} empty; pick another year")
    return notes


def load_cohort(outdir: Path, pool: int, seed: int, min_year: int | None,
                scrub_deid: bool = False, within: str | None = None,
                split_year: int | None = None):
    """Per-era evaluation pools and per-era training pools, split by patient.

    The rng is consumed in sorted corpus order and does not depend on which era
    is the training era, so every cell sees byte-identical evaluation pools. That
    is what makes the within-era difference a comparison of the same queries.
    """
    npath, cpath = outdir / "notes.parquet", outdir / "cohort.csv"
    for p in (npath, cpath):
        if not p.exists():
            raise SystemExit(f"missing {p}; pass --outdir <dir with notes.parquet, cohort.csv>")
    notes = pd.read_parquet(npath)
    cohort = pd.read_csv(cpath)
    if "hadm_id" not in cohort.columns:
        raise SystemExit(f"cohort.csv needs hadm_id; has {sorted(cohort.columns)}")
    if "text" not in notes.columns or "hadm_id" not in notes.columns:
        raise SystemExit(f"notes.parquet needs hadm_id and text; has {sorted(notes.columns)}")
    # subject_id and corpus may live in either file; only the union must have them.
    available = set(notes.columns) | set(cohort.columns)
    for c in ("subject_id", "corpus"):
        if c not in available:
            raise SystemExit(
                f"no '{c}' in either file; notes.parquet has {sorted(notes.columns)}, "
                f"cohort.csv has {sorted(cohort.columns)}"
            )

    # A duplicated hadm_id in cohort.csv silently MULTIPLIES notes through the
    # merge, inflating the pools without any error. Caught in testing; guard it.
    if cohort["hadm_id"].duplicated().any():
        n_dup = int(cohort["hadm_id"].duplicated().sum())
        log(f"  WARNING: cohort.csv has {n_dup:,} duplicate hadm_id row(s); keeping the "
            "first of each so the merge cannot duplicate notes")
        cohort = cohort.drop_duplicates(subset="hadm_id", keep="first")

    # Merge ONLY what notes.parquet is missing. Re-merging a column it already
    # carries yields corpus_x/corpus_y and no 'corpus' at all, which is how this
    # failed on the first real run.
    bring = [c for c in ("subject_id", "corpus") if c not in notes.columns]
    if bring:
        n_before = len(notes)
        notes = notes.merge(cohort[["hadm_id", *bring]], on="hadm_id", how="left")
        if len(notes) != n_before:
            raise SystemExit(
                f"the cohort merge changed the note count ({n_before:,} -> {len(notes):,}); "
                "hadm_id is not unique in cohort.csv"
            )
        log(f"  took {', '.join(bring)} from cohort.csv; "
            f"notes.parquet already had the rest")
    else:
        log("  notes.parquet already carries subject_id and corpus; no merge needed")
    for c in ("subject_id", "corpus", "text"):
        if c not in notes.columns:
            raise SystemExit(
                f"no '{c}' column after the merge; notes.parquet has "
                f"{sorted(notes.columns)} and cohort.csv has {sorted(cohort.columns)}"
            )
    notes = notes.dropna(subset=["subject_id", "text", "corpus"])
    if scrub_deid:
        # Applied BEFORE the length filter so the filter judges the scrubbed
        # text, and to every era so the operation itself cannot differ by era
        # (it is a no-op on MIMIC-IV, which has no placeholders).
        before_chars = int(notes["text"].str.len().sum())
        hits = int(notes["text"].str.count(DEID_RE.pattern).sum())
        notes = notes.assign(
            text=notes["text"].str.replace(DEID_RE, " ", regex=True)
                              .str.replace(WS_RE, " ", regex=True).str.strip())
        after_chars = int(notes["text"].str.len().sum())
        log(f"  --scrub-deid: removed {hits:,} placeholder spans, "
            f"{before_chars - after_chars:,} of {before_chars:,} characters "
            f"({100 * (before_chars - after_chars) / max(before_chars, 1):.1f}%)")
        per = notes.groupby("corpus")["text"].apply(
            lambda s: int(s.str.count(DEID_RE.pattern).sum()))
        log(f"  placeholders remaining by era: {per.to_dict()}")

    notes = notes[notes["text"].str.len() >= MIN_CHARS]
    if min_year is not None:
        notes = _restrict_mimic4(notes, cohort, min_year)
    if within is not None:
        notes = _split_within(notes, cohort, within, split_year)

    rng = np.random.default_rng(seed)
    eval_pools, train_pools = {}, {}
    for corpus, g in notes.groupby("corpus"):
        subs = np.array(sorted(g["subject_id"].unique()))
        rng.shuffle(subs)
        cut = max(pool * 2, int(0.3 * len(subs)))
        eval_subs = set(subs[:cut].tolist())
        is_eval = g["subject_id"].isin(eval_subs)
        eval_pools[corpus] = g[is_eval]["text"].tolist()
        train_pools[corpus] = g[~is_eval]["text"].tolist()
        log(f"  {corpus}: {len(subs):,} patients -> {int(is_eval.sum()):,} eval notes, "
            f"{int((~is_eval).sum()):,} held out for training")
    return eval_pools, train_pools


def qhash(pairs: list[tuple[str, str]]) -> str:
    """Fingerprint of the query list, so two cells cannot be paired by mistake."""
    h = hashlib.sha256()
    for _, q in pairs:
        h.update(q.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- retrieval ---


def encode(texts, model, tok, device, is_query: bool, batch: int):
    """Mean-pooled, L2-normalised. Targets windowed 510/stride 384, queries not."""
    import torch
    import torch.nn.functional as F

    rows, owner = [], []
    buf, buf_owner = [], []

    def flush():
        if not buf:
            return
        width = max(len(b) for b in buf)
        ids = torch.full((len(buf), width), tok.pad_token_id, dtype=torch.long)
        msk = torch.zeros((len(buf), width), dtype=torch.long)
        for r, b in enumerate(buf):
            ids[r, : len(b)] = torch.tensor(b)
            msk[r, : len(b)] = 1
        with torch.no_grad():
            h = model(input_ids=ids.to(device), attention_mask=msk.to(device)).last_hidden_state
        m = msk.to(device).unsqueeze(-1).float()
        v = F.normalize((h * m).sum(1) / m.sum(1).clamp(min=1e-9), p=2, dim=1)
        rows.append(v.float().cpu().numpy())
        owner.extend(buf_owner)
        buf.clear(); buf_owner.clear()

    for i, t in enumerate(texts):
        body = tok(t, add_special_tokens=False, truncation=False)["input_ids"]
        if is_query:
            spans = [body[:WINDOW]]
        else:
            spans = [body[s:s + WINDOW] for s in range(0, max(len(body), 1), STRIDE)] or [[]]
            spans = [s for s in spans if s] or [[]]
        for s in spans:
            buf.append([tok.cls_token_id] + s + [tok.sep_token_id])
            buf_owner.append(i)
            if len(buf) >= batch:
                flush()
    flush()
    return np.vstack(rows).astype(np.float32), np.array(owner)


def rr_at_k(Q, D, owner, n_docs):
    """Per-query reciprocal rank at TOP_K. Query i's target is document i.

    A document scores the maximum cosine over its windows, as in Paper 20.
    Returns the per-query vector, which is what the paired analysis needs.
    """
    S = Q @ D.T
    doc = np.full((len(Q), n_docs), -np.inf, dtype=np.float32)
    for w in range(S.shape[1]):
        d = owner[w]
        np.maximum(doc[:, d], S[:, w], out=doc[:, d])
    rr = np.zeros(len(Q), np.float32)
    for i in range(len(Q)):
        k = min(TOP_K, n_docs - 1)
        t = np.argpartition(-doc[i], k)[: k + 1]
        for r, j in enumerate(t[np.argsort(-doc[i, t])][:TOP_K]):
            if j == i:
                rr[i] = 1.0 / (r + 1)
                break
    return rr


def boot_mean(x, n=N_BOOT, seed=SEED):
    """Percentile bootstrap of a mean. Used for levels and paired differences."""
    g = np.random.default_rng(seed)
    idx = g.integers(0, len(x), size=(n, len(x)))
    bs = np.asarray(x)[idx].mean(1)
    return float(np.mean(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def evaluate(model_dir, pairs_by_era, device, batch, tok_src):
    """Per-era per-query reciprocal ranks for one checkpoint."""
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tok_src)
    mdl = AutoModel.from_pretrained(model_dir).to(device).eval()
    out = {}
    for era, pairs in pairs_by_era.items():
        docs = [d for d, _ in pairs]
        qs = [q for _, q in pairs]
        D, owner = encode(docs, mdl, tok, device, False, batch)
        Q, _ = encode(qs, mdl, tok, device, True, batch)
        out[era] = rr_at_k(Q, D, owner, len(docs))
    del mdl, tok
    gc.collect()
    import torch
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    return out


# ------------------------------------------------------------------ mlm arm ---


def build_mlm(device, model, head_source):
    """Retriever encoder + the language-model head its repository dropped.

    all-mpnet-base-v2 is a contrastive fine-tune of microsoft/mpnet-base and
    ships no LM head; training against a randomly initialised one wrecks the
    encoder, which is what the first pilot measured. Every encoder tensor must
    transfer and is verified bit-identical.
    """
    from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
    import torch

    if AutoTokenizer.from_pretrained(model).get_vocab() != \
       AutoTokenizer.from_pretrained(head_source).get_vocab():
        raise SystemExit(
            f"vocabulary mismatch between {model} and {head_source}; the head "
            "predicts over a different token set than the encoder produces"
        )
    mlm = AutoModelForMaskedLM.from_pretrained(head_source)
    enc = AutoModel.from_pretrained(model)
    src, tgt = enc.state_dict(), mlm.base_model.state_dict()
    transfer = {k: v for k, v in src.items()
                if k in tgt and tuple(v.shape) == tuple(tgt[k].shape)}
    missing = [k for k in tgt if k not in transfer]
    mlm.base_model.load_state_dict(transfer, strict=False)
    after = mlm.base_model.state_dict()
    altered = [k for k, v in transfer.items() if not torch.equal(v.cpu(), after[k].cpu())]
    if altered:
        raise SystemExit(f"transplant did not take for {len(altered)} tensor(s): {altered[:8]}")
    log(f"    transplanted {len(transfer)}/{len(tgt)} encoder tensors, verified bit-identical")
    hard = [k for k in missing if not k.startswith("pooler.")]
    if hard:
        raise SystemExit(f"encoder tensors did not transfer: {hard[:8]}")
    if missing:
        log(f"    not transferred (unused by mean pooling): {missing}")
    del enc
    gc.collect()
    return mlm.to(device)


def train_mlm(texts, on_checkpoint, steps, checkpoints, batch, seq, lr, device,
              seed, model, head_source, head_warmup, lr_warmup,
              clip_grad=1.0):
    """Continued MLM. One fixed sequence length throughout, fp32."""
    import torch
    torch.manual_seed(seed)
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, DataCollatorForLanguageModeling

    tok = AutoTokenizer.from_pretrained(model)
    mdl = build_mlm(device, model, head_source)
    mdl.train()

    body = seq - 2
    need = (steps + head_warmup) * batch
    chunks = []
    for t in texts:
        ids = tok(t, add_special_tokens=False, truncation=False)["input_ids"]
        for s in range(0, len(ids), body):
            piece = ids[s:s + body]
            if len(piece) < body // 2:
                continue
            chunks.append([tok.cls_token_id] + piece + [tok.sep_token_id])
        if len(chunks) >= need:
            break
    log(f"    {len(chunks):,} chunks available, {need:,} needed")
    if len(chunks) < max(batch * 2, 1):
        raise RuntimeError(f"only {len(chunks)} chunks; cannot form batches of {batch}")
    if len(chunks) < need:
        log("    note: fewer chunks than steps x batch; the epoch will repeat")

    coll = DataCollatorForLanguageModeling(tokenizer=tok, mlm=True, mlm_probability=0.15)
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader([{"input_ids": c} for c in chunks], batch_size=batch, shuffle=True,
                    collate_fn=coll, generator=g, drop_last=True)

    for p in mdl.base_model.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW([p for p in mdl.parameters() if p.requires_grad], lr=lr)
    log(f"    phase 1: {head_warmup} head-only steps, encoder frozen")

    step, tokens, unfrozen, t_start = 0, 0, False, [None]
    t0 = time.time()
    while step < steps + head_warmup:
        for b in dl:
            if step == head_warmup and not unfrozen:
                for p in mdl.base_model.parameters():
                    p.requires_grad = True
                opt = torch.optim.AdamW(mdl.parameters(), lr=lr)
                unfrozen, t_start[0] = True, time.time()
                log(f"    phase 2: encoder unfrozen, {lr_warmup}-step lr warmup")
            if unfrozen:
                sc = min(1.0, (step - head_warmup + 1) / max(lr_warmup, 1))
                for grp in opt.param_groups:
                    grp["lr"] = lr * sc
            b = {k: v.to(device) for k, v in b.items()}
            loss = mdl(**b).loss
            loss.backward()
            if clip_grad and clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), clip_grad)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if unfrozen:
                tokens += b["input_ids"].numel()
            if step % 50 == 0 or step == 1:
                el = time.time() - t0
                ph = "train" if unfrozen else "head "
                log(f"    {ph} {step - head_warmup if unfrozen else step:>5}  "
                    f"loss {loss.item():.4f}  {tokens / max(el, 1e-9):,.0f} tok/s")
            k = step - head_warmup
            if unfrozen and k in checkpoints:
                on_checkpoint(k, mdl)
            if step >= steps + head_warmup:
                break
    rate = tokens / max(time.time() - (t_start[0] or t0), 1e-9)
    del mdl, opt, dl
    gc.collect()
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    return rate


# ---------------------------------------------------------- contrastive arm ---


def prep_contrastive(ts, tok, texts, train_pool, q_seq, seed):
    """Tokenise (query, target) pairs once. Targets kept to TARGET_CAP tokens."""
    rng = np.random.default_rng(seed)
    take = min(len(texts), train_pool)
    sel = [texts[i] for i in rng.choice(len(texts), take, replace=False)]
    log(f"    building training pairs from {take:,} held-out notes")
    pairs = build_pairs(ts, sel)
    q_ids, t_ids = [], []
    for target, query in pairs:
        qi = tok(query, add_special_tokens=False, truncation=False)["input_ids"][: q_seq - 2]
        ti = tok(target, add_special_tokens=False, truncation=False)["input_ids"][:TARGET_CAP]
        if len(qi) < 4 or len(ti) < 8:
            continue
        q_ids.append(np.asarray(qi, dtype=np.int32))
        t_ids.append(np.asarray(ti, dtype=np.int32))
    log(f"    {len(q_ids):,} training pairs retained")
    return q_ids, t_ids


def make_batch(idx, q_ids, t_ids, tok, q_seq, t_seq, rng, device):
    """Fixed-width (queries, targets). Targets are random windows, not openings,
    because evaluation scores max-over-windows and the query is drawn from the
    longest prose run, which is usually past the first t_seq tokens."""
    import torch
    n = len(idx)
    qi = torch.full((n, q_seq), tok.pad_token_id, dtype=torch.long)
    qm = torch.zeros((n, q_seq), dtype=torch.long)
    ti = torch.full((n, t_seq), tok.pad_token_id, dtype=torch.long)
    tm = torch.zeros((n, t_seq), dtype=torch.long)
    body = t_seq - 2
    for r, j in enumerate(idx):
        s = [tok.cls_token_id] + q_ids[j].tolist() + [tok.sep_token_id]
        qi[r, : len(s)] = torch.tensor(s, dtype=torch.long); qm[r, : len(s)] = 1
        t = t_ids[j]
        w = t[int(rng.integers(0, len(t) - body + 1)):][:body] if len(t) > body else t
        s = [tok.cls_token_id] + w.tolist() + [tok.sep_token_id]
        ti[r, : len(s)] = torch.tensor(s, dtype=torch.long); tm[r, : len(s)] = 1
    return qi.to(device), qm.to(device), ti.to(device), tm.to(device)


def mean_pool(out, mask):
    import torch.nn.functional as F
    h = out.last_hidden_state
    m = mask.unsqueeze(-1).float()
    return F.normalize((h * m).sum(1) / m.sum(1).clamp(min=1e-9), p=2, dim=1)


def train_contrastive(ts, texts, on_checkpoint, steps, checkpoints, batch, q_seq,
                      t_seq, lr, device, seed, model, train_pool, lr_warmup,
                      clip_grad=1.0):
    """MultipleNegativesRankingLoss on local (query, target) pairs.

    Note on reproducibility: mdl.train() turns dropout on, and dropout masks are
    drawn from torch's global RNG. Seeding it here is what makes a rerun with the
    same seed produce the same ranks; without it the same inputs gave different
    per-query ranks on every run. Reductions on some accelerators remain
    non-deterministic, so this reduces run-to-run variation without guaranteeing
    bit-identity on every device.

    No LM head, so no transplant and no frozen warmup: the model trains on the
    objective it was built with. The other targets in the batch are the
    negatives, so batch size is the task difficulty, not just a throughput knob.
    """
    import torch
    torch.manual_seed(seed)
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    mdl = AutoModel.from_pretrained(model).to(device)
    mdl.train()

    q_ids, t_ids = prep_contrastive(ts, tok, texts, train_pool, q_seq, seed)
    n = len(q_ids)
    if n < batch * 2:
        raise RuntimeError(f"only {n} training pairs; need well over batch {batch} "
                           "for in-batch negatives")
    log(f"    {max(n // batch, 1):,} steps per epoch; {steps:,} steps is "
        f"{steps / max(n // batch, 1):.1f} epochs")

    opt = torch.optim.AdamW(mdl.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    order, cursor = rng.permutation(n), 0
    step, tokens, t0 = 0, 0, time.time()
    dead, collapsed_at = 0, None
    clipped, max_gn = 0, 0.0

    while step < steps:
        if cursor + batch > n:
            order, cursor = rng.permutation(n), 0
        idx = order[cursor:cursor + batch]
        cursor += batch
        for grp in opt.param_groups:
            grp["lr"] = lr * min(1.0, (step + 1) / max(lr_warmup, 1))

        qi, qm, ti, tm = make_batch(idx, q_ids, t_ids, tok, q_seq, t_seq, rng, device)
        qv = mean_pool(mdl(input_ids=qi, attention_mask=qm), qm)
        tv = mean_pool(mdl(input_ids=ti, attention_mask=tm), tm)
        scores = (qv @ tv.T) * MNRL_SCALE
        labels = torch.arange(len(idx), device=device)
        loss = F.cross_entropy(scores, labels)
        loss.backward()
        if clip_grad and clip_grad > 0:
            gn = float(torch.nn.utils.clip_grad_norm_(mdl.parameters(), clip_grad))
            if gn > clip_grad:
                clipped += 1
            if gn > max_gn:
                max_gn = gn
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        tokens += qi.numel() + ti.numel()

        # Collapse watch. A collapsed encoder puts every embedding at the same
        # point, so all in-batch similarities are equal, the softmax is uniform,
        # the loss pins at log(batch) and accuracy falls to about 1/batch. Both
        # conditions are required: at step 1 the loss can exceed log(batch) while
        # accuracy is still healthy, and accuracy alone is noisy.
        with torch.no_grad():
            acc = (scores.argmax(1) == labels).float().mean().item()
        # The test is a BAND around log(batch), not a floor. A collapsed encoder
        # sits on log(batch) almost exactly, while an untrained one overshoots it
        # (step 1 of the real runs gave 3.51 to 4.31 against log(48) = 3.871), so
        # a floor would fire on a fresh model. Arming is also delayed past warmup,
        # because a model that has not yet moved off uniform is near log(batch) by
        # construction. Accuracy is logged but not used as a test: at small batch
        # sizes chance accuracy is too high for it to separate anything.
        uniform = math.log(len(idx))
        if step > max(lr_warmup, COLLAPSE_ARM_AFTER) and \
                abs(loss.item() - uniform) <= UNIFORM_TOL * uniform:
            dead += 1
        else:
            dead = 0
        if dead >= COLLAPSE_PATIENCE:
            collapsed_at = step
            log("")
            log("  " + "!" * 70)
            log(f"  COLLAPSE at step {step:,}: loss has sat on log(batch) = "
                f"{uniform:.4f} (last {loss.item():.4f}, in-batch acc {acc:.3f})")
            log(f"  for {COLLAPSE_PATIENCE} consecutive steps. The encoder has "
                "collapsed to a point and")
            log("  will not recover. Training stops here.")
            log(f"  Checkpoints already evaluated below step {step:,} are valid. "
                "Anything at or")
            log("  above it is not. The cell records collapsed_at_step.")
            log("  " + "!" * 70)
            log("")
            break

        if step % 25 == 0 or step == 1:
            log(f"    train {step:>5}  loss {loss.item():.4f}  in-batch acc {acc:.3f}  "
                f"{tokens / max(time.time() - t0, 1e-9):,.0f} tok/s")
        if step in checkpoints:
            on_checkpoint(step, mdl)

    rate = tokens / max(time.time() - t0, 1e-9)
    if clip_grad and clip_grad > 0:
        log(f"    gradient clipping engaged on {clipped:,} of {step:,} steps "
            f"({clipped / max(step, 1):.1%}); largest norm seen {max_gn:.2f} "
            f"against a limit of {clip_grad:g}")
        if clipped == 0:
            log("    clipping never engaged, so it cannot be what keeps this run "
                "alive; if a cell still collapses, lower --lr instead")
    del mdl, opt
    gc.collect()
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    return rate, collapsed_at, {"clipped_steps": clipped,
                                "max_grad_norm": max_gn}


# ------------------------------------------------------------------ reports ---


def cell_tag(scrub: bool) -> str:
    """Scrubbed and unscrubbed cells must not share a filename.

    Without this, running the scrub condition into a workdir that already holds
    the unscrubbed cells makes the resume logic SKIP them as complete and hand
    back the previous run's numbers, silently.
    """
    return "_scrubbed" if scrub else ""


def cell_path(workdir: Path, objective: str, train_on: str, scrub: bool = False) -> Path:
    return workdir / f"cell_{objective}_{train_on}{cell_tag(scrub)}.npz"


def save_cell(path: Path, objective, train_on, steps_done, rr, qhashes, rate, meta):
    """Per-query ranks plus summaries, rewritten after every checkpoint."""
    flat = {f"rr__{s}__{e}": v for s, per in rr.items() for e, v in per.items()}
    np.savez_compressed(
        path,
        meta=np.array(json.dumps({
            "objective": objective, "train_on": train_on,
            "steps_done": sorted(steps_done), "qhashes": qhashes,
            "tokens_per_s": rate, **meta}), dtype=object),
        **flat)


def load_cells(workdir: Path, scrub: bool = False) -> dict:
    """Only cells matching the requested scrub condition, so the two never mix."""
    out = {}
    for p in sorted(workdir.glob(f"cell_*{cell_tag(scrub)}.npz")):
        if not scrub and p.stem.endswith("_scrubbed"):
            continue
        try:
            z = np.load(p, allow_pickle=True)
            meta = json.loads(str(z["meta"].item()))
            rr = {}
            for k in z.files:
                if not k.startswith("rr__"):
                    continue
                _, s, e = k.split("__")
                rr.setdefault(int(s), {})[e] = z[k]
            out[(meta["objective"], meta["train_on"])] = {"meta": meta, "rr": rr}
        except Exception as e:
            log(f"  could not read {p.name}: {type(e).__name__}: {e}")
    return out


def report(cells: dict, eras: list[str]) -> None:
    print("\n" + "=" * 78)
    print("PAPER 21c — THE 2x2")
    print("=" * 78)
    if not cells:
        print("  nothing on disk yet")
        return

    collapsed = {k: cells[k].get("meta", {}).get("collapsed_at_step")
                 for k in cells}
    if any(collapsed.values()):
        print("\n  WARNING: one or more cells collapsed during training:")
        for (o, t), at in collapsed.items():
            if at:
                print(f"    {o} on {t}: collapsed at step {at:,}")
        print("  Contrasts are suppressed at and above the earliest collapse step,\n"
              "  because a contrast needs every cell it draws on to be alive.")

    for objective in ("mlm", "contrastive"):
        keys = [(o, t) for (o, t) in cells if o == objective]
        if not keys:
            continue
        floor = min([collapsed[k] for k in keys if collapsed.get(k)], default=None)
        print(f"\n{'-' * 78}\n{objective.upper()}\n{'-' * 78}")
        trained = sorted(t for _, t in keys)

        for _, t in sorted(keys):
            c = cells[(objective, t)]
            bad = c.get("meta", {}).get("collapsed_at_step")
            print(f"\n  trained on {t}:  MRR@10 by step")
            if bad:
                print(f"    *** THIS CELL COLLAPSED AT STEP {bad:,}. Rows at or above "
                      f"that step are\n    *** measurements of a dead encoder. Do not "
                      "use them, and do not use any\n    *** contrast computed from "
                      "them below.")
            for s in sorted(c["rr"]):
                row = f"    {s:>6}" + (" X" if bad and s >= bad else "  ")
                for e in eras:
                    if e in c["rr"][s]:
                        m, lo, hi = boot_mean(c["rr"][s][e])
                        row += f"   {e} {m:.4f} [{lo:.3f},{hi:.3f}]"
                print(row)

        if len(trained) < 2:
            print(f"\n  only the {trained[0]} cell is present; the within-era "
                  "difference needs both and is not computed")
            continue

        a, b = trained[0], trained[1]
        ca, cb = cells[(objective, a)], cells[(objective, b)]
        doses = sorted((set(ca["rr"]) & set(cb["rr"])) - {0})
        if not doses:
            print("\n  the two cells share no non-zero step; rerun one with "
                  "matching --mlm-checkpoints/--con-checkpoints")
            continue

        for e in eras:
            ha = ca["meta"]["qhashes"].get(e)
            hb = cb["meta"]["qhashes"].get(e)
            if ha and hb and ha != hb:
                print(f"\n  REFUSING to pair {e}: query lists differ between cells "
                      f"({ha} vs {hb}).")
                print("  The two cells evaluated different queries, so paired "
                      "differences would be meaningless.")
                return

        for d in doses:
            if floor is not None and d >= floor:
                print(f"\n  step {d:,} — SUPPRESSED: a cell in this contrast "
                      f"collapsed at step {floor:,}")
                continue
            print(f"\n  step {d:,} — each era under both training conditions")
            print("    trained on  " + "".join(f"{'eval ' + e:>22}" for e in eras))
            for t in (a, b):
                row = f"    {t:<11} "
                for e in eras:
                    v = cells[(objective, t)]["rr"][d].get(e)
                    row += f"{(np.mean(v) if v is not None else float('nan')):>22.4f}"
                print(row)

            print("\n    d(e) = own-trained minus other-trained, paired over queries:")
            signs = []
            for e in eras:
                own = cells[(objective, e)]["rr"][d].get(e)
                oth = cells[(objective, b if e == a else a)]["rr"][d].get(e)
                if own is None or oth is None or len(own) != len(oth):
                    print(f"      {e:<9} unavailable")
                    continue
                m, lo, hi = boot_mean(own - oth)
                signs.append((m, lo, hi))
                star = "" if lo <= 0 <= hi else "  *"
                print(f"      {e:<9} {m:+.4f}  [{lo:+.4f},{hi:+.4f}]{star}")
            if len(signs) == 2:
                combined = [s[0] for s in signs]
                print(f"      locality  {np.mean(combined):+.4f}   (mean of the two)")
                if all(lo > 0 for _, lo, _ in signs):
                    print("      -> both positive and excluding zero: locality")
                elif all(hi < 0 for _, _, hi in signs):
                    print("      -> both negative: each era does worse on its own data")
                elif (signs[0][0] > 0) != (signs[1][0] > 0) and \
                     (signs[0][1] > 0 or signs[1][1] > 0):
                    print("      -> signs disagree with one interval excluding zero:")
                    print("         the eras are not exchangeable; report separately, "
                          "do not pool")
                else:
                    print("      -> intervals include zero: no locality effect detected")

            print("\n    b(e) = own-trained minus unadapted, paired over queries:")
            for e in eras:
                own = cells[(objective, e)]["rr"][d].get(e)
                base = cells[(objective, e)]["rr"].get(0, {}).get(e)
                if own is None or base is None or len(own) != len(base):
                    print(f"      {e:<9} unavailable")
                    continue
                m, lo, hi = boot_mean(own - base)
                print(f"      {e:<9} {m:+.4f}  [{lo:+.4f},{hi:+.4f}]")


# --------------------------------------------------------------------- main ---


def main() -> int:
    p = argparse.ArgumentParser(description="Paper 21c complete 2x2, one overnight run")
    p.add_argument("--two-site", type=Path,
                   default=Path.home() / "projects" / "paper9" / "two_site.py")
    p.add_argument("--outdir", type=Path,
                   default=Path.home() / "projects" / "paper21a" / "out")
    p.add_argument("--workdir", type=Path,
                   default=Path.home() / "projects" / "paper21c_x")
    p.add_argument("--cells", default=DEFAULT_CELLS,
                   help=f"objective:era pairs. era may also be {MIXED_NAME!r}, which "
                        "trains on both eras pooled and balanced, for the retraining "
                        "case")
    p.add_argument("--eras", default="mimic3,mimic4")
    p.add_argument("--pool", type=int, default=1000, help="evaluation documents per era")
    p.add_argument("--train-pool", type=int, default=20000,
                   help="held-out notes used to build contrastive training pairs")
    p.add_argument("--mlm-steps", type=int, default=3000)
    p.add_argument("--mlm-checkpoints", default="1000,3000")
    p.add_argument("--mlm-batch", type=int, default=64)
    p.add_argument("--mlm-seq", type=int, default=128)
    p.add_argument("--head-warmup", type=int, default=300)
    p.add_argument("--con-steps", type=int, default=2500)
    p.add_argument("--con-checkpoints", default="250,500,1000,2500")
    p.add_argument("--con-batch", type=int, default=48,
                   help="pairs per step; this IS the in-batch negative count")
    p.add_argument("--q-seq", type=int, default=96)
    p.add_argument("--t-seq", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--clip-grad", type=float, default=1.0,
                   help="max gradient norm; 0 disables clipping. Without it a "
                        "single bad batch can blow the weights out and the encoder "
                        "collapses with no way back. Pass 0 to reproduce a run made "
                        "before clipping existed")
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--encode-batch", type=int, default=16)
    p.add_argument("--within", default=None,
                   help="restrict to one corpus and split it into <corpus>_early and "
                        "<corpus>_late by --split-year; holds corpus identity, EHR "
                        "generation and de-identification regime fixed so only time "
                        "varies. Use with --eras <corpus>_early,<corpus>_late")
    p.add_argument("--split-year", type=int, default=None,
                   help="year lower bound at or below which a note is 'early'")
    p.add_argument("--scrub-deid", action="store_true",
                   help="strip [**...**] de-identification placeholders from note text in "
                        "BOTH the training and evaluation pools, before the length filter; "
                        "tests whether the within-era asymmetry is a de-identification "
                        "artefact. Scrubbed cells are stored under their own filenames")
    p.add_argument("--mimic4-min-year", type=int, default=None,
                   help="drop MIMIC-IV admissions at or before this year to remove the "
                        "2008-2012 era overlap; warns and proceeds if no year column exists")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--head-source", default=HEAD_SOURCE)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--keep-checkpoints", action="store_true",
                   help="keep model directories after evaluating them (~440 MB each)")
    p.add_argument("--device", default=None, help="override mps/cuda/cpu autodetect")
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="2-minute end-to-end check: tiny pool, tiny doses, all four cells")
    a = p.parse_args()

    if (a.within is None) != (a.split_year is None):
        raise SystemExit("--within and --split-year must be given together")
    if a.within is not None and a.eras == "mimic3,mimic4":
        a.eras = f"{a.within}_early,{a.within}_late"
        log(f"--within {a.within}: eras default to {a.eras}")
    if a.within is not None and a.cells == DEFAULT_CELLS:
        a.cells = ",".join(f"{o}:{a.within}_{p}" for o in ("mlm", "contrastive")
                           for p in ("early", "late"))
    eras = [e.strip() for e in a.eras.split(",") if e.strip()]
    a.workdir.mkdir(parents=True, exist_ok=True)

    if a.report_only:
        report(load_cells(a.workdir, a.scrub_deid), eras)
        return 0

    if a.smoke:
        a.pool, a.train_pool = 24, 200
        a.mlm_steps, a.mlm_checkpoints, a.head_warmup = 4, "2,4", 2
        a.con_steps, a.con_checkpoints = 4, "2,4"
        a.mlm_batch, a.con_batch, a.warmup = 4, 4, 2
        a.workdir = a.workdir / "smoke"
        a.workdir.mkdir(parents=True, exist_ok=True)
        log("SMOKE TEST: tiny pools and doses. Numbers are meaningless; this only "
            "proves the pipeline runs end to end.")

    cells = []
    for spec in a.cells.split(","):
        spec = spec.strip()
        if not spec:
            continue
        if ":" not in spec:
            raise SystemExit(f"--cells entries look like objective:era, got {spec!r}")
        o, t = spec.split(":", 1)
        if o not in ("mlm", "contrastive"):
            raise SystemExit(f"unknown objective {o!r}; use mlm or contrastive")
        if t != MIXED_NAME and t not in eras:
            raise SystemExit(f"train-on era {t!r} is not in --eras {eras} "
                             f"(or {MIXED_NAME!r})")
        cells.append((o, t))

    import torch
    device = a.device or ("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    ts = load_two_site(a.two_site)
    log(f"query convention imported from {a.two_site}")
    log(f"device {device}; {len(cells)} cell(s): " +
        ", ".join(f"{o} on {t}" for o, t in cells))

    log("building cohort with a patient-level split")
    eval_pools, train_pools = load_cohort(a.outdir, a.pool, a.seed, a.mimic4_min_year,
                                          a.scrub_deid, a.within, a.split_year)
    for e in eras:
        if e not in eval_pools:
            raise SystemExit(f"era {e!r} not in cohort; have {sorted(eval_pools)}")

    if any(t == MIXED_NAME for _, t in cells):
        per = max(a.train_pool // len(eras), 1)
        g = np.random.default_rng(a.seed)
        merged = []
        for e in eras:
            src = train_pools.get(e, [])
            if not src:
                raise SystemExit(f"no held-out training notes for {e!r}; cannot "
                                 f"build the {MIXED_NAME} pool")
            k = min(len(src), per)
            merged += [src[i] for i in g.choice(len(src), k, replace=False)]
            log(f"  {MIXED_NAME} pool: {k:,} notes from {e}")
        g.shuffle(merged)
        train_pools[MIXED_NAME] = merged
        log(f"  {MIXED_NAME} pool total {len(merged):,} notes, balanced across "
            f"{', '.join(eras)}; evaluation is unchanged and covers every era")

    rng = np.random.default_rng(a.seed)
    pairs_by_era, hashes = {}, {}
    for era in eras:
        texts = eval_pools[era]
        take = min(len(texts), a.pool * 3)
        sel = [texts[i] for i in rng.choice(len(texts), take, replace=False)]
        pr = build_pairs(ts, sel)
        if len(pr) < a.pool:
            log(f"  WARNING: {era} yielded only {len(pr):,} usable pairs "
                f"(wanted {a.pool:,})")
        pr = pr[: a.pool]
        if len(pr) < 8:
            raise SystemExit(f"{era}: only {len(pr)} usable pairs; cannot evaluate")
        pairs_by_era[era] = pr
        hashes[era] = qhash(pr)
        qw = [len(q.split()) for _, q in pr]
        log(f"  {era}: {len(pr):,} pairs, median query {int(np.median(qw))} words, "
            f"hash {hashes[era]}")

    print("\n  QUERY PREVIEW — if these are boilerplate the run is worthless")
    for era, pr in pairs_by_era.items():
        print(f"  --- {era} ---")
        for _, q in pr[:3]:
            print(f"    {q[:140]}")
    print()

    log("evaluating the unadapted baseline (shared by every cell)")
    base_rr = evaluate(a.model, pairs_by_era, device, a.encode_batch, a.model)
    for e in eras:
        m, lo, hi = boot_mean(base_rr[e])
        log(f"  step     0  {e}  MRR@10 = {m:.4f}  [{lo:.4f}, {hi:.4f}]")

    mlm_ck = sorted({int(x) for x in a.mlm_checkpoints.split(",") if x.strip()})
    con_ck = sorted({int(x) for x in a.con_checkpoints.split(",") if x.strip()})

    failures = []
    for objective, train_on in cells:
        path = cell_path(a.workdir, objective, train_on, a.scrub_deid)
        want = mlm_ck if objective == "mlm" else con_ck
        if path.exists():
            try:
                done = json.loads(str(np.load(path, allow_pickle=True)["meta"].item()))["steps_done"]
                if set(want).issubset(set(done)):
                    log(f"SKIP {objective} on {train_on}: already complete on disk")
                    continue
                log(f"REDO {objective} on {train_on}: on disk but incomplete "
                    f"(has {sorted(done)}, wants {want})")
            except Exception as e:
                log(f"REDO {objective} on {train_on}: unreadable ({type(e).__name__})")

        log("")
        log("=" * 72)
        log(f"CELL: {objective} trained on {train_on}, tested on {', '.join(eras)}")
        log("=" * 72)
        t_cell = time.time()
        rr = {0: {e: base_rr[e] for e in eras}}
        steps_done = {0}

        def on_checkpoint(step, mdl, _o=objective, _t=train_on, _rr=rr,
                          _done=steps_done, _p=path):
            d = a.workdir / f"ck_{_o}_{_t}{cell_tag(a.scrub_deid)}_{step}"
            mdl.save_pretrained(d)
            was = mdl.training
            mdl.eval()
            try:
                per = evaluate(d, pairs_by_era, device, a.encode_batch, a.model)
            finally:
                if was:
                    mdl.train()
            _rr[step] = per
            _done.add(step)
            for e in eras:
                m, lo, hi = boot_mean(per[e])
                log(f"    step {step:>5}  {e}  MRR@10 = {m:.4f}  [{lo:.4f}, {hi:.4f}]")
            save_cell(_p, _o, _t, _done, _rr, hashes, 0.0,
                      {"pool": a.pool, "seed": a.seed, "model": a.model,
                       "mimic4_min_year": a.mimic4_min_year, "scrub_deid": a.scrub_deid,
                       "partial": True})
            if not a.keep_checkpoints:
                shutil.rmtree(d, ignore_errors=True)

        try:
            if objective == "mlm":
                rate = train_mlm(train_pools[train_on], on_checkpoint, a.mlm_steps,
                                 set(mlm_ck), a.mlm_batch, a.mlm_seq, a.lr, device,
                                 a.seed, a.model, a.head_source, a.head_warmup,
                                 a.warmup, a.clip_grad)
                collapsed_at, gstat = None, {}
            else:
                rate, collapsed_at, gstat = train_contrastive(
                    ts, train_pools[train_on], on_checkpoint,
                    a.con_steps, set(con_ck), a.con_batch,
                    a.q_seq, a.t_seq, a.lr, device, a.seed,
                    a.model, a.train_pool, a.warmup, a.clip_grad)
            save_cell(path, objective, train_on, steps_done, rr, hashes, rate,
                      {"pool": a.pool, "seed": a.seed, "model": a.model,
                       "steps": a.mlm_steps if objective == "mlm" else a.con_steps,
                       "mimic4_min_year": a.mimic4_min_year, "scrub_deid": a.scrub_deid,
                       "clip_grad": a.clip_grad, "collapsed_at_step": collapsed_at,
                       **gstat, "partial": False})
            log(f"  cell done in {(time.time() - t_cell) / 3600:.2f} h, "
                f"{rate:,.0f} tok/s -> {path.name}")
        except Exception as e:
            failures.append((objective, train_on, f"{type(e).__name__}: {e}"))
            log(f"  CELL FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            log("  continuing to the next cell; rerun the same command to retry this one")

    report(load_cells(a.workdir, a.scrub_deid), eras)

    if failures:
        print("\n  CELLS THAT FAILED (rerun the same command to retry them):")
        for o, t, msg in failures:
            print(f"    {o} on {t}: {msg}")
    else:
        print("\n  all requested cells completed")
    print(f"\n  per-query ranks and summaries in {a.workdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
