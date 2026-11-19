#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "pyarrow"]
# ///
"""
Patient-clustered intervals and a BM25 lexical baseline, from one reconstruction.

    uv run paper21c_robust_v2.py --two-site ~/projects/paper9/two_site.py

About 5 minutes. No GPU. Reads notes.parquet, cohort.csv and the stored cells.

WHY BOTH IN ONE SCRIPT
----------------------
Each needs the same thing first: the exact 1,000 (target, query) pairs the cells
were scored on. Rebuilding those costs about 160 seconds per era, so doing the
two analyses separately would pay that twice.

ANALYSIS 1 — PATIENT-CLUSTERED INTERVALS
----------------------------------------
The bootstrap in the main analysis resamples queries, which treats them as
independent. They are not. The patient-level split separates training from
evaluation, but it does nothing about repetition WITHIN the evaluation set: in
this cohort 35.3% of MIMIC-III notes and 32.2% of MIMIC-IV notes belong to
patients who contribute more than one note, so one patient can supply several
queries and those queries share a writer, an admission history and a
documentation style.

The companion study on acuity-stratified retrieval reports patient-clustered
standard errors for this reason. This script recomputes b, c, d and the
asymmetry D with a cluster bootstrap: patients are resampled with replacement
and all of a drawn patient's queries enter together, so the interval carries
the within-patient correlation instead of assuming it away.

Clustering usually widens intervals. If a contrast survives it, the contrast is
not an artefact of treating correlated queries as independent. If one does not,
that must be reported.

ANALYSIS 2 — BM25 LEXICAL BASELINE
----------------------------------
Nothing in the main design separates what contrastive adaptation buys from what
simple lexical overlap already provides. A reviewer can reasonably ask whether
a model that has learnt this corpus is beating a bag of words at all, and the
companion benchmark and stratified studies both report BM25 for exactly that
reason.

BM25 is computed over the same 1,000 candidate documents per era, scoring the
same 1,000 queries, with the standard Okapi parameters k1=1.5 and b=0.75. It
needs no training, so it is identical across every cell of the design and acts
as a fixed reference line: a lexical floor that neither adaptation nor era
mismatch can move.

HOW THE QUERY-TO-PATIENT MAPPING IS MADE SAFE
---------------------------------------------
build_pairs discards the identity of the note each pair came from. Rather than
re-implement its selection rule, this script calls the SAME narrative_query on
the same texts in the same order: the notes that yield a query are exactly the
notes that yield a pair, in order, so the k-th pair belongs to the k-th such
note. The reconstruction is then checked against the SHA-256 query fingerprint
stored in each cell, and the run aborts if it differs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
N_BOOT = 2000
SEED = 42
KS = (1, 5, 10)
WORD = re.compile(r"[a-z0-9]+")


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_module(path: Path, name: str):
    if not path.exists():
        raise SystemExit(f"missing {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_cell(path: Path):
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"].item()))
    rr = {}
    for k in z.files:
        if not k.startswith("rr__"):
            continue
        _, s, e = k.split("__")
        rr.setdefault(int(s), {})[e] = z[k]
    return meta, rr


def hit_at(rr, k: int) -> np.ndarray:
    return (np.asarray(rr, np.float64) >= (1.0 / k) - 1e-6).astype(np.float64)


def boot_plain(x, n=N_BOOT, seed=SEED):
    x = np.asarray(x, np.float64)
    g = np.random.default_rng(seed)
    bs = x[g.integers(0, len(x), size=(n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def boot_cluster(x, groups, n=N_BOOT, seed=SEED):
    """Resample CLUSTERS with replacement; every member of a drawn cluster enters."""
    x = np.asarray(x, np.float64)
    idx = defaultdict(list)
    for i, g_ in enumerate(groups):
        idx[g_].append(i)
    keys = list(idx)
    members = [np.asarray(idx[k]) for k in keys]
    g = np.random.default_rng(seed)
    out = np.empty(n)
    for b in range(n):
        pick = g.integers(0, len(keys), size=len(keys))
        out[b] = x[np.concatenate([members[j] for j in pick])].mean()
    return float(x.mean()), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def star(lo, hi) -> str:
    return "*" if not lo <= 0 <= hi else " "


def fmt(v, lo, hi) -> str:
    return f"{v:+.4f}{star(lo, hi)}[{lo:+.3f},{hi:+.3f}]"


# ---------------------------------------------------------------- BM25 ------
def bm25_rr(docs: list[str], queries: list[str], k1=1.5, b=0.75, top_k=10):
    """Okapi BM25 over the candidate set. Query i's target is document i."""
    toks = [WORD.findall(d.lower()) for d in docs]
    dl = np.array([len(t) for t in toks], np.float64)
    avgdl = dl.mean() if len(dl) else 1.0
    N = len(toks)
    tf = [Counter(t) for t in toks]
    df = Counter()
    for t in tf:
        df.update(t.keys())
    idf = {w: math.log(1.0 + (N - c + 0.5) / (c + 0.5)) for w, c in df.items()}
    post = defaultdict(list)
    for j, c in enumerate(tf):
        for w, f in c.items():
            post[w].append((j, f))
    rr = np.zeros(len(queries), np.float32)
    for i, q in enumerate(queries):
        sc = np.zeros(N, np.float64)
        for w in set(WORD.findall(q.lower())):
            if w not in post:
                continue
            iw = idf[w]
            for j, f in post[w]:
                sc[j] += iw * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl[j] / avgdl))
        kk = min(top_k, N - 1)
        t = np.argpartition(-sc, kk)[: kk + 1]
        for r, j in enumerate(t[np.argsort(-sc[t])][:top_k]):
            if j == i:
                rr[i] = 1.0 / (r + 1)
                break
    return rr


def main() -> int:
    p = argparse.ArgumentParser(description="Clustered intervals and a BM25 floor")
    p.add_argument("--two-site", type=Path,
                   default=Path.home() / "projects" / "paper9" / "two_site.py")
    p.add_argument("--crossover", type=Path, default=HERE / "paper21c_crossover_v5.py")
    p.add_argument("--outdir", type=Path,
                   default=Path.home() / "projects" / "paper21a" / "out")
    p.add_argument("--workdir", type=Path,
                   default=Path.home() / "projects" / "paper21c_v8")
    p.add_argument("--eras", default="mimic3,mimic4")
    p.add_argument("--objective", default="contrastive")
    p.add_argument("--pool", type=int, default=1000)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--mimic4-min-year", type=int, default=2013)
    p.add_argument("--export-bm25", type=Path, default=None,
                   help="save the BM25 per-query reciprocal ranks, so this "
                        "never has to be recomputed")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    eras = [e.strip() for e in a.eras.split(",") if e.strip()]
    X = load_module(a.crossover, "paper21c_crossover_v5")
    ts = X.load_two_site(a.two_site)

    # ---- text -> subject_id, so a query can be attributed to a patient -------
    notes = pd.read_parquet(a.outdir / "notes.parquet")
    cohort = pd.read_csv(a.outdir / "cohort.csv", low_memory=False)
    if "subject_id" not in notes.columns:
        notes = notes.merge(cohort[["hadm_id", "subject_id"]].drop_duplicates("hadm_id"),
                            on="hadm_id", how="left")
    dup = int(notes["text"].duplicated().sum())
    if dup:
        log(f"  note: {dup:,} duplicate note texts; the first subject_id is used for each")
    t2s = dict(zip(notes["text"], notes["subject_id"]))

    eval_pools, _ = X.load_cohort(a.outdir, a.pool, a.seed, a.mimic4_min_year,
                                  False, None, None)

    cells = {}
    for e in eras:
        f = a.workdir / f"cell_{a.objective}_{e}.npz"
        if not f.exists():
            raise SystemExit(f"missing {f}")
        cells[e] = load_cell(f)

    log("rebuilding the evaluated pairs and attributing each to a patient")
    rng = np.random.default_rng(a.seed)
    pairs, subj = {}, {}
    for era in eras:
        texts = eval_pools[era]
        take = min(len(texts), a.pool * 3)
        sel = [texts[i] for i in rng.choice(len(texts), take, replace=False)]
        pr = X.build_pairs(ts, sel)[: a.pool]
        h = X.qhash(pr)
        stored = cells[era][0].get("qhashes", {}).get(era)
        if stored and stored != h:
            raise SystemExit(f"{era}: rebuilt hash {h} != stored {stored}; the flags do "
                             "not match the run and the audit cannot proceed")
        # the notes that yield a query are the notes that yield a pair, in order
        df = ts.build_df(sel)
        src = [t for t in sel if ts.narrative_query(t, df)][: a.pool]
        if len(src) != len(pr):
            raise SystemExit(f"{era}: {len(src)} source notes for {len(pr)} pairs; the "
                             "mapping is not one-to-one and must not be used")
        pairs[era] = pr
        subj[era] = [t2s.get(t, -1) for t in src]
        miss = sum(1 for s in subj[era] if s == -1)
        n_pat = len(set(subj[era]))
        rep = len(subj[era]) - n_pat
        log(f"  {era}: hash {h} OK; {n_pat:,} patients for {len(pr):,} queries "
            f"({rep} queries share a patient with another){'; ' + str(miss) + ' unmapped' if miss else ''}")
        if miss:
            raise SystemExit("some queries could not be attributed to a patient")

    doses = sorted((set(cells[eras[0]][1]) & set(cells[eras[1]][1])) - {0})
    top = doses[-1]
    results = {}

    # ---- 1. clustered vs unclustered ---------------------------------------
    print("\n" + "=" * 104)
    print(f"1. PATIENT-CLUSTERED INTERVALS, step {top:,}")
    print("=" * 104)
    print("  clusters are patients; every query of a drawn patient enters together\n")
    for era in eras:
        other = [x for x in eras if x != era][0]
        base = np.asarray(cells[era][1][0][era], np.float64)
        own = np.asarray(cells[era][1][top][era], np.float64)
        cross = np.asarray(cells[other][1][top][era], np.float64)
        g = subj[era]
        print(f"  {era} records")
        print("    " + "metric".ljust(11) + "contrast".ljust(10)
              + "by query".rjust(26) + "by patient".rjust(26) + "   width x")
        for mname, fn in (("MRR", None), *[(f"recall@{k}", k) for k in KS]):
            B, O, C = (base, own, cross) if fn is None else \
                      (hit_at(base, fn), hit_at(own, fn), hit_at(cross, fn))
            for cname, v in (("b", O - B), ("c", C - B), ("d", O - C)):
                pm, plo, phi = boot_plain(v)
                cm, clo, chi = boot_cluster(v, g)
                wx = (chi - clo) / (phi - plo) if phi > plo else float("nan")
                print(f"    {mname:<11}{cname:<10}{fmt(pm, plo, phi):>26}"
                      f"{fmt(cm, clo, chi):>26}   {wx:.2f}")
                results[f"{era}|{mname}|{cname}"] = {
                    "query": [pm, plo, phi], "patient": [cm, clo, chi]}
        print()

    # D across eras, clustered on both sides
    print("  the asymmetry D = d(earlier) - d(later), clustered in both eras")
    print("    " + "metric".ljust(11) + "by query".rjust(26) + "by patient".rjust(26))
    e0, e1 = eras
    for mname, fn in (("MRR", None), *[(f"recall@{k}", k) for k in KS]):
        def dvec(era, other):
            o = cells[era][1][top][era]; c = cells[other][1][top][era]
            return (np.asarray(o, np.float64) - np.asarray(c, np.float64)) if fn is None \
                else hit_at(o, fn) - hit_at(c, fn)
        de, dl = dvec(e0, e1), dvec(e1, e0)
        g0 = np.random.default_rng(SEED)
        pm = de.mean() - dl.mean()
        bp = (de[g0.integers(0, len(de), size=(N_BOOT, len(de)))].mean(1)
              - dl[g0.integers(0, len(dl), size=(N_BOOT, len(dl)))].mean(1))
        # clustered: resample patients independently within each era
        def cl(x, gr, seed):
            idx = defaultdict(list)
            for i, k in enumerate(gr):
                idx[k].append(i)
            keys = list(idx); mem = [np.asarray(idx[k]) for k in keys]
            r = np.random.default_rng(seed); o = np.empty(N_BOOT)
            for b in range(N_BOOT):
                pick = r.integers(0, len(keys), size=len(keys))
                o[b] = x[np.concatenate([mem[j] for j in pick])].mean()
            return o
        bc = cl(de, subj[e0], SEED) - cl(dl, subj[e1], SEED + 1)
        print(f"    {mname:<11}"
              f"{fmt(pm, np.percentile(bp,2.5), np.percentile(bp,97.5)):>26}"
              f"{fmt(pm, np.percentile(bc,2.5), np.percentile(bc,97.5)):>26}")
        results[f"D|{mname}"] = {"query": [pm, float(np.percentile(bp,2.5)), float(np.percentile(bp,97.5))],
                                 "patient": [pm, float(np.percentile(bc,2.5)), float(np.percentile(bc,97.5))]}

    # ---- 2. BM25 floor ------------------------------------------------------
    print("\n" + "=" * 104)
    print("2. BM25 LEXICAL BASELINE on the same candidate sets")
    print("=" * 104)
    print("  Okapi BM25, k1=1.5, b=0.75; no training, so identical in every cell\n")
    print("  " + "era".ljust(9) + "BM25 MRR".rjust(11) + "unadapted".rjust(11)
          + "cross".rjust(10) + "own".rjust(10) + "   " + "BM25 r@10".rjust(11)
          + "unadapted".rjust(11) + "cross".rjust(10) + "own".rjust(10))
    for era in eras:
        other = [x for x in eras if x != era][0]
        docs = [d for d, _ in pairs[era]]
        qs = [q for _, q in pairs[era]]
        log(f"  scoring BM25 on {era} ({len(docs):,} documents)")
        rr = bm25_rr(docs, qs)
        base = np.asarray(cells[era][1][0][era], np.float64)
        own = np.asarray(cells[era][1][top][era], np.float64)
        cross = np.asarray(cells[other][1][top][era], np.float64)
        print(f"  {era:<9}{rr.mean():>11.4f}{base.mean():>11.4f}{cross.mean():>10.4f}"
              f"{own.mean():>10.4f}   {hit_at(rr,10).mean():>11.4f}"
              f"{hit_at(base,10).mean():>11.4f}{hit_at(cross,10).mean():>10.4f}"
              f"{hit_at(own,10).mean():>10.4f}")
        results[f"{era}|bm25"] = {"mrr": float(rr.mean()),
                                  "r10": float(hit_at(rr, 10).mean())}
        print(f"\n    every model minus BM25 on {era} records, patient-clustered")
        print("    " + "metric".ljust(11) + "unadapted - BM25".rjust(26)
              + "cross - BM25".rjust(26) + "own - BM25".rjust(26))
        for mname, fn in (("MRR", None), *[(f"recall@{k}", k) for k in KS]):
            R = rr if fn is None else hit_at(rr, fn)
            row = f"    {mname:<11}"
            for nm, v in (("unadapted", base), ("cross", cross), ("own", own)):
                V = np.asarray(v, np.float64) if fn is None else hit_at(v, fn)
                m, lo, hi = boot_cluster(V - R, subj[era])
                row += fmt(m, lo, hi).rjust(26)
                results[f"{era}|bm25|{mname}|{nm}"] = [m, lo, hi]
            print(row)
        if a.export_bm25:
            np.save(a.export_bm25.with_suffix(f".{era}.npy"), rr)
            print(f"    saved BM25 per-query reciprocal ranks for {era}")
        print()

    print("\n" + "=" * 104)
    print("READING IT")
    print("=" * 104)
    print("""
  Clustering: compare the two interval columns. 'width x' above 1 means the
  patient-clustered interval is wider, which is the expected direction. A
  contrast starred by query but not by patient does not survive the correction
  and must be reported as such. Report the clustered intervals as primary and
  say so in the Methods; the companion acuity study already does this.

  BM25: it is a floor, not a competitor. If the adapted encoder sits above it,
  adaptation is buying something a bag of words does not already have. If BM25
  sits near the adapted models, the retrieval task is largely lexical and the
  deployment claims need rewriting around that. Either way the number belongs in
  the manuscript, because both companion studies report it and its absence is
  the first thing a reviewer of those papers will look for.
""")
    if a.out:
        a.out.write_text(json.dumps(results, indent=2, default=float))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
