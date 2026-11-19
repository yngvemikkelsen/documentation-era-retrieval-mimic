#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "pyarrow"]
# ///
"""
Is the same document present in both corpora, after the era restriction?

    uv run paper21c_overlap_v4.py --min-year 2013 --restrict mimic4

No GPU. Under 15 minutes on 90,000 notes. Reads notes.parquet only.

WHY
---
MIMIC-III and MIMIC-IV overlap in time, roughly 2008-2012, and the patient
identifiers were regenerated between the two releases. A patient-level split
within each corpus therefore establishes nothing about independence BETWEEN the
corpora: the same admission can appear on the training side of one and the
evaluation side of the other, and no identifier reveals it.

Exact text matching cannot find this. The two releases were de-identified by
different pipelines -- bracketed placeholders in one, blanks in the other -- so
one source note becomes two different strings. The comparison has to be made on
normalised text, and it has to tolerate the edits de-identification makes.

WHAT IT DOES
------------
Each note is reduced to the set of hashes of its overlapping 5-word shingles,
after stripping de-identification placeholders, lowercasing and collapsing
whitespace. Near-duplicate detection then uses a bottom-k sketch: the k smallest
shingle hashes of a document are a random sample of its shingles, so the overlap
between two documents' sketches estimates their Jaccard similarity, and an
inverted index over sketch values finds candidate pairs without comparing every
note to every other note.

Candidates are then rescored exactly, on full shingle sets, so the numbers
reported are true Jaccard similarities and not sketch estimates.

THREE THINGS ARE REPORTED, AND THE THIRD IS THE ONE THAT MATTERS
----------------------------------------------------------------
  cross-corpus    the strongest matches between the two corpora. These are the
                  contamination candidates.
  random pairs    the same statistic on randomly drawn cross-corpus pairs, which
                  gives the scale of ordinary clinical-prose similarity. A top
                  match near this level is not a duplicate.
  within-corpus   the strongest matches INSIDE one corpus, as a positive
                  control. Clinical corpora contain amended and re-filed notes,
                  so if the method finds nothing here either, it is blind rather
                  than reassuring, and the cross-corpus null means nothing.

WHY v3
------
v2 proved the unrestricted corpora share source documents: 115 pairs at Jaccard
0.50 or above, against zero within MIMIC-III. The fix is to restrict MIMIC-IV to
admissions after MIMIC-III ends, so no admission can be in both databases. This
version applies that restriction and then has to show the overlap is GONE.

Showing an absence needs a control that can produce a presence, and v2's control
cannot: it topped out at 0.4980 with zero pairs above threshold. Its sensitivity
was demonstrated only by the real cross-corpus hits, which the restriction
removes. So v3 adds a planted-duplicate control. Known documents are copied from
one corpus into the other side of the search, degraded to three similarity
levels, and the recovery rate is reported at each. A null accompanied by a high
recovery rate on planted duplicates is evidence. A null on its own is not.

v4 EXITS NON-ZERO WHEN THE CHECK FAILS
--------------------------------------
v3 printed a verdict and returned 0 whatever it said, so a run script could
print "STILL present" and then spend hours training on contaminated data. v4
returns:

    0   clean: no cross-corpus duplicate, and the planted control recovered
        enough to make that absence evidence
    2   cross-corpus duplicates are still present
    3   no duplicates found, but the planted control is too weak to support a
        null (recovery below --min-recall at the hardest level)
    4   the planted control was skipped, so no null can be supported

A calling script must stop on anything but 0. Exit 3 and 4 are failures on
purpose: an unsupported absence is not a clean result, it is an unmeasured one.

v3 also fixes a verdict bug in v1 and v2. The branch order tested the control
before the cross-corpus result, so a control at 0.4980 printed "UNINFORMATIVE"
over a confirmed positive at 0.9155. The control gates a NULL; a cross-corpus
hit is its own proof that the method detects duplication.
"""

from __future__ import annotations

import argparse
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

YEAR_COLS = ("year_group", "real_year_lower", "anchor_year_group", "anchor_year",
             "admit_year", "admityear")
DEID_RE = re.compile(r"\[\*\*.*?\*\*\]")
WORD_RE = re.compile(r"[a-z0-9]+")
SHINGLE = 5
SKETCH_K = 64
B = np.uint64(1000003)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def shingles(text: str, vocab: dict) -> np.ndarray:
    """Hashes of overlapping 5-word shingles of the normalised text."""
    words = WORD_RE.findall(DEID_RE.sub(" ", text).lower())
    if len(words) < SHINGLE:
        return np.empty(0, dtype=np.uint64)
    ids = np.empty(len(words), dtype=np.uint64)
    for i, w in enumerate(words):
        v = vocab.get(w)
        if v is None:
            v = vocab[w] = len(vocab)
        ids[i] = v
    n = len(ids) - SHINGLE + 1
    h = np.zeros(n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(SHINGLE):
            h = h * B + ids[j:j + n]
    return np.unique(h)


def sketch(h: np.ndarray, k: int = SKETCH_K) -> np.ndarray:
    """The k smallest hashes: a hash-ordered random sample of the shingles."""
    if len(h) <= k:
        return h
    return np.partition(h, k)[:k]


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    inter = np.intersect1d(a, b, assume_unique=True).size
    return inter / (len(a) + len(b) - inter)


def build(texts: list[str], vocab: dict, label: str):
    """Only the sketches are retained. Keeping every note's full shingle set
    would be about 1.4 GB at 90,000 notes; the few documents that need an exact
    score are re-shingled on demand instead."""
    sk = []
    for i, t in enumerate(texts):
        sk.append(sketch(shingles(t, vocab)))
        if (i + 1) % 20000 == 0:
            log(f"    {label}: {i + 1:,} notes shingled")
    return sk


def candidates(sk_a, sk_b, min_shared: int, max_postings: int):
    """Documents in B that share at least min_shared sketch values with a
    document in A. The inverted index is built over A.

    Shingles occurring in more than max_postings documents are dropped. A
    5-word sequence present in thousands of notes is boilerplate: it carries no
    evidence of duplication, and leaving it in makes the search quadratic in the
    size of its postings list."""
    index = defaultdict(list)
    for i, s in enumerate(sk_a):
        for v in s.tolist():
            index[v].append(i)
    dropped = [v for v, lst in index.items() if len(lst) > max_postings]
    for v in dropped:
        del index[v]
    if dropped:
        log(f"    dropped {len(dropped):,} boilerplate shingle(s) over "
            f"{max_postings:,} postings")
    out = defaultdict(int)
    for j, s in enumerate(sk_b):
        hit: dict[int, int] = defaultdict(int)
        for v in s.tolist():
            for i in index.get(v, ()):
                hit[i] += 1
        for i, c in hit.items():
            if c >= min_shared:
                out[(i, j)] = c
    return out


def report(name: str, pairs, sh_a, sh_b, texts_a, texts_b, top: int,
           thr: float = 0.5, show: int = 0, export=None):
    """Exact Jaccard for every candidate pair.

    v1 called the shingler inside the comparison, so a document appearing in k
    pairs was shingled k times; at half a million pairs that dominated the run.
    Here each distinct document is shingled once and reused."""
    if not pairs:
        print(f"\n  {name}: no candidate pairs at all")
        return []
    ca = {i for i, _ in pairs}
    cb = {j for _, j in pairs}
    log(f"    rescoring {len(pairs):,} pair(s) over {len(ca):,}+{len(cb):,} "
        "distinct documents")
    HA = {i: sh_a(i) for i in ca}
    HB = {j: sh_b(j) for j in cb}
    allscored = sorted(((jaccard(HA[i], HB[j]), i, j) for (i, j) in pairs),
                       reverse=True)
    scored = allscored[:top]
    dup = [t for t in allscored if t[0] >= thr]
    print(f"\n  {name}: {len(pairs):,} candidate pair(s); {len(dup):,} at "
          f"Jaccard >= {thr:.2f}")
    if dup:
        print(f"    distinct documents involved: {len({i for _, i, _ in dup}):,} "
              f"on the left, {len({j for _, _, j in dup}):,} on the right")
    print("    " + "rank".rjust(5) + "Jaccard".rjust(10) + "  excerpt of the "
          "left-hand note")
    for r, (jv, i, j) in enumerate(scored, 1):
        ex = WORD_RE.sub(lambda m: m.group(0), texts_a[i][:90].replace("\n", " "))
        print(f"    {r:>5}{jv:>10.4f}  {ex}")
    for t in (0.90, 0.80, 0.50, 0.30):
        print(f"    pairs at Jaccard >= {t:.2f}: {sum(1 for v, _, _ in allscored if v >= t):,}")

    for r in range(min(show, len(scored))):
        jv, i, j = scored[r]
        print("\n    " + "-" * 86)
        print(f"    PAIR {r + 1}, Jaccard {jv:.4f} — read these and decide whether "
              "they are the same admission")
        print("    " + "-" * 86)
        for lbl, t in (("LEFT ", texts_a[i]), ("RIGHT", texts_b[j])):
            body = " ".join(t.split())[:1100]
            print(f"    [{lbl}] {body}")
            print()

    if export is not None and dup:
        import csv
        with open(export, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["jaccard", "left_index", "right_index"])
            for jv, i, j in dup:
                w.writerow([f"{jv:.4f}", i, j])
        print(f"    wrote {len(dup):,} pair(s) to {export}")
    return [t[0] for t in allscored[:top]]


def main() -> int:
    p = argparse.ArgumentParser(description="Cross-corpus near-duplicate detection")
    p.add_argument("--outdir", type=Path,
                   default=Path.home() / "projects" / "paper21a" / "out")
    p.add_argument("--notes-name", default="notes.parquet")
    p.add_argument("--eras", default="mimic3,mimic4")
    p.add_argument("--min-shared", type=int, default=24,
                   help="sketch values two notes must share to be rescored. 24 of "
                        "64 screens at an estimated Jaccard near 0.38. v1 used 8, "
                        "which screens near 0.13 and admitted half a million pairs "
                        "on real clinical prose, none of which could be duplicates")
    p.add_argument("--dup-threshold", type=float, default=0.50,
                   help="exact Jaccard at or above which a pair is counted a "
                        "near-duplicate")
    p.add_argument("--show", type=int, default=2,
                   help="print the full text of this many top pairs, side by side, "
                        "so the verdict can be checked by eye")
    p.add_argument("--export", type=Path, default=None,
                   help="write every pair at or above --dup-threshold to CSV, so "
                        "the affected notes can be removed from one corpus")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--max-postings", type=int, default=1000,
                   help="drop shingles occurring in more than this many notes; "
                        "they are boilerplate and make the search quadratic")
    p.add_argument("--min-year", type=int, default=None,
                   help="drop notes of --restrict whose year lower bound is at or "
                        "below this. Must match --mimic4-min-year in the cells")
    p.add_argument("--restrict", default="mimic4",
                   help="the era --min-year applies to")
    p.add_argument("--min-recall", type=float, default=0.80,
                   help="planted-duplicate recovery required at the hardest level "
                        "before a null is accepted as evidence")
    p.add_argument("--plant", type=int, default=200,
                   help="documents copied from era 1 into the era 2 side of the "
                        "search, as a positive control. 0 disables it")
    p.add_argument("--sample", type=int, default=0,
                   help="notes per corpus, for a quick first pass; 0 = all")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    eras = [e.strip() for e in a.eras.split(",") if e.strip()]
    if len(eras) != 2:
        raise SystemExit("--eras needs exactly two")
    f = a.outdir / a.notes_name
    if not f.exists():
        raise SystemExit(f"missing {f}")
    notes = pd.read_parquet(f)
    if "corpus" not in notes.columns or "text" not in notes.columns:
        raise SystemExit(f"notes needs corpus and text; has {sorted(notes.columns)}")

    if a.min_year is not None:
        cf = a.outdir / "cohort.csv"
        if not cf.exists():
            raise SystemExit(f"--min-year needs {cf} for the year column")
        cohort = pd.read_csv(cf, low_memory=False)
        col = next((c for c in YEAR_COLS if c in cohort.columns), None)
        if col is None:
            raise SystemExit(
                f"--min-year {a.min_year} requested but none of {YEAR_COLS} is in "
                f"cohort.csv, which has {sorted(cohort.columns)}. Refusing to "
                "proceed: an unrestricted run would look like a clean result.")
        if "hadm_id" not in notes.columns:
            raise SystemExit("notes.parquet needs hadm_id to apply --min-year")
        yr = cohort[["hadm_id", col]].drop_duplicates("hadm_id", keep="first").copy()
        yr["_y"] = yr[col].astype(str).str.extract(r"(\d{4})")[0].astype(float)
        n0 = len(notes)
        notes = notes.merge(yr[["hadm_id", "_y"]], on="hadm_id", how="left")
        if len(notes) != n0:
            raise SystemExit("the year merge changed the note count; hadm_id is not "
                             "unique in cohort.csv")
        nan = int(((notes["corpus"] == a.restrict) & notes["_y"].isna()).sum())
        if nan:
            log(f"  {nan:,} {a.restrict} notes have no parsable year; DROPPING them, "
                "because a note of unknown date cannot be shown to postdate the "
                "other era")
            notes = notes[~((notes["corpus"] == a.restrict) & notes["_y"].isna())]
        drop = (notes["corpus"] == a.restrict) & (notes["_y"] <= a.min_year)
        log(f"  --min-year {a.min_year} via '{col}': dropped {int(drop.sum()):,} "
            f"{a.restrict} notes")
        notes = notes[~drop].drop(columns=["_y"])

    rng = np.random.default_rng(a.seed)
    texts = {}
    for e in eras:
        t = notes.loc[notes["corpus"] == e, "text"].astype(str).tolist()
        if not t:
            raise SystemExit(f"no notes for era {e!r}")
        if a.sample and a.sample < len(t):
            t = [t[i] for i in rng.choice(len(t), a.sample, replace=False)]
        texts[e] = t
        log(f"{e}: {len(t):,} notes")

    vocab: dict[str, int] = {}
    log("shingling (de-identification stripped, lowercased, whitespace collapsed)")
    sk = {}
    for e in eras:
        sk[e] = build(texts[e], vocab, e)

    def sh(era):
        return lambda i: shingles(texts[era][i], vocab)
    log(f"  vocabulary {len(vocab):,} distinct words")

    print("\n" + "=" * 92)
    print("CROSS-CORPUS NEAR-DUPLICATE SEARCH")
    print("=" * 92)

    log("indexing and searching across corpora")
    cross = candidates(sk[eras[0]], sk[eras[1]], a.min_shared, a.max_postings)
    top_cross = report(f"{eras[0]} against {eras[1]}", cross,
                       sh(eras[0]), sh(eras[1]), texts[eras[0]],
                       texts[eras[1]], a.top, a.dup_threshold, a.show, a.export)

    log("scoring random cross-corpus pairs, for scale")
    n = min(2000, len(texts[eras[0]]), len(texts[eras[1]]))
    ia = rng.choice(len(texts[eras[0]]), n, replace=False)
    ib = rng.choice(len(texts[eras[1]]), n, replace=False)
    rnd = np.array([jaccard(shingles(texts[eras[0]][i], vocab),
                            shingles(texts[eras[1]][j], vocab))
                    for i, j in zip(ia, ib)])
    print(f"\n  random cross-corpus pairs (n={n:,}): median {np.median(rnd):.4f}, "
          f"99th pct {np.percentile(rnd, 99):.4f}, max {rnd.max():.4f}")

    print("\n" + "=" * 92)
    print("PLANTED-DUPLICATE CONTROL — can this search find duplication at all?")
    print("=" * 92)
    plant_rates = {}
    if a.plant:
        src = texts[eras[0]]
        pick = rng.choice(len(src), min(a.plant, len(src)), replace=False)
        for lvl, frac in (("verbatim", 1.00), ("75% retained", 0.75),
                          ("55% retained", 0.55)):
            inj = []
            for i in pick:
                w = src[i].split()
                keep_n = max(int(len(w) * frac), SHINGLE + 1)
                inj.append(DEID_RE.sub("___", " ".join(w[:keep_n])))
            side = texts[eras[1]] + inj
            first = len(texts[eras[1]])
            sk_side = [sketch(shingles(t, vocab)) for t in side]
            cand = candidates(sk[eras[0]], sk_side, a.min_shared, a.max_postings)
            found = set()
            for (i, j) in cand:
                if j >= first and pick[j - first] == i:
                    if jaccard(shingles(src[i], vocab),
                               shingles(side[j], vocab)) >= a.dup_threshold:
                        found.add(i)
            rate = len(found) / len(pick)
            plant_rates[lvl] = rate
            print(f"  {lvl:<14} {len(found):>4} of {len(pick):,} planted duplicates "
                  f"recovered at Jaccard >= {a.dup_threshold:.2f}   ({100 * rate:.1f}%)")
            del sk_side, cand
        print("\n  The 55% row is the one that matters: it is the hardest case this")
        print("  search is asked to find. A null above is only as strong as that rate.")
    else:
        print("  skipped (--plant 0)")

    log("secondary control: searching for duplicates WITHIN the first corpus")
    half = len(sk[eras[0]]) // 2
    within = candidates(sk[eras[0]][:half], sk[eras[0]][half:], a.min_shared,
                        a.max_postings)
    t0, t1 = texts[eras[0]][:half], texts[eras[0]][half:]
    top_within = report(f"{eras[0]} against itself (positive control)", within,
                        lambda i: shingles(t0[i], vocab),
                        lambda j: shingles(t1[j], vocab), t0, t1, a.top,
                        a.dup_threshold, 0, None)

    print("\n" + "=" * 92)
    print("VERDICT")
    print("=" * 92)
    cmax = max(top_cross) if top_cross else 0.0
    wmax = max(top_within) if top_within else 0.0
    rmax = float(rnd.max())
    hard = plant_rates.get("55% retained")
    print(f"\n  restriction applied             "
          f"{('--min-year ' + str(a.min_year) + ' on ' + a.restrict) if a.min_year is not None else 'NONE'}")
    print(f"  strongest cross-corpus match    {cmax:.4f}")
    print(f"  strongest within-corpus match   {wmax:.4f}   (secondary control)")
    print(f"  strongest random pair           {rmax:.4f}   (scale of ordinary prose)")
    if hard is not None:
        print(f"  planted duplicates recovered    {100 * hard:.1f}%   (hardest level, "
              "55% retained)")
    print()
    rc = 0
    if cmax >= a.dup_threshold:
        rc = 2
    elif hard is None:
        rc = 4
    elif hard < a.min_recall:
        rc = 3
    if cmax >= a.dup_threshold:
        print("  Cross-corpus near-duplicates are STILL present. The restriction has")
        print("  not removed the overlap. Read the printed pairs: if they are the same")
        print("  admission, the cut year is too early or the year column mislabels")
        print("  them. Do not run the cells on this.")
    elif hard is None:
        print("  No cross-corpus duplicate was found, but the planted control was")
        print("  skipped (--plant 0), so nothing shows this search can detect")
        print("  duplication on these notes. The absence is unmeasured, not clean.")
    elif hard < a.min_recall:
        print(f"  No cross-corpus duplicate was found, but the search recovered only")
        print(f"  {100 * hard:.1f}% of planted duplicates at the hardest level, below the")
        print(f"  {100 * a.min_recall:.0f}% required. The null is weak: lower --min-shared and repeat")
        print("  before reporting an absence.")
    elif hard is not None:
        print("  No cross-corpus near-duplicate, and the search recovers planted")
        print(f"  duplicates at {100 * hard:.1f}% in the hardest case. The absence is")
        print("  evidence. This is the sentence the manuscript can carry, with the")
        print("  recovery rate reported beside it.")
    print("\n  Neither outcome addresses patient recurrence: the same patient admitted")
    print("  in both periods appears in both corpora, shares no text, and cannot be")
    print("  identified because MIMIC-III's real years were destroyed at")
    print("  de-identification. That is a limitation to state, not a thing to detect.")
    print("\n" + "=" * 92)
    print(f"  EXIT {rc} — " + {
        0: "PASS. Safe to train on this restriction.",
        2: "FAIL. Duplicates remain; do not train on this.",
        3: "FAIL. The null is not supported by the control.",
        4: "FAIL. No control was run, so the null is unmeasured.",
    }[rc])
    print("=" * 92)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
