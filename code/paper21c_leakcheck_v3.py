#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "pyarrow"]
# ///
"""
How many of the 1,000 evaluated documents were in the other era's training set?

    uv run paper21c_leakcheck_v3.py --two-site ~/projects/paper9/two_site.py

No GPU, no model, no training. Reads notes.parquet, cohort.csv and the stored
cells.

WHY THIS AND NOT A RE-RUN
-------------------------
paper21c_overlap_v2.py established that MIMIC-III and MIMIC-IV share source
documents: 115 pairs at Jaccard 0.50 or above, against zero within MIMIC-III
under identical settings. That is a property of the two corpora.

It is not yet a property of THIS experiment. Contamination can only bias a
result if a contaminated document was actually evaluated. The cells evaluate
1,000 documents per era, drawn from a patient-disjoint evaluation pool, and the
training sets hold 20,000 notes drawn from the complementary pool. A leak
requires one specific coincidence: a document that is among era A's 1,000
evaluated targets AND whose twin is among era B's 20,000 training notes.

This script counts those documents exactly. The count, not an argument about
probability, is what decides whether the cells have to be re-run.

WHY THE RECONSTRUCTION CAN BE TRUSTED
-------------------------------------
Nothing here re-implements the experiment's sampling. The crossover script is
imported and its own load_cohort, build_pairs and qhash are called, with the
same seed, in the same order. The evaluation rng is created once and consumed in
era order exactly as the crossover script consumes it; the training rng is
created fresh per era exactly as prep_contrastive creates it.

The reconstruction is then CHECKED, not assumed. Each cell stores a SHA-256
fingerprint of its query list. If the reconstructed fingerprint does not match
the stored one, the run aborts and reports the mismatch: a reconstruction that
is not byte-identical to the experiment cannot be used to audit it.

WHAT IT MEASURES, AND WHAT IT DOES NOT
--------------------------------------
Measured: each evaluated target against every training note of the other era,
by the same near-duplicate detector, imported from paper21c_overlap_v2.py.

Not measured: patient-level overlap that shares no text. The same patient can
appear in both corpora through different admissions. No text comparison can see
that, and only a date restriction excludes it.

Conservative on the training side: the training notes are compared as whole
notes, not as the query-stripped targets the model actually saw. Whole notes
contain more text, so a match is easier to find, not harder.

EXIT CODE (new in v2)
---------------------
    0   no evaluated document was in the other era's training set
    2   at least one was, so the cells carry contamination

v1 always returned 0, so a run script could not act on the result. A calling
script must stop on 2: the count is the whole point of running this.

READING THE RESULT
------------------
  0 contaminated targets   The corpora overlap; this experiment did not sample
                           any of it. Nothing to re-run. Report the count.
  few contaminated targets Report the count, and the bound it puts on the
                           effect: one query moving from rank 11+ to rank 1
                           moves a 1,000-query mean by 0.001.
  many                     Re-run with those documents removed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_module(path: Path, name: str):
    """Import a sibling script so its functions are reused, not re-implemented."""
    if not path.exists():
        raise SystemExit(
            f"{name} not found at {path}\n"
            f"Put paper21c_crossover_v5.py and paper21c_overlap_v2.py in the same "
            f"directory as this script, or pass --crossover / --overlap.")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    p = argparse.ArgumentParser(
        description="Exact count of evaluated documents present in the other "
                    "era's training set")
    p.add_argument("--two-site", type=Path,
                   default=Path.home() / "projects" / "paper9" / "two_site.py")
    p.add_argument("--crossover", type=Path,
                   default=HERE / "paper21c_crossover_v5.py")
    p.add_argument("--overlap", type=Path, default=HERE / "paper21c_overlap_v4.py",
                   help="the detector whose functions are reused. v2 default in "
                        "earlier versions was a trap: this must be the SAME "
                        "detector the run gated on, or the audit screens "
                        "differently from the check that cleared the corpus")
    p.add_argument("--outdir", type=Path,
                   default=Path.home() / "projects" / "paper21a" / "out")
    p.add_argument("--workdir", type=Path,
                   default=Path.home() / "projects" / "paper21c_v5")
    p.add_argument("--eras", default="mimic3,mimic4")
    p.add_argument("--objective", default="contrastive")
    p.add_argument("--pool", type=int, default=1000,
                   help="must equal the --pool the cells were run with")
    p.add_argument("--train-pool", type=int, default=20000,
                   help="must equal the --train-pool the cells were run with")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mimic4-min-year", type=int, default=None)
    p.add_argument("--scrub-deid", action="store_true")
    p.add_argument("--min-shared", type=int, default=8,
                   help="sketch values shared before an exact score. 8 of 64 is "
                        "an estimated Jaccard near 0.13, far below the 0.50 that "
                        "counts as a duplicate. It is affordable here because "
                        "1,000 x 20,000 is a hundredfold smaller search than the "
                        "full corpora, and a low screen makes the census complete")
    p.add_argument("--dup-threshold", type=float, default=0.50)
    p.add_argument("--max-postings", type=int, default=1000)
    p.add_argument("--show", type=int, default=2,
                   help="print this many contaminated targets in full, if any")
    p.add_argument("--export", type=Path, default=None,
                   help="CSV of contaminated query indices, for the exclusion re-run")
    a = p.parse_args()

    eras = [e.strip() for e in a.eras.split(",") if e.strip()]
    if len(eras) != 2:
        raise SystemExit("--eras needs exactly two")

    X = load_module(a.crossover, "paper21c_crossover_v5")
    O = load_module(a.overlap, "paper21c_overlap_v2")

    log("loading two_site.py and the cohort")
    ts = X.load_two_site(a.two_site)
    eval_pools, train_pools = X.load_cohort(
        a.outdir, a.pool, a.seed, a.mimic4_min_year, a.scrub_deid, None, None)
    for e in eras:
        if e not in eval_pools:
            raise SystemExit(f"no evaluation pool for era {e!r}; "
                             f"have {sorted(eval_pools)}")

    # ---- reconstruct the evaluated targets, in the crossover script's order ----
    log("rebuilding the evaluation pairs (two_site narrative extraction; this is "
        "the slow step and its rate is not known in advance)")
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    pairs_by_era, hashes = {}, {}
    for era in eras:
        texts = eval_pools[era]
        take = min(len(texts), a.pool * 3)
        sel = [texts[i] for i in rng.choice(len(texts), take, replace=False)]
        pr = X.build_pairs(ts, sel)[: a.pool]
        pairs_by_era[era] = pr
        hashes[era] = X.qhash(pr)
        log(f"  {era}: {len(pr):,} pairs, hash {hashes[era]}")
    log(f"  pair building took {time.time() - t0:.0f} s")

    # ---- prove the reconstruction is the experiment's query list --------------
    print("\n" + "=" * 92)
    print("RECONSTRUCTION CHECK")
    print("=" * 92)
    checked = 0
    for era in eras:
        f = a.workdir / f"cell_{a.objective}_{era}.npz"
        if not f.exists():
            print(f"  {f.name}: missing, cannot check against it")
            continue
        meta = json.loads(str(np.load(f, allow_pickle=True)["meta"].item()))
        for e2 in eras:
            stored = meta.get("qhashes", {}).get(e2)
            if not stored:
                continue
            ok = stored == hashes[e2]
            print(f"  {f.name}  {e2}: stored {stored}  rebuilt {hashes[e2]}  "
                  f"{'MATCH' if ok else 'MISMATCH'}")
            if not ok:
                raise SystemExit(
                    "The rebuilt query list is not the one the cell evaluated. The "
                    "audit cannot proceed: --pool, --seed, --mimic4-min-year, "
                    "--scrub-deid or notes.parquet differs from the run. Fix the "
                    "flags to match the cell's meta and repeat.")
            checked += 1
    if not checked:
        raise SystemExit("No stored query hash to check against; refusing to "
                         "audit an unverified reconstruction.")
    print(f"\n  {checked} hash comparison(s), all matching. The documents below are "
          "the ones the cells scored.")

    # ---- the census -----------------------------------------------------------
    print("\n" + "=" * 92)
    print("EVALUATED TARGETS FOUND IN THE OTHER ERA'S TRAINING SET")
    print("=" * 92)

    vocab: dict[str, int] = {}
    summary, rows = {}, []
    for era in eras:
        other = [x for x in eras if x != era][0]
        targets = [t for t, _ in pairs_by_era[era]]

        # the other era's training notes, selected as prep_contrastive selects them
        tt = train_pools[other]
        take = min(len(tt), a.train_pool)
        tsel = [tt[i] for i in np.random.default_rng(a.seed).choice(
            len(tt), take, replace=False)]

        log(f"{era} targets ({len(targets):,}) against {other} training notes "
            f"({len(tsel):,})")
        sk_t = [O.sketch(O.shingles(x, vocab)) for x in targets]
        sk_r = O.build(tsel, vocab, f"{other} train")
        cand = O.candidates(sk_t, sk_r, a.min_shared, a.max_postings)

        scored = O.report(f"{era} evaluated targets against {other} training notes",
                          cand, lambda i: O.shingles(targets[i], vocab),
                          lambda j: O.shingles(tsel[j], vocab),
                          targets, tsel, 20, a.dup_threshold, a.show, None)

        hit = sorted({i for (i, j) in cand
                      if O.jaccard(O.shingles(targets[i], vocab),
                                   O.shingles(tsel[j], vocab)) >= a.dup_threshold})
        summary[era] = hit
        rows += [(era, i) for i in hit]
        print(f"\n  >>> {era}: {len(hit)} of {len(targets):,} evaluated targets are "
              f"near-duplicates of a {other} training note")
        if hit:
            print(f"      query indices: {hit}")
        del sk_t, sk_r, cand, scored

    # ---- what it means --------------------------------------------------------
    print("\n" + "=" * 92)
    print("VERDICT")
    print("=" * 92)
    tot = sum(len(v) for v in summary.values())
    for era in eras:
        n = len(summary[era])
        bound = n / max(a.pool, 1)
        print(f"\n  {era}: {n} contaminated target(s) of {a.pool:,}")
        print(f"    largest possible effect on a mean reciprocal rank: {bound:.4f}")
        print(f"    (every contaminated query moving from unretrieved to rank 1)")
    if tot == 0:
        print("\n  No evaluated document was in the other era's training set.")
        print("  The corpora do share source documents, but this experiment did not")
        print("  sample any of that overlap into a position where it could bias a")
        print("  contrast. No cell needs re-running. Report the count and the method.")
    else:
        print(f"\n  {tot} contaminated target(s) in total. Compare each era's bound")
        print("  above with the effect it would have to explain. A bound far below")
        print("  the confidence interval of the contrast means the leak cannot")
        print("  account for it, and the honest report is the count plus the bound.")
        print("  A bound of the same order as the effect means re-running with these")
        print("  documents removed is the only defensible course.")
    print("\n  Neither outcome addresses patient-level overlap without shared text.")
    print("  Only a date restriction on MIMIC-IV excludes that.")

    rc = 2 if tot else 0
    print("\n" + "=" * 92)
    print(f"  EXIT {rc} — " + ("PASS. No evaluated document was in the other era's "
                               "training set." if rc == 0 else
                               f"FAIL. {tot} contaminated evaluated target(s)."))
    print("=" * 92)

    if a.export and rows:
        import csv
        with open(a.export, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["era", "query_index"])
            w.writerows(rows)
        print(f"\n  wrote {len(rows)} row(s) to {a.export}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
