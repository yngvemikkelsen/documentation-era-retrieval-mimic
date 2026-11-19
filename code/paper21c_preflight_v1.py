#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "pyarrow"]
# ///
"""
Check every assumption the run depends on, BEFORE the run starts.

    uv run paper21c_preflight_v1.py --two-site ~/projects/paper9/two_site.py \
        --emit /tmp/paper21c_v8.env

About 15 minutes. No GPU. Writes only the --emit file.

WHAT IT CHECKS, AND WHY EACH ONE CAN RUIN THE RUN
-------------------------------------------------
1. Every helper script is present and importable.
   A missing one fails nine hours in, not now.

2. No target workdir already holds a cell file.
   paper21c_crossover_v5.py decides to skip a cell with
   set(want).issubset(set(done)) -- CHECKPOINT STEPS ONLY. It never compares
   --model, --mimic4-min-year, --train-pool or --pool. So a PubMedBERT run into
   a workdir holding mpnet cells prints "SKIP: already complete" and returns the
   mpnet numbers under the PubMedBERT label, silently. This is the single most
   dangerous failure in the chain and the only defence is a clean workdir.

3. The real training-pool size of every arm, from load_cohort itself.
   --train-pool is one global number and prep_contrastive takes
   min(len(pool), train_pool). If one arm holds fewer notes than the flag, that
   arm trains on less data and "own beats cross" is confounded with volume. The
   flag must be below every arm's pool, so it is computed here, not guessed.

4. The realised pair count of every arm, by calling build_pairs for real.
   Counting notes is not enough: build_pairs drops notes with no extractable
   query and targets under MIN_TARGET. The only way to know 1,000 pairs are
   reachable is to build them. This is why the script takes 15 minutes.

5. Whether the scrubbed condition changes the pools.
   --scrub-deid removes characters, so notes can fall under MIN_CHARS and leave
   the cohort. The scrubbed and unscrubbed arms therefore get ONE shared
   train-pool value, the minimum over all four, or the de-identification
   contrast is confounded by training volume.

6. Which split year the within-MIMIC-IV control can actually use.
   Both candidates are evaluated and the feasible one with the larger balanced
   pool is chosen. The within control gets its own train-pool: it is compared
   against itself, not against the main 2x2.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not restrict MIMIC-IV for the within-MIMIC-IV control. That control uses
MIMIC-IV only -- _split_within drops every other corpus -- so no MIMIC-III note
is ever present and cross-corpus contamination is impossible. Restricting it
would shrink an 11-year span to 5 years and cost the control its power.

OUTPUT
------
KEY=value lines to --emit, for the run script to source, and a non-zero exit on
any failure. Nothing is written if a check fails.
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
NEEDED = ("paper21c_crossover_v5.py", "paper21c_overlap_v4.py",
          "paper21c_leakcheck_v3.py", "paper21c_margin_v1.py",
          "paper21c_verify_v1.py",
          "paper21c_mixed_v1.py")


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Preflight for the Paper 21 v8 run")
    p.add_argument("--two-site", type=Path,
                   default=Path.home() / "projects" / "paper9" / "two_site.py")
    p.add_argument("--outdir", type=Path,
                   default=Path.home() / "projects" / "paper21a" / "out")
    p.add_argument("--workdirs", default="",
                   help="comma-separated target workdirs that must be free of cells")
    p.add_argument("--pool", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-year", type=int, default=2013)
    p.add_argument("--split-years", default="2013,2016")
    p.add_argument("--emit", type=Path, default=None)
    p.add_argument("--allow-occupied", action="store_true",
                   help="accept workdirs that already hold cell files. Only pass "
                        "this after paper21c_verify_v1.py --allow-missing has "
                        "confirmed those cells match the configuration about to be "
                        "run; otherwise the crossover script will SKIP them on "
                        "checkpoint steps alone and return their old numbers")
    p.add_argument("--skip-pairs", action="store_true",
                   help="skip check 4 (the 15-minute part). Only for debugging: it "
                        "removes the only real proof that 1,000 pairs exist")
    a = p.parse_args()

    fail: list[str] = []
    out: dict[str, str] = {}

    # ---- 1. helper scripts ---------------------------------------------------
    print("=" * 92)
    print("1. HELPER SCRIPTS")
    print("=" * 92)
    for n in NEEDED:
        ok = (HERE / n).exists()
        print(f"  {'OK  ' if ok else 'MISSING'}  {n}")
        if not ok:
            fail.append(f"missing helper script {n}")
    if not a.two_site.exists():
        print(f"  MISSING  {a.two_site}")
        fail.append(f"missing {a.two_site}")
    else:
        print(f"  OK       {a.two_site}")
    if fail:
        print("\nPREFLIGHT FAILED before anything else could be checked.")
        for f in fail:
            print(f"  {f}")
        return 1

    spec = importlib.util.spec_from_file_location(
        "paper21c_crossover_v5", HERE / "paper21c_crossover_v5.py")
    X = importlib.util.module_from_spec(spec)
    sys.modules["paper21c_crossover_v5"] = X
    spec.loader.exec_module(X)
    print(f"  crossover imported: MIN_CHARS={X.MIN_CHARS}, MIN_TARGET={X.MIN_TARGET}, "
          f"TOP_K={X.TOP_K}")

    # ---- 2. workdirs must be free of cells -----------------------------------
    print("\n" + "=" * 92)
    print("2. TARGET WORKDIRS (the silent-skip trap)")
    print("=" * 92)
    wds = [Path(w.strip()).expanduser() for w in a.workdirs.split(",") if w.strip()]
    if not wds:
        print("  none given; pass --workdirs to check them")
    for w in wds:
        cells = sorted(w.glob("cell_*.npz")) if w.exists() else []
        if cells:
            print(f"  OCCUPIED  {w}")
            for c in cells:
                try:
                    m = json.loads(str(np.load(c, allow_pickle=True)["meta"].item()))
                    print(f"              {c.name}  model={m.get('model')}  "
                          f"min_year={m.get('mimic4_min_year')}  steps_done="
                          f"{m.get('steps_done')}")
                except Exception as e:
                    print(f"              {c.name}  (unreadable: {type(e).__name__})")
            if a.allow_occupied:
                print(f"              --allow-occupied: accepted as a resume")
            else:
                fail.append(
                    f"{w} already holds {len(cells)} cell file(s); the crossover "
                    "script would SKIP those cells on checkpoint steps alone and "
                    "return their old numbers. Either delete the workdir, or "
                    "confirm the cells match with paper21c_verify_v1.py "
                    "--allow-missing and re-run with --allow-occupied")
        else:
            print(f"  FREE      {w}")

    # ---- 3 to 6. pools, pairs, and the chosen flags --------------------------
    log("loading two_site.py")
    ts = X.load_two_site(a.two_site)

    def pools(min_year, scrub, within=None, split_year=None):
        ev, tr = X.load_cohort(a.outdir, a.pool, a.seed, min_year, scrub,
                               within, split_year)
        return ev, tr

    def pair_count(texts):
        if a.skip_pairs:
            return None
        take = min(len(texts), a.pool * 3)
        rng = np.random.default_rng(a.seed)
        sel = [texts[i] for i in rng.choice(len(texts), take, replace=False)]
        return len(X.build_pairs(ts, sel))

    print("\n" + "=" * 92)
    print(f"3-5. CROSS-CORPUS CONFIGURATIONS, --mimic4-min-year {a.min_year}")
    print("=" * 92)
    print("  the scrubbed and unscrubbed arms share one --train-pool, or the")
    print("  de-identification contrast is confounded by training volume\n")
    print(f"  {'condition':<12}{'era':<10}{'eval notes':>12}{'train notes':>13}"
          f"{'pairs built':>13}")
    cross_train: list[int] = []
    for label, scrub in (("plain", False), ("scrubbed", True)):
        ev, tr = pools(a.min_year, scrub)
        for era in sorted(ev):
            n_pairs = pair_count(ev[era])
            cross_train.append(len(tr[era]))
            shown = "skipped" if n_pairs is None else f"{n_pairs:,}"
            print(f"  {label:<12}{era:<10}{len(ev[era]):>12,}{len(tr[era]):>13,}"
                  f"{shown:>13}")
            if n_pairs is not None and n_pairs < a.pool:
                fail.append(f"{label}/{era} built only {n_pairs:,} pairs, "
                            f"below --pool {a.pool:,}")
    tp_cross = (min(cross_train) // 1000) * 1000
    if tp_cross < a.pool:
        fail.append(f"cross-corpus train pool {tp_cross:,} is below --pool")
    print(f"\n  smallest arm {min(cross_train):,} notes  ->  "
          f"--train-pool {tp_cross:,}")
    out["TRAINPOOL_CROSS"] = str(tp_cross)

    print("\n" + "=" * 92)
    print("6. WITHIN-MIMIC-IV CONTROL — no restriction, full 2008-2019 span")
    print("=" * 92)
    print("  MIMIC-IV only, so no MIMIC-III note is present and cross-corpus")
    print("  contamination is impossible. Restricting it here would cost it power.\n")
    print(f"  {'split':<8}{'era':<16}{'eval notes':>12}{'train notes':>13}"
          f"{'pairs built':>13}{'verdict':>10}")
    best = None
    for sy in [int(x) for x in a.split_years.split(",")]:
        try:
            ev, tr = pools(None, False, "mimic4", sy)
        except SystemExit as e:
            print(f"  {sy:<8}unusable: {e}")
            continue
        rows, ok = [], True
        for era in sorted(ev):
            n_pairs = pair_count(ev[era])
            rows.append((era, len(ev[era]), len(tr[era]), n_pairs))
            if n_pairs is not None and n_pairs < a.pool:
                ok = False
        tp = (min(r[2] for r in rows) // 1000) * 1000
        if tp < a.pool:
            ok = False
        for era, ne, nt, np_ in rows:
            shown = "skipped" if np_ is None else f"{np_:,}"
            print(f"  {sy:<8}{era:<16}{ne:>12,}{nt:>13,}{shown:>13}"
                  f"{('ok' if ok else 'FAIL'):>10}")
        print(f"  {'':<8}{'-> --train-pool':<16}{tp:>12,}")
        if ok and (best is None or tp > best[1]):
            best = (sy, tp)
    if best is None:
        fail.append("no within-MIMIC-IV split year is feasible at this --pool")
    else:
        print(f"\n  chosen: --split-year {best[0]}  --train-pool {best[1]:,}")
        out["SPLITYEAR"] = str(best[0])
        out["TRAINPOOL_WITHIN"] = str(best[1])

    # ---- verdict -------------------------------------------------------------
    print("\n" + "=" * 92)
    if fail:
        print("PREFLIGHT FAILED — nothing was written, nothing should be run")
        print("=" * 92)
        for f in fail:
            print(f"  {f}")
        return 1
    print("PREFLIGHT PASSED")
    print("=" * 92)
    for k, v in out.items():
        print(f"  {k}={v}")
    if a.emit:
        a.emit.write_text("".join(f"{k}={v}\n" for k, v in out.items()))
        print(f"\n  wrote {a.emit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
