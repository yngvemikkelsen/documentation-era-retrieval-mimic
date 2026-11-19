#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
What does training on both eras at once cost, and what does it save?

    uv run paper21c_mixed_v1.py --workdir ~/projects/paper21c_v5 \
        --mixdir ~/projects/paper21c_v5_mix

Seconds. Reads stored cells only: no cohort, no two_site.py, no pair rebuild.

WHY A SEPARATE SCRIPT
---------------------
The mixed cell lives in its own workdir, so the crossover report sees it alone
and says so: "only the mixed cell is present; the within-era difference needs
both". The three contrasts that matter all pair the mixed cell against cells in
the main workdir, so they have to be computed across the two directories.

Both workdirs must come from the same configuration, or the comparison is across
training trajectories rather than across training sets. The query hashes are
checked and the run aborts if they differ.

THE THREE CONTRASTS, on each era's records
------------------------------------------
    mixed - unadapted     the absolute benefit of training on everything
    own   - mixed         what matched-era data still buys over mixed
    mixed - cross         what keeping the other era in the mix saves you
                          against discarding it

The deployment reading is the last two together. A site retraining holds old and
new notes. It can train on both, or on recent notes only. "own - mixed" is the
price of mixing; "mixed - cross" is the price of discarding. If the second is
much larger than the first, keep everything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

N_BOOT = 2000
SEED = 42
KS = (1, 5, 10)


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


def boot(x, n=N_BOOT, seed=SEED):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    g = np.random.default_rng(seed)
    bs = x[g.integers(0, len(x), size=(n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def hit_at(rr, k: int) -> np.ndarray:
    """1 where the target reached rank k or better; rr is 1/rank inside TOP_K."""
    return (np.asarray(rr, np.float64) >= (1.0 / k) - 1e-6).astype(np.float64)


def fmt(v, lo, hi) -> str:
    return f"{v:+.4f}{'*' if not lo <= 0 <= hi else ' '} [{lo:+.3f},{hi:+.3f}]"


def main() -> int:
    p = argparse.ArgumentParser(description="Mixed-era training against the 2x2")
    p.add_argument("--workdir", type=Path,
                   default=Path.home() / "projects" / "paper21c_v5")
    p.add_argument("--mixdir", type=Path,
                   default=Path.home() / "projects" / "paper21c_v5_mix")
    p.add_argument("--eras", default="mimic3,mimic4")
    p.add_argument("--objective", default="contrastive")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    eras = [e.strip() for e in a.eras.split(",") if e.strip()]
    if len(eras) != 2:
        raise SystemExit("--eras needs exactly two")

    cells = {}
    for e in eras:
        f = a.workdir / f"cell_{a.objective}_{e}.npz"
        if not f.exists():
            raise SystemExit(f"missing {f}")
        cells[e] = load_cell(f)
    fm = a.mixdir / f"cell_{a.objective}_mixed.npz"
    if not fm.exists():
        raise SystemExit(f"missing {fm}")
    cells["mixed"] = load_cell(fm)

    print("=" * 96)
    print("MIXED-ERA TRAINING — the retraining case")
    print("=" * 96)

    # same configuration, or the comparison is across trajectories
    print("\nconfiguration check")
    ref = None
    for name, (meta, _) in cells.items():
        bad = meta.get("collapsed_at_step")
        print(f"  {name:<8} clip_grad {meta.get('clip_grad', '?')}  "
              f"clipped {meta.get('clipped_steps', '?')}  "
              f"{'COLLAPSED at ' + format(bad, ',') if bad else 'alive'}")
        if bad:
            raise SystemExit(f"{name} collapsed during training; it cannot be used")
        cfg = (meta.get("clip_grad"), meta.get("model"), meta.get("scrub_deid"))
        if ref is None:
            ref = (name, cfg)
        elif cfg != ref[1]:
            raise SystemExit(
                f"{name} was run under a different configuration than {ref[0]} "
                f"({cfg} vs {ref[1]}); the comparison would be across trajectories, "
                "not across training sets")

    for e in eras:
        hs = {n: cells[n][0]["qhashes"].get(e) for n in cells}
        if len({h for h in hs.values() if h}) > 1:
            raise SystemExit(f"query hashes differ for {e}: {hs}; the cells evaluated "
                             "different queries, so paired differences are meaningless")
    print(f"  query hashes match across all three cells")

    doses = sorted(set.intersection(*[set(cells[n][1]) for n in cells]) - {0})
    results: dict = {}

    for e in eras:
        other = [x for x in eras if x != e][0]
        base = np.asarray(cells[e][1][0][e], np.float64)
        print("\n" + "=" * 96)
        print(f"{e} RECORDS")
        print("=" * 96)
        print(f"  own = trained on {e}   cross = trained on {other}   "
              f"mixed = trained on both")
        print("\n  " + "dose".rjust(6) + "mixed - unadapted".rjust(26)
              + "own - mixed".rjust(26) + "mixed - cross".rjust(26))
        for s in doses:
            own = np.asarray(cells[e][1][s][e], np.float64)
            cro = np.asarray(cells[other][1][s][e], np.float64)
            mix = np.asarray(cells["mixed"][1][s][e], np.float64)
            row = f"  {s:>6,}"
            for lbl, v in (("abs", mix - base), ("cost", own - mix),
                           ("save", mix - cro)):
                m, lo, hi = boot(v)
                row += fmt(m, lo, hi).rjust(26)
                results[f"{e}|{s}|{lbl}"] = {"d": m, "lo": lo, "hi": hi}
            print(row)

        s = doses[-1]
        own = np.asarray(cells[e][1][s][e], np.float64)
        cro = np.asarray(cells[other][1][s][e], np.float64)
        mix = np.asarray(cells["mixed"][1][s][e], np.float64)
        print(f"\n  recall at step {s:,}")
        print("  " + "k".rjust(5) + "unadapted".rjust(11) + "cross".rjust(10)
              + "mixed".rjust(10) + "own".rjust(10) + "   "
              + "own - mixed".rjust(25) + "mixed - cross".rjust(25))
        for k in KS:
            hb, hc, hm, ho = (hit_at(x, k) for x in (base, cro, mix, own))
            c1 = boot(ho - hm)
            c2 = boot(hm - hc)
            print(f"  {k:>5}{hb.mean():>11.4f}{hc.mean():>10.4f}{hm.mean():>10.4f}"
                  f"{ho.mean():>10.4f}   {fmt(*c1):>25}{fmt(*c2):>25}")
            results[f"{e}|r{k}|cost"] = {"d": c1[0], "lo": c1[1], "hi": c1[2]}
            results[f"{e}|r{k}|save"] = {"d": c2[0], "lo": c2[1], "hi": c2[2]}

    print("\n" + "=" * 96)
    print("READING IT")
    print("=" * 96)
    print("""
  Compare the last two columns. "own - mixed" is what you give up by training on
  everything instead of only the matched era. "mixed - cross" is what you gain by
  keeping the other era instead of discarding it.

  If "mixed - cross" is much the larger, keep every note you hold when you
  retrain. If the two are similar, the training set barely matters and the
  simpler policy wins.

  A negative "own - mixed" would mean mixed beats matched on that era's own
  records, which would be a stronger result than the deployment rule needs: more
  data helping even off-era. Check its interval before claiming it.

  The recall table is there because MRR is rank-sensitive. A cost that is present
  in MRR and gone by recall@10 is a reordering inside the retrieved set, which a
  reranker absorbs. A cost that persists at recall@10 means documents leave the
  set and nothing downstream recovers them.

  The configuration check at the top is not a formality. The mixed cell and the
  2x2 cells must come from the same script and flags, or the difference measures
  the trajectory rather than the training set.
""")
    if a.out:
        a.out.write_text(json.dumps(results, indent=2, default=float))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
