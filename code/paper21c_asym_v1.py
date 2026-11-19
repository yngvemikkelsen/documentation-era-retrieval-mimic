#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
Put a confidence interval on the asymmetry itself.

    uv run paper21c_asym_v1.py --workdir ~/projects/paper21c_v8

Seconds. Reads stored cells only. No GPU, no training, no pair rebuild.

WHY THIS IS NEEDED
------------------
The paper's claim is not that d(earlier) > 0 and d(later) > 0. It is that
d(earlier) is LARGER than d(later): the asymmetry. Reporting the two shortfalls
with separate intervals and observing that one is bigger is not a test of that
claim, because two intervals that both exclude zero can still be compatible with
equal underlying values. The difference needs its own interval.

    D = d(earlier) - d(later)

HOW IT IS RESAMPLED, AND WHY NOT PAIRED
---------------------------------------
Every contrast inside one era is paired: the same 1,000 queries are scored by
every model, so b, c and d are means of per-query differences and the bootstrap
resamples those queries.

D is different. d(earlier) is computed on the 1,000 earlier-era queries and
d(later) on the 1,000 later-era queries. These are disjoint sets of documents
from different corpora, so the two estimates are INDEPENDENT, not paired. The
bootstrap therefore resamples each era's queries separately and takes the
difference of the two resampled means. Pairing them by index would impose a
correspondence between query 7 of one corpus and query 7 of the other that does
not exist.

The consequence is that D's interval is wider than either d's: it carries the
sampling variance of both. That is the honest width for this claim.

WHAT IS REPORTED
----------------
At every dose, for mean reciprocal rank and for recall at 1, 3, 5 and 10:

    d(earlier)   d(later)   D = difference   95% interval on D   ratio

A D whose interval excludes zero establishes the asymmetry as such. A D whose
interval includes zero means the two shortfalls cannot be distinguished at that
dose or that k, whatever their separate intervals do.

WHAT IT DOES NOT ADDRESS
------------------------
Bootstrap intervals express query sampling variance only. They say nothing
about variance between training runs, which needs repeated runs at different
seeds and is a separate and much more expensive exercise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

N_BOOT = 2000
SEED = 42
KS = (1, 3, 5, 10)


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


def boot_diff(x, y, n=N_BOOT, seed=SEED):
    """D = mean(x) - mean(y), with x and y INDEPENDENT samples.

    Each is resampled separately at its own length. Returns the point estimate
    and the 2.5/97.5 percentiles of the resampled difference."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    g = np.random.default_rng(seed)
    bx = x[g.integers(0, len(x), size=(n, len(x)))].mean(1)
    by = y[g.integers(0, len(y), size=(n, len(y)))].mean(1)
    bd = bx - by
    return (float(x.mean() - y.mean()),
            float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5)))


def boot_one(x, n=N_BOOT, seed=SEED):
    x = np.asarray(x, np.float64)
    g = np.random.default_rng(seed)
    bs = x[g.integers(0, len(x), size=(n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def star(lo, hi) -> str:
    return "*" if not lo <= 0 <= hi else " "


def main() -> int:
    p = argparse.ArgumentParser(description="Interval on d(earlier) - d(later)")
    p.add_argument("--workdir", type=Path,
                   default=Path.home() / "projects" / "paper21c_v8")
    p.add_argument("--eras", default="mimic3,mimic4",
                   help="earlier,later — order matters: D = d(first) - d(second)")
    p.add_argument("--objective", default="contrastive")
    p.add_argument("--scrubbed", action="store_true")
    p.add_argument("--label", default=None, help="name for this condition in the output")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    eras = [e.strip() for e in a.eras.split(",") if e.strip()]
    if len(eras) != 2:
        raise SystemExit("--eras needs exactly two")
    early, late = eras
    tag = "_scrubbed" if a.scrubbed else ""

    cells = {}
    for e in eras:
        f = a.workdir / f"cell_{a.objective}_{e}{tag}.npz"
        if not f.exists():
            raise SystemExit(f"missing {f}")
        cells[e] = load_cell(f)

    label = a.label or f"{a.workdir.name}{tag}"
    print("=" * 100)
    print(f"ASYMMETRY WITH AN INTERVAL — {label}")
    print("=" * 100)
    print(f"  D = d({early}) - d({late}), resampled independently per era")
    print(f"  model {cells[early][0].get('model')}")
    print(f"  min_year {cells[early][0].get('mimic4_min_year')}   "
          f"scrub {bool(cells[early][0].get('scrub_deid'))}")

    # the two cells must have scored the same queries, or d is not d
    for e in eras:
        hs = {n: cells[n][0].get("qhashes", {}).get(e) for n in eras}
        if len({h for h in hs.values() if h}) > 1:
            raise SystemExit(f"query hashes differ for {e}: {hs}")
    print("  query hashes match across both cells")

    doses = sorted((set(cells[early][1]) & set(cells[late][1])) - {0})
    results = {}

    for metric, k in (("MRR", None), *[(f"recall@{kk}", kk) for kk in KS]):
        print(f"\n  {metric}")
        print("  " + "dose".rjust(6) + "d(earlier)".rjust(13) + "d(later)".rjust(12)
              + "D".rjust(12) + "95% CI on D".rjust(22) + "ratio".rjust(8))
        for s in doses:
            # d(e) = own - cross, per query, within era e
            de = (np.asarray(cells[early][1][s][early], np.float64)
                  - np.asarray(cells[late][1][s][early], np.float64))
            dl = (np.asarray(cells[late][1][s][late], np.float64)
                  - np.asarray(cells[early][1][s][late], np.float64))
            if k is not None:
                de = (hit_at(cells[early][1][s][early], k)
                      - hit_at(cells[late][1][s][early], k))
                dl = (hit_at(cells[late][1][s][late], k)
                      - hit_at(cells[early][1][s][late], k))
            me, _, _ = boot_one(de)
            ml, _, _ = boot_one(dl)
            D, lo, hi = boot_diff(de, dl)
            ratio = "n/a" if abs(ml) < 1e-9 else f"{me / ml:.2f}"
            print(f"  {s:>6,}{me:>+13.4f}{ml:>+12.4f}{D:>+11.4f}{star(lo, hi)}"
                  f"{f'[{lo:+.4f},{hi:+.4f}]':>22}{ratio:>8}")
            results[f"{metric}|{s}"] = {"d_early": me, "d_late": ml,
                                        "D": D, "lo": lo, "hi": hi}

    print("\n" + "=" * 100)
    print("READING IT")
    print("=" * 100)
    print("""
  The starred rows are the ones that establish an asymmetry. A row where both
  d values exclude zero but D does not is a row where the two shortfalls are
  both real and cannot be told apart: report it that way and not as an
  asymmetry.

  D is wider than either d because it carries two eras' sampling variance and
  the two estimates are independent, not paired. That width is correct for this
  claim; narrowing it would need more queries per era, not a different test.

  The ratio column is descriptive only. It has no interval, it is unstable when
  the denominator is near zero, and it should not be reported as an effect size
  without the difference beside it.
""")
    if a.out:
        a.out.write_text(json.dumps(results, indent=2, default=float))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
