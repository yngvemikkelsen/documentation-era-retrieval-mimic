#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
The conventional era-by-era interaction, alongside the asymmetry D.

    uv run paper21c_interaction_v1.py --workdir ~/projects/paper21c_v8

Seconds. No accelerator. Reads the stored cells only. Changes nothing.

WHY THIS EXISTS
---------------
The manuscript reports D = d(earlier) - d(later), where d(e) is the matched-era
advantage on era e: the matched-era model minus the mismatched-era model, scored
on era e records. D answers the directional question the study asks, which is
whether the backward penalty exceeds the forward penalty.

D is NOT the conventional training-era by evaluation-era interaction. The two
matched-era advantages are defined with opposite training directions: d(earlier)
compares earlier-trained against later-trained, and d(later) compares
later-trained against earlier-trained. Applying one training direction to both
record sets instead gives

    I = (M_ee - M_le) - (M_el - M_ll) = d(earlier) + d(later)

where M_xy is the model trained on era x scored on era y records. A uniform
advantage for one training corpus contributes to D without being an
interaction, and two equal matched-era advantages give D = 0 even when the
interaction is large. Both quantities are worth reporting; they answer
different questions.

WHAT IT REPORTS
---------------
For each metric: the four cell means, d(earlier), d(later), their difference D
and their sum I, each with a 95 percentile interval from 2,000 resamples.

The two eras are scored on disjoint query sets, so the resample draws each era
independently and then combines the resampled means. This is the same treatment
the manuscript already applies to D, and it is correct for I for the same
reason. Note that Var(d_E - d_L) equals Var(d_E + d_L) under independence, so D
and I carry the same sampling variance and differ only in location.

The intervals here are query-level. The manuscript's primary intervals are
patient-clustered. On this cohort the two differ in width by at most about 7
percent, because only 31 of 1,000 earlier-era queries and 19 of 1,000
later-era queries are repeated observations from a patient already represented.
State in the caption which resample each reported interval came from.

READING THE RESULT
------------------
If I excludes zero, the design shows a training-era by evaluation-era
interaction. If D also excludes zero, the two matched-era advantages are
additionally unequal in magnitude, which is the directional claim. The second
does not follow from the first.
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
    rr: dict[int, dict[str, np.ndarray]] = {}
    for k in z.files:
        if not k.startswith("rr__"):
            continue
        _, step, era = k.split("__")
        rr.setdefault(int(step), {})[era] = z[k]
    return meta, rr


def hit_at(x, k: int) -> np.ndarray:
    return (np.asarray(x, np.float64) >= (1.0 / k) - 1e-6).astype(np.float64)


def pct(a) -> tuple[float, float]:
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def star(lo: float, hi: float) -> str:
    return "*" if not lo <= 0 <= hi else " "


def main() -> int:
    p = argparse.ArgumentParser(description="Interaction and asymmetry together")
    p.add_argument("--workdir", type=Path,
                   default=Path.home() / "projects" / "paper21c_v8")
    p.add_argument("--objective", default="contrastive")
    p.add_argument("--eras", default="mimic3,mimic4")
    p.add_argument("--step", type=int, default=0,
                   help="checkpoint to report; 0 means the largest available")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    eras = [e.strip() for e in a.eras.split(",") if e.strip()]
    if len(eras) != 2:
        raise SystemExit("--eras needs exactly two")
    e0, e1 = eras

    cells = {}
    for e in eras:
        f = a.workdir / f"cell_{a.objective}_{e}.npz"
        if not f.exists():
            raise SystemExit(f"missing {f}")
        cells[e] = load_cell(f)

    shared = sorted((set(cells[e0][1]) & set(cells[e1][1])) - {0})
    if not shared:
        raise SystemExit("no checkpoint is present in both cells")
    step = a.step if a.step in shared else shared[-1]
    print(f"  checkpoints in both cells: {shared}")
    print(f"  reporting step {step:,}\n")

    rng = np.random.default_rng(SEED)
    results = {}

    print("=" * 100)
    print("CELL MEANS, THE ASYMMETRY D AND THE INTERACTION I")
    print("=" * 100)
    print("  M_xy is the model trained on era x, scored on era y records\n")
    hdr = ("metric".ljust(11) + "M_ee".rjust(8) + "M_le".rjust(8)
           + "M_el".rjust(8) + "M_ll".rjust(8)
           + "d(earlier)".rjust(24) + "d(later)".rjust(24))
    print("  " + hdr)

    rows = []
    for name, k in (("MRR", None), *[(f"recall@{x}", x) for x in KS]):
        def v(cell_era: str, rec_era: str) -> np.ndarray:
            x = cells[cell_era][1][step][rec_era]
            return np.asarray(x, np.float64) if k is None else hit_at(x, k)

        m_ee, m_le = v(e0, e0), v(e1, e0)     # earlier records
        m_el, m_ll = v(e0, e1), v(e1, e1)     # later records
        d_e = m_ee - m_le                     # matched advantage, earlier records
        d_l = m_ll - m_el                     # matched advantage, later records

        be = d_e[rng.integers(0, len(d_e), size=(N_BOOT, len(d_e)))].mean(1)
        bl = d_l[rng.integers(0, len(d_l), size=(N_BOOT, len(d_l)))].mean(1)
        de_lo, de_hi = pct(be)
        dl_lo, dl_hi = pct(bl)
        print(f"  {name:<11}{m_ee.mean():>8.4f}{m_le.mean():>8.4f}"
              f"{m_el.mean():>8.4f}{m_ll.mean():>8.4f}"
              f"{f'{d_e.mean():+.4f}{star(de_lo, de_hi)}[{de_lo:+.3f},{de_hi:+.3f}]':>24}"
              f"{f'{d_l.mean():+.4f}{star(dl_lo, dl_hi)}[{dl_lo:+.3f},{dl_hi:+.3f}]':>24}")
        rows.append((name, d_e.mean(), d_l.mean(), be, bl))

    print("\n" + "=" * 100)
    print("THE TWO COMBINED QUANTITIES")
    print("=" * 100)
    print("  D = d(earlier) - d(later)   difference in the matched-era advantages")
    print("  I = d(earlier) + d(later)   conventional training-era by evaluation-era interaction\n")
    print("  " + "metric".ljust(11) + "D".rjust(30) + "I".rjust(30))
    for name, me, ml, be, bl in rows:
        bd, bi = be - bl, be + bl
        d_lo, d_hi = pct(bd)
        i_lo, i_hi = pct(bi)
        D, I = me - ml, me + ml
        print(f"  {name:<11}"
              f"{f'{D:+.4f}{star(d_lo, d_hi)}[{d_lo:+.4f},{d_hi:+.4f}]':>30}"
              f"{f'{I:+.4f}{star(i_lo, i_hi)}[{i_lo:+.4f},{i_hi:+.4f}]':>30}")
        results[name] = {"d_earlier": me, "d_later": ml,
                         "D": [D, d_lo, d_hi], "I": [I, i_lo, i_hi]}

    print("\n" + "=" * 100)
    print("READING IT")
    print("=" * 100)
    print("""
  A star marks an interval that excludes zero.

  I excluding zero means the design shows a training-era by evaluation-era
  interaction. D excluding zero means the two matched-era advantages are also
  unequal in magnitude. The second does not follow from the first, and the
  manuscript's directional claim rests on D.

  These intervals are query-level. Say so wherever they are reported, because
  the primary contrasts in the manuscript are patient-clustered.
""")
    if a.out:
        a.out.write_text(json.dumps(results, indent=2, default=float))
        print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
