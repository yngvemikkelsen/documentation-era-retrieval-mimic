#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
Does cross-era adaptation help or hurt, against no adaptation at all?

    uv run paper21c_margin_v1.py --workdir ~/projects/paper21c_x

Reads only the stored cell files. No cohort, no two_site.py, no pair rebuild.
Replaces paper21c_recall_v1.py, which reported the wrong contrast.

THE THREE CONTRASTS
-------------------
Every cell stores step 0, which is the unadapted model, so three contrasts are
available on one era's records and all of them are paired per query:

    b = own-era adapted   - unadapted      what matched adaptation buys
    c = cross-era adapted - unadapted      what mismatched adaptation buys
    d = own-era adapted   - cross-era      the shortfall from mismatch

Only d has been reported so far. d is a comparison of two adapted models, so it
cannot say whether cross-era adaptation helps or harms: a large d is consistent
both with cross-era adaptation hurting and with it helping slightly less. c is
the quantity that answers that, and b = c + d, which this script verifies.

If c > 0 there is no backward-compatibility failure. A site that adapts on newer
notes retrieves older notes better than it did before, and d is the size of the
gain it forgoes by not using matched data.

THE DOSE TREND
--------------
Whether c stays positive depends on how far the starting encoder is from
clinical text. The domain-general part of the gain is large when the encoder has
never seen clinical notes, and shrinks as the encoder gets closer to the domain.
The era-specific part does not shrink. If the era part ever exceeds the domain
part, c turns negative.

Each dose step moves the model closer to clinical text, so the dose ladder is a
proxy for that distance. Read c down the dose column:

    c holds or grows      the claim is stable over the range tested
    c shrinks towards 0   the margin is being consumed, and an encoder that
                          starts closer to clinical text could show c < 0

This is a proxy and not the same thing as starting from a near encoder, because
dose also changes how much the model has specialised to one era's documentation.
It bounds the claim rather than settling it.

RANK AGAINST RETRIEVAL
----------------------
Everything is reported in MRR and in recall at 1, 3, 5 and 10. rr_at_k stores
1/rank within TOP_K = 10 and zero outside it, so rank is recoverable exactly.
MRR is the most rank-sensitive reading and recall@10 the least. A contrast that
is present in MRR and absent at recall@10 is a reordering inside the retrieved
set, which a top-10 context window or a reranker would not see. One that
persists at recall@10 means documents leave the retrieved set, and nothing
downstream can recover them.

Recall beyond 10 is unavailable: rank 11 and rank 900 are stored identically.
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


def boot(x, n=N_BOOT, seed=SEED):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    g = np.random.default_rng(seed)
    bs = x[g.integers(0, len(x), size=(n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def hit_at(rr, k: int) -> np.ndarray:
    """1 where the target reached rank k or better. rr is 1/rank inside TOP_K and
    0 outside, so rr >= 1/k is exactly rank <= k. The epsilon absorbs float32
    storage error in 1/3, 1/7 and similar."""
    return (np.asarray(rr, np.float64) >= (1.0 / k) - 1e-6).astype(np.float64)


def cell(v, lo, hi, star_if=True) -> str:
    s = "*" if star_if and not lo <= 0 <= hi else " "
    return f"{v:+.4f}{s}[{lo:+.3f},{hi:+.3f}]"


def main() -> int:
    p = argparse.ArgumentParser(
        description="Cross-era adaptation against no adaptation")
    p.add_argument("--workdir", type=Path,
                   default=Path.home() / "projects" / "paper21c_x")
    p.add_argument("--objective", default="contrastive", choices=["contrastive", "mlm"])
    p.add_argument("--eras", default="mimic3,mimic4")
    p.add_argument("--scrubbed", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    eras = [e.strip() for e in a.eras.split(",") if e.strip()]
    if len(eras) != 2:
        raise SystemExit("--eras needs exactly two")
    tag = "_scrubbed" if a.scrubbed else ""
    cells = {}
    for e in eras:
        path = a.workdir / f"cell_{a.objective}_{e}{tag}.npz"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        cells[e] = load_cell(path)

    # integrity: step 0 is the unadapted model, so both cells must hold the same
    # per-query vector for a given era. If they differ, the two runs did not see
    # the same evaluation set and nothing below is comparable.
    print("=" * 102)
    print(f"BASELINE CHECK — step 0 must be identical in both cells")
    print("=" * 102)
    base = {}
    for era in eras:
        vs = {src: cells[src][1].get(0, {}).get(era) for src in eras}
        if any(v is None for v in vs.values()):
            raise SystemExit(f"step 0 missing for {era}; this script needs the "
                             "unadapted baseline")
        a0, b0 = (np.asarray(vs[s], np.float64) for s in eras)
        mx = float(np.max(np.abs(a0 - b0)))
        ok = "identical" if mx == 0.0 else f"MAX DIFF {mx:.6f}"
        print(f"  {era}: n={len(a0):,}  MRR {a0.mean():.4f}  {ok}")
        if mx > 1e-6:
            raise SystemExit(
                f"step 0 differs between cells for {era} by {mx:.6f}; the two runs "
                "did not evaluate the same set, so b and c are not comparable")
        base[era] = a0

    doses = sorted((set(cells[eras[0]][1]) & set(cells[eras[1]][1])) - {0})
    results: dict = {}

    for era in eras:
        other = [x for x in eras if x != era][0]
        print("\n" + "=" * 102)
        print(f"{era} RECORDS — MRR")
        print("=" * 102)
        print(f"  b = adapted on {era} minus unadapted")
        print(f"  c = adapted on {other} minus unadapted   <- answers help or harm")
        print(f"  d = adapted on {era} minus adapted on {other}")
        print("\n  " + "dose".rjust(6) + "b".rjust(25) + "c".rjust(25)
              + "d".rjust(25) + "   b-(c+d)")
        for step in doses:
            own = cells[era][1].get(step, {}).get(era)
            oth = cells[other][1].get(step, {}).get(era)
            if own is None or oth is None:
                continue
            own = np.asarray(own, np.float64)
            oth = np.asarray(oth, np.float64)
            bv, cv, dv = own - base[era], oth - base[era], own - oth
            row = f"  {step:>6,}"
            for v in (bv, cv, dv):
                row += cell(*boot(v)).rjust(25)
            row += f"   {float(np.max(np.abs(bv - (cv + dv)))):.2e}"
            print(row)
            results[f"mrr|{era}|{step}"] = {
                "b": float(bv.mean()), "c": float(cv.mean()), "d": float(dv.mean())}

        print(f"\n  c in recall terms — adapted on {other} minus unadapted, "
              f"on {era} records")
        print("  " + "dose".rjust(6) + "".join(f"recall@{k}".rjust(25) for k in KS))
        for step in doses:
            oth = cells[other][1].get(step, {}).get(era)
            if oth is None:
                continue
            row = f"  {step:>6,}"
            for k in KS:
                v = hit_at(oth, k) - hit_at(base[era], k)
                row += cell(*boot(v)).rjust(25)
                results[f"c_r{k}|{era}|{step}"] = float(v.mean())
            print(row)

        print(f"\n  d in recall terms — the shortfall, on {era} records")
        print("  " + "dose".rjust(6) + "".join(f"recall@{k}".rjust(25) for k in KS))
        for step in doses:
            own = cells[era][1].get(step, {}).get(era)
            oth = cells[other][1].get(step, {}).get(era)
            if own is None or oth is None:
                continue
            row = f"  {step:>6,}"
            for k in KS:
                v = hit_at(own, k) - hit_at(oth, k)
                row += cell(*boot(v)).rjust(25)
                results[f"d_r{k}|{era}|{step}"] = float(v.mean())
            print(row)

        print(f"\n  levels on {era} records, for scale")
        print("  " + "dose".rjust(6) + "model".rjust(16) + "MRR".rjust(10)
              + "".join(f"r@{k}".rjust(10) for k in KS))
        for step in [0, *doses]:
            for src in eras:
                v = cells[src][1].get(step, {}).get(era)
                if v is None:
                    continue
                lbl = "unadapted" if step == 0 else (
                    f"on {src}" + (" (own)" if src == era else " (cross)"))
                if step == 0 and src != eras[0]:
                    continue
                row = (f"  {step:>6,}{lbl:>16}{float(np.mean(v)):>10.4f}"
                       + "".join(f"{float(hit_at(v, k).mean()):>10.4f}" for k in KS))
                print(row)

    print("\n" + "=" * 102)
    print("READING IT")
    print("=" * 102)
    print("""
  Read c first. c is the answer to whether cross-era adaptation helps or harms.

  c > 0 and its interval above zero: adaptation on the other era improves
  retrieval on these records. There is no backward-compatibility failure, and d
  is the gain forgone rather than a loss incurred.

  c < 0 and its interval below zero: adaptation on the other era makes these
  records harder to retrieve than doing nothing. That is a real degradation.

  c straddling zero: the data cannot separate help from harm at that dose.

  Then read c down the dose column. A c that shrinks towards zero as the dose
  grows means the margin is being consumed, and an encoder starting closer to
  clinical text could show c < 0. A c that holds means the claim is stable over
  the range tested. This is a proxy for encoder distance, not a substitute for
  running a near encoder: dose also changes how far the model has specialised to
  one era, so it bounds the claim rather than settling it.

  Then compare the MRR and recall@10 columns for the same contrast. Present in
  MRR and absent at recall@10 means a reordering inside the retrieved set, which
  a top-10 context window or a cross-encoder reranker would absorb. Present at
  recall@10 means documents leave the retrieved set, and no later stage recovers
  them.

  b-(c+d) is an arithmetic check and must be at machine precision. Anything
  larger means the three contrasts were not computed on the same queries.
""")
    if a.out:
        a.out.write_text(json.dumps(results, indent=2, default=float))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
