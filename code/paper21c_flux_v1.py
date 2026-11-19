#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
How many records LEAVE the retrieved set, and how many enter it?

    uv run paper21c_flux_v1.py --workdir ~/projects/paper21c_v8 \
        --mixdir ~/projects/paper21c_v8_mix

Seconds. Reads stored cells only. No GPU, no training, no pair rebuild.

WHY A NET FIGURE IS NOT ENOUGH
------------------------------
Recall@10 falling from 0.6500 to 0.5230 is a NET change of 0.1270. It does not
mean that 127 of the 650 records the first model retrieved were lost. Some
records leave the top 10 and others enter it, and the net is the difference.
Saying "nearly one in five of what the outgoing model retrieved is no longer
retrieved" overstates what a net figure can support: the true exit count is at
least the net and can be much larger.

This script counts both directions per query, which the stored per-query
reciprocal ranks make exact.

    exits     retrieved by A at rank <= k, not retrieved by B
    entries   not retrieved by A, retrieved by B
    net       exits - entries, which equals the recall@k difference
    churn     exits + entries, the total number of records that moved

The deployment statement a site needs is the exit count, not the net: an exit is
a record a clinician could previously reach and now cannot, and an entry
somewhere else in the corpus does not compensate for it.

WHAT IS COMPARED
----------------
On each era's records, at the final dose, three replacements:

    own -> cross    retraining on the other era alone
    own -> mixed    retraining on both eras pooled
    unadapted -> own    first adoption, for scale

The first is the replacement scenario. The second is the mitigation. The third
shows what adoption gained, so the loss can be read against it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KS = (1, 5, 10)
N_BOOT = 2000
SEED = 42


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
    return (np.asarray(rr, np.float64) >= (1.0 / k) - 1e-6).astype(bool)


def boot_net(a_hit, b_hit, n=N_BOOT, seed=SEED):
    """Paired bootstrap on the per-query net change b - a."""
    d = b_hit.astype(np.float64) - a_hit.astype(np.float64)
    g = np.random.default_rng(seed)
    bs = d[g.integers(0, len(d), size=(n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main() -> int:
    p = argparse.ArgumentParser(description="Entry and exit counts behind a net recall change")
    p.add_argument("--workdir", type=Path,
                   default=Path.home() / "projects" / "paper21c_v8")
    p.add_argument("--mixdir", type=Path,
                   default=Path.home() / "projects" / "paper21c_v8_mix")
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
    have_mixed = fm.exists()
    if have_mixed:
        cells["mixed"] = load_cell(fm)
    else:
        print(f"note: {fm} not found; the pooled comparison is skipped")

    doses = sorted(set(cells[eras[0]][1]) & set(cells[eras[1]][1]) - {0})
    top = doses[-1]
    results = {}

    print("=" * 100)
    print(f"WHAT MOVES IN AND OUT OF THE TOP k, AT STEP {top:,}")
    print("=" * 100)

    for era in eras:
        other = [x for x in eras if x != era][0]
        base = cells[era][1][0][era]
        own = cells[era][1][top][era]
        cross = cells[other][1][top][era]
        comparisons = [("own -> cross  (replace with other-era model)", own, cross),
                       ("unadapted -> own  (first adoption)", base, own)]
        if have_mixed:
            comparisons.insert(1, ("own -> mixed  (replace with pooled model)",
                                   own, cells["mixed"][1][top][era]))

        print(f"\n{'=' * 100}\n{era} RECORDS   n = {len(base):,}\n{'=' * 100}")
        for name, A, B in comparisons:
            print(f"\n  {name}")
            print("    " + "k".rjust(4) + "from".rjust(9) + "to".rjust(9)
                  + "exits".rjust(9) + "entries".rjust(9) + "net".rjust(8)
                  + "churn".rjust(8) + "   exits as % of 'from'")
            for k in KS:
                ha, hb = hit_at(A, k), hit_at(B, k)
                exits = int(np.sum(ha & ~hb))
                entries = int(np.sum(~ha & hb))
                frm, to = int(ha.sum()), int(hb.sum())
                pct = 100.0 * exits / frm if frm else float("nan")
                print(f"    {k:>4}{frm:>9,}{to:>9,}{exits:>9,}{entries:>9,}"
                      f"{to - frm:>+8,}{exits + entries:>8,}   {pct:>5.1f}%")
                results[f"{era}|{name.split('(')[0].strip()}|{k}"] = {
                    "from": frm, "to": to, "exits": exits, "entries": entries,
                    "net": to - frm, "exit_pct_of_from": pct}

    print("\n" + "=" * 100)
    print("READING IT")
    print("=" * 100)
    print("""
  Report the exit count and the entry count, not the net alone. The net is what
  recall@k shows; it understates how much moved, because exits and entries
  cancel. A replacement that loses 200 records and gains 73 has the same net as
  one that loses 127 and gains nothing, and they are not the same event for a
  clinician looking for a specific old admission.

  The 'exits as % of from' column is the honest version of "one in five of what
  the outgoing model retrieved is no longer retrieved". It is computed against
  what the outgoing model actually retrieved, per query, not inferred from two
  aggregate recall figures.

  Churn under 'unadapted -> own' gives the scale: adoption moves a great many
  records, so a smaller churn under replacement is not by itself reassuring.
""")
    if a.out:
        a.out.write_text(json.dumps(results, indent=2, default=float))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
