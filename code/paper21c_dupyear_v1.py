#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "pyarrow"]
# ///
"""
Which MIMIC-IV year band does each detected duplicate sit in?

    uv run paper21c_dupyear_v1.py

Seconds. Reads paper21c_dups.csv, notes.parquet and cohort.csv. Changes nothing.

WHY THIS DECIDES THE DESIGN
---------------------------
MIMIC-III covers 2001-2012 and cannot be dated per admission: its real years
were destroyed at de-identification. So the only restriction available is on the
MIMIC-IV side. That restriction is worth doing ONLY if the duplicates we found
actually live in the bands it removes.

  every duplicate in 2008-2010 or 2011-2013
      the bands do separate the corpora. Dropping them removes every duplicate
      the detector found, and the remaining risk is patient recurrence, which
      shares no text.

  any duplicate in 2014-2016 or 2017-2019
      the bands do NOT separate the corpora. A MIMIC-IV note labelled 2014+ that
      duplicates a MIMIC-III note means the label does not bound the admission
      date, so no cut on this column can exclude the overlap. The MIMIC-III
      versus MIMIC-IV comparison is then not salvageable by restriction, and the
      only leakage-free design left is within MIMIC-IV.

HOW THE INDICES MAP BACK
------------------------
paper21c_overlap_v2.py exported positions, not identifiers. With --sample 0 it
read each era as notes.loc[notes["corpus"] == era, "text"], preserving
notes.parquet row order, so position n is the nth row of that era. This script
rebuilds the same ordering and asserts the era counts match the detector's
(52,726 and 37,056) before trusting any mapping.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

YEAR_COLS = ("year_group", "real_year_lower", "anchor_year_group", "anchor_year",
             "admit_year", "admityear")


def main() -> int:
    p = argparse.ArgumentParser(description="Year band of each detected duplicate")
    p.add_argument("--dups", type=Path,
                   default=Path.home() / "projects" / "paper21c_dups.csv")
    p.add_argument("--outdir", type=Path,
                   default=Path.home() / "projects" / "paper21a" / "out")
    p.add_argument("--eras", default="mimic3,mimic4")
    p.add_argument("--expect", default="52726,37056",
                   help="era note counts the detector reported, as a guard")
    a = p.parse_args()

    eras = [e.strip() for e in a.eras.split(",") if e.strip()]
    exp = [int(x) for x in a.expect.split(",")]
    if not a.dups.exists():
        raise SystemExit(f"missing {a.dups}")

    notes = pd.read_parquet(a.outdir / "notes.parquet")
    cohort = pd.read_csv(a.outdir / "cohort.csv", low_memory=False)
    col = next((c for c in YEAR_COLS if c in cohort.columns), None)
    if col is None:
        raise SystemExit(f"no year column in cohort.csv; has {sorted(cohort.columns)}")

    # the detector applied no length filter and no sampling, so era order is
    # notes.parquet row order restricted to that corpus
    idx = {}
    for e, n in zip(eras, exp):
        rows = notes.index[notes["corpus"] == e].tolist()
        if len(rows) != n:
            raise SystemExit(
                f"{e}: notes.parquet gives {len(rows):,} rows but the detector "
                f"reported {n:,}. The exported positions cannot be mapped; re-run "
                "the detector and the export together.")
        idx[e] = rows
    print(f"era ordering verified: " +
          ", ".join(f"{e} {len(idx[e]):,}" for e in eras))

    yr = cohort[["hadm_id", col]].drop_duplicates("hadm_id", keep="first")
    dups = pd.read_csv(a.dups)
    need = {"jaccard", "left_index", "right_index"}
    if not need <= set(dups.columns):
        raise SystemExit(f"{a.dups} needs {sorted(need)}; has {sorted(dups.columns)}")
    print(f"{len(dups):,} duplicate pair(s) read from {a.dups.name}")

    out = dups.copy()
    for side, e in (("left", eras[0]), ("right", eras[1])):
        out[f"{side}_hadm"] = [idx[e][i] for i in out[f"{side}_index"]]
        out[f"{side}_hadm"] = notes.loc[out[f"{side}_hadm"], "hadm_id"].values
        out = out.merge(yr.rename(columns={col: f"{side}_{col}"}),
                        left_on=f"{side}_hadm", right_on="hadm_id",
                        how="left").drop(columns="hadm_id")

    rc = f"right_{col}"
    print(f"\nMIMIC-IV year band of each duplicate ({rc})")
    print(out[rc].value_counts(dropna=False).to_string())

    early = out[rc].astype(str).str.extract(r"(\d{4})")[0].astype(float) <= 2013
    n_late = int((~early).sum())
    print("\n" + "=" * 78)
    if n_late == 0:
        print("EVERY detected duplicate is in a band at or before 2013.")
        print("--mimic4-min-year 2013 removes all of them. The bands do separate the")
        print("corpora, and the residual risk is patient recurrence, which shares no")
        print("text and must be stated as a limitation.")
    else:
        print(f"{n_late} of {len(out)} duplicates sit in a 2014+ band.")
        print("The year label does NOT bound the admission date, so no cut on this")
        print("column can exclude the overlap. The MIMIC-III versus MIMIC-IV")
        print("comparison cannot be cleaned by restriction. The remaining")
        print("leakage-free design is within MIMIC-IV, 2014-2016 against 2017-2019.")
        print("\nThe offending pairs:")
        print(out.loc[~early, ["jaccard", "left_index", "right_index", rc]]
              .head(20).to_string(index=False))
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
