#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "pyarrow"]
# ///
"""
Can any retained MIMIC-IV admission actually predate 2014?

    uv run paper21c_anchorcheck_v1.py --mimic4 ~/data/mimiciv/3.1

Under a minute. Reads the MIMIC-IV hosp tables, notes.parquet and cohort.csv.
Changes nothing.

THE PROBLEM THIS SETTLES
------------------------
The analysis restricted MIMIC-IV to the 2014-2016 and 2017-2019 bands of
anchor_year_group and the manuscript called the resulting separation from
MIMIC-III true by construction. That is an overstatement, and the MIMIC-IV
documentation is the reason: anchor_year is described only as "a deidentified
year for each patient". It is NOT stated to be the year of the patient's first
admission, and nothing says every admission falls at or after it.

anchor_year_group gives the real-world range of anchor_year, not of any
particular stay. A patient whose anchor_year_group is 2014-2016 can therefore
hold an admission at a shifted year EARLIER than anchor_year, and that
admission's real year would precede 2014. If it precedes 2013 it is inside
MIMIC-III's span and the overlap the restriction was meant to remove is back.

HOW THE REAL YEAR IS RECOVERED
------------------------------
Date shifting in MIMIC-IV is per patient and preserves intervals, so for one
patient the difference between two shifted years equals the difference between
the two real years. Hence, for an admission with shifted year Y:

    offset        = Y - anchor_year
    real year     is in [group_low + offset, group_high + offset]

The lower end is the one that matters: if group_low + offset <= 2012 for any
retained admission, that admission could fall inside MIMIC-III's 2001-2012 span.

WHAT IT REPORTS
---------------
  1. whether cohort.csv's year_group agrees with anchor_year_group, which
     establishes that the restriction operated on the column it was meant to
  2. the distribution of offsets, and how many admissions precede their own
     anchor year at all
  3. for the notes RETAINED by --mimic4-min-year 2013, the earliest possible
     real year, and the count whose lower bound reaches 2012 or earlier
  4. a CSV of any such admissions, so they can be excluded and the cells re-run

READING THE RESULT
------------------
  0 violators   The restriction holds. The manuscript may say the separation was
                verified against admission dates rather than assumed from the
                band, which is a stronger claim than the one it makes now.
  any violators Those notes were in the analysis and should not have been.
                Exclude them and re-run; the near-duplicate audit did not catch
                them because a shared patient need not produce shared text.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

GROUP_RE = re.compile(r"(\d{4})\s*-\s*(\d{4})")


def find(root: Path, stem: str) -> Path:
    hits = sorted(list(root.rglob(f"{stem}.csv.gz")) + list(root.rglob(f"{stem}.csv")))
    if not hits:
        raise SystemExit(
            f"could not find {stem}.csv[.gz] under {root}\n"
            f"Pass --mimic4 pointing at the MIMIC-IV root (the folder containing "
            f"the 'hosp' module).")
    return hits[0]


def main() -> int:
    p = argparse.ArgumentParser(description="Verify the MIMIC-IV date restriction")
    p.add_argument("--mimic4", type=Path, required=True,
                   help="MIMIC-IV root, containing hosp/patients and hosp/admissions")
    p.add_argument("--outdir", type=Path,
                   default=Path.home() / "projects" / "paper21a" / "out")
    p.add_argument("--era", default="mimic4")
    p.add_argument("--min-year", type=int, default=2013,
                   help="the --mimic4-min-year the cells were run with")
    p.add_argument("--mimic3-last-year", type=int, default=2012,
                   help="last year MIMIC-III can contain")
    p.add_argument("--export", type=Path,
                   default=Path.home() / "projects" / "paper21c_anchor_violations.csv")
    a = p.parse_args()

    pat_f, adm_f = find(a.mimic4, "patients"), find(a.mimic4, "admissions")
    print(f"patients:   {pat_f}")
    print(f"admissions: {adm_f}")

    pat = pd.read_csv(pat_f, usecols=["subject_id", "anchor_year", "anchor_year_group"])
    adm = pd.read_csv(adm_f, usecols=["subject_id", "hadm_id", "admittime"])
    notes = pd.read_parquet(a.outdir / "notes.parquet")
    cohort = pd.read_csv(a.outdir / "cohort.csv", low_memory=False)
    mine = notes.loc[notes["corpus"] == a.era, ["hadm_id"]].drop_duplicates()
    print(f"\n{a.era} notes in the cohort: {len(mine):,}")

    g = pat["anchor_year_group"].astype(str).str.extract(GROUP_RE)
    pat["g_low"] = pd.to_numeric(g[0], errors="coerce")
    pat["g_high"] = pd.to_numeric(g[1], errors="coerce")
    adm["adm_year"] = pd.to_datetime(adm["admittime"], errors="coerce").dt.year

    df = (mine.merge(adm, on="hadm_id", how="left")
               .merge(pat, on="subject_id", how="left"))
    miss = int(df["anchor_year"].isna().sum() + df["adm_year"].isna().sum())
    if miss:
        print(f"  WARNING: {miss:,} rows could not be matched to an anchor year or "
              "admission time; they are reported separately and NOT cleared")

    df["offset"] = df["adm_year"] - df["anchor_year"]
    df["real_low"] = df["g_low"] + df["offset"]
    df["real_high"] = df["g_high"] + df["offset"]

    # ---- 1. does cohort.csv's year_group agree with anchor_year_group? ------
    print("\n" + "=" * 92)
    print("1. PROVENANCE OF THE COLUMN THE RESTRICTION USED")
    print("=" * 92)
    if "year_group" in cohort.columns:
        chk = (cohort.loc[cohort["corpus"] == a.era, ["hadm_id", "year_group"]]
               .drop_duplicates("hadm_id")
               .merge(df[["hadm_id", "anchor_year_group"]].drop_duplicates("hadm_id"),
                      on="hadm_id", how="inner"))
        norm = lambda x: x.astype(str).str.replace(" ", "", regex=False)
        agree = int((norm(chk["year_group"]) == norm(chk["anchor_year_group"])).sum())
        print(f"  {agree:,} of {len(chk):,} compared rows agree "
              f"({100 * agree / max(len(chk), 1):.2f}%)")
        if agree != len(chk):
            print("  MISMATCH: cohort.csv's year_group is not anchor_year_group. The "
                  "restriction may not have operated on the column assumed.")
    else:
        print("  cohort.csv has no year_group column to compare")

    # ---- 2. do admissions precede their own anchor year? -------------------
    print("\n" + "=" * 92)
    print("2. OFFSET OF EACH ADMISSION FROM ITS PATIENT'S ANCHOR YEAR")
    print("=" * 92)
    off = df["offset"].dropna()
    print(f"  min {off.min():+.0f}   median {off.median():+.0f}   max {off.max():+.0f}")
    neg = int((off < 0).sum())
    print(f"  admissions BEFORE their anchor year: {neg:,} of {len(off):,} "
          f"({100 * neg / max(len(off), 1):.2f}%)")
    print("  If this is zero, anchor_year behaves as a first-admission year on this")
    print("  cohort and the band does bound the admission date from below.")

    # ---- 3. the retained set ------------------------------------------------
    kept = df[(df["g_low"] > a.min_year)].copy()
    print("\n" + "=" * 92)
    print(f"3. THE SET RETAINED BY --mimic4-min-year {a.min_year}")
    print("=" * 92)
    print(f"  retained notes: {len(kept):,}")
    if len(kept):
        print(f"  earliest possible real admission year: {kept['real_low'].min():.0f}")
        print(f"  latest possible real admission year:   {kept['real_high'].max():.0f}")
    bad = kept[kept["real_low"] <= a.mimic3_last_year]
    unresolved = kept[kept["real_low"].isna()]

    print("\n" + "=" * 92)
    print("VERDICT")
    print("=" * 92)
    print(f"\n  retained admissions whose earliest possible year is "
          f"{a.mimic3_last_year} or before: {len(bad):,}")
    if len(unresolved):
        print(f"  retained admissions with no recoverable date: {len(unresolved):,}")
    if len(bad) == 0 and len(unresolved) == 0:
        print("\n  The restriction holds against the admission dates, not only against")
        print("  the band label. The manuscript may state that calendar separation was")
        print("  verified per admission rather than assumed from anchor_year_group.")
        rc = 0
    else:
        print("\n  The restriction does NOT hold by itself. These admissions could fall")
        print("  inside MIMIC-III's span and were in the analysis. Exclude them and")
        print("  re-run the affected cells. A shared patient need not produce shared")
        print("  text, so the near-duplicate audit would not have caught them.")
        out = pd.concat([bad, unresolved]).drop_duplicates("hadm_id")
        cols = [c for c in ("hadm_id", "subject_id", "anchor_year",
                            "anchor_year_group", "adm_year", "offset",
                            "real_low", "real_high") if c in out.columns]
        try:
            a.export.parent.mkdir(parents=True, exist_ok=True)
            out[cols].to_csv(a.export, index=False)
            print(f"\n  wrote {len(out):,} row(s) to {a.export}")
        except OSError as e:
            # a failed write must not cost the verdict, which is the point of the run
            print(f"\n  could not write {a.export}: {e}")
            print("  the verdict above still stands; pass --export to a writable path")
        rc = 2
    print(f"\n  EXIT {rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
