# Reproducing the run

Everything below assumes a prepared data directory holding `notes.parquet` and
`cohort.csv`, built from PhysioNet downloads under your own credentials. See
**Data access** in `README.md`.

Paths in the commands are the ones the reported run used. Change them to suit;
do not leave them to the scripts' defaults.

---

## The whole thing, in one command

```bash
caffeinate -i -s -t 43200 &     # macOS, separate terminal: keep awake 12 h
bash code/run_v8.sh
```

About **11.5 hours**. It stops at the first failure and prints why. Nothing after
a failure is run.

`run_v8.sh` expects the Python scripts beside it, so run it from a directory
holding all of `code/`. It reads `two_site.py` from `$HOME/projects/paper9/`;
edit the `TWO_SITE` variable at the top if yours is elsewhere.

---

## What the ten stages do, and what gates each one

| Stage | What runs | Gate |
|---|---|---|
| −1 | Resume check, only if cells are already on disk | Every cell present must match the configuration about to be run, or the run refuses to continue |
| 0 | Preflight, ~15 min | Scripts present; workdirs free of cells; real pool sizes read through `load_cohort`; pairs built for real; within-era split year chosen. Emits `--train-pool` rather than taking it on trust |
| 1 | Cross-corpus overlap, ~5 min | Exit 0 only. See the exit codes below |
| 2 | The primary 2×2 and the dose ladder, ~2.4 h | Cells verified by reading their metadata, not by `$?` |
| 3 | Leak audit, ~4 min | Zero evaluated targets may have a twin in the other era's training set. Run here, not at the end, so a failure stops the remaining eight hours |
| 4 | Pooled-training arm, ~1.2 h | |
| 5 | Within-era control, ~2.4 h | Run **without** the calendar restriction, deliberately — see below |
| 6 | Second encoder, ~2.3 h | Its own workdir, or it would be silently skipped |
| 7 | De-identification arm, ~2.4 h | Same workdir, `_scrubbed` filenames |
| 8 | Full verification of every cell | Model, restriction, pool, seed, checkpoints and per-era n all checked against what was requested |
| 9 | The contrasts | |

### Overlap gate exit codes

```
0   clean: no cross-corpus duplicate, and the planted control recovered enough
    to make that absence evidence
2   cross-corpus duplicates are still present
3   no duplicates found, but planted recovery is below --min-recall
4   the planted control was skipped, so no null can be supported
```

3 and 4 are failures on purpose. An absence is only evidence if the search could
have produced a presence.

### Why the within-era control is unrestricted

It uses MIMIC-IV only — `_split_within` drops every other corpus — so no
MIMIC-III note is ever present and cross-corpus contamination is impossible by
construction. Restricting it would shrink an eleven-year span to five and cost
the control its power.

---

## The post-run analyses

These read the stored cells. None of them trains anything; all run in seconds to
minutes.

```bash
# The asymmetry D = d(earlier) - d(later), with its own interval
uv run code/paper21c_asym_v1.py --workdir ~/projects/paper21c_v8

# Patient-clustered intervals and the untrained BM25 baseline
uv run code/paper21c_robust_v2.py \
    --two-site ~/projects/paper9/two_site.py \
    --export-bm25 ~/projects/paper21c_bm25 \
    --out ~/projects/paper21c_robust.json

# Records entering and leaving the top 10
uv run code/paper21c_flux_v1.py --workdir ~/projects/paper21c_v8

# Calendar restriction verified against admission dates, not the band label
uv run code/paper21c_anchorcheck_v1.py --mimic4 ~/physionet.org/files/mimiciv/3.1

# Which MIMIC-IV year band each detected duplicate sits in
uv run code/paper21c_dupyear_v1.py
```

`paper21c_dupyear_v1.py` reads the duplicate CSV exported by an **unrestricted**
overlap run. To regenerate it, run the detector without `--min-year`:

```bash
uv run code/paper21c_overlap_v4.py --export ~/projects/paper21c_dups.csv
```

That run is expected to **fail with exit 2**. That is the finding: 115 pairs at
Jaccard ≥ 0.50 against zero within MIMIC-III, which is what the restriction
exists to remove.

---

## Order of discovery, for anyone retracing the argument

1. Run the detector unrestricted. It fails: the corpora share source documents.
2. Run `paper21c_dupyear_v1.py`. Every duplicate sits in a MIMIC-IV band at or
   before 2013, none in 2014+, where about 81 of 115 would fall if the band label
   carried no information about admission date.
3. Restrict MIMIC-IV to 2014 onward and run the detector again. It passes, and
   the planted-duplicate control shows the search would have found duplication
   had any remained.
4. Run `paper21c_anchorcheck_v1.py`. The restriction holds against each retained
   admission's date, not merely against its band label.
5. Only then run the cells.

Steps 1 to 4 are why no number in the manuscript comes from any run before
`run_v8.sh`. Every earlier cell is contaminated.
