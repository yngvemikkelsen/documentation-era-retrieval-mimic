# documentation-era-retrieval-mimic

Analysis code for:

> Mikkelsen Y. **Local contrastive adaptation of a clinical retrieval encoder
> transfers asymmetrically between documentation eras: within-institution
> experimental study.** Manuscript under review, 2026.

The study asks one question: **does adapting a dense retrieval encoder on one
documentation era degrade retrieval of records written under another era, inside
a single institution?** It is a complete within-unit 2×2 on discharge summaries
from Beth Israel Deaconess Medical Center, reached through MIMIC-III (2001–2012)
and MIMIC-IV-Note (2014 onward). Each era supplies training data in turn and both
eras are evaluated in every cell, so each era is its own control.

This repository holds the code that produced the reported results and the
aggregate result tables behind every figure in the manuscript. **It holds no
clinical text and no per-query or per-note data of any kind.**

---

## What is here

```
code/        the pipeline, exactly as it ran
results/     aggregate tables only — metric values, no identifiers, no text
REPRODUCE.md the run order, the gates, and what each exit code means
```

### `code/`

| File | What it does |
|---|---|
| `run_v8.sh` | The single entry point. Ten stages, about 11.5 hours. Stops at the first failure and says why. |
| `paper21c_crossover_v5.py` | The cell runner: cohort, patient-level split, query construction, contrastive fine-tuning, per-query evaluation at each checkpoint. |
| `two_site.py` | The query convention, imported rather than reimplemented. Supplies `sentences`, `build_df`, `narrative_query`, `is_prose`. From the companion cross-site study. |
| `paper21c_preflight_v1.py` | Checks every assumption before any compute: scripts present, workdirs free of cells, real pool sizes, realised pair counts, the feasible within-era split year. |
| `paper21c_overlap_v4.py` | Cross-corpus near-duplicate detection with a planted-duplicate control. The gate that clears the corpus. |
| `paper21c_leakcheck_v3.py` | Targeted audit: did any evaluated target have a twin in the other era's training set? |
| `paper21c_verify_v1.py` | Reads every produced cell and fails if its recorded model, restriction, pool, seed or checkpoint set is not what was requested. |
| `paper21c_margin_v1.py` | The three contrasts b, c and d, and the check that b = c + d. |
| `paper21c_mixed_v1.py` | The pooled-training arm, compared across two workdirs. |
| `paper21c_asym_v1.py` | The asymmetry D = d(earlier) − d(later), with its own interval. |
| `paper21c_robust_v2.py` | Patient-clustered intervals and the untrained BM25 baseline. |
| `paper21c_flux_v1.py` | Records entering and leaving the top 10, behind a net recall change. |
| `paper21c_anchorcheck_v1.py` | Verifies the calendar restriction against admission dates rather than against the anchor band label. |
| `paper21c_dupyear_v1.py` | Which MIMIC-IV year band each detected duplicate sits in. The analysis that decided the design. |

### `results/`

| File | Contents |
|---|---|
| `primary_contrasts.csv` | b, c and d on both eras, at four metrics, with query-level and patient-clustered intervals side by side. |
| `asymmetry_D.csv` | D in the primary condition and the three controls. |
| `levels.csv` | Absolute retrieval levels for BM25, unadapted, mismatched, pooled and matched. |
| `bm25_contrasts.csv` | Every dense condition minus the lexical baseline, patient-clustered. |
| `dose_ladder.csv` | Every contrast at 250, 500, 1,000 and 2,500 steps. |
| `record_flux.csv` | Records entering and leaving the top 10 under each model change. |
| `leakage_control.csv` | The near-duplicate census, the band locations, the planted-duplicate recovery rates and the anchor-year verification. |

Every number in these tables is reproduced from the run logs, and the internal
identities hold: `b = c + d` in every row, each contrast equals the difference of
its levels, `D` equals the difference of the two shortfalls, and each flux row's
entries minus exits equals its net recall change.

---

## Data access

The two corpora are **not** redistributed here and cannot be. Both are available
from PhysioNet to credentialed investigators who have completed the required
human-subjects training and signed the data use agreement:

- MIMIC-III v1.4 — https://doi.org/10.13026/C2XW26
- MIMIC-IV-Note v2.2 — https://doi.org/10.13026/1n74-ne17

The pipeline expects a prepared `outdir` holding `notes.parquet` (with `hadm_id`
and `text`, and `subject_id` and `corpus` in either file) and `cohort.csv` (with
`hadm_id` and a year column). Building those from the PhysioNet downloads is a
step upstream of this repository.

---

## Requirements

Every script carries a PEP 723 header, so `uv run <script>` resolves its own
dependencies and no `requirements.txt` is needed. Python 3.11 or newer.

The heavy dependencies are `torch` and `transformers`, used only by
`paper21c_crossover_v5.py`. Everything else needs `numpy`, `pandas` and
`pyarrow`.

The reported run used Apple Silicon with the Metal (MPS) backend at roughly
11,000 tokens per second, 11.3 hours wall clock for all nine cells. The code
auto-detects `mps`, then `cuda`, then `cpu`, and `--device` overrides it.

---

## Reproducibility, stated accurately

**Seeding.** Dropout masks are drawn from the tensor library's global stream.
That stream is seeded, and repeat runs on the hardware used here were
bit-identical. Bit-identity is **not** guaranteed across devices: some
accelerators order reductions non-deterministically.

**Single seed.** Each cell ran at one seed. Repeat runs reproduce, but
reproducibility is not stability: how far the estimates would move under
different seeds is not quantified, and multi-seed replication is the obvious
strengthening of this work.

**Bootstrap scope.** The intervals resample queries, or patients where stated.
They express sampling variance in the evaluation set, not variance between
training runs.

---

## Known discrepancies in the code as published

These are left uncorrected so that the code is byte-identical to what produced
the results. None of them affects a reported number.

1. **Stale filename in a docstring.** The usage examples at the top of
   `paper21c_crossover_v5.py` name `paper21c_crossover_v1.py`. Use the commands
   in `REPRODUCE.md` instead.
2. **Stale filename in two more docstrings.** `paper21c_leakcheck_v3.py` and
   `paper21c_dupyear_v1.py` refer in prose to `paper21c_overlap_v2.py`, and
   `paper21c_leakcheck_v3.py` registers the imported detector under that module
   name. The detector it actually loads is `paper21c_overlap_v4.py`, which is its
   default and is what the run passed explicitly. The two are not
   interchangeable: v4 is the version the run gated on.
3. **Home-directory defaults are hard-coded.** Several scripts default `--outdir`,
   `--workdir` and `--two-site` to paths under `~/projects/`. Pass the flags
   explicitly; `run_v8.sh` does.
4. **An absolute path in `two_site.py`.** Line 91 sets a module-level default
   pointing at a second corpus under `/Users/yngve/`. It is unused here: this
   study imports only `sentences`, `build_df`, `narrative_query` and `is_prose`.
   It is left as published so this copy stays byte-identical to the one in the
   companion cross-site repository.
5. **`--train-pool` is not recorded in cell metadata.** It is recoverable only
   from the run log line `building training pairs from N held-out notes`, which
   is why `paper21c_verify_v1.py` takes `--logdir` and `--expect-train-pool`.

---

## The one failure mode worth knowing about

`paper21c_crossover_v5.py` decides to skip an existing cell with
`set(want).issubset(set(done))` — **checkpoint steps only**. It never compares
`--model`, `--mimic4-min-year`, `--train-pool` or `--pool`. A second-encoder run
into a workdir holding first-encoder cells prints `SKIP: already complete` and
returns the first encoder's numbers under the second encoder's label, silently.

Two defences are wired into `run_v8.sh`: the preflight refuses to start if any
target workdir holds a cell file, and `paper21c_verify_v1.py` reads every
finished cell's recorded configuration and fails if it is not what was asked for.
Do not disable either.

The same script's `main()` returns 0 even when cells fail, so a calling script
that tests `$?` learns nothing. Completion is verified by reading the cell files.

---

## Licence

MIT, see `LICENSE`. The licence covers this code only. The MIMIC corpora are
governed by the PhysioNet data use agreement.
