#!/usr/bin/env bash
# Paper 21 — the whole model-side study, once, with every check wired in.
#
#   caffeinate -i -s -t 43200 &        # another terminal, 12 hours
#   bash run_v8.sh
#
# Total about 11.5 hours. It stops at the first failure and says why.
#
# ---------------------------------------------------------------------------
# WHY THIS REPLACES run_v5/v6/v7
#
# run_v6 printed the overlap verdict and trained anyway.
# run_v7 gated on the crossover script's exit code, which is useless:
#   paper21c_crossover_v5.py main() collects failed cells into a list, prints
#   "CELLS THAT FAILED", and returns 0 regardless. A queue can lose three of
#   nine cells and report success. Completion is therefore verified by READING
#   THE CELL FILES, not by testing $?.
#
# The silent failure that matters most: the crossover script decides to skip a
# cell with set(want).issubset(set(done)) -- checkpoint steps ONLY. It never
# compares --model, --mimic4-min-year, --train-pool or --pool. A PubMedBERT run
# into a workdir holding mpnet cells prints "SKIP: already complete" and hands
# back the mpnet numbers under the PubMedBERT label. Two defences here:
# preflight refuses to start if any target workdir holds a cell file, and
# paper21c_verify_v1.py reads every finished cell's recorded model and
# restriction and fails if they are not what was asked for.
#
# ---------------------------------------------------------------------------
# THE RESTRICTION, AND WHERE IT DOES NOT APPLY
#
# MIMIC-III covers 2001-2012 and cannot be dated per admission: its real years
# were destroyed at de-identification, so no cut is possible on that side.
# MIMIC-IV's 2008-2010 and 2011-2013 bands overlap MIMIC-III; 2014-2016 and
# 2017-2019 cannot. All 115 detected duplicates sat in the two early bands and
# none in the later two, so --mimic4-min-year 2013 removes every one of them.
#
# The within-MIMIC-IV control is run WITHOUT the restriction. It uses MIMIC-IV
# only -- _split_within drops every other corpus -- so no MIMIC-III note is ever
# present and cross-corpus contamination is impossible. Restricting it would
# shrink an 11-year span to 5 and cost the control its power.
#
# ---------------------------------------------------------------------------
# WHY --train-pool IS COMPUTED, NOT TYPED
#
# --train-pool is one global number and prep_contrastive takes
# min(len(pool), train_pool). If one arm holds fewer notes than the flag, that
# arm trains on less data and "own beats cross" becomes confounded with volume.
# The preflight reads the real pool of every arm through load_cohort and emits
# the largest safe value. The cross-corpus and within-corpus comparisons get
# their own values, because each is compared against itself.
#
# Nothing in this script writes to paper21c_x, _v5, _v6, _v7 or any earlier
# workdir. Those cells are contaminated and must not appear in any table.

set -u
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

TWO_SITE="$HOME/projects/paper9/two_site.py"
OUT="$HOME/projects/paper21a/out"
LOGDIR="$HOME/projects/paper21c_logs"
ENVF="$LOGDIR/v8_preflight.env"

MAIN="$HOME/projects/paper21c_v8"
MIX="$HOME/projects/paper21c_v8_mix"
PUBMED="$HOME/projects/paper21c_v8_pubmed"
WITHIN="$HOME/projects/paper21c_v8_within"

MINYEAR=2013
POOL=1000
SEED=42
MPNET="sentence-transformers/all-mpnet-base-v2"
PUBMODEL="NeuML/pubmedbert-base-embeddings"
CKPTS="250,500,1000,2500"
STEPS=2500

mkdir -p "$LOGDIR"
stamp() { date "+%Y-%m-%d %H:%M:%S"; }
die() {
  echo
  echo "############################################################"
  echo "### $(stamp)  STOPPED: $1"
  echo "### Nothing after this point was run."
  echo "############################################################"
  exit 1
}

# cells() workdir logname [extra flags...]
cells() {
  local wd="$1"; local name="$2"; shift 2
  local log="$LOGDIR/${name}.log"
  echo "=== $(stamp)  START $name  -> $log"
  uv run paper21c_crossover_v5.py --two-site "$TWO_SITE" --outdir "$OUT" \
      --workdir "$wd" --pool "$POOL" --seed "$SEED" \
      --con-steps "$STEPS" --con-checkpoints "$CKPTS" "$@" 2>&1 | tee "$log"
  # PIPESTATUS must be read before anything else runs, including the command
  # substitution in an echo, which would overwrite $?.
  local rc=${PIPESTATUS[0]}
  # rc is recorded but NOT used as a pass/fail signal: crossover_v5 main()
  # returns 0 even when cells fail. Completion is verified by reading the cell
  # files in paper21c_verify_v1.py.
  echo "=== $(stamp)  $name returned $rc (recorded, not trusted)"
  echo
}

echo "############################################################"
echo "### $(stamp)  Paper 21 v8 — full model-side run"
echo "############################################################"
echo

# =========================================================================
# STAGE -1  RESUME CHECK — makes a restart safe after a crash
# =========================================================================
# An 11-hour run that dies at hour 8 must not force a full redo. But resuming
# into an occupied workdir is only safe if the cells already there were run with
# the same model, restriction, pool and seed: the crossover script skips a cell
# on checkpoint steps alone and would otherwise hand back numbers from a
# different configuration. So: if anything is already on disk, every cell
# present must match before the run is allowed to continue.
OCCUPIED=0
for w in "$MAIN" "$MIX" "$PUBMED" "$WITHIN"; do
  if [ -d "$w" ] && compgen -G "$w/cell_*.npz" > /dev/null; then OCCUPIED=1; fi
done
ALLOW=""
if [ "$OCCUPIED" -eq 1 ]; then
  echo "=== $(stamp)  STAGE -1  cells already on disk; checking a resume is safe"
  uv run paper21c_verify_v1.py --allow-missing --skip-logs \
      --check "path=$MAIN,model=$MPNET,min_year=$MINYEAR,scrub=0,pool=$POOL,seed=$SEED,cells=contrastive:mimic3;contrastive:mimic4" \
      --check "path=$MAIN,model=$MPNET,min_year=$MINYEAR,scrub=1,pool=$POOL,seed=$SEED,cells=contrastive:mimic3;contrastive:mimic4" \
      --check "path=$MIX,model=$MPNET,min_year=$MINYEAR,scrub=0,pool=$POOL,seed=$SEED,cells=contrastive:mixed" \
      --check "path=$PUBMED,model=$PUBMODEL,min_year=$MINYEAR,scrub=0,pool=$POOL,seed=$SEED,cells=contrastive:mimic3;contrastive:mimic4" \
      --check "path=$WITHIN,model=$MPNET,min_year=None,scrub=0,pool=$POOL,seed=$SEED,cells=contrastive:mimic4_early;contrastive:mimic4_late" \
      2>&1 | tee "$LOGDIR/v8_stage-1_resume.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || die "cells on disk do not match this configuration. Delete $MAIN $MIX $PUBMED $WITHIN and start again, or resolve the mismatches listed in $LOGDIR/v8_stage-1_resume.log"
  ALLOW="--allow-occupied"
  echo "=== $(stamp)  STAGE -1 PASSED — resuming; matching cells will be skipped"
  echo
fi

# =========================================================================
# STAGE 0  PREFLIGHT — every assumption checked before any compute
# =========================================================================
echo "=== $(stamp)  STAGE 0  preflight (about 15 min)"
rm -f "$ENVF"
uv run paper21c_preflight_v1.py --two-site "$TWO_SITE" --outdir "$OUT" \
    --pool "$POOL" --seed "$SEED" --min-year "$MINYEAR" \
    --workdirs "$MAIN,$MIX,$PUBMED,$WITHIN" $ALLOW \
    --emit "$ENVF" 2>&1 | tee "$LOGDIR/v8_stage0_preflight.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || die "preflight failed; read $LOGDIR/v8_stage0_preflight.log"
[ -f "$ENVF" ] || die "preflight passed but wrote no env file"
# shellcheck disable=SC1090
. "$ENVF"
: "${TRAINPOOL_CROSS:?preflight did not emit TRAINPOOL_CROSS}"
: "${TRAINPOOL_WITHIN:?preflight did not emit TRAINPOOL_WITHIN}"
: "${SPLITYEAR:?preflight did not emit SPLITYEAR}"
echo "=== $(stamp)  STAGE 0 PASSED"
echo "###   cross-corpus --train-pool $TRAINPOOL_CROSS"
echo "###   within-corpus --train-pool $TRAINPOOL_WITHIN  --split-year $SPLITYEAR"
echo

# =========================================================================
# STAGE 1  GATE: the restriction must remove the overlap, provably
# =========================================================================
# Exit 0 pass, 2 duplicates remain, 3 planted recovery too low, 4 no control.
# 3 and 4 are failures on purpose: an absence is only evidence if the search
# could have found something. --plant copies 200 MIMIC-III notes into the
# MIMIC-IV side at three damage levels; the 55% figure is what the methods
# section quotes.
echo "=== $(stamp)  STAGE 1  overlap gate (about 5 min)"
uv run paper21c_overlap_v4.py --outdir "$OUT" \
    --min-year "$MINYEAR" --restrict mimic4 \
    --plant 200 --min-recall 0.80 --show 2 --seed "$SEED" \
    --export "$HOME/projects/paper21c_v8_residual_dups.csv" \
    2>&1 | tee "$LOGDIR/v8_stage1_overlap.log"
case "${PIPESTATUS[0]}" in
  0) echo "=== $(stamp)  STAGE 1 PASSED";;
  2) die "cross-corpus duplicates remain after --mimic4-min-year $MINYEAR (exit 2)";;
  3) die "planted-duplicate recovery below 80%: the null is unsupported (exit 3)";;
  4) die "no planted control ran: the null is unmeasured (exit 4)";;
  *) die "overlap gate exited ${PIPESTATUS[0]}";;
esac
echo

# =========================================================================
# STAGE 2  the primary 2x2 and the dose ladder   (about 2.4 h)
# =========================================================================
cells "$MAIN" v8_main \
  --mimic4-min-year "$MINYEAR" --train-pool "$TRAINPOOL_CROSS" \
  --cells contrastive:mimic3,contrastive:mimic4 --eras mimic3,mimic4

uv run paper21c_verify_v1.py --logdir "$LOGDIR" \
    --expect-train-pool "v8_main=$TRAINPOOL_CROSS" \
    --check "path=$MAIN,model=$MPNET,min_year=$MINYEAR,scrub=0,pool=$POOL,seed=$SEED,steps=250;500;1000;2500,cells=contrastive:mimic3;contrastive:mimic4" \
    2>&1 | tee "$LOGDIR/v8_stage2_verify.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || die "the primary cells did not verify; read $LOGDIR/v8_stage2_verify.log"

# =========================================================================
# STAGE 3  GATE: no evaluated document may be in the other era's training set
# =========================================================================
# Run here, not at the end: a failure must stop the remaining 8 hours.
echo "=== $(stamp)  STAGE 3  leak audit (about 4 min)"
uv run paper21c_leakcheck_v3.py --two-site "$TWO_SITE" --outdir "$OUT" \
    --workdir "$MAIN" --overlap "$HERE/paper21c_overlap_v4.py" \
    --mimic4-min-year "$MINYEAR" --train-pool "$TRAINPOOL_CROSS" \
    --pool "$POOL" --seed "$SEED" --show 2 \
    --export "$HOME/projects/paper21c_v8_contaminated.csv" \
    2>&1 | tee "$LOGDIR/v8_stage3_leakcheck.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || die "contaminated evaluated targets remain; the primary result is not reportable"
echo "=== $(stamp)  STAGE 3 PASSED"
echo

# =========================================================================
# STAGE 4  the retraining case   (about 1.2 h)
# =========================================================================
# The mixed pool is built at --train-pool/2 per era, so its total matches the
# single-era arms rather than doubling them.
cells "$MIX" v8_mixed \
  --mimic4-min-year "$MINYEAR" --train-pool "$TRAINPOOL_CROSS" \
  --cells contrastive:mixed --eras mimic3,mimic4

# =========================================================================
# STAGE 5  within-MIMIC-IV control   (about 2.4 h)  NO --mimic4-min-year
# =========================================================================
# This is the control that separates a regime change from corpus identity. It
# must keep the full 2008-2019 span, which is why the restriction is absent.
cells "$WITHIN" v8_within \
  --within mimic4 --split-year "$SPLITYEAR" --train-pool "$TRAINPOOL_WITHIN" \
  --cells "contrastive:mimic4_early,contrastive:mimic4_late" \
  --eras mimic4_early,mimic4_late

# =========================================================================
# STAGE 6  second encoder   (about 2.3 h)   own workdir, or it would be SKIPPED
# =========================================================================
cells "$PUBMED" v8_pubmed \
  --model "$PUBMODEL" \
  --mimic4-min-year "$MINYEAR" --train-pool "$TRAINPOOL_CROSS" \
  --cells contrastive:mimic3,contrastive:mimic4 --eras mimic3,mimic4

# =========================================================================
# STAGE 7  de-identification arm   (about 2.4 h)   same workdir, _scrubbed tag
# =========================================================================
cells "$MAIN" v8_scrubbed \
  --scrub-deid \
  --mimic4-min-year "$MINYEAR" --train-pool "$TRAINPOOL_CROSS" \
  --cells contrastive:mimic3,contrastive:mimic4 --eras mimic3,mimic4

# =========================================================================
# STAGE 8  verify every cell against what was requested
# =========================================================================
echo "=== $(stamp)  STAGE 8  full verification"
uv run paper21c_verify_v1.py --logdir "$LOGDIR" \
    --expect-train-pool "v8_main=$TRAINPOOL_CROSS,v8_mixed=$TRAINPOOL_CROSS,v8_pubmed=$TRAINPOOL_CROSS,v8_scrubbed=$TRAINPOOL_CROSS,v8_within=$TRAINPOOL_WITHIN" \
    --check "path=$MAIN,model=$MPNET,min_year=$MINYEAR,scrub=0,pool=$POOL,seed=$SEED,cells=contrastive:mimic3;contrastive:mimic4" \
    --check "path=$MAIN,model=$MPNET,min_year=$MINYEAR,scrub=1,pool=$POOL,seed=$SEED,cells=contrastive:mimic3;contrastive:mimic4" \
    --check "path=$MIX,model=$MPNET,min_year=$MINYEAR,scrub=0,pool=$POOL,seed=$SEED,cells=contrastive:mixed" \
    --check "path=$PUBMED,model=$PUBMODEL,min_year=$MINYEAR,scrub=0,pool=$POOL,seed=$SEED,cells=contrastive:mimic3;contrastive:mimic4" \
    --check "path=$WITHIN,model=$MPNET,min_year=None,scrub=0,pool=$POOL,seed=$SEED,cells=contrastive:mimic4_early;contrastive:mimic4_late" \
    2>&1 | tee "$LOGDIR/v8_stage8_verify.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || die "full verification failed; read $LOGDIR/v8_stage8_verify.log"
echo "=== $(stamp)  STAGE 8 PASSED"
echo

# =========================================================================
# STAGE 9  the contrasts
# =========================================================================
echo "=== $(stamp)  STAGE 9  contrasts"
uv run paper21c_margin_v1.py --workdir "$MAIN" \
    --out "$HOME/projects/paper21c_v8_margin.json" 2>&1 | tee "$LOGDIR/v8_margin_main.log"
uv run paper21c_margin_v1.py --workdir "$MAIN" --scrubbed \
    --out "$HOME/projects/paper21c_v8_margin_scrubbed.json" 2>&1 | tee "$LOGDIR/v8_margin_scrubbed.log"
uv run paper21c_margin_v1.py --workdir "$PUBMED" \
    --out "$HOME/projects/paper21c_v8_margin_pubmed.json" 2>&1 | tee "$LOGDIR/v8_margin_pubmed.log"
uv run paper21c_margin_v1.py --workdir "$WITHIN" --eras mimic4_early,mimic4_late \
    --out "$HOME/projects/paper21c_v8_margin_within.json" 2>&1 | tee "$LOGDIR/v8_margin_within.log"
uv run paper21c_mixed_v1.py --workdir "$MAIN" --mixdir "$MIX" \
    --out "$HOME/projects/paper21c_v8_mixed.json" 2>&1 | tee "$LOGDIR/v8_mixed.log"

echo
echo "COLLAPSE AND CLIPPING SUMMARY"
for p in "$MAIN" "$MIX" "$PUBMED" "$WITHIN"; do
  [ -d "$p" ] || continue
  echo "  $p"
  uv run --with numpy python - "$p" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
for f in sorted(Path(sys.argv[1]).glob("cell_*.npz")):
    m = json.loads(str(np.load(f, allow_pickle=True)["meta"].item()))
    bad = m.get("collapsed_at_step")
    print(f"    {f.name:44} "
          f"{'COLLAPSED at ' + format(bad, ',') if bad else 'alive'}"
          f"   clipped {m.get('clipped_steps', '?')}"
          f"   min_year {m.get('mimic4_min_year')}"
          f"   {str(m.get('model', ''))[-22:]}")
PY
done

echo
echo "############################################################"
echo "### $(stamp)  FINISHED — every gate and both verifications passed"
echo "############################################################"
echo
echo "Numbers the manuscript needs, and where they are:"
echo "  planted-duplicate recovery at 55%   $LOGDIR/v8_stage1_overlap.log"
echo "  contaminated evaluated targets (0)  $LOGDIR/v8_stage3_leakcheck.log"
echo "  b, c, d and recall, primary         $LOGDIR/v8_margin_main.log"
echo "  de-identification decomposition     $LOGDIR/v8_margin_scrubbed.log"
echo "  second-encoder replication          $LOGDIR/v8_margin_pubmed.log"
echo "  within-regime control               $LOGDIR/v8_margin_within.log"
echo "  retraining case                     $LOGDIR/v8_mixed.log"
echo
echo "Not run: the MLM arm. Add it only if the objective comparison is reported."
echo "Logs: $LOGDIR"
