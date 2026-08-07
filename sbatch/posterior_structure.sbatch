#!/bin/bash
#SBATCH -J aionflow-poststruct
#SBATCH --account=finkbeiner_lab
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -c 8
#SBATCH --mem 80G
#SBATCH -t 0-00:30
#SBATCH -o logs/poststruct_%j.out
#SBATCH -e logs/poststruct_%j.err
#SBATCH --open-mode=append
#
# Does conditioning on the optical data REVERSE the sign of a correlation that
# is positive in the population? Draws the joint posterior for every validation
# source and compares four correlation matrices (population, predicted,
# conditional, residual). See scripts/posterior_structure.py for what each one
# means and which comparison is the defensible claim.
#
# One forward pass over the val split plus SAMPLES draws per source. Minutes,
# not hours; the 30 min wall time is slack, not a target.
#
#   DATASET=dr2 CKPT=$AIONFLOW_ROOT/outputs/p1-dr2-<jobid>/snapshots/epoch_028.pt \
#     scripts/submit.sh --export=ALL sbatch/posterior_structure.sbatch
#
# Was `-p gpu_h200` (H200, TRES weight 2651.5 against A100's 836.5). This job is
# a single forward plus sampling, so it is nowhere near the >3.2x speedup that
# would make an H200 hour cheaper than an A100 hour in fairshare terms.
#
# This file used to be the ONLY launcher defaulting to the DR2 trio while every
# other one defaulted to DR1, so it would happily analyse a DR1-trained
# checkpoint against DR2 labels and say nothing. sbatch/_dataset.sh now makes
# the dataset an explicit argument for all three.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/fasrc_env.sh
source sbatch/_dataset.sh

SAMPLES="${SAMPLES:-512}"
# Linear combinations of the joint dims (sSFR = SFR - M*). Only a joint
# posterior can produce these: a difference needs the covariance.
DERIVED="${DERIVED:-}"
# Pool held-out splits and/or split the report by a class column. The GALAXY
# subsample is small, so pooling val+test is usually needed for any power.
SPLITS="${SPLITS:-val}"
# cigale_spectype lives in the sidecar, so --group-csv defaults to the same file
# _dataset.sh resolved. It is NaN for 1,060 of 25,582 rows and contains no STARs
# (measured 2026-08-06); posterior_structure reports the ungrouped count.
GROUP_CSV="${GROUP_CSV:-}"
GROUP_COL="${GROUP_COL:-cigale_spectype}"
# Modalities to condition on. Same checkpoint, no retraining: the model was
# trained with modality dropout. COMBO="spectra z image" drops WISE.
COMBO="${COMBO:-spectra z wise image}"
: "${CKPT:?set CKPT to the body whose posterior you want to inspect}"
OUT="${OUT:-$(dirname "$CKPT")/posterior_structure.json}"

python scripts/posterior_structure.py \
  --checkpoint "$CKPT" \
  --staged-dir "$STAGED_DIR" \
  --clean-split-csv "$CLEAN_SPLIT_CSV" \
  --extra-targets-csv "$EXTRA_TARGETS_CSV" \
  --samples "$SAMPLES" \
  ${DERIVED:+--derived $DERIVED} \
  --splits $SPLITS \
  --combo $COMBO \
  ${GROUP_CSV:+--group-csv "$GROUP_CSV" --group-col "$GROUP_COL"} \
  --out "$OUT"
