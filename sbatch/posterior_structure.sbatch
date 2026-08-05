#!/bin/bash
#SBATCH -J aionflow-poststruct
#SBATCH -p gpu_h200
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
#   CKPT=$AIONFLOW_ROOT/outputs/p1-dr2-<jobid>/snapshots/epoch_028.pt \
#     scripts/submit.sh --export=ALL sbatch/posterior_structure.sbatch
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/fasrc_env.sh

CLEAN_SPLIT_CSV="${CLEAN_SPLIT_CSV:-$AIONFLOW_DATA/clean_split_dr2.csv}"
EXTRA_TARGETS_CSV="${EXTRA_TARGETS_CSV:-$AIONFLOW_DATA/targets_sidecar_dr2.csv}"
STAGED_DIR="${STAGED_DIR:-$AIONFLOW_STAGED}"
SAMPLES="${SAMPLES:-512}"
# Linear combinations of the joint dims (sSFR = SFR - M*). Only a joint
# posterior can produce these: a difference needs the covariance.
DERIVED="${DERIVED:-}"
# Pool held-out splits and/or split the report by a class column. The GALAXY
# subsample is small, so pooling val+test is usually needed for any power.
SPLITS="${SPLITS:-val}"
GROUP_CSV="${GROUP_CSV:-}"
GROUP_COL="${GROUP_COL:-cigale_spectype}"
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
  ${GROUP_CSV:+--group-csv "$GROUP_CSV" --group-col "$GROUP_COL"} \
  --out "$OUT"
