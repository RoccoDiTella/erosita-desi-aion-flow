#!/bin/bash
#SBATCH -J aionflow-eval
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -t 0-02:00
#SBATCH -o logs/eval_%j.out
#SBATCH -e logs/eval_%j.err
#SBATCH --open-mode=append
# Test-set eval of a train-multi checkpoint: the 15-combo per-head tables, the
# SFR-vs-mass baseline check, and the implied-HR quadrature. Everything the deck
# needs from a run comes out of this one job.
#
#   CKPT=$AIONFLOW_ROOT/outputs/mt-v4-sfr-35828655/best.pt \
#     scripts/submit.sh --export=ALL sbatch/eval_multi.sbatch
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/fasrc_env.sh

CKPT="${CKPT:?set CKPT to the checkpoint to evaluate}"
STAGED_DIR="${STAGED_DIR:-$AIONFLOW_STAGED}"
CLEAN_SPLIT_CSV="${CLEAN_SPLIT_CSV:-$AIONFLOW_DATA/clean_split.csv}"
# must be the SFR-bearing sidecar; targets_bands.csv silently drops the head
EXTRA_TARGETS_CSV="${EXTRA_TARGETS_CSV:-$AIONFLOW_DATA/targets_sidecar.csv}"
HR_REF_CSV="${HR_REF_CSV:-$AIONFLOW_DATA/targets_extra.csv}"

COMMON=(--checkpoint "$CKPT" --staged-dir "$STAGED_DIR"
        --clean-split-csv "$CLEAN_SPLIT_CSV"
        --extra-targets-csv "$EXTRA_TARGETS_CSV")

echo "=== per-head tables + SFR-vs-mass baseline ==="
python scripts/eval_multitarget.py "${COMMON[@]}" --hr-ref-csv "$HR_REF_CSV"

echo "=== HR32 as an implied target (exact quadrature) ==="
python scripts/hr_from_joint.py "${COMMON[@]}" --hr-ref-csv "$HR_REF_CSV"

echo "=== outputs ==="
ls -la "$(dirname "$CKPT")/eval"
