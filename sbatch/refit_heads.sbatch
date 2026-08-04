#!/bin/bash
#SBATCH -J aionflow-refit
#SBATCH -p gpu_h200
#SBATCH --gpus 1
#SBATCH -c 8
#SBATCH --mem 80G
#SBATCH -t 0-01:00
#SBATCH -o logs/refit_%j.out
#SBATCH -e logs/refit_%j.err
#SBATCH --open-mode=append
#
# Phase 2: freeze the phase-1 body, refit every head with its OWN early stopping.
#
# Cheap by construction. The expensive frozen-AION forward runs ONCE per split to
# cache the context vectors; every head is then a small flow fit on cached
# tensors, so per-head early stopping costs almost nothing and cannot drag any
# other head, because the body does not move.
#
# CKPT selects which body. Phase 1 writes snapshots every SNAPSHOT_EVERY epochs
# precisely so this can be run against several and the body chosen by POST-REFIT
# validation, which is not the same epoch as the joint's own val minimum.
#
#   CKPT=$AIONFLOW_ROOT/outputs/p1-joint-<jobid>/best.pt \
#     scripts/submit.sh --export=ALL sbatch/refit_heads.sbatch
#
#   # or sweep the snapshots, one job, sequential:
#   SWEEP_SNAPSHOTS=1 RUN_DIR=$AIONFLOW_ROOT/outputs/p1-joint-<jobid> \
#     scripts/submit.sh --export=ALL sbatch/refit_heads.sbatch
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/fasrc_env.sh

CLEAN_SPLIT_CSV="${CLEAN_SPLIT_CSV:-$AIONFLOW_DATA/clean_split.csv}"
EXTRA_TARGETS_CSV="${EXTRA_TARGETS_CSV:-$AIONFLOW_DATA/targets_sidecar.csv}"
STAGED_DIR="${STAGED_DIR:-$AIONFLOW_STAGED}"
# 'all' fits every target on the frozen body, including the ones phase 1 never
# trained. That is the whole point of freezing: the body is not target-specific.
REFIT_HEADS="${REFIT_HEADS:-all}"
CONTEXT_SLOT="${CONTEXT_SLOT:-joint}"
EPOCHS="${EPOCHS:-200}"
PATIENCE="${PATIENCE:-20}"
LR="${LR:-1e-3}"

run_one () {
  local ckpt="$1" out="$2"
  echo "[refit] === $ckpt -> $out ===" >&2
  python scripts/refit_heads.py \
    --checkpoint "$ckpt" \
    --staged-dir "$STAGED_DIR" \
    --clean-split-csv "$CLEAN_SPLIT_CSV" \
    --extra-targets-csv "$EXTRA_TARGETS_CSV" \
    --refit-heads $REFIT_HEADS \
    --context-slot "$CONTEXT_SLOT" \
    --epochs "$EPOCHS" --patience "$PATIENCE" --lr "$LR" \
    --out "$out"
}

if [ -n "${SWEEP_SNAPSHOTS:-}" ]; then
  : "${RUN_DIR:?set RUN_DIR when sweeping snapshots}"
  for snap in "$RUN_DIR"/snapshots/epoch_*.pt; do
    [ -e "$snap" ] || { echo "[refit] no snapshots in $RUN_DIR" >&2; exit 1; }
    run_one "$snap" "$RUN_DIR/refit/$(basename "${snap%.pt}")"
  done
  echo "[refit] sweep done; compare refit_report.json across bodies" >&2
else
  : "${CKPT:?set CKPT to a phase-1 checkpoint}"
  run_one "$CKPT" "${OUT:-$(dirname "$CKPT")/refit}"
fi
