#!/bin/bash
#SBATCH -J aionflow-multi
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -c 8
#SBATCH --mem 80G
#SBATCH -t 0-10:00
#SBATCH -o logs/multi_%j.out
#SBATCH -e logs/multi_%j.err
#SBATCH --open-mode=append
# Multi-target read-only-CLS training (train-multi). Every earlier multi-target
# run was launched with an ad-hoc `sbatch --wrap`, which left no record of the
# invocation anywhere in the repo -- this file is that record.
#
# Defaults reproduce the current best run (mt-v3-lrfix-35416432: bs 896 bucketed
# with bucket accumulation, lr 3e-4 constant, adapters on their own 1e-4, P4
# dropped) and add the log_sfr head, which arrives through the sidecar.
#
#   scripts/submit.sh --export=ALL sbatch/train_multi.sbatch
#   EPOCHS=2 RUN_TAG=mt-smoke scripts/submit.sh -p gpu_test -t 00:40:00 \
#     --export=ALL sbatch/train_multi.sbatch
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/fasrc_env.sh

STAGED_DIR="${STAGED_DIR:-$AIONFLOW_STAGED}"
CLEAN_SPLIT_CSV="${CLEAN_SPLIT_CSV:-$AIONFLOW_DATA/clean_split.csv}"
# targets_sidecar.csv = band targets + log_sfr; targets_bands.csv is the older
# band-only sidecar and drops the SFR head silently if used by mistake.
EXTRA_TARGETS_CSV="${EXTRA_TARGETS_CSV:-$AIONFLOW_DATA/targets_sidecar.csv}"
RUN_TAG="${RUN_TAG:-mt-v4-sfr}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-896}"
LR="${LR:-3e-4}"
ADAPTER_LR="${ADAPTER_LR:-1e-4}"
ADAPTER_WD="${ADAPTER_WD:-1e-1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
BETA1="${BETA1:-0.95}"
INJECT_SAMPLES="${INJECT_SAMPLES:-50}"
# Current scope: the joint over (M*, SFR, Lx, P3) plus a marginal head for each,
# and the flux marginal (the paper's headline target, 100% coverage, free).
# Everything else is parked. NOTE: dropping P2 removes the old P2xP3 joint, so
# this run produces no HR32.
DROP_HEADS="${DROP_HEADS:-logmstar log_flux_p1 log_flux_p2 log_flux_p4 log_mbh_pan25 log_mbh_vo09}"
HEAD_LR="${HEAD_LR:-}"                 # flows; empty -> same as LR
HEAD_LOSS_WEIGHTS="${HEAD_LOSS_WEIGHTS:-}"   # e.g. "joint=2 log_flux_p3=0.5"
# Phase 1 knobs. JOINT_ONLY=1 trains the joint flow alone: no loss weighting to
# balance, because there is only one loss. LR_SCHEDULE was hardcoded to
# constant here; it is a variable now.
JOINT_ONLY="${JOINT_ONLY:-}"
LR_SCHEDULE="${LR_SCHEDULE:-constant}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
GRAD_CLIP="${GRAD_CLIP:-5.0}"
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-0}"
DIAG_EVERY="${DIAG_EVERY:-50}"
JOINT_QUAD_NODES="${JOINT_QUAD_NODES:-}"
JOINT_QUAD_SPAN="${JOINT_QUAD_SPAN:-}"

# Fail before burning GPU-hours if the sidecar is missing or has no SFR column.
python - <<PY
import sys, pandas as pd
p = "$EXTRA_TARGETS_CSV"
d = pd.read_csv(p, nrows=5)
missing = [c for c in ("targetid", "log_sfr", "log_sfr_sig_lo", "log_sfr_sig_hi")
           if c not in d.columns]
if missing:
    sys.exit(f"FAIL: {p} is missing {missing} -- did you point at targets_bands.csv?")
print(f"[gate] sidecar OK: {p}")
PY

python -m shareable_aion_flow.main train-multi \
  --staged-dir "$STAGED_DIR" \
  --clean-split-csv "$CLEAN_SPLIT_CSV" \
  --extra-targets-csv "$EXTRA_TARGETS_CSV" \
  ${DROP_HEADS:+--drop-heads $DROP_HEADS} \
  --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
  --bucketed --accumulate-buckets --grad-checkpoint \
  --lr "$LR" --lr-schedule "$LR_SCHEDULE" \
  --warmup-steps "$WARMUP_STEPS" --grad-clip "$GRAD_CLIP" \
  --snapshot-every "$SNAPSHOT_EVERY" --diag-every "$DIAG_EVERY" \
  ${JOINT_QUAD_NODES:+--joint-quad-nodes $JOINT_QUAD_NODES} \
  ${JOINT_QUAD_SPAN:+--joint-quad-span $JOINT_QUAD_SPAN} \
  ${JOINT_ONLY:+--joint-only} \
  --adapter-lr "$ADAPTER_LR" --adapter-wd "$ADAPTER_WD" \
  --weight-decay "$WEIGHT_DECAY" --beta1 "$BETA1" \
  --inject-samples "$INJECT_SAMPLES" \
  ${HEAD_LR:+--head-lr $HEAD_LR} \
  $(for w in $HEAD_LOSS_WEIGHTS; do printf -- "--head-loss-weight %s " "$w"; done) \
  --num-workers 8 \
  --run-id "${RUN_TAG}-${SLURM_JOB_ID}" \
  --output-dir "$AIONFLOW_ROOT/outputs" \
  --wandb
