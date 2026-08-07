#!/bin/bash
#SBATCH -J aionflow-multi
#SBATCH --account=finkbeiner_lab
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
#   DATASET=dr2 RUN_TAG=mt-dr2 scripts/submit.sh --export=ALL sbatch/train_multi.sbatch
#   DATASET=dr2 RUN_TAG=mt-smoke EPOCHS=2 scripts/submit.sh -p gpu_test -t 00:40:00 \
#     --export=ALL sbatch/train_multi.sbatch
#
# `scripts/submit.sh` passes -A from .fasrc.env, which overrides the header
# above; the header is here so a bare `sbatch` charges the right account too.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/fasrc_env.sh
# Resolves + gates CLEAN_SPLIT_CSV, EXTRA_TARGETS_CSV, STAGED_DIR. Requires
# DATASET; exits here rather than at the first forward if the sidecar is wrong.
source sbatch/_dataset.sh

# Named, not defaulted: two runs both called "mt-v4-sfr" is how the registry in
# docs/decisions.md stops being readable. The SLURM job id is appended below.
RUN_TAG="${RUN_TAG:?set RUN_TAG to name this run, e.g. RUN_TAG=mt-dr2-joint}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-896}"
LR="${LR:-3e-4}"
ADAPTER_LR="${ADAPTER_LR:-1e-4}"
ADAPTER_WD="${ADAPTER_WD:-1e-1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
BETA1="${BETA1:-0.95}"
# Current scope: the joint over (M*, SFR, Lx, P3) plus a marginal head for each,
# and the flux marginal (the paper's headline target, 100% coverage, free).
# Everything else is parked. NOTE: dropping P2 removes the old P2xP3 joint, so
# this run produces no HR32.
DROP_HEADS="${DROP_HEADS:-logmstar log_flux_p1 log_flux_p2 log_flux_p4 log_mbh_pan25 log_mbh_vo09}"
HEAD_LR="${HEAD_LR:-}"                 # flows; empty -> same as LR
HEAD_LOSS_WEIGHTS="${HEAD_LOSS_WEIGHTS:-}"   # e.g. "joint=2 log_flux_p3=0.5"
# Two-phase training was REMOVED 2026-08-06: JOINT_ONLY/--joint-only,
# --snapshot-every, refit_heads.py and refit_compare.py are gone. Models train
# from scratch on a declared head set, and a different head set is a different
# job. The EMA pin that JOINT_ONLY used to trigger survives, re-derived from
# N_HEADS == 1 -- the condition it was always really about, since with a single
# loss the 1/EMA normaliser is a second LR schedule fighting the cosine.
LR_SCHEDULE="${LR_SCHEDULE:-constant}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
GRAD_CLIP="${GRAD_CLIP:-5.0}"
DIAG_EVERY="${DIAG_EVERY:-50}"
JOINT_QUAD_NODES="${JOINT_QUAD_NODES:-}"
JOINT_QUAD_SPAN="${JOINT_QUAD_SPAN:-}"

# --no-inject is passed unconditionally, and there is no INJECT_SAMPLES knob any
# more. Injection smears each label by its own split-normal error bar, which on
# the X-ray targets is most of the residual variance, so it trains the flow
# toward the measurement kernel rather than toward p(y|x). The estimand is
# p(y|x) and the errors belong in the reported floor, not in the objective.
# `--inject-samples` still exists in main.py and defaults to 50; it does nothing
# once --no-inject is set, and nothing here should ever set it.
python -m shareable_aion_flow.main train-multi \
  --staged-dir "$STAGED_DIR" \
  --clean-split-csv "$CLEAN_SPLIT_CSV" \
  --extra-targets-csv "$EXTRA_TARGETS_CSV" \
  ${DROP_HEADS:+--drop-heads $DROP_HEADS} \
  --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
  --bucketed --accumulate-buckets --grad-checkpoint \
  --lr "$LR" --lr-schedule "$LR_SCHEDULE" \
  --warmup-steps "$WARMUP_STEPS" --grad-clip "$GRAD_CLIP" \
  --diag-every "$DIAG_EVERY" \
  ${JOINT_QUAD_NODES:+--joint-quad-nodes $JOINT_QUAD_NODES} \
  ${JOINT_QUAD_SPAN:+--joint-quad-span $JOINT_QUAD_SPAN} \
  --adapter-lr "$ADAPTER_LR" --adapter-wd "$ADAPTER_WD" \
  --weight-decay "$WEIGHT_DECAY" --beta1 "$BETA1" \
  --no-inject \
  ${HEAD_LR:+--head-lr $HEAD_LR} \
  $(for w in $HEAD_LOSS_WEIGHTS; do printf -- "--head-loss-weight %s " "$w"; done) \
  --num-workers 8 \
  --run-id "${RUN_TAG}-${SLURM_JOB_ID}" \
  --output-dir "$AIONFLOW_ROOT/outputs" \
  --wandb
