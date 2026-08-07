#!/bin/bash
#SBATCH -J aionflow-train
#SBATCH --account=finkbeiner_lab
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -t 0-12:00
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err
#SBATCH --open-mode=append
#
# SINGLE-TARGET training (`main.py train`) over the staged superset. The
# multi-target run of record is sbatch/train_multi.sbatch; this file is the
# one-head path, kept for the paper reproduction and for single-target probes.
#
# Runs on `gpu` (A100-80GB) under account finkbeiner_lab. Do NOT retarget this
# at `-p siag_gpu`: siag_lab has not landed (verified against the scheduler
# 2026-08-06 -- `sbatch --test-only -A siag_lab -p siag_gpu` returns "Invalid
# account or account/partition combination"). siag_gpu IS visible in sinfo, so
# seeing it there is not evidence of membership.
#
#   DATASET=dr2 EPOCHS=15 scripts/submit.sh --export=ALL -t 06:00:00 sbatch/train.sbatch
# Paper reproduction = native staged splits, paper head:
#   DATASET=dr2 HEAD_ARGS="" NATIVE_SPLITS=1 RUN_TAG=repro \
#     scripts/submit.sh --export=ALL sbatch/train.sbatch
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/fasrc_env.sh
# Names the dataset and resolves + gates CLEAN_SPLIT_CSV / EXTRA_TARGETS_CSV /
# STAGED_DIR. Requires DATASET; there is no default. This file defaulted to the
# DR1 trio (clean_split.csv, staged_paper) until 2026-08-06, which is how a DR1
# checkpoint could be produced under a DR2-labelled run.
source sbatch/_dataset.sh

HEAD_ARGS="${HEAD_ARGS---num-queries 1 --num-layers 1 --context-hidden 128}"
TARGET="${TARGET:-log_ml_flux_1}"
RUN_TAG="${RUN_TAG:-v1-clean}"
EPOCHS="${EPOCHS:-50}"
# NATIVE_SPLITS=1 uses the splits baked into the staged HDF5 instead of the
# split CSV. Note CLEAN_SPLIT_CSV="" no longer does this: _dataset.sh resolves
# it with `${CLEAN_SPLIT_CSV:-<default>}`, which treats empty as unset and puts
# the default straight back. This flag is the supported way to ask for it.
NATIVE_SPLITS="${NATIVE_SPLITS:-}"
[ -n "$NATIVE_SPLITS" ] && CLEAN_SPLIT_CSV=""

# Gate: never burn GPU-hours on bad data. Any FAIL aborts before training.
# The superset legitimately contains duplicates + NWAY rejects (paper design),
# so the gate runs in superset mode; the cleaned config is a view on top.
#
# 30,762 is the row count of the CURRENT staged copy (the paper-layout superset:
# 32,092 manifest rows less 1,330 missing-cutout drops). It is a property of the
# staged HDF5, not of the DR2 labels, which is why DR2 did not move it -- DR2
# relabelled through the sidecar and nothing was re-staged. Plan step 17
# re-stages inputs-only and this number changes with it; override with
# AIONFLOW_EXPECT_ROWS rather than editing, and see docs/DATA.md §0.
#
# --labels names which staging contract to assert. dr2v2 stages INPUTS ONLY, so
# the validator must demand the X-ray labels are ABSENT -- the same thing
# _dataset.sh gates on. It used to demand the opposite unconditionally, which
# made this gate reject every dr2v2 staged dir and fail a fresh clone before
# training started. --target none goes with it: an inputs-only file carries no
# label to select rows on.
if [ "${DATASET:-}" = "dr2v2" ]; then
  LABEL_ARGS=(--labels forbidden --target none)
else
  LABEL_ARGS=(--labels required --target "$TARGET")
fi
echo "=== validating $STAGED_DIR ==="
python scripts/validate_staged.py \
  --staged-dir "$STAGED_DIR" \
  --match-quality-csv "$AIONFLOW_DATA/match_quality.csv" \
  --paper-manifest-csv "$AIONFLOW_DATA/manifests/aion_tvsplit_manifest.csv" \
  --allow-duplicates \
  --expect-rows "${AIONFLOW_EXPECT_ROWS:-30762}" \
  "${LABEL_ARGS[@]}"
echo "=== validation passed, starting training ==="

VIEW_ARGS=()
[ -n "$CLEAN_SPLIT_CSV" ] && VIEW_ARGS=(--clean-split-csv "$CLEAN_SPLIT_CSV")
[ -n "$MAX_TARGET_SIGMA" ] && VIEW_ARGS+=(--max-target-sigma "$MAX_TARGET_SIGMA")

# `--error-mode none` is HARDCODED and there is no ERROR_MODE knob any more.
# The estimand is p(y|x). Injection smears each label by its own split-normal
# error bar, and on the X-ray targets that bar is most of the residual variance,
# so it fits the measurement kernel rather than the conditional; errors belong
# in the reported floor. This is the same decision train_multi.sbatch expresses
# as `--no-inject` -- that flag exists only on `train-multi`, so on this
# single-target path `--error-mode none` IS the equivalent. Until 2026-08-06
# this line read `--error-mode "$ERROR_MODE"` under a comment advertising
# `inject` as "the adopted training mode", which is the reverse of the decision
# in docs/decisions.md §3 and §11.6.
python -m shareable_aion_flow.main train \
  --staged-dir "$STAGED_DIR" \
  --target "$TARGET" \
  --error-mode none \
  "${VIEW_ARGS[@]}" \
  $HEAD_ARGS \
  --epochs "$EPOCHS" --batch-size "${BATCH_SIZE:-448}" --eval-batch-size "${BATCH_SIZE:-448}" --num-workers 16 \
  --run-id "${RUN_TAG}-${TARGET}-${SLURM_JOB_ID}" \
  --output-dir "$AIONFLOW_ROOT/outputs" \
  --eval-after-train \
  --wandb --wandb-mode "${AIONFLOW_WANDB_MODE:-online}"
