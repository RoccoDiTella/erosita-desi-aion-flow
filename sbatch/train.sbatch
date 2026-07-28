#!/bin/bash
#SBATCH -J aionflow-train
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -t 0-12:00
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err
#SBATCH --open-mode=append
# Full training over the CANONICAL staged superset ($AIONFLOW_STAGED=staged_paper).
# Runs on `gpu` (A100-80GB) under finkbeiner_lab; move to `-p siag_gpu` once
# siag_lab membership lands.
#
# Defaults = the new-science config: V1 head + cleaned VIEW + convolution.
#   EPOCHS=15 scripts/submit.sh --export=ALL -t 06:00:00 sbatch/train.sbatch
# Paper reproduction = native paper splits, paper head, no error handling:
#   HEAD_ARGS="" ERROR_MODE=none CLEAN_SPLIT_CSV="" RUN_TAG=repro \
#     scripts/submit.sh --export=ALL sbatch/train.sbatch
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/fasrc_env.sh

HEAD_ARGS="${HEAD_ARGS---num-queries 1 --num-layers 1 --context-hidden 128}"
TARGET="${TARGET:-log_ml_flux_1}"
ERROR_MODE="${ERROR_MODE:-none}"   # p(y|x); pass ERROR_MODE=inject for the adopted training mode
STAGED_DIR="${STAGED_DIR:-$AIONFLOW_STAGED}"
# Cleaned re-split view over the superset; set CLEAN_SPLIT_CSV="" for native (paper) splits.
CLEAN_SPLIT_CSV="${CLEAN_SPLIT_CSV-$AIONFLOW_DATA/clean_split.csv}"
RUN_TAG="${RUN_TAG:-v1-clean}"
EPOCHS="${EPOCHS:-50}"

# Gate: never burn GPU-hours on bad data. Any FAIL aborts before training.
# The superset legitimately contains duplicates + NWAY rejects (paper design),
# so the gate runs in superset mode; the cleaned config is a view on top.
echo "=== validating $STAGED_DIR ==="
python scripts/validate_staged.py \
  --staged-dir "$STAGED_DIR" \
  --match-quality-csv "$AIONFLOW_DATA/match_quality.csv" \
  --paper-manifest-csv "$AIONFLOW_DATA/manifests/aion_tvsplit_manifest.csv" \
  --allow-duplicates \
  --expect-rows "${AIONFLOW_EXPECT_ROWS:-30762}" \
  --target "$TARGET"
echo "=== validation passed, starting training ==="

VIEW_ARGS=()
[ -n "$CLEAN_SPLIT_CSV" ] && VIEW_ARGS=(--clean-split-csv "$CLEAN_SPLIT_CSV")
[ -n "$MAX_TARGET_SIGMA" ] && VIEW_ARGS+=(--max-target-sigma "$MAX_TARGET_SIGMA")

python -m shareable_aion_flow.main train \
  --staged-dir "$STAGED_DIR" \
  --target "$TARGET" \
  --error-mode "$ERROR_MODE" \
  "${VIEW_ARGS[@]}" \
  $HEAD_ARGS \
  --epochs "$EPOCHS" --batch-size "${BATCH_SIZE:-448}" --eval-batch-size "${BATCH_SIZE:-448}" --num-workers 16 \
  --run-id "${RUN_TAG}-${TARGET}-${SLURM_JOB_ID}" \
  --output-dir "$AIONFLOW_ROOT/outputs" \
  --eval-after-train \
  --wandb --wandb-mode "${AIONFLOW_WANDB_MODE:-online}"
