#!/bin/bash
#SBATCH -J aionflow-v1
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -t 0-12:00
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err
#SBATCH --open-mode=append
# Full V1 training. Runs on `gpu` (full A100s) under finkbeiner_lab; switch to
# `-p siag_gpu` once siag_lab membership lands. Head flags default to the V1
# minimal head; override via env (HEAD_ARGS) for other variants.
#   scripts/submit.sh sbatch/train.sbatch
#   TARGET=log_lx scripts/submit.sh sbatch/train.sbatch
#   scripts/submit.sh -p gpu_requeue sbatch/train.sbatch      # override partition
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/fasrc_env.sh

HEAD_ARGS="${HEAD_ARGS:---num-queries 1 --num-layers 1 --context-hidden 128}"
TARGET="${TARGET:-log_ml_flux_1}"
ERROR_MODE="${ERROR_MODE:-convolve}"

# Gate: never burn GPU-hours on bad data. Any FAIL aborts before training.
echo "=== validating $AIONFLOW_STAGED ==="
python scripts/validate_staged.py \
  --staged-dir "$AIONFLOW_STAGED" \
  --match-quality-csv "$AIONFLOW_DATA/match_quality.csv" \
  --expect-clean \
  --expect-rows "${AIONFLOW_EXPECT_ROWS:-25200}" \
  --target "$TARGET"
echo "=== validation passed, starting training ==="

python -m shareable_aion_flow.main train \
  --staged-dir "$AIONFLOW_STAGED" \
  --target "$TARGET" \
  --error-mode "$ERROR_MODE" \
  $HEAD_ARGS \
  --epochs 50 --batch-size "${BATCH_SIZE:-448}" --eval-batch-size "${BATCH_SIZE:-448}" --num-workers 16 \
  --run-id "v1-${TARGET}-${SLURM_JOB_ID}" \
  --output-dir "$AIONFLOW_ROOT/outputs" \
  --eval-after-train \
  --wandb --wandb-mode "${AIONFLOW_WANDB_MODE:-online}"
