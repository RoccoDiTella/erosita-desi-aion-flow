#!/bin/bash
#SBATCH -J aionflow-smoke
#SBATCH -p gpu_test
#SBATCH --gpus 1
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -t 0-02:00
#SBATCH -o logs/smoke_%j.out
#SBATCH -e logs/smoke_%j.err
#SBATCH --open-mode=append
# Training smoke on a MIG A100 slice (gpu_test). Parameterized by env so one
# script covers both configurations:
#   V1 new-science smoke (default):    scripts/submit.sh sbatch/train_smoke.sbatch
#   paper-repro smoke:  STAGED=$AIONFLOW_DATA/staged_paper_smoke ERROR_MODE=none \
#                       HEAD_ARGS="" RUN_TAG=repro scripts/submit.sh --export=ALL sbatch/train_smoke.sbatch
# (empty HEAD_ARGS = the paper q4/l2 head defaults.)
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/fasrc_env.sh

STAGED="${STAGED:-${AIONFLOW_STAGED_SMOKE:-$AIONFLOW_STAGED}}"
TARGET="${TARGET:-log_ml_flux_1}"
ERROR_MODE="${ERROR_MODE:-convolve}"
HEAD_ARGS="${HEAD_ARGS---num-queries 1 --num-layers 1 --context-hidden 128}"
RUN_TAG="${RUN_TAG:-v1}"
echo "smoke: staged=$STAGED target=$TARGET error_mode=$ERROR_MODE head='${HEAD_ARGS:-paper-default}'"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true

# gpu_test hands out a ~20 GB A100 MIG slice that can already be partly occupied,
# and AION's image tokenizer dominates activation memory -> keep the smoke batch
# small. Real runs on `gpu` (full A100) use sbatch/train.sbatch.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BS="${AIONFLOW_SMOKE_BS:-32}"

python -m shareable_aion_flow.main train \
  --staged-dir "$STAGED" \
  --target "$TARGET" \
  --error-mode "$ERROR_MODE" \
  $HEAD_ARGS \
  --epochs 2 --batch-size "$BS" --eval-batch-size "$BS" --num-workers 8 \
  --run-id "smoke-${RUN_TAG}-${SLURM_JOB_ID}" \
  --output-dir "$AIONFLOW_ROOT/outputs" \
  --eval-after-train \
  --wandb --wandb-mode "${AIONFLOW_WANDB_MODE:-online}" --log-every 5
