#!/bin/bash
#SBATCH -J aionflow-v1smoke
#SBATCH -p gpu_test
#SBATCH --gpus 1
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -t 0-02:00
#SBATCH -o logs/v1smoke_%j.out
#SBATCH -e logs/v1smoke_%j.err
#SBATCH --open-mode=append
# V1 minimal-head smoke on a MIG A100 slice (gpu_test). Uses a small staged
# subset if AIONFLOW_STAGED_SMOKE is set (build it with prepare-data --limit).
# Submit with: scripts/submit.sh sbatch/train_smoke.sbatch
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/fasrc_env.sh

STAGED="${AIONFLOW_STAGED_SMOKE:-$AIONFLOW_STAGED}"
echo "V1 smoke on staged dir: $STAGED"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true

# gpu_test hands out a ~20 GB A100 MIG slice that can already be partly occupied,
# and AION's image tokenizer dominates activation memory -> keep the smoke batch
# small. Real runs on `gpu` (full 80 GB A100) use sbatch/train.sbatch.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BS="${AIONFLOW_SMOKE_BS:-32}"

python -m shareable_aion_flow.main train \
  --staged-dir "$STAGED" \
  --target log_ml_flux_1 \
  --error-mode convolve \
  --num-queries 1 --num-layers 1 --context-hidden 128 \
  --epochs 2 --batch-size "$BS" --eval-batch-size "$BS" --num-workers 8 \
  --run-id "v1-minimal-smoke-${SLURM_JOB_ID}" \
  --output-dir "$AIONFLOW_ROOT/outputs" \
  --eval-after-train \
  --wandb --wandb-mode "${AIONFLOW_WANDB_MODE:-online}" --log-every 5
