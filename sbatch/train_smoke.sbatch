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

python -m shareable_aion_flow.main train \
  --staged-dir "$STAGED" \
  --target log_ml_flux_1 \
  --num-queries 1 --num-layers 1 --context-hidden 128 \
  --epochs 2 --batch-size 256 --eval-batch-size 256 --num-workers 8 \
  --run-id v1-minimal-smoke \
  --output-dir outputs \
  --eval-after-train
