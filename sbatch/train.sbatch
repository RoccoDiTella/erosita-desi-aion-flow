#!/bin/bash
#SBATCH -J aionflow-v1
#SBATCH -p siag_gpu
#SBATCH --gpus 1
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -t 0-12:00
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err
#SBATCH --open-mode=append
# Full V1 training on the siag_gpu A100s. Head flags default to the V1 minimal
# head; override via env (HEAD_ARGS) for other variants.
# Submit with: scripts/submit.sh sbatch/train.sbatch
#   (or override partition: scripts/submit.sh -p gpu_requeue sbatch/train.sbatch)
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/fasrc_env.sh

HEAD_ARGS="${HEAD_ARGS:---num-queries 1 --num-layers 1 --context-hidden 128}"

python -m shareable_aion_flow.main train \
  --staged-dir "$AIONFLOW_STAGED" \
  --target log_ml_flux_1 \
  $HEAD_ARGS \
  --epochs 50 --batch-size 448 --eval-batch-size 448 --num-workers 16 \
  --run-id "v1-minimal-$SLURM_JOB_ID" \
  --output-dir outputs \
  --eval-after-train
