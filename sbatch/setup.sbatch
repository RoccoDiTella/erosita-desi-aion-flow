#!/bin/bash
#SBATCH -J aionflow-setup
#SBATCH -p shared
#SBATCH -c 8
#SBATCH --mem 24G
#SBATCH -t 0-03:00
#SBATCH -o logs/setup_%j.out
#SBATCH -e logs/setup_%j.err
#SBATCH --open-mode=append
# One-time setup ON A COMPUTE NODE (never the login node): build the conda env
# and unzip the FITS pool. Submit with: scripts/submit.sh sbatch/setup.sbatch
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source .fasrc.env

echo "[1/2] build conda env (idempotent)"
bash scripts/fasrc_build_env.sh

echo "[2/2] unzip fits_pool if needed"
if [ ! -d "$AIONFLOW_DATA/raw/legacysurvey/fits_pool" ]; then
  ( cd "$AIONFLOW_DATA/raw/legacysurvey" && unzip -q -o fits_pool.zip )
fi
echo "fits count: $(ls "$AIONFLOW_DATA/raw/legacysurvey/fits_pool" 2>/dev/null | wc -l)"
echo "SETUP DONE"
