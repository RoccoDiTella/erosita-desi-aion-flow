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

echo "[2/2] unzip fits_pool if incomplete"
# Idempotency must be COUNT-based, not directory-existence based: an interrupted
# unzip leaves a populated-but-partial directory, and an existence check then
# skips the repair forever. (This bit us once: 10,355 of 27,373 cutouts on disk,
# which silently cut the staged sample from ~22k rows to 9,592.)
LS_DIR="$AIONFLOW_DATA/raw/legacysurvey"
want=$(unzip -l "$LS_DIR/fits_pool.zip" | tail -1 | awk '{print $2}')
have=$(ls "$LS_DIR/fits_pool" 2>/dev/null | wc -l)
echo "fits_pool: have=$have want=$want"
if [ "$have" -lt "$want" ]; then
  echo "incomplete -> unzipping (this can take a while)"
  ( cd "$LS_DIR" && unzip -q -o fits_pool.zip )
  have=$(ls "$LS_DIR/fits_pool" | wc -l)
fi
echo "fits count: $have / $want"
[ "$have" -ge "$want" ] || { echo "ERROR: fits_pool still incomplete"; exit 1; }
echo "SETUP DONE"
