#!/bin/bash
#SBATCH -J aionflow-prepare
#SBATCH -p shared
#SBATCH -c 16
#SBATCH --mem 48G
#SBATCH -t 0-01:00
#SBATCH -o logs/prepare_%j.out
#SBATCH -e logs/prepare_%j.err
#SBATCH --open-mode=append
# Build the CLEANED, image-backed staged train/val/test HDF5 ON A COMPUTE NODE.
# NWAY clean filter + dedup + re-split + per-source error columns.
# Submit AFTER sbatch/setup.sbatch (env + unzip). e.g.:
#   scripts/submit.sh sbatch/prepare_data.sbatch
#   (or gate on setup: scripts/submit.sh --dependency=afterok:<setup_jobid> sbatch/prepare_data.sbatch)
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/fasrc_env.sh

python -m shareable_aion_flow.main prepare-data \
  --source-hdf5 "$AIONFLOW_DATA/raw/erosita_desi/erosita_spectra_merged_32k.hdf5" \
  --split-manifest-csv "$AIONFLOW_DATA/manifests/aion_tvsplit_manifest.csv" \
  --fits-pool-dir "$AIONFLOW_DATA/raw/legacysurvey/fits_pool" \
  --match-quality-csv "$AIONFLOW_DATA/match_quality.csv" \
  --targets-extra-csv "$AIONFLOW_DATA/targets_extra.csv" \
  --clean \
  --output-dir "$AIONFLOW_STAGED" \
  --overwrite

echo "cleaned staged splits written to $AIONFLOW_STAGED"
