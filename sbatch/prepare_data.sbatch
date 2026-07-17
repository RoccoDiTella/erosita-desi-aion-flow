#!/bin/bash
#SBATCH -J aionflow-prepare
#SBATCH -p shared
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -t 0-03:00
#SBATCH -o logs/prepare_%j.out
#SBATCH -e logs/prepare_%j.err
#SBATCH --open-mode=append
# Build the image-backed staged train/val/test HDF5 from the raw merged catalog +
# manifest + FITS pool. CPU job. Submit with: scripts/submit.sh sbatch/prepare_data.sbatch
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/fasrc_env.sh

python -m shareable_aion_flow.main prepare-data \
  --source-hdf5 "$AIONFLOW_DATA/raw/erosita_desi/erosita_spectra_merged_32k.hdf5" \
  --split-manifest-csv "$AIONFLOW_DATA/manifests/aion_tvsplit_manifest.csv" \
  --fits-pool-dir "$AIONFLOW_DATA/raw/legacysurvey/fits_pool" \
  --output-dir "$AIONFLOW_STAGED" \
  --overwrite

echo "staged splits written to $AIONFLOW_STAGED"
