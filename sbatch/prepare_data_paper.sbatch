#!/bin/bash
#SBATCH -J aionflow-prep-paper
#SBATCH -p shared
#SBATCH -c 16
#SBATCH --mem 48G
#SBATCH -t 0-06:00
#SBATCH -o logs/prep_paper_%j.out
#SBATCH -e logs/prep_paper_%j.err
#SBATCH --open-mode=append
# PAPER-REPRODUCTION staging: the original split manifest, NO clean filter, NO
# extra target/error columns — the exact data configuration the PAI26 table was
# trained on. Builds the smoke subset first (cheap), then the full set, so a
# repro smoke can be chained on this job. Keeps the cleaned `staged/` untouched.
#   scripts/submit.sh sbatch/prepare_data_paper.sbatch
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/fasrc_env.sh

echo "[1/2] paper smoke subset (--limit 2000)"
python -m shareable_aion_flow.main prepare-data \
  --source-hdf5 "$AIONFLOW_DATA/raw/erosita_desi/erosita_spectra_merged_32k.hdf5" \
  --split-manifest-csv "$AIONFLOW_DATA/manifests/aion_tvsplit_manifest.csv" \
  --fits-pool-dir "$AIONFLOW_DATA/raw/legacysurvey/fits_pool" \
  --limit 2000 \
  --output-dir "$AIONFLOW_DATA/staged_paper_smoke" \
  --overwrite

echo "[2/2] paper full set"
python -m shareable_aion_flow.main prepare-data \
  --source-hdf5 "$AIONFLOW_DATA/raw/erosita_desi/erosita_spectra_merged_32k.hdf5" \
  --split-manifest-csv "$AIONFLOW_DATA/manifests/aion_tvsplit_manifest.csv" \
  --fits-pool-dir "$AIONFLOW_DATA/raw/legacysurvey/fits_pool" \
  --output-dir "$AIONFLOW_DATA/staged_paper" \
  --overwrite

echo "paper staged splits written to $AIONFLOW_DATA/staged_paper{,_smoke}"
