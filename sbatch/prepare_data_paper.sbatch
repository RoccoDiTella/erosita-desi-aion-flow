#!/bin/bash
#SBATCH -J aionflow-prep-paper
#SBATCH -p shared
#SBATCH -c 16
#SBATCH --mem 48G
#SBATCH -t 0-06:00
#SBATCH -o logs/prep_paper_%j.out
#SBATCH -e logs/prep_paper_%j.err
#SBATCH --open-mode=append
# CANONICAL (superset) staging: the original split manifest, NO clean filter,
# WITH the extra target/error columns — this single copy serves BOTH configs:
#   paper reproduction: native files as-is (error_mode=none ignores sigma columns)
#   cleaned science:    runtime row-selection via clean_split.csv (no 2nd copy)
# Builds the smoke subset first (cheap), then the full set, then the clean-split
# sidecar CSV. Submit: scripts/submit.sh sbatch/prepare_data_paper.sbatch
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/fasrc_env.sh

echo "[1/3] paper smoke subset (--limit 2000)"
python -m shareable_aion_flow.main prepare-data \
  --source-hdf5 "$AIONFLOW_DATA/raw/erosita_desi/erosita_spectra_merged_32k.hdf5" \
  --split-manifest-csv "$AIONFLOW_DATA/manifests/aion_tvsplit_manifest.csv" \
  --fits-pool-dir "$AIONFLOW_DATA/raw/legacysurvey/fits_pool" \
  --targets-extra-csv "$AIONFLOW_DATA/targets_extra.csv" \
  --limit 2000 \
  --output-dir "$AIONFLOW_DATA/staged_paper_smoke" \
  --overwrite

echo "[2/3] paper full (superset) set"
python -m shareable_aion_flow.main prepare-data \
  --source-hdf5 "$AIONFLOW_DATA/raw/erosita_desi/erosita_spectra_merged_32k.hdf5" \
  --split-manifest-csv "$AIONFLOW_DATA/manifests/aion_tvsplit_manifest.csv" \
  --fits-pool-dir "$AIONFLOW_DATA/raw/legacysurvey/fits_pool" \
  --targets-extra-csv "$AIONFLOW_DATA/targets_extra.csv" \
  --output-dir "$AIONFLOW_DATA/staged_paper" \
  --overwrite

echo "[3/3] clean-split sidecar (runtime view of the superset)"
python scripts/make_clean_split.py \
  --split-manifest-csv "$AIONFLOW_DATA/manifests/aion_tvsplit_manifest.csv" \
  --match-quality-csv "$AIONFLOW_DATA/match_quality.csv" \
  --output-csv "$AIONFLOW_DATA/clean_split.csv"

echo "canonical staged data written to $AIONFLOW_DATA/staged_paper{,_smoke} + clean_split.csv"
