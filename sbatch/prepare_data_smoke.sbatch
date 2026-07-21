#!/bin/bash
#SBATCH -J aionflow-prep-smoke
#SBATCH -p shared
#SBATCH -c 8
#SBATCH --mem 24G
#SBATCH -t 0-01:00
#SBATCH -o logs/prep_smoke_%j.out
#SBATCH -e logs/prep_smoke_%j.err
#SBATCH --open-mode=append
# Small CLEANED staged subset for GPU smoke tests -> $AIONFLOW_STAGED_SMOKE.
# Same cleaning path as the full job, just --limit'ed, so the smoke exercises the
# real code (clean filter + dedup + re-split + per-source error columns).
#   scripts/submit.sh sbatch/prepare_data_smoke.sbatch
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
  --limit 2000 \
  --output-dir "$AIONFLOW_STAGED_SMOKE" \
  --overwrite

echo "smoke staged splits written to $AIONFLOW_STAGED_SMOKE"
