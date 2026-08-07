#!/bin/bash
#SBATCH -J aionflow-prepare
#SBATCH --account=finkbeiner_lab
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
#
# DR1-ERA INPUTS, DELIBERATELY, and the SPLIT IT DERIVES IS TARGETID-GROUPED.
# Every path below is DR1 (merged HDF5, paper manifest, fits_pool, targets_extra)
# because DR2 relabelled through the sidecar and nothing was re-staged. The
# `--clean` re-split inside prepare-data groups on desi_targetid, which does not
# separate X-ray photons: 773 detuids carry more than one DESI fibre (measured
# 2026-08-06). Prefer sbatch/prepare_data_paper.sbatch (superset staging, no
# split) plus scripts/make_split.py, which groups on ero_detuid -- the five-step
# chain in docs/DATA.md section 7. Plan step 17 supersedes this file.
#
# It does NOT source sbatch/_dataset.sh: that gate asserts the staged HDF5
# already exists, which is what this job creates.
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
