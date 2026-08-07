#!/bin/bash
#SBATCH -J aionflow-prepare-v2
#SBATCH --account=finkbeiner_lab
#SBATCH -p shared
#SBATCH -c 16
#SBATCH --mem 8G
# 8G is MEASURED, not guessed: job 37694119 staged all 22,800 rows with
# MaxRSS 963 MB. It asked for 48G, i.e. 2% utilisation. Over-requesting memory
# on a shared cluster delays our own jobs, holds RAM other users cannot have,
# and counts against fairshare -- and an inflated request is what got the first
# gpu_test submission rejected outright. 8G is ~8x the measured peak.
#SBATCH -t 0-02:00
#SBATCH -o logs/prepare_v2_%j.out
#SBATCH -e logs/prepare_v2_%j.err
#SBATCH --open-mode=append
#
# Build `staged_v2/` -- the INPUTS-ONLY staged train/val/test HDF5 that every
# current launcher consumes via `sbatch/_dataset.sh` DATASET=dr2v2.
#
# THIS IS THE MISSING COLD-START STEP. Until it existed, nothing in the repo
# could produce staged_v2 on a fresh machine: `_dataset.sh` asserted the
# directory, `fasrc_stage_data.sh` shipped the CSVs, and no launcher built the
# HDF5 from them. A clone could resolve every label and still not train.
#
#   scripts/submit.sh sbatch/prepare_data_v2.sbatch
#   (gate on setup: scripts/submit.sh --dependency=afterok:<setup_jobid> ...)
#
# HOW THIS DIFFERS FROM prepare_data.sbatch, which it supersedes:
#
#  * It stages from `manifest_dr2.csv`, the availability table built by
#    scripts/build_manifest.py, which already carries the split column and the
#    four presence flags. prepare_data.sbatch used the DR1 paper manifest.
#  * NO `--clean`. That flag re-splits inside prepare-data grouped on
#    desi_targetid, which does not separate X-ray photons: 756 detections carry
#    more than one DESI fibre. The split is decided upstream by
#    scripts/make_split.py, grouped on ero_detuid, and arrives in the manifest.
#  * NO `--targets-extra-csv`. Staging writes NO labels at all. Every label
#    comes from the sidecar at load time, so a staged copy cannot go stale
#    against the catalogue. `_dataset.sh` gates on their absence and
#    `validate_staged.py --labels forbidden` asserts it.
#
# It does NOT source sbatch/_dataset.sh: that gate asserts the staged HDF5
# already exists, which is exactly what this job creates.
#
# Cutouts are OPTIONAL. Sources with no Legacy Survey cutout are staged with a
# zero image and has_image=False rather than dropped -- dropping them once
# turned ~22k rows into 9,592 when a pool was half-unzipped, and Run A does not
# use images at all.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/fasrc_env.sh

MANIFEST="${AIONFLOW_MANIFEST:-$AIONFLOW_DATA/manifest_dr2.csv}"
OUT="${AIONFLOW_STAGED_V2:-$AIONFLOW_DATA/staged_v2}"

for f in "$MANIFEST" \
         "$AIONFLOW_DATA/raw/erosita_desi/erosita_spectra_merged_32k.hdf5"; do
  [ -e "$f" ] || { echo "FAIL: missing input $f -- run scripts/fasrc_stage_data.sh first"; exit 1; }
done

echo "=== staging $MANIFEST -> $OUT (inputs only) ==="
python -m shareable_aion_flow.main prepare-data \
  --source-hdf5 "$AIONFLOW_DATA/raw/erosita_desi/erosita_spectra_merged_32k.hdf5" \
  --split-manifest-csv "$MANIFEST" \
  --fits-pool-dir "$AIONFLOW_DATA/raw/legacysurvey/fits_pool" \
  --output-dir "$OUT" \
  --overwrite

# Assert the contract this staging exists to satisfy, before any GPU is booked.
echo "=== verifying the inputs-only contract ==="
python scripts/validate_staged.py \
  --staged-dir "$OUT" \
  --labels forbidden \
  --target none \
  --skip-dataloader

echo "staged_v2 written to $OUT and verified inputs-only"
