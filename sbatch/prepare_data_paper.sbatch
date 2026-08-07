#!/bin/bash
#SBATCH -J aionflow-prep-paper
#SBATCH --account=finkbeiner_lab
#SBATCH -p shared
#SBATCH -c 16
#SBATCH --mem 48G
#SBATCH -t 0-01:00
#SBATCH -o logs/prep_paper_%j.out
#SBATCH -e logs/prep_paper_%j.err
#SBATCH --open-mode=append
# CANONICAL (superset) staging: the original split manifest, NO clean filter,
# WITH the extra target/error columns — this single copy serves BOTH configs:
#   paper reproduction: native files as-is (error_mode=none ignores sigma columns)
#   cleaned science:    runtime row-selection via a split CSV (no 2nd copy)
# Builds the smoke subset first (cheap), then the full set. Submit:
#   scripts/submit.sh sbatch/prepare_data_paper.sbatch
#
# DR1-ERA INPUTS, DELIBERATELY. Every path below is the DR1 merged HDF5, the DR1
# paper manifest and the DR1 fits_pool, because DR2 relabelled through the
# sidecar and nothing was re-staged. This job therefore reproduces the CURRENT
# staged copy (30,762 rows); it does not produce a DR2 one. Plan step 17
# re-stages inputs-only and supersedes this file. It does NOT source
# sbatch/_dataset.sh: _dataset.sh asserts that desi_{train,val,test}.hdf5
# already exist, which is exactly what this job creates.
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

echo "[3/3] split: NOT BUILT HERE ANY MORE"
cat >&2 <<'EOF'

This step used to call scripts/make_clean_split.py and write clean_split.csv.
It has been REMOVED, not merely defaulted away from.

make_clean_split.py groups the split on desi_targetid. 773 ero_detuids in the
current sample carry more than one DESI fibre (measured 2026-08-06 on the
25,582-row clean_split_dr2.csv), so a targetid-grouped split puts the SAME
X-RAY PHOTONS -- the same ML_FLUX_1, the same log_lx up to a redshift -- in
train and in test. That is a leak the metrics cannot see. 50 detuids are also
shared with the 104,945-row expansion despite zero targetid overlap, so the
same mistake recurs the moment the two are merged.

The split is now built by scripts/make_split.py, which groups on ero_detuid via
a keyed blake2b hash with no RNG, and it needs the manifest, so it does not
belong in this staging job. The chain is five steps and builds the manifest
twice; it is written out with real invocations in docs/DATA.md section 7.

make_clean_split.py is retained in the tree for reproducing DR1-era artifacts
and must not be called by any launcher.
EOF

echo "canonical staged data written to $AIONFLOW_DATA/staged_paper{,_smoke}"
echo "next: build the target table, sidecar, manifest and split per docs/DATA.md section 7"
