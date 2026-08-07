#!/bin/bash
# STAGE DATA: push ONLY the allowlisted inputs to the CONFIRMED FASRC path.
# No blanket rsync. Refuses to run unless the destination is writable (run
# scripts/fasrc_preflight.sh first). Verifies every SOURCE exists before sending.
#   bash scripts/fasrc_stage_data.sh --dry-run   # preview
#   bash scripts/fasrc_stage_data.sh             # transfer
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/.fasrc.env"
SOCK="${FASRC_SOCKET:-$HOME/.ssh/cm-fasrc}"
HOST="${FASRC_HOST:-rditella@login.rc.fas.harvard.edu}"
RSH="ssh -S $SOCK"
DRY=""; [ "$1" = "--dry-run" ] && DRY="--dry-run"

# --- SOURCE allowlist (verified to exist below) ---
#
# THE DR2 PAIR IS THE POINT OF THIS SCRIPT NOW. `sbatch/_dataset.sh` requires
# DATASET=dr2 and resolves $AIONFLOW_DATA/clean_split_dr2.csv +
# $AIONFLOW_DATA/targets_sidecar_dr2.csv. Until 2026-08-06 this allowlist pushed
# only the DR1 pair, so a fresh stage-in produced a cluster on which every
# launcher exits at the _dataset.sh gate with "does not exist" -- the data was
# never wrong, it was never sent. Both files go FLAT into $AIONFLOW_DATA, not
# into a dr2/ subdirectory as they sit locally; see .fasrc.env.
SD=/home/roccoditella/astroai/stanford_deadline
ERO="$SD/aion_project/shareable_aion_flow/data/raw/erosita_desi"
MAN="$SD/aion_project/shareable_aion_flow/data/manifests"
declare -A PLAN=(
  ["$ERO/erosita_spectra_merged_32k.hdf5"]="$AIONFLOW_DATA/raw/erosita_desi"
  ["$ERO/erosita_desi_matches_Xray_properties.csv"]="$AIONFLOW_DATA/raw/erosita_desi"
  # Only source of DESI spectype/zwarn for the current sample; build_manifest.py
  # needs it for has_z, and plan step 12 needs it for the class column.
  ["$ERO/erosita_desi_dr1_matches_all_properties.csv"]="$AIONFLOW_DATA/raw/erosita_desi"
  # --- dr2v2 quartet: what a CURRENT run actually consumes ---
  # This block used to hold ONLY the un-suffixed dr2 pair, so every dr2v2 path
  # `sbatch/_dataset.sh` resolves existed on the workstation and nowhere else,
  # and `DATASET=dr2v2 source sbatch/_dataset.sh` on the cluster exited
  # "targets_sidecar_dr2_v2.csv does not exist". That is the same failure this
  # file's own header describes for the DR1 -> DR2 move: the data was never
  # wrong, it was never sent. manifest_dr2.csv rides along because it is what
  # sbatch/prepare_data_v2.sbatch stages FROM; without it the cluster can
  # resolve the labels but cannot build the inputs.
  ["$SD/data/dr2/clean_split_dr2_v2.csv"]="$AIONFLOW_DATA"
  ["$SD/data/dr2/targets_sidecar_dr2_v2.csv"]="$AIONFLOW_DATA"
  ["$SD/data/dr2/manifest_dr2.csv"]="$AIONFLOW_DATA"
  # --- superseded dr2 pair: DATASET=dr2 only, to reproduce a published number ---
  ["$SD/data/dr2/clean_split_dr2.csv"]="$AIONFLOW_DATA"
  ["$SD/data/dr2/targets_sidecar_dr2.csv"]="$AIONFLOW_DATA"
  # --- DR1 leftovers, retained for a named reason each ---
  # match_quality: the spec-z audit, an input to scripts/make_split.py.
  ["$SD/data/match_quality.csv"]="$AIONFLOW_DATA"
  # targets_extra: the ONLY source of hr32 (--hr-ref-csv). Not trainable.
  ["$SD/data/targets_extra.csv"]="$AIONFLOW_DATA"
  # DR1 sidecar. Cannot be trained on -- it has none of the nine detection
  # columns the DR2 target spec requires, and _dataset.sh's dr1 branch is a hard
  # refusal. Kept only so a DR1-era artifact can be reproduced from history.
  ["$SD/data/targets_sidecar.csv"]="$AIONFLOW_DATA"
  ["$MAN/"]="$AIONFLOW_DATA/manifests"
  ["$SD/fits_pool.zip"]="$AIONFLOW_DATA/raw/legacysurvey"
  ["$HOME/.cache/huggingface/hub/models--polymathic-ai--aion-base"]="$AIONFLOW_ROOT/hf_cache/hub"
)

# --- guards ---
: "${AIONFLOW_DATA:?set AIONFLOW_DATA in .fasrc.env}"
echo "== destination write-check: $AIONFLOW_ROOT =="
$RSH "$HOST" "test -w '$AIONFLOW_ROOT'" || { echo "ABORT: $AIONFLOW_ROOT not writable — run scripts/fasrc_preflight.sh first."; exit 1; }
echo "== verifying every source exists =="
for s in "${!PLAN[@]}"; do [ -e "$s" ] || { echo "ABORT: missing source $s"; exit 1; }; done
echo "all sources present ✓"

$RSH "$HOST" "mkdir -p '$AIONFLOW_DATA/raw/erosita_desi' '$AIONFLOW_DATA/raw/legacysurvey' '$AIONFLOW_DATA/manifests' '$AIONFLOW_ROOT/hf_cache/hub'"
for s in "${!PLAN[@]}"; do
  d="${PLAN[$s]}"
  echo ">> $s  ->  $HOST:$d/"
  rsync -a --info=progress2 $DRY --no-owner --no-group -e "$RSH" "$s" "$HOST:$d/"
done
echo "STAGED (allowlist only). Next on the cluster:"
echo "  1. unzip fits_pool.zip into raw/legacysurvey/fits_pool/ (compare file COUNTS, not existence)"
echo "  2. confirm the DR2 pair landed flat where _dataset.sh looks for it:"
echo "       ls -l '$AIONFLOW_DATA/clean_split_dr2.csv' '$AIONFLOW_DATA/targets_sidecar_dr2.csv'"
echo "  3. DATASET=dr2 RUN_TAG=... scripts/submit.sh --export=ALL sbatch/train_multi.sbatch"
echo "     (_dataset.sh gates on the sidecar columns AND on median(flux_sig_lo) in (0.10,0.14),"
echo "      so a DR1 file under a DR2 name is caught before any GPU time is spent.)"
