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
SD=/home/roccoditella/astroai/stanford_deadline
ERO="$SD/aion_project/shareable_aion_flow/data/raw/erosita_desi"
MAN="$SD/aion_project/shareable_aion_flow/data/manifests"
declare -A PLAN=(
  ["$ERO/erosita_spectra_merged_32k.hdf5"]="$AIONFLOW_DATA/raw/erosita_desi"
  ["$ERO/erosita_desi_matches_Xray_properties.csv"]="$AIONFLOW_DATA/raw/erosita_desi"
  ["$ERO/erosita_desi_dr1_matches_all_properties.csv"]="$AIONFLOW_DATA/raw/erosita_desi"
  ["$SD/data/match_quality.csv"]="$AIONFLOW_DATA"
  ["$SD/data/targets_extra.csv"]="$AIONFLOW_DATA"
  # Runtime sidecar for train-multi: band targets + log_sfr, joined by targetid.
  # ~9 MB, rebuilt locally by scripts/make_sfr_sidecar.py -- the 7.3 GB CIGALE
  # VAC it is derived from is NEVER pushed.
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
echo "STAGED (allowlist only). Next on the cluster: unzip fits_pool.zip in raw/legacysurvey, then run prepare-data."
