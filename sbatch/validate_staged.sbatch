#!/bin/bash
#SBATCH -J aionflow-validate
#SBATCH -p gpu_test
#SBATCH --gpus 1
#SBATCH -c 8
#SBATCH --mem 24G
#SBATCH -t 0-00:30
#SBATCH -o logs/validate_%j.out
#SBATCH -e logs/validate_%j.err
#SBATCH --open-mode=append
# Validate the staged data before spending real GPU time on it: schema, row
# counts, split integrity (dedup/leakage/fractions), NWAY clean filter, per-source
# sigma, value ranges, and a real batch through AION + the head.
#   scripts/submit.sh sbatch/validate_staged.sbatch            # both staged dirs
#   AIONFLOW_VALIDATE_DIR=<dir> scripts/submit.sh sbatch/validate_staged.sbatch
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/fasrc_env.sh

# The full cleaned sample is 25,200 rows (NWAY keep + one row per targetid).
EXPECT_ROWS="${AIONFLOW_EXPECT_ROWS:-25200}"
# Set AIONFLOW_VALIDATE_CLEAN=0 for paper-repro dirs: they intentionally keep
# the NWAY-rejected matches, so asserting the clean filter would always fail.
CLEAN_FLAG="--expect-clean"
[ "${AIONFLOW_VALIDATE_CLEAN:-1}" = "0" ] && CLEAN_FLAG=""
status=0

validate() {
  local dir="$1"; shift
  if [ ! -d "$dir" ]; then
    echo "SKIP (not staged yet): $dir"
    return 0
  fi
  echo "================ $dir ================"
  python scripts/validate_staged.py \
    --staged-dir "$dir" \
    --match-quality-csv "$AIONFLOW_DATA/match_quality.csv" \
    --paper-manifest-csv "$AIONFLOW_DATA/manifests/aion_tvsplit_manifest.csv" \
    $CLEAN_FLAG \
    --check-model \
    "$@" || status=1
  echo
}

if [ -n "$AIONFLOW_VALIDATE_DIR" ]; then
  validate "$AIONFLOW_VALIDATE_DIR"
else
  # The smoke dir is --limit'ed, so its total row count is not the full 25,200.
  validate "$AIONFLOW_STAGED_SMOKE"
  validate "$AIONFLOW_STAGED" --expect-rows "$EXPECT_ROWS"
fi

echo "VALIDATION EXIT: $status"
exit $status
