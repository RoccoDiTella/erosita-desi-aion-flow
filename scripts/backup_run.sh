#!/bin/bash
# Copy a run's keeper artifacts off purge-eligible netscratch into home.
# Keeps: best checkpoint, config, prior, curves, eval CSVs/tables, packet.
# Skips: last.pt (transient), wandb/ (synced to the cloud already).
#   bash scripts/backup_run.sh <run-id> [<run-id> ...]
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/.fasrc.env"
DEST_ROOT="${AIONFLOW_BACKUP:-$HOME/aionflow_results}"

for run_id in "$@"; do
  src="$AIONFLOW_ROOT/outputs/$run_id"
  [ -d "$src" ] || { echo "SKIP (no such run): $src"; continue; }
  dest="$DEST_ROOT/$run_id"
  mkdir -p "$dest"
  rsync -a \
    --include='best.pt' --include='config.json' --include='kde_prior.npz' \
    --include='epoch_metrics.jsonl' --include='metrics.json' \
    --include='test_flow_metrics.csv' --include='test_predictions.csv' \
    --include='results_table.*' --include='packet/***' \
    --exclude='*' \
    "$src/" "$dest/"
  echo "backed up $run_id -> $dest ($(du -sh "$dest" | cut -f1))"
done
