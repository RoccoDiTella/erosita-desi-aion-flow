#!/bin/bash
# PREFLIGHT: verify the FASRC destination is the CORRECT, writable place BEFORE
# staging any data. Run from the workstation over the ControlMaster socket:
#     bash scripts/fasrc_preflight.sh
# It (a) checks the socket, (b) SETTLES account+partition eligibility with a
# dry-run submit, (c) lists candidate lab/scratch storage so you can set
# AIONFLOW_ROOT correctly, and (d) write-tests the configured AIONFLOW_ROOT and
# reports free space. Nothing is transferred and no job is queued.
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO_ROOT/.fasrc.env" ] || { echo "ERROR: copy .fasrc.env.example -> .fasrc.env and fill it"; exit 1; }
source "$REPO_ROOT/.fasrc.env"
SOCK="${FASRC_SOCKET:-$HOME/.ssh/cm-fasrc}"
HOST="${FASRC_HOST:-rditella@login.rc.fas.harvard.edu}"
run(){ ssh -S "$SOCK" "$HOST" "$@"; }

echo "== 1. socket =="
ssh -S "$SOCK" -O check "$HOST" || { echo "ABORT: socket not up (see FASRC_NOTES.md)"; exit 1; }

echo "== 2. identity / SLURM associations =="
run 'echo "user=$(whoami)  primary_group=$(id -gn)"; groups | tr " " "\n" | sort; sacctmgr -nP show assoc user=$USER format=account,partition | sort -u'

# THE MEMBERSHIP TEST IS --test-only, NOT sinfo.
#
# This step used to run `sinfo -p siag_gpu` and read "partition not visible" as
# "membership pending". That is no longer a valid inference: siag_gpu IS visible
# in sinfo without siag_lab membership (observed 2026-08-06), so visibility says
# nothing either way. A dry-run submit is the only thing that settles it, it
# costs seconds, and it queues nothing.
#
# Measured 2026-08-06, which is why FASRC_ACCOUNT is finkbeiner_lab:
#   -A siag_lab      -p siag_gpu -> "Invalid account or account/partition combination"
#   -A finkbeiner_lab -p gpu     -> schedules
#   -A finkbeiner_lab -p gpu_test-> schedules
ACC="${FASRC_ACCOUNT:?set FASRC_ACCOUNT in .fasrc.env}"
echo "== 3. eligibility: dry-run submit for -A $ACC (nothing is queued) =="
for part in "${FASRC_PART:-gpu}" "${FASRC_PART_SMOKE:-gpu_test}"; do
  printf '   -A %-16s -p %-10s : ' "$ACC" "$part"
  run "sbatch --test-only -A '$ACC' -p '$part' --gpus 1 -t 00:05:00 --wrap true" 2>&1 | tail -1 \
    || echo "   (!) NOT eligible for -p $part under -A $ACC — do not launch"
done
echo "   (sinfo visibility is NOT a membership test; only the line above is.)"

echo "== 4. candidate storage (pick a WRITABLE one for AIONFLOW_ROOT) =="
run 'for d in /n/netscratch/'"$ACC"' /n/netscratch/$USER /n/holylabs/LABS/'"$ACC"' /n/holylabs/LABS/'"$ACC"'/Users/$USER $HOME; do [ -e "$d" ] && echo "$(ls -ld "$d" 2>/dev/null)"; done'

echo "== 5. configured AIONFLOW_ROOT write-test: ${AIONFLOW_ROOT} =="
if run "mkdir -p '$AIONFLOW_DATA' '$AIONFLOW_ROOT/hf_cache' 2>/dev/null && touch '$AIONFLOW_ROOT/.write_test' 2>/dev/null && rm -f '$AIONFLOW_ROOT/.write_test'"; then
  echo "   WRITABLE ✓"
  run "df -h '$AIONFLOW_ROOT' | tail -1; echo 'need >= ~25G free for data (16G) + env (~6G)'"
else
  echo "   (!) NOT writable — fix AIONFLOW_ROOT in .fasrc.env to a path from step 4, then re-run."
  exit 1
fi
echo "PREFLIGHT PASSED — destination confirmed. Safe to run scripts/fasrc_stage_data.sh"
echo "   That pushes the DR2 pair (clean_split_dr2.csv + targets_sidecar_dr2.csv) FLAT"
echo "   into \$AIONFLOW_DATA, which is where sbatch/_dataset.sh looks. Without them"
echo "   every launcher exits at the gate."
