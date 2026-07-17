#!/bin/bash
# PREFLIGHT: verify the FASRC destination is the CORRECT, writable place BEFORE
# staging any data. Run from the workstation over the ControlMaster socket:
#     bash scripts/fasrc_preflight.sh
# It (a) checks the socket, (b) confirms account + siag_gpu, (c) lists candidate
# lab/scratch storage so you can set AIONFLOW_ROOT correctly, and (d) write-tests
# the configured AIONFLOW_ROOT and reports free space. Nothing is transferred.
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO_ROOT/.fasrc.env" ] || { echo "ERROR: copy .fasrc.env.example -> .fasrc.env and fill it"; exit 1; }
source "$REPO_ROOT/.fasrc.env"
SOCK="${FASRC_SOCKET:-$HOME/.ssh/cm-fasrc}"
HOST="${FASRC_HOST:-rditella@login.rc.fas.harvard.edu}"
run(){ ssh -S "$SOCK" "$HOST" "$@"; }

echo "== 1. socket =="
ssh -S "$SOCK" -O check "$HOST" || { echo "ABORT: socket not up (see FASRC_NOTES.md)"; exit 1; }

echo "== 2. identity / SLURM account =="
run 'echo "user=$(whoami)  primary_group=$(id -gn)"; groups | tr " " "\n" | grep -i siag || echo "(!) not in a siag group yet"; sacctmgr -nP show assoc user=$USER format=account,partition | sort -u'

echo "== 3. partition ${FASRC_PART:-siag_gpu} visible? =="
run "sinfo -p ${FASRC_PART:-siag_gpu} -o '%P %a %l %D %G' 2>&1 | head || echo '(!) partition not visible — siag_lab membership pending?'"

echo "== 4. candidate storage (pick a WRITABLE one for AIONFLOW_ROOT) =="
run 'for d in /n/netscratch/'"${FASRC_ACCOUNT:-siag_lab}"' /n/netscratch/$USER /n/holylabs/LABS/'"${FASRC_ACCOUNT:-siag_lab}"' /n/holylabs/LABS/'"${FASRC_ACCOUNT:-siag_lab}"'/Users/$USER $HOME; do [ -e "$d" ] && echo "$(ls -ld "$d" 2>/dev/null)"; done'

echo "== 5. configured AIONFLOW_ROOT write-test: ${AIONFLOW_ROOT} =="
if run "mkdir -p '$AIONFLOW_DATA' '$AIONFLOW_ROOT/hf_cache' 2>/dev/null && touch '$AIONFLOW_ROOT/.write_test' 2>/dev/null && rm -f '$AIONFLOW_ROOT/.write_test'"; then
  echo "   WRITABLE ✓"
  run "df -h '$AIONFLOW_ROOT' | tail -1; echo 'need >= ~25G free for data (16G) + env (~6G)'"
else
  echo "   (!) NOT writable — fix AIONFLOW_ROOT in .fasrc.env to a path from step 4, then re-run."
  exit 1
fi
echo "PREFLIGHT PASSED — destination confirmed. Safe to run scripts/fasrc_stage_data.sh"
