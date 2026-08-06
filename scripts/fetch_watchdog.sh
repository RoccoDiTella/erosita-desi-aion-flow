#!/bin/bash
# Keep a fetcher alive across laptop suspends, network drops and crashes.
#
# Both fetchers are exactly resumable (a shard or cutout either exists complete
# or does not exist), so restarting one is always safe and never redoes work.
# The failure that actually cost us 11 hours was neither a crash nor an exit:
# the process stayed ALIVE holding dead sockets and produced nothing. So dying
# is not the only thing worth watching for -- silence is.
#
# This watchdog therefore restarts on BOTH conditions:
#   * the process exited, or
#   * no new output file has appeared for --stall-secs.
#
#   scripts/fetch_watchdog.sh spectra   # or: cutouts
#   scripts/fetch_watchdog.sh both
#
# Stop it with:  pkill -f fetch_watchdog
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
D="${DR2_DIR:-$HOME/astroai/stanford_deadline/data/dr2}"
STALL="${STALL_SECS:-900}"          # 15 min of no new files = wedged
POLL="${POLL_SECS:-60}"

newest_age () {                      # seconds since the newest matching file
  local dir="$1" pat="$2" newest
  newest=$(find "$dir" -maxdepth 1 -name "$pat" -printf '%T@\n' 2>/dev/null | sort -rn | head -1)
  [ -z "$newest" ] && { echo 999999; return; }
  echo $(( $(date +%s) - ${newest%.*} ))
}

start_spectra () {
  nohup python3 "$REPO/scripts/fetch_desi_spectra.py" \
    --targets "$D/new_targets_nway.csv" --out "$D/spectra_dr2_new.h5" --workers 6 \
    >> "$D/fetch.log" 2>&1 &
  echo "[watchdog] started spectra pid $!" >&2
}

start_cutouts () {
  nohup python3 "$REPO/scripts/fetch_ls_cutouts.py" \
    --targets "$D/new_targets_nway.csv" --out "$D/fits_pool_dr2" \
    --prioritize "$D/spectra_dr2_new.shards" --rescan-every 500 \
    >> "$D/cutouts.log" 2>&1 &
  echo "[watchdog] started cutouts pid $!" >&2
}

supervise () {                       # $1 name  $2 pattern  $3 dir  $4 starter
  local name="$1" pat="$2" dir="$3" starter="$4"
  # Time of the last (re)start. Stall is judged against BOTH the newest file
  # and this, because a freshly started process must not be condemned by a
  # timestamp it had no chance to beat. Without the grace period, clearing old
  # output makes every restart look instantly stalled and the watchdog kills
  # the fetcher in a loop, which is exactly what happened once a batch of
  # 696 MB coadds needed longer than STALL to produce their first file.
  local started=0
  while true; do
    local running age since_start
    case "$name" in
      spectra) pgrep -f "[f]etch_desi_spectra.py" >/dev/null && running=1 || running=0 ;;
      cutouts) pgrep -f "[f]etch_ls_cutouts.py"   >/dev/null && running=1 || running=0 ;;
    esac
    age=$(newest_age "$dir" "$pat")
    since_start=$(( $(date +%s) - started ))
    if [ "$running" -eq 0 ]; then
      echo "[watchdog] $name not running -- starting ($(date '+%F %T'))" >&2
      started=$(date +%s)
      $starter
    elif [ "$age" -gt "$STALL" ] && [ "$since_start" -gt "$STALL" ]; then
      echo "[watchdog] $name STALLED: no new file for ${age}s -- restarting ($(date '+%F %T'))" >&2
      case "$name" in
        spectra) pkill -9 -f "[f]etch_desi_spectra.py" ;;
        cutouts) pkill -9 -f "[f]etch_ls_cutouts.py" ;;
      esac
      sleep 5
      started=$(date +%s)
      $starter
    fi
    sleep "$POLL"
  done
}

case "${1:-both}" in
  spectra) supervise spectra "*.npz"  "$D/spectra_dr2_new.shards" start_spectra ;;
  cutouts) supervise cutouts "*.fits" "$D/fits_pool_dr2"          start_cutouts ;;
  both)
    supervise spectra "*.npz"  "$D/spectra_dr2_new.shards" start_spectra &
    supervise cutouts "*.fits" "$D/fits_pool_dr2"          start_cutouts &
    wait ;;
  *) echo "usage: $0 [spectra|cutouts|both]" >&2; exit 2 ;;
esac
