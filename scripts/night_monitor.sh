#!/bin/bash
# Overnight watch over the fetchers, one line every INTERVAL to a log.
#
# The per-fetcher watchdog already restarts a dead or silent process. This sits
# above it and catches what it cannot see:
#
#   * the WATCHDOG ITSELF dying, which would leave nothing supervising anything
#   * empty shards reappearing, which is the FileNotFoundError-as-permanent bug
#     coming back and is worth knowing before it poisons another 4,000 groups
#   * the disk filling
#   * no progress at all across several intervals while everything looks alive
#
# It restarts the watchdog but never the fetchers: that is the watchdog's job,
# and two supervisors racing to start the same process is how duplicates appear.
#
#   scripts/night_monitor.sh &            # log to data dir, 300 s cadence
#   INTERVAL=600 scripts/night_monitor.sh &
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
D="${DR2_DIR:-$HOME/astroai/stanford_deadline/data/dr2}"
LOG="${NIGHT_LOG:-$D/night_monitor.log}"
INTERVAL="${INTERVAL:-300}"
STALL_SECS="${STALL_SECS:-2700}"
SHARDS="$D/spectra_dr2_new.shards"
POOL="$D/fits_pool_dr2"

count () { ls "$1"/*."$2" 2>/dev/null | wc -l; }
newest_age () {
  local n
  n=$(find "$1" -maxdepth 1 -name "*.$2" -printf '%T@\n' 2>/dev/null | sort -rn | head -1)
  [ -z "$n" ] && { echo 999999; return; }
  echo $(( $(date +%s) - ${n%.*} ))
}

prev_s=$(count "$SHARDS" npz)
prev_c=$(count "$POOL" fits)
flat=0

echo "[night] started $(date '+%F %T'), interval ${INTERVAL}s, log $LOG" >> "$LOG"
while true; do
  sleep "$INTERVAL"
  s=$(count "$SHARDS" npz); c=$(count "$POOL" fits)
  ds=$(( s - prev_s )); dc=$(( c - prev_c ))
  rate_s=$(( ds * 3600 / INTERVAL )); rate_c=$(( dc * 3600 / INTERVAL ))
  # empty shards are the signature of the transient-as-permanent bug returning
  empty=$(find "$SHARDS" -maxdepth 1 -name "*.npz" -size -2k 2>/dev/null | wc -l)
  age_s=$(newest_age "$SHARDS" npz); age_c=$(newest_age "$POOL" fits)
  free=$(df -BG --output=avail "$D" | tail -1 | tr -dc '0-9')
  up_w=$(pgrep -f "[f]etch_watchdog" >/dev/null && echo 1 || echo 0)
  up_s=$(pgrep -f "[f]etch_desi_spectra" >/dev/null && echo 1 || echo 0)
  up_c=$(pgrep -f "[f]etch_ls_cutouts"   >/dev/null && echo 1 || echo 0)

  flags=""
  [ "$empty" -gt 0 ] && flags="$flags EMPTY_SHARDS=$empty"
  [ "$free" -lt 20 ] && flags="$flags LOW_DISK=${free}G"
  [ "$up_w" -eq 0 ] && flags="$flags WATCHDOG_DOWN"
  if [ "$ds" -eq 0 ] && [ "$dc" -eq 0 ]; then
    flat=$(( flat + 1 ))
    [ "$flat" -ge 3 ] && flags="$flags NO_PROGRESS_x$flat"
  else
    flat=0
  fi

  printf '[night] %s shards %s (+%s, %s/h, age %ss) cutouts %s (+%s, %s/h, age %ss) up w%s s%s c%s free %sG%s\n' \
    "$(date '+%F %T')" "$s" "$ds" "$rate_s" "$age_s" "$c" "$dc" "$rate_c" "$age_c" \
    "$up_w" "$up_s" "$up_c" "$free" "$flags" >> "$LOG"

  # Restart the supervisor if it died. Never the fetchers themselves.
  if [ "$up_w" -eq 0 ]; then
    echo "[night] watchdog down, restarting it ($(date '+%F %T'))" >> "$LOG"
    STALL_SECS="$STALL_SECS" setsid nohup bash "$REPO/scripts/fetch_watchdog.sh" both \
      >> "$D/watchdog.log" 2>&1 < /dev/null &
    disown 2>/dev/null || true
  fi
  prev_s=$s; prev_c=$c
done
