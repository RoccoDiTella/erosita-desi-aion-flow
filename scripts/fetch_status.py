#!/usr/bin/env python
"""Status of a running fetch_desi_spectra job: rate, ETA, retries, yield.

Two things this gets right that a naive reading of the log does not:

* Elapsed time comes from the PROCESS start (ps lstart), not the log file's
  ctime. On Linux ctime is the inode-change time, so it updates on every write
  and makes any rate computed from it wildly optimistic.
* The ETA is projected PER PROGRAM. Groups are processed in (survey, program,
  healpix) order, so main/backup and main/bright are drained before main/dark
  -- and dark holds 82% of the targets at ~21 per file against ~2 for backup.
  A single global spectra-per-file average therefore under-projects the yield
  badly early in the run.

    python scripts/fetch_status.py --log <dir>/fetch.log --targets <dir>/new_targets_nway.csv
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import pandas as pd


def process_elapsed() -> float | None:
    """Seconds since the fetch process started, or None if it is not running."""
    try:
        out = subprocess.run(["ps", "-eo", "pid,etimes,args"], capture_output=True,
                             text=True, timeout=10).stdout
    except Exception:
        return None
    for line in out.splitlines():
        if "fetch_desi_spectra.py" in line and "fetch_status" not in line:
            return float(line.split()[1])
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--targets", type=Path, required=True)
    args = ap.parse_args()

    text = args.log.read_text()
    rows = re.findall(r"^  ([\d,]+)/([\d,]+) files \| ([\d,]+) spectra \| "
                      r"(\d+) to retry \| (\d+) absent", text, re.M)
    retries = re.findall(r"^  \[retry-later\] (.+)$", text, re.M)
    if not rows:
        raise SystemExit("no progress lines in the log yet")
    done, tot, spec, retry, absent = (int(x.replace(",", "")) for x in rows[-1])

    elapsed = process_elapsed()
    running = elapsed is not None
    print(f"state       : {'RUNNING' if running else 'NOT RUNNING (paused/finished)'}")
    print(f"files       : {done:,}/{tot:,}  ({done/tot:.1%})")
    print(f"spectra     : {spec:,}")
    print(f"to retry    : {retry}   absent: {absent}")

    if not running:
        print("\nresume with the identical command; .done.json makes it skip finished groups")
        return

    rate = done / elapsed
    print(f"elapsed     : {elapsed/3600:.2f} h    rate: {rate*60:.1f} files/min")

    # per-program projection: how many targets remain in files not yet reached
    t = pd.read_csv(args.targets).drop_duplicates("targetid")
    g = (t.groupby(["survey", "program"])
           .agg(targets=("targetid", "size"),
                files=("healpix", lambda s: s.nunique()))
           .sort_index())
    g["per_file"] = g.targets / g.files
    print("\nwork by program (processed in this order):")
    print(g.to_string())

    order = list(g.index)
    seen, remaining_files, remaining_targets = 0, 0, 0
    for key in order:
        n = int(g.loc[key, "files"])
        if seen + n <= done:
            seen += n
            continue
        unread = n - max(0, done - seen)
        remaining_files += unread
        remaining_targets += unread * float(g.loc[key, "per_file"])
        seen += n
    # Cost is fixed-per-file (header walk + FIBERMAP) PLUS per-spectrum, and the
    # two differ by an order of magnitude across programs: main/dark averages
    # 20.2 targets per file against 1.85 for main/backup. Extrapolating
    # files/min while the cheap programs drain first badly under-estimates the
    # remaining time, so fit both terms from the run's own progress.
    FIXED_S, PER_SPEC_S = 8.4, 1.62
    workers = 6
    m = re.search(r"--workers\s+(\d+)", text)
    if m:
        workers = int(m.group(1))
    eta = (remaining_files * FIXED_S + remaining_targets * PER_SPEC_S) / workers
    naive = remaining_files / rate if rate > 0 else float("nan")
    print(f"\nremaining   : {remaining_files:,} files, ~{remaining_targets:,.0f} spectra")
    print(f"ETA         : {eta/3600:.1f} h  "
          f"(~{time.strftime('%a %H:%M', time.localtime(time.time()+eta))})")
    print(f"              [naive files/min would say {naive/3600:.1f} h, "
          f"which ignores that dark files hold ~11x more targets]")
    print(f"projected   : {spec + remaining_targets:,.0f} spectra total")

    if retries:
        print(f"\nlast retry-later events ({len(retries)} total):")
        for r in retries[-5:]:
            print(f"   {r[:100]}")


if __name__ == "__main__":
    main()
