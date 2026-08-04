#!/usr/bin/env python
"""Fetch Legacy Survey grz cutouts, one file per target, resumable and polite.

The cutout service is HARD rate limited: a second concurrent request returns
HTTP 429, so this is deliberately SEQUENTIAL. Measured ~5.2 s and 0.40 MB per
cutout, i.e. ~6 days for 105k targets. It is designed to be left running for
that long and interrupted freely.

Robustness, mirroring fetch_desi_spectra.py:
  * one file per target, written to a temp name and renamed (atomic), so a kill
    can never leave a half-written cutout that a later run would trust;
  * resume is file existence, so it is exact and needs no side-car state;
  * 429 and 5xx back off and retry rather than being recorded as permanent,
    because treating throttling as failure would silently lose sources.

Ordering matters when the job will not finish in one sitting. --prioritize
puts targets whose SPECTRUM has already been fetched at the front, so stopping
early yields complete (spectrum, image) pairs instead of unusable orphans. The
priority set is re-read periodically, since the spectra job is still running.

    python scripts/fetch_ls_cutouts.py --targets t.csv --out fits_pool_dr2 \
        --prioritize spectra_dr2_new.shards
"""

from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

URL = ("https://www.legacysurvey.org/viewer/fits-cutout"
       "?ra={ra:.6f}&dec={dec:.6f}&layer={layer}&pixscale={pixscale}&size={size}&bands={bands}")
EXPECT_BYTES = 414720          # 4 bands x 160 x 160 float32 + FITS headers


def spectra_available(shard_dir: Path) -> set[int]:
    """Targetids that already have a spectrum on disk."""
    out: set[int] = set()
    for sp in shard_dir.glob("*.npz"):
        try:
            out.update(int(t) for t in np.load(sp)["tid"])
        except Exception:
            continue                      # shard mid-write or unreadable; skip
    return out


def fetch_one(ra: float, dec: float, dest: Path, args, attempts: int = 5) -> str:
    """Returns 'ok', 'skip' (already present) or raises after exhausting retries."""
    if dest.exists():
        return "skip"
    url = URL.format(ra=ra, dec=dec, layer=args.layer, pixscale=args.pixscale,
                     size=args.size, bands=args.bands)
    delay = args.backoff
    last: Exception = RuntimeError("no attempt made")
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=args.timeout) as r:
                blob = r.read()
            if len(blob) < 10_000:
                raise ValueError(f"suspiciously small response: {len(blob)} bytes")
            tmp = dest.with_name(dest.name + ".tmp")
            tmp.write_bytes(blob)
            tmp.replace(dest)             # atomic
            return "ok"
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(delay)
                delay = min(delay * 2, args.max_backoff)
                continue
            raise                         # 404 etc: a real, permanent answer
        except Exception as e:            # timeout, truncated read, DNS
            last = e
            time.sleep(delay)
            delay = min(delay * 2, args.max_backoff)
    raise last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=Path, required=True,
                    help="CSV with targetid, mean_fiber_ra, mean_fiber_dec")
    ap.add_argument("--out", type=Path, required=True, help="cutout pool directory")
    ap.add_argument("--prioritize", type=Path, default=None,
                    help="spectra shard dir; targets with a spectrum go first")
    ap.add_argument("--rescan-every", type=int, default=500,
                    help="re-read the priority set every N attempts (spectra job is live)")
    ap.add_argument("--layer", default="ls-dr10")
    ap.add_argument("--pixscale", type=float, default=0.262)
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--bands", default="griz")
    ap.add_argument("--sleep", type=float, default=0.4,
                    help="pause between requests. Do not set to 0: the service 429s.")
    ap.add_argument("--backoff", type=float, default=5.0)
    ap.add_argument("--max-backoff", type=float, default=120.0)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--limit", type=int, default=0, help="0 = all (use for a trial)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(args.targets).drop_duplicates("targetid")
    t["targetid"] = t.targetid.astype(np.int64)
    todo = t[~t.targetid.map(lambda i: (args.out / f"{i}.fits").exists())]
    print(f"{len(t):,} targets, {len(t)-len(todo):,} already fetched, {len(todo):,} to go",
          flush=True)
    if args.limit:
        todo = todo.head(args.limit)

    prio: set[int] = set()
    if args.prioritize and args.prioritize.exists():
        prio = spectra_available(args.prioritize)
        print(f"prioritising {len(prio):,} targets that already have a spectrum", flush=True)

    def ordered(frame):
        """Targets whose spectrum already exists go first.

        The job takes ~6 days, so it will be interrupted. Ordering this way
        means an early stop leaves complete (spectrum, image) pairs rather than
        images for sources we may never have a spectrum for.
        """
        if not prio:
            return frame
        key = frame.targetid.isin(prio)
        return pd.concat([frame.loc[key], frame.loc[~key]])

    ok = skip = fail = 0
    t0 = time.time()
    total = len(todo)
    remaining = todo
    done_n = 0
    # Work in chunks and RE-ORDER between them. The spectra job is running
    # concurrently, so the set of targets with a spectrum grows underneath us;
    # materialising the whole queue once would freeze the priority at startup
    # and defeat the point.
    while len(remaining):
        chunk = ordered(remaining).head(max(1, args.rescan_every))
        for r in chunk.itertuples():
            dest = args.out / f"{int(r.targetid)}.fits"
            try:
                res = fetch_one(float(r.mean_fiber_ra), float(r.mean_fiber_dec), dest, args)
                ok += res == "ok"
                skip += res == "skip"
            except Exception as e:
                fail += 1
                print(f"  [fail] {int(r.targetid)}: {type(e).__name__}: {str(e)[:80]}",
                      flush=True)
            done_n += 1
            if args.sleep:
                time.sleep(args.sleep)      # the service 429s without a gap
            if done_n % 100 == 0:
                el = time.time() - t0
                rate = ok / max(el, 1e-9)
                left = total - done_n
                print(f"  {done_n:,}/{total:,} | {ok:,} fetched | {fail} failed | "
                      f"{rate*3600:.0f}/h | ETA {left/max(rate,1e-9)/3600:.1f} h", flush=True)
        remaining = remaining[~remaining.targetid.isin(chunk.targetid)]
        if args.prioritize:
            fresh = spectra_available(args.prioritize)
            if len(fresh) > len(prio):
                prio = fresh                # newly-fetched spectra move up the queue
    print(f"done: {ok:,} fetched, {skip:,} already present, {fail} failed")


if __name__ == "__main__":
    main()
