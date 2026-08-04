#!/usr/bin/env python
"""Fetch DESI coadd spectra for a target list WITHOUT downloading whole coadd files.

A DESI healpix coadd is ~500 MB and holds a couple of thousand spectra; we
typically want ~20 of them. Downloading whole files to extract 61 KB each is a
~670x waste (8.7 TB to obtain 6 GB). Instead this opens each coadd lazily over
HTTP -- the server sends accept-ranges -- and reads only the FIBERMAP plus the
flux/ivar rows belonging to our targets. Measured cost: ~1.4 MB read from a
984 MB file, ~12.6 GB for a 105k-target job.

Output is SHARDED: one small .npz per (survey, program, healpix), written under a
temporary name and renamed into place. HDF5 appends are NOT atomic, so a single
growing file is unreadable while the job runs and a kill at the wrong moment
tears it -- while a separate done-list would still claim those groups finished,
quietly dropping them on resume. A shard either exists and is complete, or does
not exist. That makes resume exact, lets you inspect partial output at any time,
and survives kill -9.

Spectra are combined across the B, R and Z cameras by inverse-variance weighting
onto ONE uniform 0.8 A grid over 3600-9824.38 A (7781 bins), matching the
existing erosita_spectra_merged_32k.hdf5. The cameras overlap, so overlaps are
coadded rather than concatenated or trimmed. Verified bit-exact against a
spectrum already in that file.

    python scripts/fetch_desi_spectra.py --targets t.csv --out spectra.h5
    python scripts/fetch_desi_spectra.py --targets t.csv --out spectra.h5 --merge
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

LAM0, DLAM, NBIN = 3600.0, 0.8, 7781
BASE = ("https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/healpix"
        "/{survey}/{program}/{group}/{pix}/coadd-{survey}-{program}-{pix}.fits")
CAMERAS = ("B", "R", "Z")


def coadd_url(survey: str, program: str, pix: int) -> str:
    return BASE.format(survey=survey, program=program, group=pix // 100, pix=pix)


def shard_path(d: Path, survey: str, program: str, pix: int) -> Path:
    return d / f"{survey}__{program}__{pix}.npz"


class MissingFile(Exception):
    """The coadd genuinely does not exist: never worth retrying."""


def fetch_group(survey: str, program: str, pix: int, want: np.ndarray, attempts: int = 4):
    """Read only the rows of one coadd that belong to `want`, with backoff.

    A public archive answers 503 when pushed, and treating that as permanent
    would silently drop sources. A genuinely absent file raises MissingFile so
    the caller can record it instead of retrying it on every resume.
    """
    import time

    last = None
    for a in range(attempts):
        try:
            return _fetch_once(survey, program, pix, want)
        except FileNotFoundError as e:
            raise MissingFile(str(e)) from e
        except Exception as e:                      # 503, timeout, truncated read
            last = e
            if a < attempts - 1:
                time.sleep(2 ** a * 3)              # 3, 6, 12 s
    raise last


def _fetch_once(survey: str, program: str, pix: int, want: np.ndarray):
    from astropy.io import fits

    with fits.open(coadd_url(survey, program, int(pix)), use_fsspec=True) as hdul:
        tid = np.asarray(hdul["FIBERMAP"].data["TARGETID"], dtype=np.int64)
        pos = {t: i for i, t in enumerate(tid)}
        rows = [(t, pos[t]) for t in want if t in pos]
        if not rows:
            return None
        idx = np.array([r for _, r in rows])
        order = np.argsort(idx)                     # ascending rows read faster
        idx, keep = idx[order], np.array([t for t, _ in rows])[order]

        num = np.zeros((len(idx), NBIN), dtype=np.float64)
        den = np.zeros((len(idx), NBIN), dtype=np.float64)
        for cam in CAMERAS:
            wave = np.asarray(hdul[f"{cam}_WAVELENGTH"].data, dtype=np.float64)
            col = np.rint((wave - LAM0) / DLAM).astype(int)   # DESI grids are 0.8 A aligned
            ok = (col >= 0) & (col < NBIN)
            col = col[ok]
            fsec, isec = hdul[f"{cam}_FLUX"].section, hdul[f"{cam}_IVAR"].section
            for k, r in enumerate(idx):
                f = np.asarray(fsec[r, :], dtype=np.float64)[ok]
                v = np.asarray(isec[r, :], dtype=np.float64)[ok]
                v = np.where(np.isfinite(v) & (v > 0) & np.isfinite(f), v, 0.0)
                num[k, col] += f * v
                den[k, col] += v
        flux = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        return keep, flux.astype(np.float32), den.astype(np.float32)


def write_shard(d: Path, survey: str, program: str, pix: int, res) -> None:
    """All-or-nothing: write under a temp name, then rename (atomic on POSIX)."""
    p = shard_path(d, survey, program, pix)
    # The temp name MUST already end in .npz: np.savez silently appends .npz to
    # anything that does not, so a ".tmp" suffix produces ".npz.tmp.npz" and the
    # subsequent rename of ".npz.tmp" fails.
    tmp = p.with_name(p.name[:-4] + ".tmp.npz")
    if res is None:                                  # absent coadd, or no rows of ours
        np.savez(tmp, tid=np.empty(0, np.int64),
                 flux=np.empty((0, NBIN), np.float32), ivar=np.empty((0, NBIN), np.float32))
    else:
        np.savez(tmp, tid=res[0], flux=res[1], ivar=res[2])
    tmp.replace(p)


def merge(shard_dir: Path, out: Path) -> None:
    import h5py

    shards = sorted(shard_dir.glob("*.npz"))
    if not shards:
        raise SystemExit(f"no shards in {shard_dir}")
    tids, fluxes, ivars = [], [], []
    for sp in shards:
        z = np.load(sp)
        if len(z["tid"]):
            tids.append(z["tid"]); fluxes.append(z["flux"]); ivars.append(z["ivar"])
    tid = np.concatenate(tids)
    flux = np.concatenate(fluxes)
    ivar = np.concatenate(ivars)
    # a target observed in two programs appears twice; keep the higher-ivar copy
    order = np.argsort(-(ivar > 0).sum(axis=1))
    tid, flux, ivar = tid[order], flux[order], ivar[order]
    _, first = np.unique(tid, return_index=True)
    first.sort()
    print(f"merging {len(shards):,} shards: {len(tid):,} rows -> {len(first):,} unique targets")
    with h5py.File(out, "w") as h5:
        h5.create_dataset("desi_targetid", data=tid[first])
        h5.create_dataset("spectra_flux", data=flux[first], chunks=(64, NBIN), compression="lzf")
        h5.create_dataset("spectra_ivar", data=ivar[first], chunks=(64, NBIN), compression="lzf")
        h5.create_dataset("spectra_lambda", data=(LAM0 + DLAM * np.arange(NBIN)).astype("f4"))
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=Path, required=True,
                    help="CSV with targetid, survey, program, healpix")
    ap.add_argument("--out", type=Path, required=True,
                    help="Final HDF5. Shards live in <out>.shards/ until --merge.")
    ap.add_argument("--merge", action="store_true",
                    help="Combine existing shards into --out and exit.")
    ap.add_argument("--workers", type=int, default=6,
                    help="Concurrent coadd files. Keep modest: the archive returns 503 "
                         "when pushed, and 8-way already triggered it.")
    ap.add_argument("--limit-groups", type=int, default=0, help="0 = all (use for a trial)")
    args = ap.parse_args()

    shard_dir = args.out.with_suffix(".shards")
    shard_dir.mkdir(parents=True, exist_ok=True)
    if args.merge:
        merge(shard_dir, args.out)
        return

    t = pd.read_csv(args.targets)
    need = {"targetid", "survey", "program", "healpix"}
    if not need <= set(t.columns):
        raise SystemExit(f"--targets needs {sorted(need)}; has {sorted(t.columns)[:8]}")
    t = t.drop_duplicates("targetid")
    groups = [(s, p, int(h), g.targetid.to_numpy(np.int64))
              for (s, p, h), g in t.groupby(["survey", "program", "healpix"])]
    if args.limit_groups:
        groups = groups[:args.limit_groups]
    todo = [g for g in groups if not shard_path(shard_dir, g[0], g[1], g[2]).exists()]
    print(f"{len(t):,} targets across {len(groups):,} coadd files", flush=True)
    print(f"{len(groups) - len(todo):,} shards present; {len(todo):,} to go", flush=True)

    ok = fail = missing = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_group, s, p, h, w): (s, p, h) for s, p, h, w in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            key = futs[fut]
            try:
                res = fut.result()
            except MissingFile:
                missing += 1
                write_shard(shard_dir, *key, None)   # record it: do not retry on resume
                continue
            except Exception as e:                   # transient: no shard, so resume retries
                fail += 1
                print(f"  [retry-later] {key}: {type(e).__name__}: {str(e)[:90]}", flush=True)
                continue
            try:
                write_shard(shard_dir, *key, res)
            except Exception as e:
                # Never let a write error escape the loop: leaving the `with`
                # block calls executor.shutdown(wait=True), which silently keeps
                # every queued future running while discarding all results.
                fail += 1
                print(f"  [write-failed] {key}: {type(e).__name__}: {e}", flush=True)
                continue
            ok += 0 if res is None else len(res[0])
            if i % 25 == 0:
                print(f"  {i:,}/{len(todo):,} files | {ok:,} spectra | "
                      f"{fail} to retry | {missing} absent", flush=True)
    print(f"done: {ok:,} spectra in {shard_dir}")
    print(f"  {missing} coadds absent, {fail} transient failures "
          f"(rerun the same command to retry them)")
    print(f"  then run with --merge to build {args.out}")


if __name__ == "__main__":
    main()
