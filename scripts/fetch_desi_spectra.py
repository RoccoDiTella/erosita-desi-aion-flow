#!/usr/bin/env python
"""Fetch DESI coadd spectra for a target list WITHOUT downloading whole coadd files.

A DESI healpix coadd is ~500 MB and holds a couple of thousand spectra; we
typically want ~20 of them. Downloading whole files to extract 61 KB each is a
~670x waste (4.1 TB to obtain 6 GB). Instead this opens each coadd lazily over
HTTP -- the server supports byte ranges -- and reads only the FIBERMAP plus the
flux/ivar rows belonging to our targets.

Output matches the existing `erosita_spectra_merged_32k.hdf5` schema: B, R and Z
are combined by inverse-variance weighting onto ONE uniform 0.8 A grid spanning
3600-9824.38 A (7781 bins), which is what the original merge produced. Cameras
overlap, so the overlaps are coadded rather than concatenated or trimmed.

Resumable: completed (survey, program, healpix) groups are recorded and skipped,
so an interrupted run continues where it stopped.

    python scripts/fetch_desi_spectra.py --targets new_targets.csv --out spectra.h5
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

LAM0, DLAM, NBIN = 3600.0, 0.8, 7781
BASE = ("https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/healpix"
        "/{survey}/{program}/{group}/{pix}/coadd-{survey}-{program}-{pix}.fits")
CAMERAS = ("B", "R", "Z")
_lock = threading.Lock()


def coadd_url(survey: str, program: str, pix: int) -> str:
    return BASE.format(survey=survey, program=program, group=pix // 100, pix=pix)


def fetch_group(survey: str, program: str, pix: int, want: np.ndarray):
    """Read only the rows of one coadd that belong to `want`.

    Returns (targetids, flux[n, NBIN], ivar[n, NBIN]) or None.
    """
    from astropy.io import fits

    url = coadd_url(survey, program, int(pix))
    with fits.open(url, use_fsspec=True) as hdul:
        tid = np.asarray(hdul["FIBERMAP"].data["TARGETID"], dtype=np.int64)
        pos = {t: i for i, t in enumerate(tid)}
        rows = [(t, pos[t]) for t in want if t in pos]
        if not rows:
            return None
        idx = np.array([r for _, r in rows])
        order = np.argsort(idx)                    # ascending rows read faster
        idx, keep = idx[order], np.array([t for t, _ in rows])[order]

        num = np.zeros((len(idx), NBIN), dtype=np.float64)
        den = np.zeros((len(idx), NBIN), dtype=np.float64)
        for cam in CAMERAS:
            wave = np.asarray(hdul[f"{cam}_WAVELENGTH"].data, dtype=np.float64)
            # map this camera onto the common grid; DESI grids are 0.8 A aligned
            col = np.rint((wave - LAM0) / DLAM).astype(int)
            ok = (col >= 0) & (col < NBIN)
            col = col[ok]
            fsec = hdul[f"{cam}_FLUX"].section
            isec = hdul[f"{cam}_IVAR"].section
            for k, r in enumerate(idx):
                f = np.asarray(fsec[r, :], dtype=np.float64)[ok]
                v = np.asarray(isec[r, :], dtype=np.float64)[ok]
                v = np.where(np.isfinite(v) & (v > 0) & np.isfinite(f), v, 0.0)
                num[k, col] += f * v
                den[k, col] += v
        flux = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        return keep, flux.astype(np.float32), den.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=Path, required=True,
                    help="CSV with targetid, survey, program, healpix")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=12,
                    help="Concurrent coadd files. Keep modest: this is a public server.")
    ap.add_argument("--limit-groups", type=int, default=0, help="0 = all (use for a trial run)")
    args = ap.parse_args()

    t = pd.read_csv(args.targets)
    need = {"targetid", "survey", "program", "healpix"}
    if not need <= set(t.columns):
        raise SystemExit(f"--targets needs {sorted(need)}; has {sorted(t.columns)[:8]}")
    t = t.drop_duplicates("targetid")
    groups = [(s, p, int(h), g.targetid.to_numpy(np.int64))
              for (s, p, h), g in t.groupby(["survey", "program", "healpix"])]
    if args.limit_groups:
        groups = groups[:args.limit_groups]
    print(f"{len(t):,} targets across {len(groups):,} coadd files", flush=True)

    done_path = args.out.with_suffix(".done.json")
    done = set(map(tuple, json.loads(done_path.read_text()))) if done_path.exists() else set()
    todo = [g for g in groups if (g[0], g[1], g[2]) not in done]
    print(f"{len(done):,} groups already done; {len(todo):,} to go", flush=True)

    import h5py
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "a") as h5:
        if "spectra_flux" not in h5:
            h5.create_dataset("spectra_flux", (0, NBIN), maxshape=(None, NBIN),
                              dtype="f4", chunks=(64, NBIN), compression="lzf")
            h5.create_dataset("spectra_ivar", (0, NBIN), maxshape=(None, NBIN),
                              dtype="f4", chunks=(64, NBIN), compression="lzf")
            h5.create_dataset("desi_targetid", (0,), maxshape=(None,), dtype="i8")
            h5.create_dataset("spectra_lambda",
                              data=(LAM0 + DLAM * np.arange(NBIN)).astype("f4"))

        def append(tids, flux, ivar):
            n = len(tids)
            for name, arr in (("spectra_flux", flux), ("spectra_ivar", ivar),
                              ("desi_targetid", tids)):
                d = h5[name]
                d.resize(d.shape[0] + n, axis=0)
                d[-n:] = arr

        ok = fail = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(fetch_group, s, p, h, w): (s, p, h) for s, p, h, w in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                key = futs[fut]
                try:
                    res = fut.result()
                except Exception as e:                     # keep going; log and retry later
                    fail += 1
                    print(f"  [fail] {key}: {type(e).__name__}: {e}", flush=True)
                    continue
                if res is not None:
                    with _lock:
                        append(*res)
                        ok += len(res[0])
                done.add(key)
                if i % 25 == 0:
                    done_path.write_text(json.dumps([list(k) for k in done]))
                    print(f"  {i:,}/{len(todo):,} files | {ok:,} spectra | {fail} failed",
                          flush=True)
        done_path.write_text(json.dumps([list(k) for k in done]))
    print(f"done: {ok:,} spectra written to {args.out} ({fail} files failed)")


if __name__ == "__main__":
    main()
