#!/usr/bin/env python
"""Assemble the MERGED source HDF5 that `prepare-data` stages from.

The expansion's spectra arrive as .npz shards (tid, flux, ivar) and the current
sample's live in the old `erosita_spectra_merged_32k.hdf5`. `prepare-data` reads
ONE source file, so until this existed the merged sample could not be staged at
all -- `manifest_merged.csv` had `source_row >= 0` on only 24,058 of 129,486
rows, exactly the rows the old file holds, and the stager drops the rest.

    python scripts/build_merged_source.py \
        --shards data/dr2/spectra_dr2_new.shards \
        --old-hdf5 aion_project/shareable_aion_flow/data/raw/erosita_desi/erosita_spectra_merged_32k.hdf5 \
        --sidecar data/dr2/targets_sidecar_merged.csv \
        --out data/dr2/spectra_merged_source.h5

Then rebuild the manifest against it and re-run the split pass, because
`source_row` is an INDEX INTO THIS FILE and every row moved:

    python scripts/build_manifest.py --targets data/dr2/merged_targets.csv \\
        --source-hdf5 data/dr2/spectra_merged_source.h5 ... --out data/dr2/manifest_merged.csv
    python scripts/build_manifest.py ... --split data/dr2/clean_split_merged.csv \\
        --out data/dr2/manifest_merged.csv

WHAT THIS WRITES, AND WHY IT IS SHORTER THAN THE OLD FILE. Only the datasets
`prepare_staged_data` actually reads:

    desi_targetid  spectra  spectra_ivar  spectra_lambda
    redshift (as desi_z)  flux_w1  flux_w2  flux_w3
    target_ra  target_dec        (optional, copied when present)

The old file also carried ml_flux_1, ml_rate_p2/p3/p4, desi_flux_g/r/z,
dist_arcsec, spectype and survey/program columns. NONE of them are read by
staging any more: every label comes from the sidecar at load time, which is the
whole point of inputs-only staging. Carrying an X-ray rate in the spectrum file
is how a DR1 flux ended up selecting a DR2 sample, so this file deliberately
does not have one.

THE WAVELENGTH GRID IS THE LOAD-BEARING ASSUMPTION. Every spectrum must live on
the same grid or the arrays cannot share one dataset, and a silent mismatch
would misalign flux against wavelength for a whole subpopulation -- invisible in
any loss curve. Both sources are on DESI's standard 7,781-bin 3600-9824 A grid;
this script ASSERTS that per shard rather than trusting it, and refuses any
shard whose bin count differs.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def shard_paths(shard_dir: Path) -> list[Path]:
    return sorted(Path(e.path) for e in os.scandir(shard_dir) if e.name.endswith(".npz"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", type=Path, required=True, help="dir of <name>.npz (tid, flux, ivar)")
    ap.add_argument("--old-hdf5", type=Path, required=True, help="source file for the current sample")
    ap.add_argument("--sidecar", type=Path, required=True,
                    help="merged sidecar: supplies z, WISE fluxes and coordinates")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"{args.out} exists; pass --overwrite to replace it")

    sc = pd.read_csv(args.sidecar, low_memory=False).drop_duplicates("targetid")
    sc["targetid"] = sc["targetid"].astype(np.int64)
    sc = sc.set_index("targetid")
    print(f"[merge] sidecar: {len(sc):,} unique targetids")

    with h5py.File(args.old_hdf5, "r") as old:
        lam = old["spectra_lambda"][:].astype(np.float32)
        old_tid = old["desi_targetid"][:].astype(np.int64)
        old_flux = old["spectra_flux"][:]
        old_ivar = old["spectra_ivar"][:]
    n_bins = lam.shape[0]
    print(f"[merge] old file: {len(old_tid):,} spectra on a {n_bins}-bin grid "
          f"[{lam[0]:.1f}, {lam[-1]:.1f}] A")

    # Old file first, shards second; first writer of a targetid wins. The old
    # file is the one every published number was measured against, so where the
    # two disagree we keep the spectrum the existing results describe.
    seen: set[int] = set()

    def first_time(values: np.ndarray) -> np.ndarray:
        """Boolean mask of the rows whose targetid has not been taken yet."""
        mask = np.zeros(len(values), dtype=bool)
        for i, v in enumerate(values):
            if v not in seen:
                seen.add(int(v))
                mask[i] = True
        return mask

    # STREAM the spectra straight into the file. Accumulating them in lists and
    # concatenating at the end costs ~4x the final array size at the moment of
    # the concatenate (both lists live, plus the destination), which is ~14.5 GB
    # for 125k spectra on a 7,781-bin grid. That OOM-killed this script on a
    # 15 GB machine at 8,000/9,316 shards, after 40 minutes of work, with
    # nothing written. Resizable datasets keep peak memory at one shard.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tid_parts: list[np.ndarray] = []          # 8 bytes/row, safe to hold
    n_written = 0
    n_dup = 0
    bad_grid: list[str] = []

    with h5py.File(args.out, "w") as out:
        gz = dict(compression="gzip", compression_opts=4)
        d_flux = out.create_dataset("spectra_flux", shape=(0, n_bins), maxshape=(None, n_bins),
                                    dtype=np.float32, chunks=(64, n_bins), **gz)
        d_ivar = out.create_dataset("spectra_ivar", shape=(0, n_bins), maxshape=(None, n_bins),
                                    dtype=np.float32, chunks=(64, n_bins), **gz)

        def append(t: np.ndarray, fl: np.ndarray, iv: np.ndarray) -> None:
            nonlocal n_written
            if not len(t):
                return
            d_flux.resize(n_written + len(t), axis=0)
            d_ivar.resize(n_written + len(t), axis=0)
            d_flux[n_written:n_written + len(t)] = fl.astype(np.float32)
            d_ivar[n_written:n_written + len(t)] = iv.astype(np.float32)
            n_written += len(t)
            tid_parts.append(t)

        # Old file first, in blocks so its 32k x 7781 array is never fully
        # resident either.
        n_old = 0
        with h5py.File(args.old_hdf5, "r") as old:
            for lo in range(0, len(old_tid), 4096):
                hi = min(lo + 4096, len(old_tid))
                keep = first_time(old_tid[lo:hi])
                if keep.any():
                    append(old_tid[lo:hi][keep],
                           old["spectra_flux"][lo:hi][keep],
                           old["spectra_ivar"][lo:hi][keep])
                    n_old += int(keep.sum())

        paths = shard_paths(args.shards)
        print(f"[merge] {len(paths):,} shards to read")
        for i, p in enumerate(paths):
            with np.load(p) as z:
                t, fl, iv = z["tid"].astype(np.int64), z["flux"], z["ivar"]
            if fl.shape[1] != n_bins:
                bad_grid.append(f"{p.name}: {fl.shape[1]} bins")
                continue
            m = first_time(t)
            n_dup += int((~m).sum())
            append(t[m], fl[m], iv[m])
            if (i + 1) % 2000 == 0:
                print(f"[merge]   {i + 1:,}/{len(paths):,} shards, {n_written:,} spectra",
                      flush=True)

        if bad_grid:
            raise SystemExit(
                f"{len(bad_grid)} shard(s) are on a different wavelength grid than "
                f"{args.old_hdf5.name} ({n_bins} bins). They cannot share one dataset and a "
                f"silent mismatch would misalign flux against wavelength. First few: "
                f"{bad_grid[:5]}")

        tid = np.concatenate(tid_parts)
        print(f"[merge] {len(tid):,} unique spectra ({n_old:,} from the old file, "
              f"{len(tid) - n_old:,} from shards); {n_dup:,} shard rows dropped as duplicates")
        _write_scalars(out, tid, lam, sc, args, n_old)

    print(f"[merge] wrote {args.out} ({args.out.stat().st_size / 2**30:.2f} GB)")
    print("[merge] NEXT: source_row is an index into THIS file, so rebuild the "
          "manifest (twice, the second pass with --split) before staging.")
    return 0


def _write_scalars(out, tid: np.ndarray, lam: np.ndarray, sc: pd.DataFrame,
                   args: argparse.Namespace, n_old: int) -> None:
    """Everything the stager needs that does not live with the spectra."""
    aligned = sc.reindex(tid)
    missing = int(aligned["z"].isna().sum())
    if missing:
        print(f"[merge] WARNING {missing:,} spectra have no sidecar row; their z and WISE "
              f"fluxes will be NaN and the manifest will mark them unusable")

    def col(name: str, *alts: str) -> np.ndarray:
        for c in (name, *alts):
            if c in aligned.columns:
                return aligned[c].to_numpy(dtype=np.float64)
        raise SystemExit(f"sidecar has none of {(name, *alts)}")

    gz = dict(compression="gzip", compression_opts=4)
    out.create_dataset("desi_targetid", data=tid, **gz)
    out.create_dataset("spectra_lambda", data=lam)
    out.create_dataset("desi_z", data=col("z").astype(np.float32), **gz)
    for band in ("w1", "w2", "w3"):
        out.create_dataset(f"flux_{band}",
                           data=col(f"flux_{band}", f"ls10_flux_{band}").astype(np.float32), **gz)
    for c in ("target_ra", "target_dec"):
        if c in aligned.columns:
            out.create_dataset(c, data=aligned[c].to_numpy(dtype=np.float64), **gz)
    out.attrs["n_spectra"] = len(tid)
    out.attrs["n_from_old"] = n_old
    out.attrs["n_from_shards"] = len(tid) - n_old
    out.attrs["source_old_hdf5"] = str(args.old_hdf5)
    out.attrs["source_shards"] = str(args.shards)


if __name__ == "__main__":
    raise SystemExit(main())
