#!/usr/bin/env python
"""Rebuild the X-ray targets for our existing sample from eROSITA DR2 (eRASS:3).

Same sources, same optical data, three sky passes instead of one. Nothing here
needs a new spectrum or cutout: only the X-ray labels change, and they get
~1.5x tighter while many more sub-band detections appear.

The DESI-side targets (CIGALE M*, SFR) are carried over unchanged -- eROSITA
depth has no bearing on them.

Errors follow the existing convention: a flux F with asymmetric linear errors
becomes log10(F) with
    sig_lo = -log10(1 - LOWERR/F)      (capped where LOWERR >= F)
    sig_hi =  log10(1 + UPERR/F)
so a source whose lower error swallows its flux is treated as unmeasured rather
than given an infinite bar.

    python scripts/make_dr2_targets.py --counterparts eRASSc3_Main_LS10.fits \
        --main eRASS3_Main_v1.3.fits --old-sidecar targets_sidecar.csv --out dr2_targets.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u

BANDS = (1, 2, 3, 4)
SIG_CAP = 1.5          # dex; beyond this the measurement carries no information


def log_with_asym_errors(flux, lowerr, uperr):
    """log10 flux plus split-normal sigmas, NaN where the flux is not a measurement."""
    f = np.asarray(flux, float)
    lo = np.asarray(lowerr, float)
    hi = np.asarray(uperr, float)
    good = np.isfinite(f) & (f > 0)
    out = np.full(f.shape, np.nan)
    slo = np.full(f.shape, np.nan)
    shi = np.full(f.shape, np.nan)
    out[good] = np.log10(f[good])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(good & np.isfinite(lo), lo / f, np.nan)
        # LOWERR >= F means the flux is consistent with zero: cap rather than -inf
        slo_all = np.where(ratio < 1.0, -np.log10(np.clip(1.0 - ratio, 1e-12, None)), SIG_CAP)
        shi_all = np.where(good & np.isfinite(hi), np.log10(1.0 + hi / f), np.nan)
    slo[good] = slo_all[good]
    shi[good] = shi_all[good]
    bad = ~np.isfinite(slo) | ~np.isfinite(shi) | (slo > SIG_CAP) | (shi > SIG_CAP)
    out[bad] = np.nan
    slo[bad] = np.nan
    shi[bad] = np.nan
    return out, slo, shi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counterparts", type=Path, required=True)
    ap.add_argument("--main", type=Path, required=True)
    ap.add_argument("--old-sidecar", type=Path, required=True)
    ap.add_argument("--desi", type=Path, required=True,
                    help="CSV with targetid, target_ra, target_dec, z")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--radius", type=float, default=1.0, help="match radius, arcsec")
    args = ap.parse_args()

    desi = (pd.read_csv(args.desi, usecols=["targetid", "target_ra", "target_dec", "z"])
            .dropna(subset=["target_ra", "target_dec"]).drop_duplicates("targetid"))
    cp = fits.open(args.counterparts, memmap=True)[1].data
    cra, cdec = np.asarray(cp["LS10_RA"], float), np.asarray(cp["LS10_DEC"], float)
    g = np.isfinite(cra) & np.isfinite(cdec)
    idx, sep, _ = SkyCoord(desi.target_ra.to_numpy()*u.deg,
                           desi.target_dec.to_numpy()*u.deg).match_to_catalog_sky(
                               SkyCoord(cra[g]*u.deg, cdec[g]*u.deg))
    m = sep.arcsec < args.radius
    print(f"DESI targets: {len(desi):,}   matched to a DR2 counterpart: {int(m.sum()):,}"
          f"  ({m.mean():.1%})")

    det = np.asarray(cp["DETUID"])[g][idx][m]
    mn = fits.open(args.main, memmap=True)[1].data
    order = {d: i for i, d in enumerate(np.asarray(mn["DETUID"]))}
    row = np.array([order.get(d, -1) for d in det])
    have = row >= 0
    r = row[have]
    out = desi[m].iloc[have].reset_index(drop=True)
    print(f"of those, present in the DR2 main catalogue: {len(out):,}")

    col = lambda c: np.asarray(mn[c], float)[r]
    # broad band -> log flux and log Lx
    lf, lo, hi = log_with_asym_errors(col("ML_FLUX_1"), col("ML_FLUX_LOWERR_1"),
                                      col("ML_FLUX_UPERR_1"))
    out["log_ml_flux_1"] = lf
    out["flux_sig_lo"], out["flux_sig_hi"] = lo, hi
    from astropy.cosmology import Planck18
    dl = Planck18.luminosity_distance(np.clip(out.z.to_numpy(float), 1e-4, None)).to(u.cm).value
    out["log_lx"] = out.log_ml_flux_1 + np.log10(4 * np.pi * dl ** 2)

    for b in BANDS:
        f, l, h = log_with_asym_errors(col(f"ML_FLUX_P{b}"), col(f"ML_FLUX_LOWERR_P{b}"),
                                       col(f"ML_FLUX_UPERR_P{b}"))
        out[f"log_flux_p{b}"] = f
        out[f"log_flux_p{b}_sig_lo"], out[f"log_flux_p{b}_sig_hi"] = l, h
    out["ero_detuid"] = det[have]
    out["ero_exp"] = col("ML_EXP_1")
    # Detection likelihood per band. Availability is gated on DETECTION rather
    # than on the error bar: a band can pass a sigma cut while being an upper
    # limit, which feeds the flow a non-measurement as though it were measured.
    out["det_like_0"] = col("DET_LIKE_0")
    for b in BANDS:
        out[f"det_like_p{b}"] = col(f"DET_LIKE_P{b}")

    # DESI-side targets are unchanged by X-ray depth: carry them across
    old = pd.read_csv(args.old_sidecar)
    keep = [c for c in old.columns if c == "targetid" or c.startswith(
        ("logmstar_cigale", "log_sfr", "ref_log_ssfr", "log_mbh", "cigale_"))]
    out = out.merge(old[keep], on="targetid", how="left")
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(out):,} rows, {len(out.columns)} columns)")


if __name__ == "__main__":
    main()
