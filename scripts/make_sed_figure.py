#!/usr/bin/env python
"""Rest-frame SED interpretation figure (Buchner suggestion).

For every test source, assemble a rest-frame nu-L-nu SED from the DESI spectrum
(f_lambda), the three WISE bands, and the MODEL-PREDICTED 0.2-2.3 keV flux
(posterior median of the all-inputs combo), slice sources by predicted L_X, and
median-stack each slice into a clean track. The X-ray point is the model's
extension of the observed SED into a band it never saw; the measured flux is
overplotted for a per-slice bias check.

Conventions: eRASS1 ML_FLUX_1 assumes an absorbed power law Gamma = 2.0
(Merloni+2024), so nuF_nu is flat in energy and nuF_nu(1 keV) =
F_band / ln(2.3/0.2). WISE fluxes are Legacy Survey nanomaggies (3631 Jy zero
point). nuL_nu(rest) = 4 pi D_L^2 nuF_nu(obs), exact for nuF_nu (no slope
K-correction). Caveat: DESI fiber spectra under-collect extended-source light
relative to total photometry (aperture mismatch vs WISE).

    python scripts/make_sed_figure.py --run-dir <run> --staged-dir <staged_paper> \
        --clean-split-csv <clean_split.csv> [--n-slices 5]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.cosmology import Planck18

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shareable_aion_flow.data_to_aion_embeddings import clean_view_row_maps  # noqa: E402

INK = "#1a1a1a"
DESI_FLAM_UNIT = 1e-17  # erg/s/cm^2/A
WISE_LAMBDA_UM = (3.368, 4.618, 12.082)  # W1, W2, W3 effective wavelengths
NMGY_TO_FNU = 3.631e-29  # erg/s/cm^2/Hz per nanomaggy (3631 Jy zero point)
C_ANGSTROM = 2.998e18  # speed of light in A/s
XRAY_BAND_TO_MONO = 1.0 / np.log(2.3 / 0.2)  # Gamma=2: nuF_nu(1 keV) = F_band * this
KEV_ANGSTROM = 12.398


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--staged-dir", type=Path, required=True)
    ap.add_argument("--clean-split-csv", type=Path, required=True)
    ap.add_argument("--n-slices", type=int, default=5)
    ap.add_argument("--min-per-bin", type=int, default=30)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (args.run_dir / "sed_interpretation.png")

    preds = pd.read_csv(args.run_dir / "test_predictions.csv")
    preds = preds[preds.input_group == "spectra+z+wise+image"].set_index("targetid")

    maps = clean_view_row_maps(args.staged_dir, args.clean_split_csv)
    spec_parts, z_parts, wise_parts, tid_parts = [], [], [], []
    wave = None
    for path, rows in maps["test"]:
        with h5py.File(path, "r") as h:
            wave = h["spectra_lambda"][:].astype(np.float64)
            spec_parts.append(h["spectra"][:][rows].astype(np.float64))
            z_parts.append(h["redshift"][:][rows].astype(np.float64))
            wise_parts.append(
                np.stack([h[f"flux_w{i}"][:][rows] for i in (1, 2, 3)], axis=1).astype(np.float64)
            )
            tid_parts.append(h["desi_targetid"][:][rows].astype(np.int64))
    spec = np.concatenate(spec_parts)
    z = np.concatenate(z_parts)
    wise = np.concatenate(wise_parts)
    tid = np.concatenate(tid_parts)

    keep = pd.Index(tid).isin(preds.index)
    spec, z, wise, tid = spec[keep], z[keep], wise[keep], tid[keep]
    p = preds.loc[tid]
    d_l = Planck18.luminosity_distance(z).to("cm").value
    lum_factor = 4.0 * np.pi * d_l**2

    lx_pred = np.log10(lum_factor * 10.0 ** p.posterior_p50.values * XRAY_BAND_TO_MONO)
    lx_meas = np.log10(lum_factor * 10.0 ** p.y_true.values * XRAY_BAND_TO_MONO)
    edges = np.quantile(lx_pred, np.linspace(0, 1, args.n_slices + 1))
    slice_of = np.clip(np.searchsorted(edges, lx_pred, side="right") - 1, 0, args.n_slices - 1)

    # rest-wavelength grid: 3 A (4 keV) to 20 um
    grid = np.geomspace(3.0, 2.0e5, 61)
    centers = np.sqrt(grid[:-1] * grid[1:])
    tracks = np.full((args.n_slices, len(centers)), np.nan)
    counts = np.zeros((args.n_slices, len(centers)), dtype=int)
    per_bin: dict[tuple[int, int], list[float]] = {}

    for i in range(len(z)):
        rest = wave / (1.0 + z[i])
        nulnu = wave * np.clip(spec[i], 0, None) * DESI_FLAM_UNIT * lum_factor[i]
        good = nulnu > 0
        idx = np.digitize(rest[good], grid) - 1
        vals = nulnu[good]
        for b in np.unique(idx):
            if 0 <= b < len(centers):
                per_bin.setdefault((slice_of[i], b), []).append(
                    float(np.median(vals[idx == b]))
                )
        for k in range(3):
            if wise[i, k] > 0:
                lam = WISE_LAMBDA_UM[k] * 1e4
                v = (C_ANGSTROM / lam) * wise[i, k] * NMGY_TO_FNU * lum_factor[i]
                b = int(np.digitize(lam / (1.0 + z[i]), grid)) - 1
                if 0 <= b < len(centers):
                    per_bin.setdefault((slice_of[i], b), []).append(float(v))

    for (s, b), vals in per_bin.items():
        counts[s, b] = len(vals)
        if len(vals) >= args.min_per_bin:
            tracks[s, b] = np.median(vals)

    # per-slice X-ray medians (predicted and measured), at the slice-median rest energy
    xray_rows = []
    for s in range(args.n_slices):
        m = slice_of == s
        xray_rows.append({
            "slice": s,
            "rest_lambda_A": float(np.median(KEV_ANGSTROM / (1.0 + z[m]))),
            "nuLnu_pred": float(np.median(10.0 ** lx_pred[m])),
            "nuLnu_meas": float(np.median(10.0 ** lx_meas[m])),
            "n": int(m.sum()),
            "lx_lo": float(edges[s]), "lx_hi": float(edges[s + 1]),
        })
    xr = pd.DataFrame(xray_rows)

    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for s in range(args.n_slices):
        color = cmap(s / max(args.n_slices - 1, 1))
        ok = np.isfinite(tracks[s]) & (centers > 800)
        ax.plot(centers[ok], tracks[s][ok], color=color, lw=1.8,
                label=f"log $\\nu L_\\nu$(1 keV) {xr.lx_lo[s]:.1f}-{xr.lx_hi[s]:.1f}")
        ax.plot(xr.rest_lambda_A[s], xr.nuLnu_pred[s], "o", color=color, ms=9, zorder=5)
        ax.plot(xr.rest_lambda_A[s], xr.nuLnu_meas[s], "s", mfc="none", mec=color, ms=9, zorder=5)
        # dotted connector: model's extrapolation gap from UV to X-ray
        left = np.flatnonzero(ok)
        if len(left):
            ax.plot([xr.rest_lambda_A[s], centers[left[0]]],
                    [xr.nuLnu_pred[s], tracks[s][left[0]]], ls=":", color=color, lw=1.0, alpha=0.7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("rest wavelength [$\\AA$]")
    ax.set_ylabel("$\\nu L_\\nu$ [erg s$^{-1}$]")
    ax.set_title("Median rest-frame SEDs by PREDICTED X-ray luminosity "
                 "(filled = predicted 1 keV, open = measured)")
    secax = ax.secondary_xaxis("top", functions=(lambda x: 12398.0 / np.maximum(x, 1e-9),
                                                 lambda e: 12398.0 / np.maximum(e, 1e-9)))
    secax.set_xlabel("rest energy [eV]")
    ax.legend(fontsize=9, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    pd.DataFrame({"rest_lambda_A": centers, **{f"slice{s}": tracks[s] for s in range(args.n_slices)}}
                 ).to_csv(out.with_suffix(".csv"), index=False)
    xr.to_csv(out.parent / "sed_xray_points.csv", index=False)
    print(xr.to_string(index=False))
    print(f"written {out}")


if __name__ == "__main__":
    main()
