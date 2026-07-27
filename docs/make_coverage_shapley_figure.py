#!/usr/bin/env python
"""Flux information by spectral region, drawn on the coverage plane.

The line-coverage figure and the Shapley heatmap show the same physics on
different axes. This merges them: rest wavelength on x, redshift on y, and the
region each spectrum actually covers shaded by how much flux information that
part of the spectrum carries.

Coverage is exact, not sampled: a source at redshift z covers observed
3600-9824 A, i.e. rest [3600/(1+z), 9824/(1+z)], so the covered region is a
band that slides blueward as z grows.

    python docs/make_coverage_shapley_figure.py --shapley shapley_table.csv \
        --redshifts z.npy [--by-zbin shapley_by_zbin.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OBS_LO, OBS_HI = 3600.0, 9824.0
INK, MUTED = "#1a1a1a", "#6a6a6a"

# half-window per line, matching the sweep's catalog at the time it ran
HALF_WIDTH = {
    "NeV": 30.0, "OII": 30.0, "HeII": 30.0, "OIII": 30.0, "NeIII": 30.0,
    "Hbeta": 120.0, "MgII": 120.0, "Halpha": 120.0, "Hdelta": 120.0,
    "Hgamma": 120.0, "Lyalpha": 120.0, "CIV": 120.0, "CIII": 120.0,
    "Hbeta+OIII": 147.8,
}

LABEL = {
    "Halpha": r"H$\alpha$", "Hbeta": r"H$\beta$", "Hgamma": r"H$\gamma$",
    "Hdelta": r"H$\delta$", "OIII": "[O III]", "OII": "[O II]", "NeIII": "[Ne III]",
    "NeV": "[Ne V]", "HeII": "He II", "MgII": "Mg II", "CIII": "C III]",
    "CIV": "C IV", "Lyalpha": r"Ly$\alpha$", "Hbeta+OIII": r"H$\beta$+[O III]",
}


def phi_profile(tab: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """Shapley value as a step function of rest wavelength.

    Continuum bins are painted first and lines second, matching the sweep's own
    claiming order (lines take their tokens, continuum gets the remainder).
    Painting in table order would let the wide continuum bins bury the lines.
    """
    prof = np.full(grid.shape, np.nan)
    for kind in ("cont", "line"):
        for r in tab[tab.kind == kind].itertuples():
            lo, hi = float(r.rest_lo), float(r.rest_hi)
            prof[(grid >= lo) & (grid <= hi)] = float(r.phi_nats)
    return prof


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shapley", type=Path, required=True)
    ap.add_argument("--redshifts", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("docs/figures/fig_coverage_shapley.png"))
    ap.add_argument("--zmax", type=float, default=3.2)
    args = ap.parse_args()

    tab = pd.read_csv(args.shapley)
    if "rest_lo" not in tab.columns:
        # Derive windows from the table itself rather than the live catalog:
        # the catalog evolves (players get merged), the saved table does not.
        los, his = [], []
        for r in tab.itertuples():
            name, c = str(r.player), float(r.rest_center_A)
            if name.startswith("cont_"):
                _, lo, hi = name.split("_")
                los.append(float(lo)); his.append(float(hi))
            else:
                hw = HALF_WIDTH.get(name, 120.0 if c > 4000 else 30.0)
                los.append(c - hw); his.append(c + hw)
        tab["rest_lo"], tab["rest_hi"] = los, his
    z = np.load(args.redshifts)
    z = z[np.isfinite(z) & (z >= 0) & (z <= args.zmax)]

    lam = np.linspace(600.0, 9900.0, 2400)
    zs = np.linspace(0.0, args.zmax, 420)
    prof = phi_profile(tab, lam)                      # [n_lam], nats
    img = np.tile(prof, (len(zs), 1))                 # [n_z, n_lam]
    covered = (lam[None, :] >= OBS_LO / (1 + zs[:, None])) & \
              (lam[None, :] <= OBS_HI / (1 + zs[:, None]))
    img = np.where(covered, img, np.nan)
    img_mnats = img * 1e3

    fig, (ax, axz) = plt.subplots(
        1, 2, figsize=(13.2, 6.4), gridspec_kw={"width_ratios": [4.2, 1.0], "wspace": 0.04},
        sharey=True)
    vmax = float(np.nanpercentile(np.abs(img_mnats), 99.5)) or 1.0
    im = ax.imshow(img_mnats, origin="lower", aspect="auto", cmap="RdBu_r",
                   norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax),
                   extent=[lam[0], lam[-1], zs[0], zs[-1]])
    # coverage boundaries
    ax.plot(OBS_LO / (1 + zs), zs, color=INK, lw=1.4)
    ax.plot(OBS_HI / (1 + zs), zs, color=INK, lw=1.4)

    lines = tab[tab.kind == "line"].sort_values("rest_center_A")
    for r in lines.itertuples():
        c = float(r.rest_center_A)
        zmin_vis = max(0.0, OBS_LO / c - 1.0)
        zmax_vis = min(args.zmax, OBS_HI / c - 1.0)
        if zmax_vis <= zmin_vis:
            continue
        ax.vlines(c, zmin_vis, zmax_vis, color=INK, lw=0.8, ls=":", alpha=0.55)
        ax.text(c, zmax_vis + 0.03, LABEL.get(r.player, r.player), rotation=90,
                fontsize=8.5, ha="center", va="bottom", color=INK)

    ax.set_xlim(700, 9900)
    ax.set_ylim(0, args.zmax + 0.35)
    ax.set_xlabel("rest-frame wavelength (Å)", fontsize=12)
    ax.set_ylabel("redshift", fontsize=12)
    ax.set_title("Flux information by spectral region, over the range each spectrum actually covers",
                 fontsize=12.5, color=INK)
    cb = fig.colorbar(im, ax=[ax, axz], pad=0.02, fraction=0.035, location="right")
    cb.set_label("Shapley value (millinats)", fontsize=10)

    # redshift distribution, shared y
    axz.hist(z, bins=90, orientation="horizontal", color="#0072B2", alpha=0.85)
    axz.set_xlabel("spectra", fontsize=11)
    axz.set_title("sample", fontsize=11, color=MUTED)
    axz.tick_params(labelleft=False, labelsize=9)
    for s in ("top", "right"):
        axz.spines[s].set_visible(False)

    fig.text(0.085, 0.005,
             "Black curves: the observed 3600-9824 Å window mapped to rest frame. Outside them a "
             "spectrum carries no information because the light never lands on the detector.",
             fontsize=9.5, color=MUTED)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=175, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
