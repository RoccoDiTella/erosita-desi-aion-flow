#!/usr/bin/env python
"""Legible redshift-resolved Shapley heatmap.

Splits the players into two panels instead of interleaving 37 mixed columns on
one categorical axis: named lines (labelled individually) and continuum bins
(labelled by wavelength). Both share one symmetric colour scale, cells carry
their value, and unavailable (player, z-bin) cells are struck out rather than
drawn as zero.

    python docs/make_shapley_heatmap.py --by-zbin shapley_by_zbin.csv \
        --shapley shapley_table.csv [--out docs/figures/fig_shapley_heatmap.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

INK, MUTED = "#1a1a1a", "#6a6a6a"
Z_LABELS = ["z < 0.5", "0.5 - 1.0", "1.0 - 1.7", "z > 1.7"]
LABEL = {
    "Halpha": r"H$\alpha$", "Hbeta": r"H$\beta$", "Hgamma": r"H$\gamma$",
    "Hdelta": r"H$\delta$", "OIII": "[O III]", "OII": "[O II]", "NeIII": "[Ne III]",
    "NeV": "[Ne V]", "HeII": "He II", "MgII": "Mg II", "CIII": "C III]",
    "CIV": "C IV", "Lyalpha": r"Ly$\alpha$", "Hbeta+OIII": r"H$\beta$+[O III]",
}


def panel(ax, mat, cols, labels, norm, title, annotate: bool) -> None:
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_xticks(range(len(cols)), labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(Z_LABELS)), Z_LABELS, fontsize=10)
    ax.set_title(title, fontsize=11.5, color=INK, pad=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "·", ha="center", va="center", fontsize=11, color=MUTED)
            elif annotate:
                strong = abs(v) > 0.45 * max(abs(norm.vmin), abs(norm.vmax))
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if strong else INK)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(Z_LABELS)), minor=True)
    ax.grid(which="minor", color="white", lw=1.4)
    ax.tick_params(which="minor", length=0)
    return im


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--by-zbin", type=Path, required=True)
    ap.add_argument("--shapley", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("docs/figures/fig_shapley_heatmap.png"))
    args = ap.parse_args()

    z = pd.read_csv(args.by_zbin, index_col=0)          # [4 z-bins x players], nats
    tab = pd.read_csv(args.shapley)
    kind = dict(zip(tab.player, tab.kind))
    centre = dict(zip(tab.player, tab.rest_center_A))

    lines = [c for c in z.columns if kind.get(c) == "line"]
    cont = [c for c in z.columns if kind.get(c) == "cont"]
    lines.sort(key=lambda c: centre[c])
    cont.sort(key=lambda c: centre[c])

    ml = z[lines].to_numpy() * 1e3                      # millinats
    mc = z[cont].to_numpy() * 1e3
    ml[ml == 0] = np.nan                                # unavailable in that z-bin
    mc[mc == 0] = np.nan
    vmax = float(np.nanpercentile(np.abs(np.concatenate([ml.ravel(), mc.ravel()])), 98)) or 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13.0, 6.2),
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.75})
    im = panel(ax1, ml, lines, [LABEL.get(c, c) for c in lines], norm,
               "Emission lines", annotate=True)
    panel(ax2, mc, cont, [f"{centre[c]:.0f}" for c in cont], norm,
          "Continuum bins (rest-frame centre, Å)", annotate=False)
    cb = fig.colorbar(im, ax=[ax1, ax2], fraction=0.028, pad=0.015)
    cb.set_label("Shapley value (millinats)", fontsize=10)
    fig.text(0.5, 0.005, "· marks a region the spectra in that redshift bin do not cover",
             fontsize=9.5, color=MUTED, ha="center")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=175, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
