#!/usr/bin/env python
"""Flux information per TOKEN, across redshift.

The unit of measurement is the AION spectrum token: a fixed 25.6 Å window in
the OBSERVED frame. Its rest-frame position and width therefore both depend on
redshift (width 25.6/(1+z) Å), so the token grid is not a fixed wavelength
grid. This draws the attribution on that grid: one row per redshift bin, cells
at the token boundaries for that bin's redshift, coloured by the Shapley value
of whichever player owns that token there.

Two things it shows that a wavelength heatmap cannot: the token grid getting
finer in the rest frame as z grows, and the coverage edges where tokens stop
landing inside the observed window.

    python docs/make_token_redshift_figure.py --by-zbin shapley_by_zbin.csv \
        [--out docs/figures/fig_token_redshift.png]
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
from shareable_aion_flow.line_shapley import (  # noqa: E402
    N_SPEC_TOKENS, player_catalog, player_token_map, token_obs_wavelength_edges, usable_tokens,
)

INK, MUTED = "#1a1a1a", "#6a6a6a"
LABEL = {
    "Halpha": r"H$\alpha$", "Hbeta": r"H$\beta$", "Hgamma": r"H$\gamma$",
    "Hdelta": r"H$\delta$", "OIII": "[O III]", "OII": "[O II]", "NeIII": "[Ne III]",
    "NeV": "[Ne V]", "HeII": "He II", "MgII": "Mg II", "CIII": "C III]",
    "CIV": "C IV", "Lyalpha": r"Ly$\alpha$", "Hbeta+OIII": r"H$\beta$+[O III]",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--by-zbin", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("docs/figures/fig_token_redshift.png"))
    ap.add_argument("--guard", type=int, default=2)
    args = ap.parse_args()

    z = pd.read_csv(args.by_zbin, index_col=0)
    z_lo = z.pop("z_lo").to_numpy() if "z_lo" in z.columns else None
    z_hi = z.pop("z_hi").to_numpy() if "z_hi" in z.columns else None
    if z_lo is None:                       # legacy 4-bin table without edges
        z_lo = np.array([0.0, 0.5, 1.0, 1.7])
        z_hi = np.array([0.5, 1.0, 1.7, 3.5])
    z_hi = np.minimum(z_hi, 4.0)
    players = player_catalog()
    name_to_idx = {p["name"]: i for i, p in enumerate(players)}
    phi = z.to_numpy() * 1e3               # [n_z, n_players] millinats
    cols = list(z.columns)

    lo_obs, hi_obs = token_obs_wavelength_edges()
    usable = usable_tokens()
    vmax = float(np.nanpercentile(np.abs(phi[np.isfinite(phi) & (phi != 0)]), 98)) or 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")

    fig, ax = plt.subplots(figsize=(13.4, 6.6))
    for b in range(len(z_lo)):
        zc = 0.5 * (z_lo[b] + z_hi[b])
        tok_of_player = player_token_map(float(zc), players, line_guard_tokens=args.guard)
        owner = np.full(N_SPEC_TOKENS, -1, dtype=int)
        for pi, toks in enumerate(tok_of_player):
            owner[toks] = pi
        # token edges in the REST frame at this bin's redshift
        rest_lo = lo_obs / (1.0 + zc)
        rest_hi = hi_obs / (1.0 + zc)
        for t in range(N_SPEC_TOKENS):
            if not usable[t] or owner[t] < 0:
                continue
            pname = players[owner[t]]["name"]
            if pname not in name_to_idx or pname not in cols:
                continue
            v = phi[b, cols.index(pname)]
            if not np.isfinite(v) or v == 0:
                continue
            ax.add_patch(plt.Rectangle(
                (rest_lo[t], z_lo[b]), rest_hi[t] - rest_lo[t], z_hi[b] - z_lo[b],
                facecolor=cmap(norm(v)), edgecolor="none"))
        ax.hlines(z_hi[b], 0, 1e4, color="white", lw=0.7)

    # named lines sit at fixed REST wavelength: vertical, unlike the token grid
    for p in players:
        if p["kind"] != "line":
            continue
        c = p["rest_center"]
        zmin = max(z_lo[0], 3600.0 / c - 1.0)
        zmax = min(z_hi[-1], 9824.0 / c - 1.0)
        if zmax <= zmin:
            continue
        ax.vlines(c, zmin, zmax, color=INK, lw=0.9, ls=":", alpha=0.7)
        ax.text(c, zmax + 0.02, LABEL.get(p["name"], p["name"]), rotation=90,
                fontsize=8.5, ha="center", va="bottom", color=INK)

    zs = np.linspace(z_lo[0], z_hi[-1], 200)
    ax.plot(3600.0 / (1 + zs), zs, color=INK, lw=1.5)
    ax.plot(9824.0 / (1 + zs), zs, color=INK, lw=1.5)
    ax.set_xlim(700, 9900)
    ax.set_ylim(z_lo[0], z_hi[-1] + 0.3)
    ax.set_xlabel("rest-frame wavelength (Å)", fontsize=12)
    ax.set_ylabel("redshift", fontsize=12)
    ax.set_title("Flux information per spectrum token, across redshift", fontsize=13, color=INK)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.04)
    cb.set_label("Shapley value (millinats)", fontsize=10)
    fig.subplots_adjust(bottom=0.20)
    fig.text(0.09, 0.035,
             "Cells are tokens: 25.6 Å observed, so 25.6/(1+z) Å in the rest frame, and the grid "
             "gets finer with redshift. Adjacent tokens owned by the\nsame player share a value, so "
             "they merge into blocks. Dotted verticals are emission lines, fixed in the rest frame; "
             "black curves bound the observed window.",
             fontsize=9.5, color=MUTED)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=175, bbox_inches="tight")
    print(f"wrote {args.out}  ({len(z_lo)} redshift bins)")


if __name__ == "__main__":
    main()
