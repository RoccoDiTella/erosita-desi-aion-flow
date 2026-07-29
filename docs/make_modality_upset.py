#!/usr/bin/env python
"""UpSet view of what each input modality is worth, in information gain.

Replaces the modality-Shapley pair of slides. A Shapley interaction index like
"spectra x z = -0.679" is a second-order quantity nobody can read off a slide;
the same fact is obvious here as a bar that fails to clear a dotted line.

One figure per target, full slide width. Combination bars are filled with
diagonal stripes in the colours of the modalities they contain, so membership
is readable from the bar itself. Dotted references mark each modality alone, in
its own colour; there is deliberately no line for all-four, since the rightmost
bar is that value. Palette matches the paper's redshift figure.

    python docs/make_modality_upset.py --metrics .../multi_test_metrics.csv
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

INK, MUTED, GRID = "#1a1a1a", "#6a6a6a", "#d5d5d5"
MODALITIES = ["spectra", "z", "wise", "image"]
NICE = {"spectra": "Spectra", "z": "Redshift", "wise": "WISE", "image": "Images"}
# the paper's performance-by-redshift palette
COLOR = {"spectra": "#7B4FA3", "z": "#D62728", "wise": "#8C5A2B", "image": "#2E86C1"}
HEADS = [("log_ml_flux_1", "log flux"), ("log_lx", r"log $L_X$"), ("log_sfr", "log SFR"),
         ("logmstar", r"log $M_*$")]


def combo_order() -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    for k in (1, 2, 3, 4):
        out.extend(combinations(MODALITIES, k))
    return out


def key(combo: tuple[str, ...]) -> str:
    return "+".join(m for m in MODALITIES if m in combo)


def striped_bar(ax, x, height, width, colors, stripe_px: int = 9) -> None:
    """Fill a bar with diagonal stripes cycling through `colors`.

    matplotlib hatching cannot do multi-colour stripes, so paint a small RGBA
    raster of diagonal bands and clip it to the bar rectangle.
    """
    if height <= 0 or not np.isfinite(height):
        return
    n = 256
    ii, jj = np.mgrid[0:n, 0:n]
    band = ((ii + jj) // stripe_px) % len(colors)
    rgba = np.zeros((n, n, 4))
    for k, c in enumerate(colors):
        rgba[band == k] = matplotlib.colors.to_rgba(c)
    rect = Rectangle((x - width / 2, 0), width, height, transform=ax.transData)
    im = ax.imshow(rgba, extent=(x - width / 2, x + width / 2, 0, height),
                   origin="lower", aspect="auto", zorder=3, interpolation="nearest")
    im.set_clip_path(rect)


def panel(ax, values: dict[str, float], order, title: str) -> None:
    xs = np.arange(len(order))
    vals = np.array([values.get(key(c), np.nan) for c in order], float)
    hi = np.nanmax(vals) if np.isfinite(vals).any() else 1.0
    for x, combo, v in zip(xs, order, vals):
        striped_bar(ax, x, v, 0.74, [COLOR[m] for m in MODALITIES if m in combo])
        if np.isfinite(v):
            ax.text(x, v + 0.018 * hi, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=9.5, color=INK)
    # one dotted reference per modality, in its own colour. The lines sit at the
    # true values; only the LABELS are nudged apart, since three modalities can
    # land within a few hundredths of a nat and overprint each other.
    solo = [(m, values.get(key((m,)), np.nan)) for m in MODALITIES]
    solo = sorted([(m, v) for m, v in solo if np.isfinite(v)], key=lambda t: t[1])
    min_sep = 0.055 * hi
    label_y: list[float] = []
    for _, v in solo:
        y = v if not label_y else max(v, label_y[-1] + min_sep)
        label_y.append(y)
    for (m, v), ly in zip(solo, label_y):
        ax.axhline(v, ls=":", lw=1.6, color=COLOR[m], alpha=0.95, zorder=2)
        ax.text(len(order) - 0.30, ly, f"  {NICE[m]}", fontsize=10.5,
                color=COLOR[m], va="center", ha="left", fontweight="bold")
    ax.set_ylabel("information gain [nats]", fontsize=12)
    ax.set_title(title, fontsize=16, color=INK, loc="left", pad=10)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.7, len(order) + 2.1)
    ax.set_ylim(0, hi * 1.14)
    ax.set_xticks([])
    ax.tick_params(labelsize=11)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def matrix(ax, order) -> None:
    for i, m in enumerate(MODALITIES):
        y = len(MODALITIES) - 1 - i
        ax.axhline(y, color="#f4f4f4", lw=13, zorder=0)
        ax.text(-0.9, y, NICE[m], fontsize=11, color=COLOR[m], ha="right",
                va="center", fontweight="bold")
    for x, combo in enumerate(order):
        ys = [len(MODALITIES) - 1 - MODALITIES.index(m) for m in combo]
        if len(ys) > 1:
            ax.plot([x, x], [min(ys), max(ys)], color="#b0b0b0", lw=1.5, zorder=2)
        for i, m in enumerate(MODALITIES):
            y = len(MODALITIES) - 1 - i
            ax.scatter([x], [y], s=70, zorder=3,
                       color=COLOR[m] if m in combo else "#dedede")
    ax.set_xlim(-0.7, len(order) + 2.1)
    ax.set_ylim(-0.6, len(MODALITIES) - 0.4)
    ax.axis("off")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--figdir", type=Path, default=Path("docs/figures"))
    ap.add_argument("--single-label", default="log flux")
    args = ap.parse_args()

    df = pd.read_csv(args.metrics)
    order = combo_order()
    args.figdir.mkdir(parents=True, exist_ok=True)

    if "head" in df.columns:
        targets = [(h, lbl, df[df["head"] == h]) for h, lbl in HEADS]
        targets = [(h, lbl, d) for h, lbl, d in targets if len(d)]
    else:
        targets = [("flux", args.single_label, df)]

    for head, lbl, d in targets:
        vals = dict(zip(d.input_group.astype(str), d.info_gain_nats.astype(float)))
        fig = plt.figure(figsize=(13.2, 6.6))
        gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.06)
        panel(fig.add_subplot(gs[0, 0]), vals, order,
              f"{lbl}: information gain by input combination")
        matrix(fig.add_subplot(gs[1, 0]), order)
        out = args.figdir / f"fig_upset_{head}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
