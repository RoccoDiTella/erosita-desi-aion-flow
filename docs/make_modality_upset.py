#!/usr/bin/env python
"""UpSet view of what each input modality is worth, in information gain.

Replaces the modality-Shapley pair of slides. A Shapley interaction index like
"spectra x z = -0.679" is a second-order quantity nobody can read off a slide;
the same fact is obvious here as a bar that fails to clear a dotted line.

Layout: combinations grouped left-to-right as singles | pairs | triples | all
four, in a fixed order spelled out by the matrix underneath. Dotted references
mark the four single modalities and the all-together total, so redundancy (bar
below the line it contains) and synergy (bar above the sum) read off the
geometry directly.
One shared UpSet membership matrix sits under the panels since the combination
order is identical for every target.

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

INK, MUTED, GRID = "#1a1a1a", "#6a6a6a", "#d5d5d5"
MODALITIES = ["spectra", "z", "wise", "image"]
NICE = {"spectra": "spectra", "z": "z", "wise": "WISE", "image": "image"}
ZONE_COLOR = {1: "#0072B2", 2: "#009E73", 3: "#E69F00", 4: "#D55E00"}
HEADS = [("log_ml_flux_1", "log flux"), ("log_lx", r"log $L_X$"), ("log_sfr", "log SFR")]


def combo_order() -> list[tuple[str, ...]]:
    """singles, then pairs, then triples, then the 4-way."""
    out: list[tuple[str, ...]] = []
    for k in (1, 2, 3, 4):
        out.extend(combinations(MODALITIES, k))
    return out


def key(combo: tuple[str, ...]) -> str:
    """The eval writes groups in MODALITIES order joined by '+'."""
    return "+".join(m for m in MODALITIES if m in combo)


def panel(ax, values: dict[str, float], title: str, order: list[tuple[str, ...]]) -> None:
    xs = np.arange(len(order))
    vals = np.array([values.get(key(c), np.nan) for c in order], float)
    colors = [ZONE_COLOR[len(c)] for c in order]
    ax.bar(xs, np.nan_to_num(vals), color=colors, width=0.72, zorder=3)

    # labels live in the right margin, clear of the last bar (index len-1)
    lab_x = len(order) - 0.1
    singles = {m: values.get(key((m,)), np.nan) for m in MODALITIES}
    for m, v in singles.items():
        if np.isfinite(v):
            ax.axhline(v, ls=":", lw=1.0, color=MUTED, alpha=0.75, zorder=2)
            ax.text(lab_x, v, f" {NICE[m]}", fontsize=7.4, color=MUTED,
                    va="center", ha="left")
    allv = values.get(key(tuple(MODALITIES)), np.nan)
    if np.isfinite(allv):
        ax.axhline(allv, ls="--", lw=1.3, color=INK, alpha=0.85, zorder=2)
        ax.text(lab_x, allv, " all four", fontsize=7.8, color=INK,
                va="center", ha="left", fontweight="bold")

    ax.set_ylabel("info gain [nats]", fontsize=9.5)
    ax.set_title(title, fontsize=11.5, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.7, len(order) + 2.4)
    ax.set_xticks([])
    ax.tick_params(labelsize=8.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    lo = min(0.0, np.nanmin(vals) if np.isfinite(vals).any() else 0.0)
    hi = np.nanmax(vals) if np.isfinite(vals).any() else 1.0
    ax.set_ylim(lo, hi * 1.18)


def matrix(ax, order: list[tuple[str, ...]]) -> None:
    for i, m in enumerate(MODALITIES):
        y = len(MODALITIES) - 1 - i
        ax.axhline(y, color="#f2f2f2", lw=10, zorder=0)
        ax.text(-0.9, y, NICE[m], fontsize=9, color=INK, ha="right", va="center")
    for x, combo in enumerate(order):
        ys = [len(MODALITIES) - 1 - MODALITIES.index(m) for m in combo]
        ax.plot([x, x], [min(ys), max(ys)], color=ZONE_COLOR[len(combo)], lw=1.6, zorder=2)
        for i, m in enumerate(MODALITIES):
            y = len(MODALITIES) - 1 - i
            on = m in combo
            ax.scatter([x], [y], s=42, zorder=3,
                       color=ZONE_COLOR[len(combo)] if on else "#dcdcdc")
    ax.set_xlim(-0.7, len(order) + 2.4)
    ax.set_ylim(-0.6, len(MODALITIES) - 0.4)
    ax.axis("off")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", type=Path, required=True,
                    help="multi_test_metrics.csv (head/input_group/info_gain_nats), or a "
                         "single-target metrics CSV with no 'head' column")
    ap.add_argument("--single-label", default="log flux",
                    help="panel title when --metrics has no 'head' column")
    ap.add_argument("--out", type=Path, default=Path("docs/figures/fig_modality_upset.png"))
    args = ap.parse_args()

    df = pd.read_csv(args.metrics)
    order = combo_order()

    if "head" in df.columns:
        # bracket access, not df.head -- that attribute is the DataFrame method
        panels = [(lbl, df[df["head"] == h]) for h, lbl in HEADS]
        panels = [(lbl, d) for lbl, d in panels if len(d)]
    else:
        panels = [(args.single_label, df)]
    if not panels:
        raise SystemExit("no matching heads in the metrics file")

    fig = plt.figure(figsize=(13.0, 3.0 + 1.85 * len(panels)))
    gs = fig.add_gridspec(len(panels) + 1, 1,
                          height_ratios=[1.85] * len(panels) + [1.05], hspace=0.30)
    for i, (lbl, d) in enumerate(panels):
        vals = dict(zip(d.input_group.astype(str), d.info_gain_nats.astype(float)))
        panel(fig.add_subplot(gs[i, 0]), vals, lbl, order)
    matrix(fig.add_subplot(gs[len(panels), 0]), order)

    fig.suptitle("Information gain by input combination", fontsize=13.5, color=INK, y=0.995)
    fig.text(0.5, 0.005,
             "dotted = each modality alone   ·   dashed = all four together   ·   "
             "a bar below a dotted line it contains is redundancy, above the sum is synergy",
             ha="center", fontsize=8.6, color=MUTED)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
