#!/usr/bin/env python
"""What each input modality is worth, in information gain, per target.

Replaces the modality-Shapley pair of slides. A Shapley interaction index like
"spectra x z = -0.679" is a second-order quantity nobody can read off a slide;
the same fact is obvious here as a bar that fails to clear a dotted line.

Each bar is divided into one horizontal band per input it contains, always in
the same top-to-bottom order (spectra, redshift, WISE, images), so a given
modality sits at the same relative height in every bar. Band boundaries are cut
at 45 degrees on the page rather than flat, and each band carries its initial.
Dotted references mark each modality alone, in its own colour; there is no line
for all-four, since the rightmost bar is that value.

Palette matches the paper's performance-by-redshift figure.

This figure compares bars AGAINST EACH OTHER -- a bar below a dotted line it
contains is redundancy -- so it is only meaningful within one row set. That is
what `--sample common` buys, and it is why this script now filters on the sample
column instead of `dict(zip(input_group, ig))`, which kept whichever row pandas
saw last and therefore threw the common-sample numbers away.

    python docs/make_modality_upset.py --metrics .../multi_test_metrics.csv \
        --sample common
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deck_sample import (  # noqa: E402
    add_sample_argument, cross_combo_warning, sample_caption, sample_meaning, select_sample,
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
from matplotlib.patches import Patch, Polygon, Rectangle  # noqa: E402

INK, MUTED, GRID = "#1a1a1a", "#6a6a6a", "#d5d5d5"
# order is load-bearing: it fixes the vertical order of the bands
MODALITIES = ["spectra", "z", "wise", "image"]
NICE = {"spectra": "Spectra", "z": "Redshift", "wise": "WISE", "image": "Images"}
LETTER = {"spectra": "S", "z": "Z", "wise": "W", "image": "I"}
COLOR = {"spectra": "#7B4FA3", "z": "#D62728", "wise": "#8C5A2B", "image": "#2E86C1"}
BASELINE_C = "#6a6a6a"          # emission-lines-only level, shown as the "EL" y tick
HEADS = [("log_ml_flux_1", "log flux"), ("log_lx", r"log $L_X$"), ("log_sfr", "log SFR"),
         ("logmstar", r"log $M_*$")]


def combo_order() -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    for k in (1, 2, 3, 4):
        out.extend(combinations(MODALITIES, k))
    return out


def key(combo: tuple[str, ...]) -> str:
    return "+".join(m for m in MODALITIES if m in combo)


def banded_bar(ax, x, height, width, members) -> None:
    """Split a bar into equal horizontal bands, one per member, cut at 45 degrees.

    Painted from the bottom member upward: each band fills everything ABOVE its
    own lower boundary, so later (higher) members overwrite the ones below and
    the result is clean bands without computing polygon intersections. Everything
    is clipped to the bar rectangle, so the bar keeps square outer edges and only
    the internal divisions are slanted.
    """
    if not np.isfinite(height) or height <= 0 or not members:
        return
    k = len(members)
    x0, x1, xc = x - width / 2, x + width / 2, x
    rect = Rectangle((x0, 0), width, height, transform=ax.transData)

    # slope giving 45 degrees on the page, in data units
    t = ax.transData
    px_per_x = t.transform((1.0, 0.0))[0] - t.transform((0.0, 0.0))[0]
    px_per_y = t.transform((0.0, 1.0))[1] - t.transform((0.0, 0.0))[1]
    slope = abs(px_per_x / px_per_y) if px_per_y else 0.0

    top, bottom = height * 2.0, -height
    for j in range(k - 1, -1, -1):          # bottom member first
        m = members[j]
        b = height * (k - 1 - j) / k        # nominal lower boundary of this band
        if j == k - 1:
            # the lowest band's lower edge IS the bar's base, not a divider --
            # slanting it would shave the bottom corners off the bar
            lo = [[x0, bottom], [x1, bottom]]
        else:
            lo = [[x0, b + slope * (x0 - xc)], [x1, b + slope * (x1 - xc)]]
        poly = Polygon(lo + [[x1, top], [x0, top]],
                       closed=True, facecolor=COLOR[m], edgecolor="none", zorder=3)
        ax.add_patch(poly)
        poly.set_clip_path(rect)

    # a letter per band, only where it fits
    band_px = abs(px_per_y) * height / k
    if band_px >= 15:
        for j, m in enumerate(members):
            yc = height * (k - 1 - j) / k + height / (2 * k)
            ax.text(xc, yc, LETTER[m], ha="center", va="center", zorder=5,
                    fontsize=min(13.0, 0.55 * band_px), color="white",
                    fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=1.6, foreground="#00000055")])


def value_label(ax, x, v, hi) -> None:
    """Bold, on a white plate: the reference lines cross these numbers."""
    ax.text(x, v + 0.020 * hi, f"{v:.2f}", ha="center", va="bottom", zorder=8,
            fontsize=10.5, color=INK, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.88,
                      boxstyle="round,pad=0.16"))


def panel(ax, values: dict[str, float], order, title: str,
          baseline: float = np.nan, subtitle: str = "", warning: str | None = None) -> None:
    xs = np.arange(len(order))
    vals = np.array([values.get(key(c), np.nan) for c in order], float)
    hi = np.nanmax(vals) if np.isfinite(vals).any() else 1.0
    # limits FIRST: banded_bar reads transData to get the 45-degree slope right
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.set_ylim(0, hi * 1.16)
    for x, combo, v in zip(xs, order, vals):
        members = [m for m in MODALITIES if m in combo]     # fixed vertical order
        banded_bar(ax, x, v, 0.78, members)
        if np.isfinite(v):
            value_label(ax, x, v, hi)
    # solid, but UNDER the bars
    for m in MODALITIES:
        v = values.get(key((m,)), np.nan)
        if np.isfinite(v):
            ax.axhline(v, ls="-", lw=2.0, color=COLOR[m], alpha=0.95, zorder=2)
    ax.set_ylabel("information gain [nats]", fontsize=13)
    ax.set_title(title, fontsize=17, color=INK, loc="left", pad=26 if subtitle else 12)
    if subtitle:
        # The row set is part of the number, so it goes on the figure, not in a
        # commit message: these bars are read against each other.
        ax.annotate(subtitle, xy=(0.0, 1.0), xycoords="axes fraction",
                    xytext=(0, 10), textcoords="offset points",
                    fontsize=10.5, color=MUTED, ha="left", va="bottom")
    if warning:
        ax.annotate(warning, xy=(1.0, 1.0), xycoords="axes fraction",
                    xytext=(0, 10), textcoords="offset points",
                    fontsize=10.5, color="#D55E00", ha="right", va="bottom",
                    fontweight="bold")
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks([])
    ax.tick_params(labelsize=12)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    # the emission-lines-only baseline as an extra y tick rather than a bar --
    # it is not an input combination, so it does not belong among them
    if np.isfinite(baseline):
        lo, up = ax.get_ylim()
        ticks = [t for t in ax.get_yticks() if lo <= t <= up]
        # keep the decimals matplotlib would have used; f"{t:g}" silently turns
        # 0.0 and 1.0 into "0" and "1" and the axis stops looking uniform
        step = (ticks[1] - ticks[0]) if len(ticks) > 1 else 1.0
        dec = max(0, int(np.ceil(-np.log10(step)))) if step > 0 else 1
        # drop a numeric tick that EL would print on top of
        ticks = [t for t in ticks if abs(t - baseline) > 0.4 * step]
        labels = [f"{t:.{dec}f}" for t in ticks]
        ticks.append(baseline)
        labels.append("EL")
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        for lab, t in zip(ax.get_yticklabels(), ticks):
            if t == baseline:
                lab.set_color(BASELINE_C)
                lab.set_fontweight("bold")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--figdir", type=Path, default=Path("docs/figures"))
    ap.add_argument("--baseline", type=Path, default=None,
                    help="baseline_lines_only_clean.csv; drawn as the leftmost grey bar")
    ap.add_argument("--single-label", default="log flux")
    add_sample_argument(ap)
    args = ap.parse_args()

    df = pd.read_csv(args.metrics)
    df, sample = select_sample(df, args.sample, source=str(args.metrics))
    warning = cross_combo_warning(sample)
    order = combo_order()
    args.figdir.mkdir(parents=True, exist_ok=True)

    if "head" in df.columns:
        targets = [(h, lbl, df[df["head"] == h]) for h, lbl in HEADS]
        targets = [(h, lbl, d) for h, lbl, d in targets if len(d)]
    else:
        targets = [("flux", args.single_label, df)]

    base = pd.read_csv(args.baseline) if args.baseline and args.baseline.exists() else None
    handles = [Patch(facecolor=COLOR[m], edgecolor="none",
                     label=f"{NICE[m]}  ({LETTER[m]})") for m in MODALITIES]
    for head, lbl, d in targets:
        # select_sample has already guaranteed one row per input_group, so this
        # dict can no longer silently drop half the table.
        vals = dict(zip(d.input_group.astype(str), d.info_gain_nats.astype(float)))
        fig, ax = plt.subplots(figsize=(13.2, 6.0))
        bval = np.nan
        if base is not None:
            r = base[base["head"] == head]
            if len(r):
                bval = float(r.info_gain_nats.iloc[0])
        panel(ax, vals, order, f"{lbl}: information gain by input combination",
              baseline=bval,
              subtitle=f"{sample_caption(sample, d)} · {sample_meaning(sample, short=True)}",
              warning=warning)
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
                   fontsize=14, bbox_to_anchor=(0.5, -0.035), handlelength=1.8,
                   handleheight=1.4, columnspacing=1.9)
        out = args.figdir / f"fig_upset_{head}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
