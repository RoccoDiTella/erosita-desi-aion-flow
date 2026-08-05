#!/usr/bin/env python
"""Distribution of the PER-SOURCE posterior correlation for one pair.

The summary tables report a mean over sources, and a mean is ambiguous: near
zero it cannot distinguish "every source has no correlation" from "the
population splits into two opposite signs". Only the histogram separates them.

Two panels, because they answer different questions:
  r   signed, so a split population is visible
  r^2 magnitude only, so "how much structure is there at all" is visible,
      with the sampling floor drawn in. With S draws per source, a TRUE zero
      correlation still yields r^2 of order 1/S, so anything below that line
      is noise rather than signal.

    python scripts/posterior_corr_hist.py --npz poststruct_noWISE.npz \
        --pair log_lx log_ssfr --group GALAXY --out fig.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID = "#1a1a1a", "#6a6a6a", "#d5d5d5"
FILL, EDGE, ACCENT = "#0072B2", "#004c78", "#D55E00"


def style(ax) -> None:
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--pair", nargs=2, required=True)
    ap.add_argument("--group", default="GALAXY")
    ap.add_argument("--samples", type=int, default=512,
                    help="draws per source, used to draw the sampling floor")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    dims = [str(x) for x in d["dims"]]
    a, b = args.pair
    i, j = dims.index(a), dims.index(b)
    keep = d["group"].astype(str) == args.group
    r = d["per_source"][keep, i, j].astype(float)
    r = r[np.isfinite(r)]
    n = len(r)
    if not n:
        raise SystemExit(f"no sources in group {args.group!r}")

    # A correlation estimated from S independent draws has sd ~ 1/sqrt(S-3)
    # under the null, so this is where "structure" has to clear to count.
    floor = 1.0 / np.sqrt(max(args.samples - 3, 1))
    mean, med = float(np.mean(r)), float(np.median(r))
    se = float(np.std(r, ddof=1) / np.sqrt(n))
    print(f"{args.group}: n={n:,}  mean r={mean:+.4f} +/- {se:.4f}  median={med:+.4f}")
    print(f"  frac r<0 = {np.mean(r < 0):.1%}   sampling floor |r| ~ {floor:.3f}")
    print(f"  mean r^2 = {np.mean(r**2):.4f}   floor r^2 ~ {floor**2:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))

    ax = axes[0]
    ax.hist(r, bins=40, color=FILL, edgecolor=EDGE, lw=0.5, alpha=0.85)
    ax.axvline(0, color=MUTED, lw=1.2, ls=":")
    ax.axvspan(-floor, floor, color=ACCENT, alpha=0.10, zorder=0)
    ax.axvline(mean, color=ACCENT, lw=2.0)
    ax.set_title(f"per-source posterior corr, {args.group}", color=INK, fontsize=12)
    ax.set_xlabel(f"r ({a} , {b})", color=MUTED)
    ax.set_ylabel("sources", color=MUTED)
    ax.text(0.02, 0.97, f"n = {n:,}\nmean {mean:+.4f} ± {se:.4f}\n"
                        f"median {med:+.4f}\nfrac < 0  {np.mean(r < 0):.1%}",
            transform=ax.transAxes, va="top", fontsize=9, color=INK)
    ax.text(0.98, 0.97, f"shaded: |r| below the\nsampling floor for "
                        f"{args.samples} draws", transform=ax.transAxes,
            va="top", ha="right", fontsize=8.5, color=ACCENT)
    style(ax)

    ax = axes[1]
    ax.hist(r ** 2, bins=40, color=FILL, edgecolor=EDGE, lw=0.5, alpha=0.85)
    ax.axvline(floor ** 2, color=ACCENT, lw=2.0)
    ax.set_title("shared variance per source", color=INK, fontsize=12)
    ax.set_xlabel(r"$r^2$", color=MUTED)
    ax.set_ylabel("sources", color=MUTED)
    ax.text(0.98, 0.97, f"mean $r^2$ = {np.mean(r**2):.4f}\n"
                        f"floor = {floor**2:.4f}\n"
                        f"frac above floor = {np.mean(r**2 > floor**2):.1%}",
            transform=ax.transAxes, va="top", ha="right", fontsize=9, color=INK)
    style(ax)

    fig.tight_layout()
    fig.savefig(args.out, dpi=190, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
