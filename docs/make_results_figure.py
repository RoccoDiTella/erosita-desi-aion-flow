#!/usr/bin/env python
"""Per-target results for the multi-target model, against the lines-only baseline.

Left panel R^2, right panel information gain, one row per head. The reference
marks are the emission-line-only flow (scripts/train_lines_baseline.py), NOT
V_PAI: V_PAI is the same frozen-AION architecture with a different head, so
beating it says nothing about whether the learned representation is worth
anything. The line-flux baseline is the classical alternative, so the distance
between bar and mark is the part of the answer that AION supplies.

    python docs/make_results_figure.py --metrics .../multi_test_metrics.csv \
        --baseline results/baseline_lines_only_clean.csv [--hr-r2 ..] [--hr-ig ..]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID = "#1a1a1a", "#6a6a6a", "#d5d5d5"
BAR, BAR_DIM, MARK = "#0072B2", "#9ecae1", "#D55E00"
ALL_INPUTS = "spectra+z+wise+image"
HEADS = [
    ("log_ml_flux_1", "log flux"),
    ("log_lx", r"log $L_X$"),
    ("log_sfr", "log SFR"),
    ("logmstar", r"log $M_*$"),
    ("log_flux_p1", "P1  0.2-0.5 keV"),
    ("log_flux_p2", "P2  0.5-1.0 keV"),
    ("log_flux_p3", "P3  1.0-2.0 keV"),
    ("log_flux_p4", "P4  2.0-5.0 keV"),
]
# the three targets the lines-only baseline was trained for
PRIMARY = {"log_ml_flux_1", "log_lx", "log_sfr"}


def panel(ax, labels, values, base, title, xlabel) -> None:
    y = np.arange(len(labels))[::-1]
    colors = [BAR if np.isfinite(v) and lbl_key in PRIMARY else BAR_DIM
              for v, lbl_key in zip(values, [k for k, _ in HEADS][:len(values)])]
    ax.barh(y, np.nan_to_num(values), color=colors, height=0.68, zorder=3)
    for yi, v in zip(y, values):
        if np.isfinite(v):
            ax.text(v + 0.012 * max(1e-9, np.nanmax(values)), yi, f"{v:+.3f}",
                    va="center", fontsize=8.6, color=INK)
    drew = False
    for yi, b in zip(y, base):
        if np.isfinite(b):
            ax.plot([b, b], [yi - 0.40, yi + 0.40], color=MARK, lw=2.4, zorder=4,
                    label=None if drew else "emission lines only")
            drew = True
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_title(title, fontsize=13, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=9.5)
    ax.grid(axis="x", color=GRID, lw=0.7, alpha=0.95)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8.8)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    lo = min(0.0, float(np.nanmin(values)) if np.isfinite(values).any() else 0.0)
    hi = float(np.nanmax(np.concatenate([values, base]))) if len(values) else 1.0
    ax.set_xlim(lo, hi * 1.16 if hi > 0 else 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, default=None)
    ap.add_argument("--hr-csv", type=Path, default=None,
                    help="hr_implied_target.csv; R2 and IG are computed from it directly "
                         "rather than transcribed by hand")
    ap.add_argument("--hr-r2", type=float, default=np.nan, help="override --hr-csv")
    ap.add_argument("--hr-ig", type=float, default=np.nan, help="override --hr-csv")
    ap.add_argument("--out", type=Path, default=Path("docs/figures/fig_results_v3.png"))
    args = ap.parse_args()

    t = pd.read_csv(args.metrics)
    t = t[t["input_group"] == ALL_INPUTS]
    base = pd.read_csv(args.baseline) if args.baseline and args.baseline.exists() else None

    labels, r2, ig, br2, big = [], [], [], [], []
    for key, label in HEADS:
        row = t[t["head"] == key]
        if not len(row):
            continue
        labels.append(label)
        r2.append(float(row.r2.iloc[0]))
        ig.append(float(row.info_gain_nats.iloc[0]))
        b = base[base["head"] == key] if base is not None else None
        br2.append(float(b.r2.iloc[0]) if b is not None and len(b) else np.nan)
        big.append(float(b.info_gain_nats.iloc[0]) if b is not None and len(b) else np.nan)
    hr_r2, hr_ig = args.hr_r2, args.hr_ig
    if args.hr_csv and args.hr_csv.exists() and not (np.isfinite(hr_r2) and np.isfinite(hr_ig)):
        h = pd.read_csv(args.hr_csv)
        if len(h) > 30:
            hr_r2 = 1.0 - float(np.sum((h.hr_meas - h.hr_p50) ** 2)
                                / np.sum((h.hr_meas - h.hr_meas.mean()) ** 2))
            hr_ig = float(h.info_gain.mean())
    if np.isfinite(hr_r2) or np.isfinite(hr_ig):
        labels.append("HR32 (implied)")
        r2.append(hr_r2); ig.append(hr_ig)
        br2.append(np.nan); big.append(np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 0.62 * len(labels) + 2.2))
    panel(axes[0], labels, np.array(r2, float), np.array(br2, float),
          r"$R^2$", "test $R^2$ (posterior mean)")
    panel(axes[1], labels, np.array(ig, float), np.array(big, float),
          "Info gain", "nats over the KDE prior")
    axes[1].set_yticklabels([])
    handles, lab = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, lab, frameon=False, fontsize=9.5,
                   loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}  ({len(labels)} heads"
          f"{', with lines-only marks' if base is not None else ''})")


if __name__ == "__main__":
    main()
