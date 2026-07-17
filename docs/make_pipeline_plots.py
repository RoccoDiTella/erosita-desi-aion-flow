#!/usr/bin/env python3
"""Generate the figures embedded in docs/pipeline.pdf (the living pipeline doc).

Slide-like illustrations: original architecture, paper results, the NWAY
crossmatch cleanup, the split-normal error treatment, and the target table.
Colorblind-safe Okabe-Ito palette; status colors for the cleanup panels.
Run after any pipeline change:  python docs/make_pipeline_plots.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures"
FIGS.mkdir(exist_ok=True)
SD = Path("/home/roccoditella/astroai/stanford_deadline/data")

# Okabe-Ito (CVD-safe categorical) + status
BLUE, ORANGE, GREEN, VERM, PURPLE, SKY, YELLOW = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442",
)
GRAY, INK, MUTED, GRID = "#8a8a8a", "#1a1a1a", "#666666", "#e2e2e2"
GOOD, WARN, BAD, NEUT = GREEN, ORANGE, VERM, GRAY

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#bbbbbb",
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlecolor": INK, "figure.dpi": 150,
})


def _save(fig, name):
    fig.savefig(FIGS / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", FIGS / name)


def fig_architecture():
    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")

    def box(x, y, w, h, text, fc, ec, tc=INK, fs=8.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                     lw=1.3, color=MUTED))

    # inputs
    inputs = ["DESI spectra\n273 tok", "Legacy image\n576 tok", "WISE W1-3\n3 tok", "redshift\n1 tok"]
    for i, t in enumerate(inputs):
        box(1, 30 - i * 8.3, 15, 6.5, t, "#eef4fb", BLUE, fs=7.5)
    # frozen AION
    box(21, 8, 16, 24, "Frozen\nAION-base\n(300M)", "#f0f0f0", GRAY, tc=MUTED, fs=10)
    ax.text(29, 5.2, "frozen", ha="center", fontsize=7.5, color=MUTED, style="italic")
    # attention head (trainable)
    box(42, 8, 20, 24,
        "Attention head\n(trainable)\n\nmodality affine ->\nlearned queries ->\nattention blocks\n-> MLP -> ctx(256)",
        "#eafaf3", GREEN, fs=7.3)
    # flow
    box(67, 12, 14, 16, "Zuko NSF\n1-D flow\n(trainable)", "#f7edf5", PURPLE, fs=8)
    # target
    box(85, 12, 13, 16, "p(y | x)\n+ errors", "#fdf0e7", VERM, fs=8.5)

    for i in range(4):
        arrow(16, 33.2 - i * 8.3, 21, 20 + (1.5 - i) * 2)
    arrow(37, 20, 42, 20); arrow(62, 20, 67, 20); arrow(81, 20, 85, 20)

    ax.text(52, 34.5, "ORIGINAL (paper q4/l2): 4 queries x 2 blocks, MLP 3072->512->512->256   |   "
                      "V1 (ours): 1 query x 1 block, ->128", ha="center", fontsize=7.5, color=INK)
    ax.text(50, 1.5, "Per-batch a modality combo is sampled; AION frozen; KDE(target) prior anchors information gain.",
            ha="center", fontsize=7.3, color=MUTED)
    _save(fig, "fig_architecture.png")


def fig_results():
    # Paper q4/l2 epoch-13 R^2 per modality combo (Fig-1 numbers) + line baseline.
    combos = [
        ("S+W+I", 0.5539), ("all inputs", 0.5489), ("S+I", 0.5297), ("S+Z+I", 0.5253),
        ("S+W", 0.5152), ("S+Z+W", 0.5074), ("Z+W+I", 0.4976), ("S only", 0.4795),
        ("I only", 0.4176), ("W only", 0.3020), ("Z only", 0.1396),
    ]
    baseline = 0.4230  # emission-line baseline lines_oii_z
    combos = sorted(combos, key=lambda t: t[1])
    labels = [c[0] for c in combos]; vals = [c[1] for c in combos]
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    y = np.arange(len(labels))
    bars = ax.barh(y, vals, color=BLUE, height=0.66, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + 0.006, yi, f"{v:.3f}", va="center", fontsize=8, color=INK)
    ax.axvline(baseline, color=ORANGE, lw=1.6, ls="--", zorder=4)
    ax.text(baseline, len(labels) - 0.3, f"  emission-line baseline {baseline:.3f}",
            color=ORANGE, fontsize=8, va="bottom")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("held-out R²  (predicting log X-ray flux)")
    ax.set_xlim(0, 0.62); ax.set_title("Original paper result — R² by modality combination (S=spectra, W=WISE, I=image, Z=z)")
    ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    _save(fig, "fig_results.png")


def fig_cleanup():
    mq = pd.read_csv(SD / "match_quality.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))

    # (a) match-quality breakdown (status colors)
    order = ["correct", "spurious", "wrong", "ambiguous", "not_in_NWAY"]
    colcol = {"correct": GOOD, "spurious": BAD, "wrong": ORANGE, "ambiguous": NEUT, "not_in_NWAY": "#c9c9c9"}
    counts = mq.match_class.value_counts()
    frac = [100 * counts.get(k, 0) / len(mq) for k in order]
    ax = axes[0]
    bars = ax.bar(range(len(order)), frac, color=[colcol[k] for k in order], zorder=3, width=0.7)
    for i, f in enumerate(frac):
        ax.text(i, f + 1.2, f"{f:.1f}%", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["correct", "spurious", "wrong", "ambig.", "absent"], fontsize=8, rotation=20)
    ax.set_ylabel("% of matches"); ax.set_ylim(0, 100)
    ax.set_title("(a) NWAY verdict on our 5″ matches", fontsize=9.5)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)

    # (b) delta-z bimodality
    d = mq.dropna(subset=["z", "zsp"]).copy(); dz = (d.z - d.zsp).abs().clip(1e-6, 3)
    ax = axes[1]
    ax.hist(np.log10(dz), bins=60, color=BLUE, zorder=3)
    ax.axvline(np.log10(0.01), color=VERM, lw=1.6, ls="--")
    ax.text(np.log10(0.01), ax.get_ylim()[1] * 0.9, " cut=0.01", color=VERM, fontsize=8)
    ax.set_xlabel("log₁₀ |z_ours − z_NWAY|"); ax.set_ylabel("sources")
    ax.set_title("(b) z agreement is bimodal → cut is natural", fontsize=9.5)

    # (c) model performance by match quality (the headline)
    ax = axes[2]
    grp = ["CLEAN", "MISMATCH"]; r2 = [0.567, 0.291]; ig = [0.349, 0.119]
    x = np.arange(2)
    ax.bar(x - 0.19, r2, width=0.36, color=GOOD, label="R²", zorder=3)
    ax.bar(x + 0.19, ig, width=0.36, color=PURPLE, label="IG (nats)", zorder=3)
    for xi, a, b in zip(x, r2, ig):
        ax.text(xi - 0.19, a + 0.012, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(xi + 0.19, b + 0.012, f"{b:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(grp)
    ax.set_ylim(0, 0.65); ax.set_ylabel("test metric")
    ax.set_title("(c) model is 2× worse on mismatches", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    _save(fig, "fig_cleanup.png")


def fig_errors():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    # (a) split-normal vs Gaussian kernel
    ax = axes[0]
    t = np.linspace(-4, 4, 400)
    slo, shi = 1.4, 0.7
    sig = np.where(t < 0, slo, shi)
    split = np.exp(-0.5 * (t / sig) ** 2) * (np.sqrt(2 / np.pi) / (slo + shi))
    gauss = np.exp(-0.5 * t ** 2) / np.sqrt(2 * np.pi)
    ax.plot(t, gauss, color=GRAY, lw=2, label="symmetric (Gaussian)")
    ax.plot(t, split, color=BLUE, lw=2.2, label=f"split-normal (σ⁻={slo}, σ⁺={shi})")
    ax.fill_between(t, split, color=BLUE, alpha=0.12)
    ax.axvline(0, color=MUTED, lw=0.8, ls=":")
    ax.set_yticks([]); ax.set_xlabel("target − measured")
    ax.set_title("(a) asymmetric measurement error", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    # (b) error magnitude vs target spread (log flux)
    ax = axes[1]
    te = pd.read_csv(SD / "targets_extra.csv")
    sig = (te.flux_sig_lo + te.flux_sig_hi) / 2
    sig = sig[np.isfinite(sig) & (sig > 0) & (sig < 0.6)]
    ax.hist(sig, bins=50, color=ORANGE, zorder=3)
    ax.axvline(0.17, color=INK, lw=1.4, ls="--")
    ax.text(0.17, ax.get_ylim()[1] * 0.92, "  median 0.17 dex", fontsize=8)
    ax.axvline(0.334, color=VERM, lw=1.6)
    ax.text(0.334, ax.get_ylim()[1] * 0.75, "  target spread\n  0.33 dex", fontsize=8, color=VERM)
    ax.set_xlabel("per-source σ of log X-ray flux (dex)"); ax.set_ylabel("sources")
    ax.set_title("(b) error ≈ 2/3 of the model residual", fontsize=9.5)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    _save(fig, "fig_errors.png")


def fig_targets():
    fig, ax = plt.subplots(figsize=(9.6, 2.5)); ax.axis("off")
    rows = [
        ["target", "source", "error model", "mode", "trainable"],
        ["log_ml_flux_1", "eROSITA ML_FLUX_1", "split-normal (LO/HI), ~0.17 dex", "convolve", "full sample ✓"],
        ["log_lx", "flux + z (Planck18)", "same dex σ as flux", "convolve", "full (z-dominated)"],
        ["logmstar", "DESI photometric mass", "spectype floor 0.2/0.3 dex", "convolve", "clean for GALAXY; 82% QSO"],
        ["hr32_u", "eROSITA band rates (arctanh)", "propagated σ_HR", "convolve", "IG-primary; ~17% S/N>2"],
    ]
    colw = [0.16, 0.24, 0.30, 0.11, 0.19]
    x0 = 0.01
    for r, row in enumerate(rows):
        y = 0.86 - r * 0.20
        x = x0
        for c, cell in enumerate(row):
            head = r == 0
            ax.add_patch(plt.Rectangle((x, y - 0.09), colw[c], 0.18,
                                       fc=("#eef4fb" if head else "white"), ec="#cccccc", lw=0.7,
                                       transform=ax.transAxes))
            ax.text(x + 0.008, y, cell, fontsize=8 if not head else 8.5,
                    fontweight="bold" if head else "normal",
                    color=INK, va="center", ha="left", transform=ax.transAxes)
            x += colw[c]
    ax.set_title("Targets & error treatment (four separate V1 runs, all --error-mode convolve)",
                 fontsize=9.5, loc="left")
    _save(fig, "fig_targets.png")


if __name__ == "__main__":
    fig_architecture()
    fig_results()
    fig_cleanup()
    fig_errors()
    fig_targets()
    print("all figures written to", FIGS)
