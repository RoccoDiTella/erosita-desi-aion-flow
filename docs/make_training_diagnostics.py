#!/usr/bin/env python
"""Training-diagnostic figure for a train-multi run.

Three panels per head is too many; this is the compact view that answers the
questions we actually ask of a run:
  (1) does it still have room to run?      train mean / probe / val per head
  (2) is it overfitting, and how fast?     val - probe gap
  (3) what makes the loss look spiky?      per-bucket decomposition
  (4) which head drives the shared trunk?  influence share
  (5) which group needs its own LR?        relative movement per group

    python docs/make_training_diagnostics.py --epoch-history v2_epoch_history.json \
        --diag-history v2_diag_history.json [--out docs/figures/fig_v3b_training.png]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, ACCENT, MUTED = "#1a1a1a", "#0072B2", "#6a6a6a"
WARM, GREEN, PURPLE = "#D55E00", "#009E73", "#CC79A7"

HEADS = [
    ("log_ml_flux_1", "log flux"),
    ("log_lx", "log L$_X$"),
    ("logmstar", "log M$_*$"),
    ("log_flux_p1", "P1"),
    ("log_flux_p2", "P2"),
    ("log_flux_p3", "P3"),
    ("p2xp3_joint", "P2$\\times$P3"),
]
BUCKETS = [("tiny", "z / W"), ("spectra", "spectra"), ("image", "image"), ("heavy", "all")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epoch-history", type=Path, required=True)
    ap.add_argument("--diag-history", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("docs/figures/fig_v3b_training.png"))
    args = ap.parse_args()

    ep = json.load(open(args.epoch_history))
    dg = json.load(open(args.diag_history))
    e = np.array([r["epoch"] for r in ep])

    fig = plt.figure(figsize=(14.5, 8.2))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.15, 1.0, 1.0], hspace=0.52, wspace=0.30)

    # --- row 1: convergence, three comparable curves for the leading heads
    for k, (key, label) in enumerate(HEADS[:4]):
        ax = fig.add_subplot(gs[0, k])
        tm = [r.get(f"trainmean/nll_{key}") for r in ep]
        pr = [r.get(f"probe/nll_{key}") for r in ep]
        vl = [r.get(f"val/nll_{key}") for r in ep]
        ax.plot(e, tm, color=WARM, lw=1.6, label="train mean (injected)")
        ax.plot(e, pr, color=GREEN, lw=1.8, label="train probe (val protocol)")
        ax.plot(e, vl, color=ACCENT, lw=2.0, label="validation")
        ax.set_title(label, fontsize=11, color=INK)
        ax.tick_params(labelsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if k == 0:
            ax.set_ylabel("NLL (nats)", fontsize=10)
    h, l = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 1.005))

    # --- row 2 left: generalization gap
    ax = fig.add_subplot(gs[1, 0:2])
    gap = [r.get("gap/pair_mean") for r in ep]
    ax.plot(e, gap, color=PURPLE, lw=2.2)
    ax.axhline(0, color=INK, lw=0.8)
    ax.fill_between(e, 0, gap, color=PURPLE, alpha=0.15)
    ax.set_title("Generalization gap (validation $-$ train probe, same protocol)",
                 fontsize=11, color=INK)
    ax.set_xlabel("epoch", fontsize=10); ax.set_ylabel("nats", fontsize=10)
    ax.tick_params(labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if gap[-1] is not None:
        ax.annotate(f"+{gap[-1]:.3f} at epoch {int(e[-1])}", xy=(e[-1], gap[-1]),
                    xytext=(-8, 8), textcoords="offset points", ha="right",
                    fontsize=10, color=PURPLE)

    # --- row 2 right: per-bucket decomposition (the spikiness, made explicit)
    ax = fig.add_subplot(gs[1, 2:4])
    key = "log_lx"
    for (bk, blabel), c in zip(BUCKETS, [MUTED, GREEN, PURPLE, WARM]):
        y = [r.get(f"bucket/{bk}/nll_{key}") for r in ep]
        if any(v is not None for v in y):
            ax.plot(e, y, lw=1.7, color=c, label=blabel)
    ax.plot(e, [r.get(f"val/nll_{key}") for r in ep], lw=2.2, color=ACCENT, ls="--",
            label="validation")
    ax.set_title("Where the spread comes from: log L$_X$ train loss by input bucket",
                 fontsize=11, color=INK)
    ax.set_xlabel("epoch", fontsize=10); ax.set_ylabel("NLL (nats)", fontsize=10)
    ax.legend(fontsize=9, frameon=False, ncol=5, loc="upper right")
    ax.tick_params(labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # --- row 3 left: influence share on the shared representation
    ax = fig.add_subplot(gs[2, 0:2])
    ds = np.array([r["_step"] for r in dg])
    bottom = np.zeros(len(ds))
    cmap = plt.get_cmap("tab10")
    for i, (key, label) in enumerate(HEADS):
        y = np.array([r.get(f"influence_share/{key}", np.nan) for r in dg], float)
        y = np.nan_to_num(y)
        ax.fill_between(ds, bottom, bottom + y, label=label, color=cmap(i), alpha=0.85, lw=0)
        bottom = bottom + y
    ax.set_ylim(0, 1)
    ax.set_title("Share of gradient pull on the SHARED representation", fontsize=11, color=INK)
    ax.set_xlabel("optimizer step", fontsize=10); ax.set_ylabel("share", fontsize=10)
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="lower center")
    ax.tick_params(labelsize=9)

    # --- row 3 right: relative movement per parameter group
    ax = fig.add_subplot(gs[2, 2:4])
    for g, c in zip(["adapters_low", "adapters_mid", "adapters_high", "cls_tokens", "shared_mlp"],
                    [WARM, PURPLE, GREEN, ACCENT, MUTED]):
        y = [r.get(f"move/{g}") for r in dg]
        if any(v is not None for v in y):
            ax.semilogy(ds, y, lw=1.8, color=c, label=g.replace("_", " "))
    ax.set_title("Relative movement per group (|$\\Delta w$| / |$w$| between checks)",
                 fontsize=11, color=INK)
    ax.set_xlabel("optimizer step", fontsize=10)
    ax.legend(fontsize=9, frameon=False)
    ax.tick_params(labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=165, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
