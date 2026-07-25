#!/usr/bin/env python
"""V3b multi-target figures: per-target results and per-head training diagnostics.

    python docs/make_v3b_figures.py --history v3b_history.json \
        --train-history v3b_train_history.json [--metrics multi_test_metrics.csv]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGS = Path(__file__).resolve().parent / "figures"
INK, ACCENT, MUTED = "#1a1a1a", "#0072B2", "#6a6a6a"
WARM, GREEN = "#D55E00", "#009E73"

HEADS = [
    ("log_ml_flux_1", "log flux"),
    ("log_lx", "log L$_X$"),
    ("logmstar", "log M$_*$"),
    ("log_flux_p1", "P1 0.2-0.6"),
    ("log_flux_p2", "P2 0.6-2.3"),
    ("log_flux_p3", "P3 2.3-5.0"),
    ("log_flux_p4", "P4 5.0-8.0"),
    ("p2xp3_joint", "P2$\\times$P3 joint"),
]


def load_hist(path: Path, key_filter: str) -> list[dict]:
    d = json.load(open(path))["data"]["project"]["run"]["sampledHistory"][0]
    return [r for r in d if r.get(key_filter) is not None]


def results_figure(metrics_csv: Path | None, hr: dict | None, out: Path) -> None:
    """Per-target R2 and info gain for the multi-target model (HR implied)."""
    rows = []
    if metrics_csv and metrics_csv.exists():
        t = pd.read_csv(metrics_csv)
        t = t[t["input_group"] == "spectra+z+wise+image"]
        for key, label in HEADS[:-1]:
            r = t[t["head"] == key]
            if len(r):
                rows.append({"label": label, "r2": float(r.iloc[0].r2),
                             "ig": float(r.iloc[0].info_gain_nats)})
    if hr:
        rows.append({"label": "HR32 (implied)", "r2": hr.get("r2", np.nan),
                     "ig": hr.get("ig", np.nan)})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.6))
    labels = [r["label"] for r in rows]
    ypos = np.arange(len(rows))[::-1]

    r2 = np.array([r["r2"] for r in rows])
    cols = [GREEN if v > 0.3 else (ACCENT if v > 0.05 else MUTED) for v in r2]
    ax1.barh(ypos, np.clip(r2, 0, None), color=cols, height=0.62)
    for y, v in zip(ypos, r2):
        ax1.text(max(v, 0) + 0.015, y, f"{v:+.3f}", va="center", fontsize=11, color=INK)
    ax1.axvline(0.549, color=WARM, lw=1.6, ls="--")
    ax1.text(0.562, 0.55, "paper flux 0.549", color=WARM, fontsize=9.5,
             ha="left", va="center")
    ax1.set_yticks(ypos, labels, fontsize=11)
    ax1.set_xlim(0, 1.05)
    ax1.set_xlabel("test R$^2$ (all inputs)", fontsize=11)
    ax1.set_title("Point accuracy", fontsize=12, color=INK)

    ig = np.array([r["ig"] for r in rows])
    ax2.barh(ypos, np.clip(ig, 0, None), color=ACCENT, height=0.62)
    for y, v in zip(ypos, ig):
        finite = np.isfinite(v)
        txt = f"{np.exp(v):.2f}$\\times$" if finite else "pending"
        col = INK if finite else MUTED
        ax2.text((max(v, 0) if finite else 0.0) + 0.02, y, txt, va="center",
                 fontsize=11, color=col)
    ax2.set_yticks(ypos, [""] * len(rows))
    ax2.set_xlabel("information gain over the prior (nats)", fontsize=11)
    ax2.set_title("Posterior sharpness", fontsize=12, color=INK)
    for ax in (ax1, ax2):
        ax.set_ylim(-0.75, len(rows) - 0.25)         # shared: rows stay aligned
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def curves_figure(val_rows: list[dict], train_rows: list[dict], best_epoch: int, out: Path) -> None:
    """Per-head training diagnostics: train (injected) and val (plain) NLL."""
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 6.2), sharex=True)
    steps_per_epoch = max(r["_step"] for r in train_rows) / max(r["epoch"] for r in val_rows)
    for ax, (key, label) in zip(axes.ravel(), HEADS):
        tx = [r["_step"] / steps_per_epoch for r in train_rows if r.get(f"train/nll_{key}") is not None]
        ty = [r[f"train/nll_{key}"] for r in train_rows if r.get(f"train/nll_{key}") is not None]
        if tx:
            ax.plot(tx, ty, color=MUTED, lw=0.7, alpha=0.55, label="train (injected)")
            w = 9
            if len(ty) > w:
                sm = np.convolve(ty, np.ones(w) / w, mode="valid")
                ax.plot(tx[w - 1:], sm, color=WARM, lw=1.4, label="train (smoothed)")
        vx = [r["epoch"] for r in val_rows]
        vy = [r.get(f"val/nll_{key}") for r in val_rows]
        ax.plot(vx, vy, color=ACCENT, lw=2.0, marker="o", ms=2.6, label="val (plain LL)")
        ax.axvline(best_epoch, color=GREEN, lw=1.2, ls=":")
        ax.set_title(label, fontsize=11, color=INK)
        ax.tick_params(labelsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("epoch", fontsize=10)
    axes[0][0].set_ylabel("NLL (nats)", fontsize=10)
    axes[1][0].set_ylabel("NLL (nats)", fontsize=10)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l + ["best epoch"], loc="upper center", ncol=4, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--history", type=Path, required=True)
    ap.add_argument("--train-history", type=Path, required=True)
    ap.add_argument("--metrics", type=Path, default=None)
    ap.add_argument("--best-epoch", type=int, default=19)
    ap.add_argument("--hr-r2", type=float, default=float("nan"))
    ap.add_argument("--hr-ig", type=float, default=float("nan"))
    args = ap.parse_args()

    FIGS.mkdir(exist_ok=True)
    val_rows = load_hist(args.history, "val/nll_log_ml_flux_1")
    train_rows = load_hist(args.train_history, "train/nll_log_ml_flux_1")
    results_figure(args.metrics, {"r2": args.hr_r2, "ig": args.hr_ig},
                   FIGS / "fig_v3b_results.png")
    curves_figure(val_rows, train_rows, args.best_epoch, FIGS / "fig_v3b_curves.png")
    print(f"wrote {FIGS/'fig_v3b_results.png'} and {FIGS/'fig_v3b_curves.png'}")


if __name__ == "__main__":
    main()
