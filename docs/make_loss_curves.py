#!/usr/bin/env python
"""Train vs validation loss for the multi-target run.

Two figures:
  fig_loss_total.png      the whole model: train-probe vs validation, summed
  fig_loss_by_target.png  the same pair, one panel per head

The train curve plotted here is the TRAIN PROBE, not the running training mean.
The probe is a fixed slice of the training split scored with the exact
validation protocol, so probe and val measure the same estimand and the gap
between them is the generalization gap. The running train mean is not
comparable -- it is computed under noise injection and across a moving model --
so it is deliberately not drawn.

    python docs/make_loss_curves.py --epoch-history v3_epoch_history.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID = "#1a1a1a", "#6a6a6a", "#d5d5d5"
TRAIN_C, VAL_C = "#E69F00", "#0072B2"
PRETTY = {
    "log_ml_flux_1": "log flux", "log_lx": r"log $L_X$", "logmstar": r"log $M_*$",
    "log_flux_p1": "P1", "log_flux_p2": "P2", "log_flux_p3": "P3", "log_flux_p4": "P4",
    "log_sfr": "log SFR", "p2xp3_joint": r"P2$\times$P3 joint",
}
ORDER = ["log_ml_flux_1", "log_lx", "logmstar", "log_sfr",
         "log_flux_p1", "log_flux_p2", "log_flux_p3", "log_flux_p4", "p2xp3_joint"]


def discover_heads(rows: list[dict]) -> list[str]:
    """Heads actually present as a probe/val PAIR -- a run may drop some."""
    keys = set().union(*(set(r.keys()) for r in rows)) if rows else set()
    probe = {re.sub(r"^probe/nll_", "", k) for k in keys if k.startswith("probe/nll_")}
    val = {re.sub(r"^val/nll_", "", k) for k in keys if k.startswith("val/nll_")}
    both = probe & val
    ranked = [h for h in ORDER if h in both] + sorted(both - set(ORDER))
    return ranked


def series(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for r in rows:
        v = r.get(key)
        if v is not None and np.isfinite(v):
            xs.append(r.get("epoch", len(xs)))
            ys.append(float(v))
    return np.asarray(xs, float), np.asarray(ys, float)


def style(ax) -> None:
    ax.grid(color=GRID, lw=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def total_figure(rows: list[dict], heads: list[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    for label, prefix, colour in (("train (probe)", "probe/nll_", TRAIN_C),
                                  ("validation", "val/nll_", VAL_C)):
        tot: dict[float, float] = {}
        for h in heads:
            x, y = series(rows, prefix + h)
            for xi, yi in zip(x, y):
                tot[xi] = tot.get(xi, 0.0) + yi
        xs = np.array(sorted(tot))
        ys = np.array([tot[x] for x in xs])
        ax.plot(xs, ys, color=colour, lw=2.2, marker="o", ms=3.0, label=label)
    # shade the generalization gap where both curves exist
    tp, tv = {}, {}
    for h in heads:
        for d, pre in ((tp, "probe/nll_"), (tv, "val/nll_")):
            x, y = series(rows, pre + h)
            for xi, yi in zip(x, y):
                d[xi] = d.get(xi, 0.0) + yi
    common = np.array(sorted(set(tp) & set(tv)))
    if len(common):
        ax.fill_between(common, [tp[c] for c in common], [tv[c] for c in common],
                        color=MUTED, alpha=0.11, lw=0, label="generalization gap")
    ax.set_xlabel("epoch", fontsize=10.5)
    ax.set_ylabel("summed NLL over heads [nats]", fontsize=10.5)
    ax.set_title("Train vs validation, all heads summed", fontsize=13, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9.5)
    style(ax)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def by_target_figure(rows: list[dict], heads: list[str], out: Path) -> None:
    n = len(heads)
    ncol = 3 if n <= 9 else 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.15 * ncol, 2.85 * nrow), squeeze=False)
    flat = axes.ravel()
    for k, h in enumerate(heads):
        ax = flat[k]
        for label, prefix, colour in (("train (probe)", "probe/nll_", TRAIN_C),
                                      ("validation", "val/nll_", VAL_C)):
            x, y = series(rows, prefix + h)
            if len(x):
                ax.plot(x, y, color=colour, lw=1.9, label=label)
        ax.set_title(PRETTY.get(h, h), fontsize=11, color=INK, loc="left")
        style(ax)
        if k % ncol == 0:
            ax.set_ylabel("NLL [nats]", fontsize=9.5)
        if k // ncol == nrow - 1:
            ax.set_xlabel("epoch", fontsize=9.5)
    for k in range(n, len(flat)):
        flat[k].axis("off")
    handles, labels = flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, fontsize=10,
                   loc="lower right", bbox_to_anchor=(0.99, 0.02))
    fig.suptitle("Train vs validation, per head (independent y-scales)",
                 fontsize=13, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epoch-history", type=Path, required=True,
                    help="JSON list of per-epoch dicts with probe/nll_* and val/nll_* keys")
    ap.add_argument("--figdir", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()

    rows = json.load(open(args.epoch_history))
    if isinstance(rows, dict):                     # tolerate {"history": [...]}
        rows = rows.get("history", rows.get("rows", []))
    heads = discover_heads(rows)
    if not heads:
        raise SystemExit("no probe/val head pairs found in the history file")
    print(f"[loss] {len(rows)} epochs, heads: {heads}")
    total_figure(rows, heads, args.figdir / "fig_loss_total.png")
    by_target_figure(rows, heads, args.figdir / "fig_loss_by_target.png")


if __name__ == "__main__":
    main()
