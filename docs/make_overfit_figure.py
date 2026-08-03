#!/usr/bin/env python
"""Why per-head early stopping: heads peak at different epochs on ONE shared body.

Two panels:
  left   each head's validation NLL measured from its OWN minimum, so the panels
         are comparable in nats. The per-head minima are marked; the vertical
         line is the single global checkpoint the run actually keeps.
  right  what that single checkpoint costs each head, in nats.

The per-head curves are plotted as (val - own best) precisely because the raw
per-panel view is misleading: independent y-scales make a head whose NLL falls
1.6 nats look smooth and a head that barely moves look violently noisy, when the
epoch-to-epoch scatter is the same size.

    python docs/make_overfit_figure.py --epoch-history docs/figures/v4_epoch_history.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID = "#1a1a1a", "#6a6a6a", "#d5d5d5"
# Okabe-Ito, CVD safe
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
           "#E69F00", "#56B4E9", "#F0E442", "#8C5A2B"]
PRETTY = {
    "log_ml_flux_1": "log flux", "log_lx": r"log $L_X$", "logmstar": r"log $M_*$",
    "log_flux_p1": "P1", "log_flux_p2": "P2", "log_flux_p3": "P3",
    "log_flux_p4": "P4", "log_sfr": "log SFR", "p2xp3_joint": r"P2$\times$P3 joint",
}
ORDER = ["log_ml_flux_1", "log_lx", "logmstar", "log_sfr",
         "log_flux_p1", "log_flux_p2", "log_flux_p3", "log_flux_p4", "p2xp3_joint"]


def load(path: Path):
    rows = sorted(json.load(open(path)), key=lambda r: r.get("epoch", 0))
    heads = {k.split("val/nll_")[1] for r in rows for k in r if k.startswith("val/nll_")}
    heads = [h for h in ORDER if h in heads] + sorted(heads - set(ORDER))
    ep = np.array([r.get("epoch", i + 1) for i, r in enumerate(rows)], float)
    val = {h: np.array([r.get(f"val/nll_{h}", np.nan) for r in rows], float) for h in heads}
    return ep, heads, val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch-history", type=Path,
                    default=Path("docs/figures/v4_epoch_history.json"))
    ap.add_argument("--figdir", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()

    ep, heads, val = load(args.epoch_history)
    # the run keeps ONE checkpoint: the epoch minimising the summed val NLL
    stack = np.array([val[h] for h in heads])
    g = int(np.nanargmin(stack.sum(axis=0)))

    best_ep, costs = {}, {}
    for h in heads:
        v = val[h]
        b = int(np.nanargmin(v))
        best_ep[h], costs[h] = ep[b], v[g] - v[b]

    fig = plt.figure(figsize=(14.4, 4.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.62, 1.0], wspace=0.60)
    axL, axR = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # ---- left: when each head peaks, and how long it trains past that ----
    by_peak = sorted(heads, key=lambda h: best_ep[h])
    last = ep[-1]
    for row, h in enumerate(by_peak):
        c = PALETTE[heads.index(h) % len(PALETTE)]
        axL.hlines(row, 1, last, color=GRID, lw=6, zorder=1,
                   capstyle="round")
        axL.hlines(row, best_ep[h], last, color=c, lw=6, alpha=0.85, zorder=2,
                   capstyle="round")
        axL.plot(best_ep[h], row, "o", color=c, ms=11, mec="white", mew=2, zorder=4)
        n = int(last - best_ep[h])
        axL.text(last + 1.1, row, f"{n} epoch{'s' if n != 1 else ''} past peak",
                 va="center", fontsize=9.5, color=MUTED)

    axL.axvline(ep[g], color=INK, ls="--", lw=1.6, zorder=3)
    axL.text(ep[g] - 0.8, len(by_peak) - 0.35,
             f"the single checkpoint we keep (epoch {int(ep[g])})",
             fontsize=10.5, color=INK, ha="right", va="center")
    axL.set_yticks(range(len(by_peak)))
    axL.set_yticklabels([PRETTY.get(h, h) for h in by_peak], fontsize=11)
    axL.set_xlabel("epoch")
    axL.set_title("Each head peaks at its own epoch, then trains past it",
                  fontsize=13, loc="left")
    axL.set_xlim(0, last + 14)
    axL.set_ylim(-0.7, len(by_peak) - 0.15)
    axL.invert_yaxis()
    axL.grid(True, axis="x", color=GRID, lw=0.6, alpha=0.8)
    axL.set_axisbelow(True)

    ordered = sorted(heads, key=lambda h: costs[h])
    ypos = np.arange(len(ordered))
    axR.barh(ypos, [costs[h] for h in ordered],
             color=[PALETTE[heads.index(h) % len(PALETTE)] for h in ordered],
             height=0.66, zorder=2)
    for y, h in zip(ypos, ordered):
        axR.text(costs[h] + 0.0009, y, f"{costs[h]:.3f}", va="center",
                 fontsize=10, color=INK)
    axR.set_yticks(ypos)
    axR.set_yticklabels([f"{PRETTY.get(h, h)}  (best ep {int(best_ep[h])})" for h in ordered],
                        fontsize=10)
    axR.set_xlabel("nats lost at the shared checkpoint")
    axR.set_title(f"Cost of one stopping point: {sum(costs.values()):.3f} nats",
                  fontsize=13, loc="left")
    axR.set_xlim(0, max(costs.values()) * 1.28)
    axR.grid(True, axis="x", color=GRID, lw=0.6, alpha=0.8)
    axR.set_axisbelow(True)
    for s in ("top", "right"):
        axL.spines[s].set_visible(False)
        axR.spines[s].set_visible(False)

    plt.tight_layout()
    out = args.figdir / "fig_overfit.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"global checkpoint epoch {int(ep[g])}; "
          f"per-head best spread {int(min(best_ep.values()))}-{int(max(best_ep.values()))}; "
          f"total recoverable {sum(costs.values()):.4f} nats")


if __name__ == "__main__":
    main()
