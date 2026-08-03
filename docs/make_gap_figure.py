#!/usr/bin/env python
"""Generalization gap per head over training (run v4).

gap = validation NLL - train-probe NLL, both scored under the IDENTICAL protocol
(the probe is a fixed seeded slice of the training split), so the difference is
the generalization gap and not a scale offset. The running training mean is not
comparable and is never used here.

  left   every head in the run, raw gap. FastSpecFit logmstar is the outlier.
  right  only the heads the next run keeps, with the COMMON MODE drawn behind
         them. Loss weights can only ever reallocate the spread around that
         common line: scaling every head's weight together is algebraically a
         learning-rate change, so the shared component is invisible to them.

    python docs/make_gap_figure.py --epoch-history docs/figures/v4_epoch_history.json
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
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
           "#E69F00", "#56B4E9", "#8C5A2B", "#333333"]
PRETTY = {
    "log_ml_flux_1": "log flux", "log_lx": r"log $L_X$",
    "logmstar": r"log $M_*$ (FastSpecFit)", "logmstar_cigale": r"log $M_*$ (CIGALE)",
    "log_flux_p1": "P1", "log_flux_p2": "P2", "log_flux_p3": "P3",
    "log_flux_p4": "P4", "log_sfr": "log SFR", "p2xp3_joint": r"P2$\times$P3 joint",
}
ORDER = ["log_ml_flux_1", "log_lx", "logmstar", "log_sfr",
         "log_flux_p1", "log_flux_p2", "log_flux_p3", "p2xp3_joint"]
KEEP = ["log_ml_flux_1", "log_lx", "log_sfr", "log_flux_p2", "log_flux_p3", "p2xp3_joint"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch-history", type=Path,
                    default=Path("docs/figures/v4_epoch_history.json"))
    ap.add_argument("--figdir", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()

    rows = sorted(json.load(open(args.epoch_history)), key=lambda r: r.get("epoch", 0))
    heads = {k.split("val/nll_")[1] for r in rows for k in r if k.startswith("val/nll_")}
    heads = [h for h in ORDER if h in heads] + sorted(heads - set(ORDER))
    ep = np.array([r.get("epoch", i + 1) for i, r in enumerate(rows)], float)
    gap = {h: np.array([r.get(f"val/nll_{h}", np.nan) for r in rows], float)
              - np.array([r.get(f"probe/nll_{h}", np.nan) for r in rows], float)
           for h in heads}

    _fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.8, 5.4))

    for i, h in enumerate(heads):
        axL.plot(ep, gap[h], color=PALETTE[i % len(PALETTE)], lw=2.0,
                 label=PRETTY.get(h, h))
    axL.axhline(0, color=MUTED, lw=1.0, ls=":")
    axL.set_title("Generalization gap, every head in the run", fontsize=13, loc="left")
    axL.set_xlabel("epoch")
    axL.set_ylabel("validation NLL minus train-probe NLL [nats]")
    axL.legend(frameon=False, fontsize=9.5, ncol=2, loc="upper left")
    axL.annotate("FastSpecFit mass runs away\n(this head is being dropped)",
                 xy=(33, 0.315), xytext=(12.5, 0.325), fontsize=9.5, color=MUTED,
                 va="center", arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.1))

    keep = [h for h in KEEP if h in gap]
    common = np.mean([gap[h] for h in keep], axis=0)
    axR.plot(ep, common, color="#9a9a9a", lw=7, alpha=0.55, solid_capstyle="round",
             label="common mode (shared body)", zorder=1)
    for h in keep:
        axR.plot(ep, gap[h], color=PALETTE[heads.index(h) % len(PALETTE)], lw=2.0,
                 label=PRETTY.get(h, h), zorder=2)
    axR.axhline(0, color=MUTED, lw=1.0, ls=":")
    axR.set_title("Only the heads the next run keeps", fontsize=13, loc="left")
    axR.text(0.985, 0.11, "loss weights move a head only RELATIVE to the grey line",
             transform=axR.transAxes, ha="right", fontsize=9.5, color=MUTED)
    axR.set_xlabel("epoch")
    axR.set_ylabel("validation NLL minus train-probe NLL [nats]")
    axR.legend(frameon=False, fontsize=9.5, loc="upper left")

    for ax in (axL, axR):
        ax.grid(True, color=GRID, lw=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        ax.set_xlim(0, ep[-1] + 1)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    plt.tight_layout()
    out = args.figdir / "fig_gap.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")
    for h in keep:
        print(f"  {h:16s} gap ep2-5 {gap[h][1:5].mean():+.4f} -> ep40 {gap[h][-1]:+.4f}")
    print(f"  common mode      {common[1:5].mean():+.4f} -> {common[-1]:+.4f}")


if __name__ == "__main__":
    main()
