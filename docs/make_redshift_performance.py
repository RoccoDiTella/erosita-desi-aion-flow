#!/usr/bin/env python
"""Performance as a function of redshift, in rolling windows.

Two panels, because one of them is misleading on its own:

  R^2 WITHIN the window   how much of the spread AT THIS REDSHIFT we explain
  RMSE (dex)              the absolute error, which no normalisation can flatter

The distinction matters most for log Lx. Its global R^2 of about 0.92 is
largely redshift: Lx is derived from flux times distance squared, so once z is
given, most of the population variance is already determined. Inside a narrow
z window that free ride is gone and the model has to predict the residual
scatter, so the within-window R^2 is far lower and is the honest number.

Windows are rolling in log z with a fixed source count, so every point carries
the same statistical weight, and the band is a bootstrap over sources.

    python docs/make_redshift_performance.py --npz poststruct_allmod.npz \
        --sidecar targets_sidecar_dr2.csv --out docs/figures/fig_perf_vs_z.png
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
PRETTY = {"log_lx": r"log $L_X$", "log_sfr": "log SFR",
          "logmstar_cigale": r"log $M_*$", "log_flux_p3": "P3 flux",
          "log_ssfr": "log sSFR"}
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9"]


def style(ax) -> None:
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def rolling(z, truth, pred, width, n_boot, rng):
    """Rolling window in sorted z with a fixed count; returns z, R2, RMSE, bands."""
    ok = np.isfinite(z) & np.isfinite(truth) & np.isfinite(pred)
    z, truth, pred = z[ok], truth[ok], pred[ok]
    order = np.argsort(z)
    z, truth, pred = z[order], truth[order], pred[order]
    n = len(z)
    if n < width * 2:
        return None
    out = []
    for s in range(0, n - width, max(1, width // 4)):
        sl = slice(s, s + width)
        t, p = truth[sl], pred[sl]
        res = t - p
        denom = np.sum((t - t.mean()) ** 2)
        r2 = 1.0 - np.sum(res ** 2) / denom if denom > 0 else np.nan
        rmse = float(np.sqrt(np.mean(res ** 2)))
        idx = rng.integers(0, width, size=(n_boot, width))
        tb, pb = t[idx], p[idx]
        rb = tb - pb
        db = ((tb - tb.mean(1, keepdims=True)) ** 2).sum(1)
        r2b = np.where(db > 0, 1.0 - (rb ** 2).sum(1) / np.maximum(db, 1e-30), np.nan)
        out.append((float(np.median(z[sl])), r2, rmse,
                    float(np.nanpercentile(r2b, 16)), float(np.nanpercentile(r2b, 84))))
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--sidecar", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--targets", nargs="+",
                    default=["log_lx", "logmstar_cigale", "log_sfr", "log_flux_p3"])
    ap.add_argument("--group", default=None, help="restrict to one spectype class")
    ap.add_argument("--width", type=int, default=400, help="sources per window")
    ap.add_argument("--n-boot", type=int, default=400)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    dims = [str(x) for x in d["dims"]]
    tid = d["targetid"].astype(np.int64)
    keep = np.ones(len(tid), bool)
    if args.group:
        keep = d["group"].astype(str) == args.group
    sd = (pd.read_csv(args.sidecar, usecols=["targetid", "z"])
          .drop_duplicates("targetid").set_index("targetid"))
    z = sd.reindex(tid).z.to_numpy(float)[keep]
    truth, mean = d["truth"][keep], d["mean"][keep]
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(2, 1, figsize=(9.6, 7.4), sharex=True)
    print(f"{'target':>18s} {'n':>6s} {'global R2':>10s}  within-window R2 range")
    for c, name in zip(COLORS, args.targets):
        k = dims.index(name)
        res = rolling(z, truth[:, k].astype(float), mean[:, k].astype(float),
                      args.width, args.n_boot, rng)
        if res is None:
            print(f"{name:>18s}  too few sources for a window")
            continue
        zz, r2, rmse, lo, hi = res.T
        ok = np.isfinite(truth[:, k]) & np.isfinite(z)
        t, p = truth[ok, k].astype(float), mean[ok, k].astype(float)
        g = 1 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2)
        print(f"{name:>18s} {int(ok.sum()):>6,} {g:>10.3f}  "
              f"{np.nanmin(r2):+.2f} to {np.nanmax(r2):+.2f}")
        lab = PRETTY.get(name, name)
        axes[0].plot(zz, r2, color=c, lw=2.0, label=f"{lab}  (global {g:.2f})")
        axes[0].fill_between(zz, lo, hi, color=c, alpha=0.15, lw=0)
        axes[1].plot(zz, rmse, color=c, lw=2.0, label=lab)

    axes[0].axhline(0, color=MUTED, lw=1.0, ls=":")
    axes[0].set_ylabel(r"$R^2$ within the window", color=MUTED)
    axes[0].set_title("Performance across redshift, rolling window of "
                      f"{args.width} sources"
                      + (f", {args.group} only" if args.group else ""),
                      color=INK, fontsize=13)
    axes[0].legend(frameon=False, fontsize=9, ncol=2)
    axes[1].set_ylabel("RMSE (dex)", color=MUTED)
    axes[1].set_xlabel("redshift", color=MUTED)
    axes[1].set_xscale("log")
    axes[1].legend(frameon=False, fontsize=9, ncol=2)
    for a in axes:
        style(a)
    fig.tight_layout()
    fig.savefig(args.out, dpi=190, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
