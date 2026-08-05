#!/usr/bin/env python
"""Performance as a function of redshift, in rolling windows.

Two panels, because one of them is misleading on its own:

  INFORMATION GAIN        nats gained over the prior, so a sharper posterior is
                          rewarded and not only a better point estimate
  RMSE (dex)              the absolute error, which no normalisation can flatter

IG here is the GAUSSIAN form, computed from the posterior mean and width
against a Gaussian prior fitted on the same targets:

    IG = log(sd_prior / sd_post)
         - 0.5 * [ ((y-mu_post)/sd_post)^2 - ((y-mu_prior)/sd_prior)^2 ]

That is exact if both densities are Gaussian and approximate otherwise. The
flow's own log-density would be better, but phase 1 trained ONLY the joint
head, so the marginal flows in this checkpoint were never fitted and their
log-probs are meaningless. Use the refit heads if an exact per-target IG is
needed.

Read IG per window, not globally. log Lx is flux times distance squared, so
conditioning on z already spends most of its population variance; a global
score flatters that head in a way a within-window score does not.

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


def rolling(z, truth, pred, sd, width, n_boot, rng):
    """Rolling window in sorted z with a fixed count; returns z, IG, RMSE, bands."""
    ok = np.isfinite(z) & np.isfinite(truth) & np.isfinite(pred) & np.isfinite(sd) & (sd > 0)
    z, truth, pred, sd = z[ok], truth[ok], pred[ok], sd[ok]
    # prior: a Gaussian on the target's own marginal over these sources
    mu0, sd0 = float(np.mean(truth)), float(np.std(truth))
    ig_all = (np.log(sd0 / sd)
              - 0.5 * (((truth - pred) / sd) ** 2 - ((truth - mu0) / sd0) ** 2))
    order = np.argsort(z)
    z, truth, pred, ig_all = z[order], truth[order], pred[order], ig_all[order]
    n = len(z)
    if n < width * 2:
        return None
    out = []
    for s in range(0, n - width, max(1, width // 4)):
        sl = slice(s, s + width)
        t, p, g = truth[sl], pred[sl], ig_all[sl]
        rmse = float(np.sqrt(np.mean((t - p) ** 2)))
        ig = float(np.mean(g))
        idx = rng.integers(0, width, size=(n_boot, width))
        igb = g[idx].mean(1)
        out.append((float(np.median(z[sl])), ig, rmse,
                    float(np.nanpercentile(igb, 16)), float(np.nanpercentile(igb, 84))))
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
    zsrc = (pd.read_csv(args.sidecar, usecols=["targetid", "z"])
            .drop_duplicates("targetid").set_index("targetid"))
    z = zsrc.reindex(tid).z.to_numpy(float)[keep]
    truth, mean, sd = d["truth"][keep], d["mean"][keep], d["sd"][keep]
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(2, 1, figsize=(9.6, 7.4), sharex=True)
    print(f"{'target':>18s} {'wins':>6s} {'mean IG':>10s}  window IG range")
    for c, name in zip(COLORS, args.targets):
        k = dims.index(name)
        res = rolling(z, truth[:, k].astype(float), mean[:, k].astype(float),
                      sd[:, k].astype(float), args.width, args.n_boot, rng)
        if res is None:
            print(f"{name:>18s}  too few sources for a window")
            continue
        zz, ig, rmse, lo, hi = res.T
        print(f"{name:>18s} {len(zz):>6,} {np.mean(ig):>10.3f}  "
              f"{np.nanmin(ig):+.2f} to {np.nanmax(ig):+.2f}")
        lab = PRETTY.get(name, name)
        axes[0].plot(zz, ig, color=c, lw=2.0, label=f"{lab}  (mean {np.mean(ig):.2f})")
        axes[0].fill_between(zz, lo, hi, color=c, alpha=0.15, lw=0)
        axes[1].plot(zz, rmse, color=c, lw=2.0, label=lab)

    axes[0].axhline(0, color=MUTED, lw=1.0, ls=":")
    axes[0].set_ylabel("information gain (nats)", color=MUTED)
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
