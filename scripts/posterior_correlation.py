#!/usr/bin/env python
"""RUN A's actual result: the per-source conditional correlation r(log Lx, log SFR | log M*).

Run A learns p(log Lx, log SFR, log M* | spectrum, z). Conditioning on M* makes
r(Lx, SFR | M*) identical to r(Lx/M*, sSFR) -- the quenching signature. The claim
is not about a mean; it is about the DISTRIBUTION of that correlation ACROSS
sources: which objects come out negative, and what else is true of them.

This script consumes the per-source posterior DRAWS written by
`scripts/posterior_structure.py --save-draws` and computes, post hoc and with no
retraining, TWO estimators and TWO nulls.

THE TWO ESTIMATORS (both are reported; neither is picked for you)
----------------------------------------------------------------
1. PARTIAL CORRELATION
       rho = (r_LS - r_LM r_SM) / sqrt((1 - r_LM^2)(1 - r_SM^2))
   equivalently -P_ij / sqrt(P_ii P_jj) from the precision matrix (both are
   implemented and the test suite asserts they agree). This is exactly "linear
   regression with M* as a control".
   CAVEAT, and it is the whole reason estimator 2 exists: this equals the true
   conditional correlation ONLY under joint Gaussianity. It is a fast
   approximation that must be VALIDATED, not the answer.

2. KERNEL-WEIGHTED CONDITIONAL CORRELATION
   Weight a source's own draws by a Gaussian kernel in log M* and take the
   weighted correlation of log Lx and log SFR at each of several M* values.
   This yields rho AS A FUNCTION OF M*, which is both the honest estimand and
   the diagnostic for whether the Gaussian shortcut held: if rho(M*) is flat and
   equal to the partial correlation, estimator 1 was fine; if rho(M*) swings with
   M*, estimator 1 was reporting an average over a structure it cannot see.
   Its price is variance. Kernel weighting throws away most of the S draws, so
   the effective sample size (reported per source) is well below S and the null
   spread is correspondingly wider. That is not a defect -- it is why the nulls
   below are run through BOTH estimators rather than only the cheap one.

WHERE THEY DISAGREE is reported explicitly, per group and per source
(`rho_kernel_minus_partial` in the CSV).

THE SANITY CHECK
----------------
Within a posterior, r_LM and r_SM are both expected positive, so the partial
correlation should come out MORE NEGATIVE than the raw r_LS. This script checks
it and shouts if it fails. One correction, because the check is stated in the run
plan as if it were a theorem and it is not quite one. Writing c = r_LM r_SM and
d = sqrt((1-r_LM^2)(1-r_SM^2)):

    rho < r_LS   <=>   r_LS (1 - d) < c

which is UNCONDITIONALLY true whenever r_LS <= 0 (that direction *is* a theorem,
and the code hard-asserts it), and true for r_LS > 0 in every symmetric case
(r_LM ~ r_SM gives the bound r_LS < 1). It can fail legitimately only when r_LM
and r_SM are strongly ASYMMETRIC and r_LS is large and positive -- e.g.
r_LM = 0.9, r_SM = 0.1, r_LS = 0.30 gives rho = +0.48 > r_LS. So a handful of
violations concentrated in that corner is algebra, not a bug; a violation with
r_LS <= 0, or |rho| > 1, or a non-PSD posterior correlation matrix, is a bug and
raises. The summary prints the violating population's r_LM/r_SM asymmetry so the
two cases can be told apart at a glance.

THE TWO NULLS -- these GATE any reading of the result (RUN_PLAN §0, tracker #44)
--------------------------------------------------------------------------------
A claim about the SKEW of a per-source correlation distribution is
uninterpretable without a null, because S finite draws produce a whole spread of
sample correlations even when the true conditional correlation is exactly zero.

  PERMUTATION null -- reassign each source's log SFR draws to a DIFFERENT source
    (a derangement, draw order reshuffled). Per-source structure is destroyed
    while N, S and every marginal are preserved. Note what it does NOT preserve:
    it also destroys r_SM, so the 1/sqrt((1-r_LM^2)(1-r_SM^2)) inflation factor
    goes with it, and this null therefore UNDER-states the null width exactly
    where the conditioning is strongest.

  SYNTHETIC-INDEPENDENT null -- per source, draw S samples from a 3-D Gaussian
    that keeps that source's own sds and its own r_LM and r_SM but sets
    r_LS = r_LM * r_SM, which makes the true conditional correlation exactly zero
    at EVERY M* while leaving the conditioning structure intact. This is the
    sharper null, and it is the one that covers the inflation the permutation
    null misses.

Both are reported, and the summary states what fraction of the observed negative
tail each explains. Without that, "some sources are negative" means nothing.

STRATIFICATION AND THE NEGATIVE CONTROL
---------------------------------------
Everything is reported per spectype. The QSO arm is an explicit NEGATIVE CONTROL:
CIGALE's SFR is known-biased for QSOs in the same direction as the physical
signal, so a matching result in QSOs means we measured the pipeline, not the
universe. z and log M* are carried per source so the distribution can be cut on
them later.

    python scripts/posterior_correlation.py --npz poststruct.npz \
        --source-csv ../stanford_deadline/data/dr2/targets_sidecar_dr2_v2.csv \
        --out-csv per_source_corr.csv --out-fig corr_dist.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TINY = 1e-300
# Okabe-Ito, the repo house palette. Validated as a 3-slot categorical set
# (lightness band / chroma / CVD separation / normal-vision floor / contrast all
# pass in light and dark). Series are also separated by fill-vs-stroke and by
# line style, so identity never rests on hue alone.
INK, MUTED, GRID = "#1a1a1a", "#6a6a6a", "#d5d5d5"
OBS, PERM, SYNTH = "#0072B2", "#D55E00", "#009E73"


# --------------------------------------------------------------------------
# estimator 1: partial correlation
# --------------------------------------------------------------------------
def partial_corr_closed_form(c_xy: np.ndarray, c_xk: np.ndarray,
                             c_yk: np.ndarray) -> np.ndarray:
    """rho_xy.k = (r_xy - r_xk r_yk) / sqrt((1-r_xk^2)(1-r_yk^2)), vectorised."""
    den = np.sqrt(np.clip((1.0 - c_xk ** 2) * (1.0 - c_yk ** 2), 0.0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (c_xy - c_xk * c_yk) / np.where(den > 0, den, np.nan)
    return np.clip(out, -1.0, 1.0)


def partial_corr_precision(corr: np.ndarray) -> np.ndarray:
    """rho_01.rest = -P_01 / sqrt(P_00 P_11) from the precision matrix.

    `corr` is [N, D, D] with columns ordered (x, y, control...). Kept as a
    separate implementation so the test suite can assert the two agree: the run
    plan asserts the equivalence, and an equivalence nobody executes is a
    comment.
    """
    corr = np.asarray(corr, dtype=np.float64)
    n, d, _ = corr.shape
    out = np.full(n, np.nan)
    eye = np.eye(d) * 1e-12
    for i in range(n):
        c = corr[i]
        if not np.isfinite(c).all():
            continue
        try:
            p = np.linalg.inv(c + eye)
        except np.linalg.LinAlgError:
            continue
        den = np.sqrt(p[0, 0] * p[1, 1])
        if not np.isfinite(den) or den <= 0:
            continue
        out[i] = np.clip(-p[0, 1] / den, -1.0, 1.0)
    return out


# --------------------------------------------------------------------------
# estimator 2: kernel-weighted conditional correlation, rho as a function of M*
# --------------------------------------------------------------------------
def silverman_bandwidth(x: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Per-source Silverman rule on the control draws. x is [N, S]."""
    s = x.shape[1]
    sd = x.std(axis=1, ddof=1)
    q75, q25 = np.percentile(x, [75, 25], axis=1)
    iqr = (q75 - q25) / 1.349
    spread = np.where((iqr > 0) & (iqr < sd), iqr, sd)
    return scale * 1.06 * spread * s ** (-0.2)


def kernel_conditional_corr(trio: np.ndarray, quantiles=(0.1, 0.3, 0.5, 0.7, 0.9),
                            bw_scale: float = 1.0, min_ess: float = 25.0,
                            chunk: int = 512) -> dict[str, np.ndarray]:
    """Corr(x, y | control = m) from a source's own draws, at several m.

    `trio` is [N, S, 3] with columns ordered (x, y, control). Returns rho, the
    effective sample size behind each estimate, the m grid, and the bandwidth.

    The bandwidth is the bias/variance dial and it is reported rather than
    hidden: too wide leaves residual control-variation inside the window, which
    drags rho back toward the RAW r_xy; too narrow drops the effective sample
    size until rho is noise. `--bandwidth-scale` sweeps it, and rho should be
    stable across a factor of two if it is real.
    """
    trio = np.asarray(trio)
    n, s, _ = trio.shape
    q = np.asarray(quantiles, dtype=np.float64)
    g = len(q)
    rho = np.full((n, g), np.nan)
    ess = np.full((n, g), np.nan)
    grid = np.full((n, g), np.nan)
    bw = np.full(n, np.nan)

    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        x = trio[a:b, :, 0].astype(np.float64)
        y = trio[a:b, :, 1].astype(np.float64)
        k = trio[a:b, :, 2].astype(np.float64)
        ok = np.isfinite(x).all(1) & np.isfinite(y).all(1) & np.isfinite(k).all(1)
        h = silverman_bandwidth(k, bw_scale)
        h = np.where(np.isfinite(h) & (h > 0) & ok, h, np.nan)
        m = np.quantile(k, q, axis=1).T                              # [nb, G]
        u = (k[:, None, :] - m[:, :, None]) / h[:, None, None]
        w = np.exp(-0.5 * u ** 2)
        sw = w.sum(-1)
        e = sw ** 2 / np.maximum((w ** 2).sum(-1), TINY)
        with np.errstate(invalid="ignore", divide="ignore"):
            mx = (w * x[:, None, :]).sum(-1) / sw
            my = (w * y[:, None, :]).sum(-1) / sw
            dx = x[:, None, :] - mx[..., None]
            dy = y[:, None, :] - my[..., None]
            cxy = (w * dx * dy).sum(-1) / sw
            cxx = (w * dx * dx).sum(-1) / sw
            cyy = (w * dy * dy).sum(-1) / sw
            den = np.sqrt(cxx * cyy)
            r = cxy / np.where(den > 0, den, np.nan)
        r = np.where(e >= min_ess, np.clip(r, -1.0, 1.0), np.nan)
        rho[a:b], ess[a:b], grid[a:b], bw[a:b] = r, e, m, h
    return {"rho": rho, "ess": ess, "grid": grid, "bandwidth": bw,
            "quantiles": q}


def trend_slope(grid: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """d rho / d control, least squares across the grid, NaN-tolerant."""
    ok = np.isfinite(grid) & np.isfinite(rho)
    n_ok = ok.sum(1)
    gm = np.where(ok, grid, 0.0)
    rm = np.where(ok, rho, 0.0)
    denom = np.maximum(n_ok, 1)
    dg = np.where(ok, grid - (gm.sum(1) / denom)[:, None], 0.0)
    dr = np.where(ok, rho - (rm.sum(1) / denom)[:, None], 0.0)
    num, den = (dg * dr).sum(1), (dg * dg).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / np.where(den > 0, den, np.nan)
    return np.where(n_ok >= 3, out, np.nan)


# --------------------------------------------------------------------------
# shared machinery
# --------------------------------------------------------------------------
def corr_from_draws(trio: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Per-source correlation matrix of the draws. [N, S, D] -> [N, D, D]."""
    trio = np.asarray(trio)
    n, s, d = trio.shape
    out = np.full((n, d, d), np.nan)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        v = trio[a:b].astype(np.float64)
        v = v - v.mean(axis=1, keepdims=True)
        cov = np.einsum("nsi,nsj->nij", v, v) / (s - 1)
        sd = np.sqrt(np.einsum("nii->ni", cov))
        with np.errstate(invalid="ignore", divide="ignore"):
            c = cov / np.where(sd > 0, sd, np.nan)[:, :, None]
            c = c / np.where(sd > 0, sd, np.nan)[:, None, :]
        out[a:b] = np.clip(c, -1.0, 1.0)
    return out


def estimate(trio: np.ndarray, quantiles, bw_scale: float, min_ess: float,
             chunk: int) -> dict[str, np.ndarray]:
    """Both estimators on one [N, S, 3] block, observed or null alike.

    Running the nulls through the IDENTICAL code path is the point: a null that
    goes down a different branch tests a different estimator.
    """
    c = corr_from_draws(trio)
    r_xy, r_xk, r_yk = c[:, 0, 1], c[:, 0, 2], c[:, 1, 2]
    kern = kernel_conditional_corr(trio, quantiles, bw_scale, min_ess, chunk)
    mid = int(np.argmin(np.abs(kern["quantiles"] - 0.5)))
    return {
        "corr": c, "r_xy": r_xy, "r_xk": r_xk, "r_yk": r_yk,
        "rho_partial": partial_corr_closed_form(r_xy, r_xk, r_yk),
        "rho_kernel": kern["rho"][:, mid],
        "rho_kernel_grid": kern["rho"],
        "kernel_grid": kern["grid"],
        "kernel_ess": kern["ess"][:, mid],
        "kernel_bandwidth": kern["bandwidth"],
        "rho_kernel_lo": kern["rho"][:, 0],
        "rho_kernel_hi": kern["rho"][:, -1],
        "rho_kernel_slope": trend_slope(kern["grid"], kern["rho"]),
    }


# --------------------------------------------------------------------------
# nulls
# --------------------------------------------------------------------------
def _derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """A permutation with no fixed point (n >= 2), so no source keeps its own y."""
    if n < 2:
        return np.arange(n)
    p = rng.permutation(n)
    fixed = np.flatnonzero(p == np.arange(n))
    for i in fixed:
        j = int(rng.integers(0, n))
        while j == i or p[j] == i:
            j = int(rng.integers(0, n))
        p[i], p[j] = p[j], p[i]
    return p


def permutation_null(trio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Give every source somebody ELSE's log-SFR draws.

    N, S and all three marginals survive; the per-source joint structure does
    not, so the true conditional correlation is exactly zero by construction.
    Draw order is reshuffled independently as well, so nothing can survive
    through a shared sampling order.
    """
    n, s, _ = trio.shape
    out = np.array(trio, copy=True)
    src = _derangement(n, rng)
    order = np.argsort(rng.random((n, s)), axis=1)
    out[:, :, 1] = np.take_along_axis(trio[src, :, 1], order, axis=1)
    return out


def synthetic_independent_null(corr: np.ndarray, sd: np.ndarray, mean: np.ndarray,
                               n_draws: int, rng: np.random.Generator,
                               clip: float = 0.99) -> np.ndarray:
    """Per-source Gaussian with the SAME r_xk and r_yk but zero true conditional corr.

    Setting r_xy = r_xk * r_yk makes Cov(x, y | k) vanish identically, so the
    true conditional correlation is exactly zero at every value of the control,
    for both estimators. The resulting matrix is always PSD: its determinant is
    (1 - r_xk^2)(1 - r_yk^2) >= 0.
    """
    a = np.clip(np.nan_to_num(corr[:, 0, 2]), -clip, clip)
    b = np.clip(np.nan_to_num(corr[:, 1, 2]), -clip, clip)
    n = len(a)
    r = np.empty((n, 3, 3))
    r[:, 0, 0] = r[:, 1, 1] = r[:, 2, 2] = 1.0
    r[:, 0, 1] = r[:, 1, 0] = a * b
    r[:, 0, 2] = r[:, 2, 0] = a
    r[:, 1, 2] = r[:, 2, 1] = b
    r += np.eye(3) * 1e-9
    chol = np.linalg.cholesky(r)
    z = rng.standard_normal((n, n_draws, 3))
    v = np.einsum("nsj,nij->nsi", z, chol)
    s = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
    return (v * s[:, None, :] + np.nan_to_num(mean)[:, None, :]).astype(np.float32)


# --------------------------------------------------------------------------
# reporting helpers
# --------------------------------------------------------------------------
def describe(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "median": np.nan,
                "p05": np.nan, "p95": np.nan, "frac_neg": np.nan, "skew": np.nan}
    c = x - x.mean()
    sd = x.std(ddof=1) if x.size > 1 else np.nan
    return {"n": int(x.size), "mean": float(x.mean()), "sd": float(sd),
            "median": float(np.median(x)),
            "p05": float(np.percentile(x, 5)), "p95": float(np.percentile(x, 95)),
            "frac_neg": float(np.mean(x < 0)),
            "skew": float((c ** 3).mean() / sd ** 3) if sd and sd > 0 else np.nan}


def _row(label: str, d: dict[str, float]) -> str:
    return (f"{label:>26s} {d['n']:>7d} {d['mean']:>+9.4f} {d['sd']:>8.4f} "
            f"{d['median']:>+9.4f} {d['p05']:>+9.4f} {d['p95']:>+9.4f} "
            f"{d['frac_neg']:>8.1%} {d['skew']:>+8.3f}")


HEADER = (f"{'':>26s} {'n':>7s} {'mean':>9s} {'sd':>8s} {'median':>9s} "
          f"{'p05':>9s} {'p95':>9s} {'frac<0':>8s} {'skew':>8s}")


def tail_table(obs: np.ndarray, perm: np.ndarray, synth: np.ndarray,
               thresholds=(0.0, -0.1, -0.2, -0.3, -0.5)) -> list[dict]:
    """How much of the observed negative tail each null already accounts for."""
    o = obs[np.isfinite(obs)]
    p = perm[np.isfinite(perm)]
    s = synth[np.isfinite(synth)]
    rows = []
    for t in thresholds:
        fo = float(np.mean(o < t)) if o.size else np.nan
        fp = float(np.mean(p < t)) if p.size else np.nan
        fs = float(np.mean(s < t)) if s.size else np.nan
        rows.append({
            "threshold": t, "obs": fo, "perm": fp, "synth": fs,
            "explained_perm": min(fp / fo, 1.0) if fo > 0 else np.nan,
            "explained_synth": min(fs / fo, 1.0) if fo > 0 else np.nan,
            "excess_perm": fo - fp, "excess_synth": fo - fs})
    return rows


def print_tail_table(rows: list[dict]) -> None:
    print(f"  {'rho <':>8s} {'observed':>10s} {'perm null':>10s} {'synth null':>11s} "
          f"{'expl.perm':>10s} {'expl.synth':>11s} {'excess(synth)':>14s}")
    for r in rows:
        print(f"  {r['threshold']:>8.2f} {r['obs']:>10.2%} {r['perm']:>10.2%} "
              f"{r['synth']:>11.2%} {r['explained_perm']:>10.1%} "
              f"{r['explained_synth']:>11.1%} {r['excess_synth']:>+14.2%}")


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------
def make_figure(df: pd.DataFrame, groups: list[str], out: Path,
                xlab: str, ylab: str, klab: str, grid_vals: np.ndarray,
                grid_rho: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncol = len(groups)
    fig, axes = plt.subplots(3, ncol, figsize=(4.6 * ncol, 11.4), squeeze=False)

    def style(ax):
        ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)

    def hist_panel(ax, obs, perm, synth, title):
        obs = obs[np.isfinite(obs)]
        perm = perm[np.isfinite(perm)]
        synth = synth[np.isfinite(synth)]
        if obs.size == 0:
            ax.set_axis_off()
            return
        lo = float(np.nanmin([obs.min(), perm.min() if perm.size else 0,
                              synth.min() if synth.size else 0]))
        hi = float(np.nanmax([obs.max(), perm.max() if perm.size else 0,
                              synth.max() if synth.size else 0]))
        bins = np.linspace(lo, hi, 46)
        ax.hist(obs, bins=bins, density=True, color=OBS, alpha=0.80,
                edgecolor="white", lw=0.4, label=f"observed  (n={obs.size:,})")
        if perm.size:
            ax.hist(perm, bins=bins, density=True, histtype="step", lw=2.0,
                    color=PERM, label="permutation null")
        if synth.size:
            ax.hist(synth, bins=bins, density=True, histtype="step", lw=2.0,
                    ls="--", color=SYNTH, label="synthetic-independent null")
        ax.axvline(0.0, color=MUTED, lw=1.1, ls=":")
        ax.set_title(title, color=INK, fontsize=11)
        ax.set_ylabel("density", color=MUTED, fontsize=9)
        ax.text(0.02, 0.97, f"median {np.median(obs):+.3f}\n"
                            f"frac < 0  {np.mean(obs < 0):.1%}",
                transform=ax.transAxes, va="top", fontsize=9, color=INK)
        style(ax)

    for c, g in enumerate(groups):
        m = np.ones(len(df), bool) if g == "ALL" else (df["group"].values == g)
        hist_panel(axes[0][c], df["rho_partial"].values[m],
                   df["null_perm_rho_partial"].values[m],
                   df["null_synth_rho_partial"].values[m],
                   f"{g} — partial correlation")
        hist_panel(axes[1][c], df["rho_kernel"].values[m],
                   df["null_perm_rho_kernel"].values[m],
                   df["null_synth_rho_kernel"].values[m],
                   f"{g} — kernel, at median {klab}")
        axes[1][c].set_xlabel(rf"$\rho$({xlab} , {ylab} | {klab})",
                              color=MUTED, fontsize=9)

        # rho as a function of the control: the estimand the partial correlation
        # cannot see. Binned median with a 16-84 band.
        ax = axes[2][c]
        gv, gr = grid_vals[m].ravel(), grid_rho[m].ravel()
        ok = np.isfinite(gv) & np.isfinite(gr)
        cen: list[float] = []
        if ok.sum() >= 30:
            gv, gr = gv[ok], gr[ok]
            # bin count follows the sample, so a small group (a few dozen
            # galaxies) still gets a profile instead of an empty panel
            nb = int(np.clip(gv.size // 40, 3, 12))
            per = max(8, gv.size // (4 * nb))
            edges = np.unique(np.percentile(gv, np.linspace(0, 100, nb + 1)))
            idx = np.clip(np.digitize(gv, edges[1:-1]), 0, len(edges) - 2)
            med, lo, hi = [], [], []
            for b in range(len(edges) - 1):
                s = idx == b
                if s.sum() < per:
                    continue
                cen.append(float(np.median(gv[s])))
                med.append(float(np.median(gr[s])))
                lo.append(float(np.percentile(gr[s], 16)))
                hi.append(float(np.percentile(gr[s], 84)))
            if cen:
                ax.fill_between(cen, lo, hi, color=OBS, alpha=0.18, lw=0,
                                label="16–84 across sources")
                ax.plot(cen, med, color=OBS, lw=2.0, marker="o", ms=5,
                        label="median across sources")
        if not cen:
            ax.text(0.5, 0.5, f"too few sources for a profile\n(n = {int(m.sum())})",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color=MUTED)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.legend(loc="best", fontsize=8, frameon=False, labelcolor=INK)
        ax.axhline(0.0, color=MUTED, lw=1.1, ls=":")
        ax.set_title(f"{g} — kernel " + r"$\rho$" + f" vs {klab}", color=INK,
                     fontsize=11)
        ax.set_xlabel(f"posterior {klab}", color=MUTED, fontsize=9)
        ax.set_ylabel(r"$\rho$ (16–84 band)", color=MUTED, fontsize=9)
        style(ax)

    axes[0][0].legend(loc="upper right", fontsize=8, frameon=False,
                      labelcolor=INK)
    fig.suptitle(f"per-source posterior correlation  {xlab} x {ylab}  |  {klab}",
                 color=INK, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", type=Path, required=True,
                    help="output of posterior_structure.py --save-draws")
    ap.add_argument("--x", default="log_lx", help="first joint dim (accretion)")
    ap.add_argument("--y", default="log_sfr", help="second joint dim (star formation)")
    ap.add_argument("--control", default="logmstar_cigale",
                    help="dim to condition on; conditioning on M* is what turns "
                         "r(Lx, SFR) into r(Lx/M*, sSFR)")
    ap.add_argument("--source-csv", type=Path, default=None,
                    help="sidecar CSV to join by targetid, for z / spectype / labels")
    ap.add_argument("--source-cols", nargs="*",
                    default=["z", "spectype", "cigale_spectype", "logmstar_cigale",
                             "log_sfr", "log_lx", "det_like_0", "nway_p_any"],
                    help="columns to carry from --source-csv where present")
    ap.add_argument("--kernel-quantiles", nargs="+", type=float,
                    default=[0.1, 0.3, 0.5, 0.7, 0.9],
                    help="where in the source's own control-posterior to evaluate rho")
    ap.add_argument("--bandwidth-scale", type=float, default=1.0,
                    help="multiplier on the Silverman bandwidth. Sweep it: a real "
                         "rho is stable across a factor of two")
    ap.add_argument("--min-ess", type=float, default=25.0,
                    help="kernel rho is NaN below this effective sample size")
    ap.add_argument("--null-reps", type=int, default=4,
                    help="null realisations per source. Rep 0 goes in the CSV; all "
                         "reps pool into the null distribution and its percentiles")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--out-fig", type=Path, default=None)
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args(argv)

    d = np.load(args.npz, allow_pickle=True)
    if "draws" not in d.files:
        raise SystemExit(
            f"{args.npz} has no 'draws' array -- it was written before --save-draws "
            "existed, or without it. The per-source correlation matrices alone fix "
            "the estimator to the linear/Gaussian one, so the kernel estimator and "
            "both nulls are impossible from them. Re-run:\n"
            "  python scripts/posterior_structure.py ... --save-draws")
    dims = [str(v) for v in d["dims"]]
    for nm in (args.x, args.y, args.control):
        if nm not in dims:
            raise SystemExit(f"{nm!r} is not a joint dim; have {dims}")
    ix, iy, ik = dims.index(args.x), dims.index(args.y), dims.index(args.control)

    draws = d["draws"]
    trio = np.ascontiguousarray(draws[:, :, [ix, iy, ik]])
    n, s, _ = trio.shape
    tid = d["targetid"].astype(np.int64)
    group = d["group"].astype(str)
    print(f"=== INPUT ===\n{args.npz}")
    print(f"  {n:,} sources x {s} draws;  joint dims {dims}")
    print(f"  estimating  rho( {args.x} , {args.y} | {args.control} )")
    print(f"  groups: " + ", ".join(f"{g}={int((group == g).sum()):,}"
                                    for g in sorted(set(group))))

    obs = estimate(trio, args.kernel_quantiles, args.bandwidth_scale,
                   args.min_ess, args.chunk)

    # ---- hard invariants: these are theorems, so a failure is a bug ----------
    corr = obs["corr"]
    fin = np.isfinite(corr).all(axis=(1, 2))
    if fin.any():
        eig = np.linalg.eigvalsh(corr[fin])
        bad = int((eig.min(axis=1) < -1e-6).sum())
        if bad:
            raise AssertionError(
                f"{bad} posterior correlation matrices are not positive "
                "semi-definite. corr_from_draws is broken; nothing downstream "
                "is meaningful.")
    for nm in ("rho_partial", "rho_kernel"):
        v = obs[nm][np.isfinite(obs[nm])]
        if v.size and np.abs(v).max() > 1.0 + 1e-9:
            raise AssertionError(f"{nm} left [-1, 1]: max |rho| = {np.abs(v).max()}")

    # ---- the run plan's sanity check ---------------------------------------
    r_xy, r_xk, r_yk = obs["r_xy"], obs["r_xk"], obs["r_yk"]
    rho_p = obs["rho_partial"]
    pos = np.isfinite(rho_p) & np.isfinite(r_xy) & (r_xk > 0) & (r_yk > 0)
    viol = pos & (rho_p > r_xy + 1e-9)
    theorem = viol & (r_xy <= 0)                 # impossible; a genuine bug
    print("\n=== SANITY: partial must sit BELOW the raw correlation ===")
    print(f"  sources with r_xk > 0 and r_yk > 0: {int(pos.sum()):,} of {n:,} "
          f"({pos.mean():.1%})")
    print(f"  mean raw r({args.x},{args.y})   = {np.nanmean(r_xy[pos]):+.4f}")
    print(f"  mean partial rho              = {np.nanmean(rho_p[pos]):+.4f}   "
          f"(shift {np.nanmean(rho_p[pos] - r_xy[pos]):+.4f})")
    if int(theorem.sum()):
        raise AssertionError(
            f"{int(theorem.sum())} sources have r_xy <= 0, r_xk > 0, r_yk > 0 and "
            "STILL a partial correlation above the raw one. That is algebraically "
            "impossible -- the estimator is wrong.")
    if int(viol.sum()):
        vf = viol.sum() / max(pos.sum(), 1)
        asym = np.abs(r_xk[viol] - r_yk[viol])
        print(f"\n  !! LOUD WARNING: {int(viol.sum()):,} of {int(pos.sum()):,} "
              f"({vf:.2%}) sources have partial ABOVE raw.")
        print("     rho < r holds iff r_xy(1-d) < r_xk r_yk, with "
              "d = sqrt((1-r_xk^2)(1-r_yk^2)).")
        print("     It is unconditional only for r_xy <= 0 (asserted above) and "
              "for symmetric r_xk ~ r_yk.")
        print(f"     violators: median r_xy = {np.median(r_xy[viol]):+.3f}, "
              f"median |r_xk - r_yk| = {np.median(asym):.3f} "
              f"(non-violators {np.median(np.abs(r_xk[pos & ~viol] - r_yk[pos & ~viol])):.3f})")
        print("     Concentrated at positive r_xy with ASYMMETRIC r_xk/r_yk is "
              "algebra. Anywhere else, treat it as a bug.")
        if vf > 0.05:
            print("     >5% is too many to be the asymmetric corner alone. "
                  "INVESTIGATE BEFORE READING ANY RESULT BELOW.")
    else:
        print("  OK: every eligible source has partial < raw.")

    # ---- nulls --------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    null = {"perm": {"partial": [], "kernel": [], "slope": []},
            "synth": {"partial": [], "kernel": [], "slope": []}}
    csv_null: dict[str, dict[str, np.ndarray]] = {}
    sd_trio = d["sd"][:, [ix, iy, ik]].astype(np.float64)
    mean_trio = d["mean"][:, [ix, iy, ik]].astype(np.float64)
    for rep in range(max(args.null_reps, 1)):
        for kind, block in (("perm", permutation_null(trio, rng)),
                            ("synth", synthetic_independent_null(
                                corr, sd_trio, mean_trio, s, rng))):
            e = estimate(block, args.kernel_quantiles, args.bandwidth_scale,
                         args.min_ess, args.chunk)
            null[kind]["partial"].append(e["rho_partial"])
            null[kind]["kernel"].append(e["rho_kernel"])
            null[kind]["slope"].append(e["rho_kernel_slope"])
            if rep == 0:
                csv_null[kind] = e
    pooled = {k: {m: np.concatenate(v) for m, v in kv.items()}
              for k, kv in null.items()}

    # The synthetic null is Gaussian, so its partial-correlation spread has a
    # closed form: Fisher-z with df = S - 3 gives sd ~ 1/sqrt(S - 4). If the
    # measured spread does not match, the null machinery itself is broken and
    # every "excess over null" below is wrong.
    analytic = 1.0 / np.sqrt(max(s - 4, 1))
    meas = float(np.nanstd(pooled["synth"]["partial"]))
    print("\n=== NULL MACHINERY CHECK (synthetic null vs closed form) ===")
    print(f"  analytic Gaussian null sd for partial corr, S={s}: {analytic:.4f}")
    print(f"  measured synthetic-null sd:                        {meas:.4f}  "
          f"(ratio {meas / analytic:.3f})")
    if not 0.85 <= meas / analytic <= 1.15:
        print("  !! the synthetic null does not reproduce its own closed form. "
              "Treat every null-based statement below as unverified.")

    # ---- per-source table ---------------------------------------------------
    df = pd.DataFrame({
        "targetid": tid, "group": group, "n_draws": s,
        "r_xy_raw": r_xy, "r_x_control": r_xk, "r_y_control": r_yk,
        "rho_partial": rho_p,
        "rho_kernel": obs["rho_kernel"],
        "rho_kernel_lo_mstar": obs["rho_kernel_lo"],
        "rho_kernel_hi_mstar": obs["rho_kernel_hi"],
        "rho_kernel_slope": obs["rho_kernel_slope"],
        "rho_kernel_minus_partial": obs["rho_kernel"] - rho_p,
        "kernel_ess": obs["kernel_ess"],
        "kernel_bandwidth": obs["kernel_bandwidth"],
        "partial_below_raw": np.where(pos, rho_p < r_xy, np.nan),
        "post_mean_x": d["mean"][:, ix], "post_mean_y": d["mean"][:, iy],
        "post_mean_control": d["mean"][:, ik],
        "post_sd_x": d["sd"][:, ix], "post_sd_y": d["sd"][:, iy],
        "post_sd_control": d["sd"][:, ik],
        "truth_x": d["truth"][:, ix], "truth_y": d["truth"][:, iy],
        "truth_control": d["truth"][:, ik],
        "null_perm_rho_partial": csv_null["perm"]["rho_partial"],
        "null_perm_rho_kernel": csv_null["perm"]["rho_kernel"],
        "null_synth_rho_partial": csv_null["synth"]["rho_partial"],
        "null_synth_rho_kernel": csv_null["synth"]["rho_kernel"],
    })
    df.attrs["x"], df.attrs["y"], df.attrs["control"] = args.x, args.y, args.control

    if args.source_csv:
        have = pd.read_csv(args.source_csv, nrows=0).columns
        cols = [c for c in args.source_cols if c in have]
        extra = (pd.read_csv(args.source_csv, usecols=["targetid", *cols])
                 .drop_duplicates("targetid"))
        extra["targetid"] = extra["targetid"].astype(np.int64)
        before = len(df)
        df = df.merge(extra, on="targetid", how="left", suffixes=("", "_cat"))
        assert len(df) == before, "the sidecar join duplicated rows"
        miss = df[cols[0]].isna().mean() if cols else 1.0
        print(f"\n  joined {len(cols)} sidecar columns ({', '.join(cols)}); "
              f"{miss:.1%} of sources unmatched")

    # per-source flags against the POOLED null 5th percentile. Pooled, not
    # per-source: a per-source threshold would need tens of reps each. It is an
    # approximation in the direction of ignoring that sources with strong
    # r_x_control / r_y_control have wider nulls.
    for kind in ("perm", "synth"):
        for est in ("partial", "kernel"):
            thr = float(np.nanpercentile(pooled[kind][est], 5))
            df[f"sig_neg_{kind}_{est}"] = df[f"rho_{est}"] < thr

    groups = ["ALL"] + sorted(set(group) - {"ALL"})

    # ---- summaries ----------------------------------------------------------
    for est, title in (("partial", "ESTIMATOR 1: PARTIAL CORRELATION "
                                   "(Gaussian shortcut)"),
                       ("kernel", "ESTIMATOR 2: KERNEL-WEIGHTED CONDITIONAL "
                                  f"(at median {args.control})")):
        print(f"\n{'=' * 108}\n### {title}\n{'=' * 108}")
        print(HEADER)
        for g in groups:
            m = np.ones(len(df), bool) if g == "ALL" else (df["group"].values == g)
            print(_row(f"{g} observed", describe(df[f"rho_{est}"].values[m])))
            print(_row(f"{g} perm null", describe(df[f"null_perm_rho_{est}"].values[m])))
            print(_row(f"{g} synth null", describe(df[f"null_synth_rho_{est}"].values[m])))
        print("\n  NEGATIVE TAIL vs NULLS (pooled over all reps, ALL sources)")
        print_tail_table(tail_table(df[f"rho_{est}"].values,
                                    pooled["perm"][est], pooled["synth"][est]))
        for kind in ("perm", "synth"):
            f = float(df[f"sig_neg_{kind}_{est}"].mean())
            print(f"  frac below the {kind} null's 5th percentile: {f:.2%}  "
                  f"(5.00% expected by chance; excess {f - 0.05:+.2%})")

    print(f"\n  kernel diagnostics: median ESS {np.nanmedian(df.kernel_ess):.1f} "
          f"of {s} draws; median bandwidth {np.nanmedian(df.kernel_bandwidth):.4f} dex "
          f"({np.nanmedian(df.kernel_bandwidth / df.post_sd_control):.2f} x the "
          f"posterior sd of {args.control})")
    print(f"  NaN kernel rho (ESS below --min-ess): "
          f"{df.rho_kernel.isna().mean():.2%} of sources")
    sl = df.rho_kernel_slope.values
    nsl = pooled["synth"]["slope"]
    print(f"  d rho / d {args.control}: median {np.nanmedian(sl):+.4f} per dex, "
          f"observed sd {np.nanstd(sl):.4f} vs synthetic-null sd "
          f"{np.nanstd(nsl):.4f}  -> "
          + ("rho VARIES with the control beyond the null: the Gaussian shortcut "
             "is averaging over real structure"
             if np.nanstd(sl) > 1.3 * np.nanstd(nsl)
             else "no M*-dependence beyond the null"))

    # ---- where the estimators disagree -------------------------------------
    print(f"\n{'=' * 108}\n### WHERE THE TWO ESTIMATORS DISAGREE\n{'=' * 108}")
    print(f"{'group':>12s} {'n both':>8s} {'corr(p,k)':>10s} {'mean k-p':>10s} "
          f"{'med |k-p|':>10s} {'sign disagree':>14s} {'p:frac<0':>9s} {'k:frac<0':>9s}")
    for g in groups:
        m = np.ones(len(df), bool) if g == "ALL" else (df["group"].values == g)
        p, k = df.rho_partial.values[m], df.rho_kernel.values[m]
        ok = np.isfinite(p) & np.isfinite(k)
        if ok.sum() < 3:
            continue
        cc = float(np.corrcoef(p[ok], k[ok])[0, 1])
        print(f"{g:>12s} {int(ok.sum()):>8d} {cc:>10.3f} "
              f"{np.mean(k[ok] - p[ok]):>+10.4f} "
              f"{np.median(np.abs(k[ok] - p[ok])):>10.4f} "
              f"{np.mean(np.sign(p[ok]) != np.sign(k[ok])):>13.1%} "
              f"{np.mean(p[ok] < 0):>9.1%} {np.mean(k[ok] < 0):>9.1%}")
    print("  A high corr(p,k) with a small offset means the Gaussian shortcut held "
          "and either estimator can\n  headline the result. Sign disagreement well "
          "above the null rate means it did not, and the kernel\n  estimator is the "
          "one to report.")

    # ---- the negative control ----------------------------------------------
    print(f"\n{'=' * 108}\n### NEGATIVE CONTROL: GALAXY (science arm) vs QSO "
          f"(pipeline arm)\n{'=' * 108}")
    gal = df["group"].values == "GALAXY"
    qso = df["group"].values == "QSO"
    if gal.sum() and qso.sum():
        print(f"{'quantity':>34s} {'GALAXY':>12s} {'QSO':>12s} {'difference':>12s}")
        for lab, col in (("median rho (partial)", "rho_partial"),
                         ("median rho (kernel)", "rho_kernel"),
                         ("frac < 0 (partial)", "rho_partial"),
                         ("frac < 0 (kernel)", "rho_kernel"),
                         ("frac sig-neg vs synth (partial)", "sig_neg_synth_partial"),
                         ("frac sig-neg vs synth (kernel)", "sig_neg_synth_kernel")):
            if lab.startswith("median"):
                a = np.nanmedian(df[col].values[gal])
                b = np.nanmedian(df[col].values[qso])
            elif lab.startswith("frac <"):
                a = float(np.nanmean(df[col].values[gal] < 0))
                b = float(np.nanmean(df[col].values[qso] < 0))
            else:
                a = float(np.nanmean(df[col].values[gal]))
                b = float(np.nanmean(df[col].values[qso]))
            print(f"{lab:>34s} {a:>12.4f} {b:>12.4f} {a - b:>+12.4f}")
        print("\n  CIGALE's SFR is biased for QSOs in the SAME direction as the "
              "physical signal.\n  If the QSO arm matches the GALAXY arm, the "
              "measurement is of the pipeline, not the universe.\n  The claim needs "
              "the GALAXY arm to be negative AND the QSO arm to differ.")
    else:
        print(f"  only one spectype present (GALAXY={int(gal.sum())}, "
              f"QSO={int(qso.sum())}); no control available.")

    out_csv = args.out_csv or args.npz.with_name(args.npz.stem + "_percorr.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}  ({len(df):,} rows x {df.shape[1]} cols)")

    if not args.no_fig:
        out_fig = args.out_fig or args.npz.with_name(args.npz.stem + "_percorr.png")
        try:
            make_figure(df, groups, out_fig, args.x, args.y, args.control,
                        obs["kernel_grid"], obs["rho_kernel_grid"])
        except Exception as exc:                                  # noqa: BLE001
            print(f"figure skipped: {type(exc).__name__}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
