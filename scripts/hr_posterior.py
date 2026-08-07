#!/usr/bin/env python
"""Bayesian hardness-ratio posteriors from eROSITA aperture counts.

Park, Kashyap, Siemiginowska, van Dyk, Zezas, Heinke & Wargelin 2006, ApJ 652,
610 (arXiv:astro-ph/0606247) -- "BEHR". Their motivation is exactly our case:
"the classical hardness ratio method cannot be applied when a source is not
detected in one of the two bands". A source detected in one band and not the
other is not a missing measurement; it is a STRONG statement about hardness,
and it sits at the end of the HR range where the classical estimator dies.

WHY WE DO NOT NEED THEIR CODE. BEHR marginalises over an unknown background
estimated from a small off-source aperture, which needs MCMC or their
quadrature. eROSITA's APE_BKG comes from a smooth background MAP with nearby
sources masked out (see the TCOMM on APE_BKG_Pn), so the background is
effectively KNOWN. With b known the posterior collapses to a closed form:

    S ~ Poisson(lambda + b),  flat prior on lambda >= 0
    p(lambda | S, b) proportional to (lambda + b)^S exp(-lambda)

Expanding the binomial makes that an exact finite mixture of Gammas,

    p(lambda | S, b) = sum_k w_k Gamma(lambda | k+1, 1),
    w_k proportional to C(S, k) b^(S-k) k!        k = 0..S

which samples in closed form, vectorised, for 10^5 sources in seconds. S = 0
gives w = (1,), i.e. Exponential(1) -- the classic no-counts posterior, and the
reason a non-detection is data rather than a hole.

HR is formed on RATES, not raw counts: the per-band exposures differ (P4 median
228 s against P2's 298 s from vignetting), so a counts ratio would fold the
instrument's effective area into the "hardness".

    HR = (r_hard - r_soft) / (r_hard + r_soft),  r = lambda / t

matching eROSITA's own sign convention (hard minus soft).

Emits per source: HR posterior mean, sd, and quantiles; plus optional draws.
NEVER emit a point estimate with a Gaussian sigma: at eRASS depth the median
source has ~4 counts and the 95% interval spans over half the available range,
so the posterior is the measurement.

COLUMN NAMES DO NOT MATCH THE EVAL PATH, DELIBERATELY. This writes
`hr<soft><hard>_{mean,sd,q025,q16,q50,q84,q975}` -- so the P2/P3 pair is
`hr23_*`, soft band first. The eval path (`scripts/hr_from_joint.py`,
`eval_core.hr_summary_rows`) consumes a scalar column literally named `hr32`
plus an `hr32_ok` flag, which is the OLD point-estimate-with-sigma
representation this file exists to replace. Do not wire this output into
`--hr-ref-csv` by renaming a column: the two disagree about digit order AND
about what a hardness ratio IS. Reconciling them is a modelling decision, not
a rename.

Nothing in the repo consumes this file yet. It is the analytical reference that
Run B's implied HR posterior gets validated against.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def rate_draws(counts: np.ndarray, bkg: np.ndarray, exp: np.ndarray,
               n_draw: int, rng: np.random.Generator,
               max_terms: int = 60) -> np.ndarray:
    """Draws from p(rate | counts, bkg) per source. Shape [n_draw, n_src].

    Exact for the Gamma-mixture posterior. `max_terms` caps the binomial
    expansion for very bright sources, where the posterior is Gaussian to any
    precision we care about and the mixture is numerically pointless.
    """
    counts = np.asarray(counts, float)
    bkg = np.clip(np.asarray(bkg, float), 1e-12, None)
    exp = np.asarray(exp, float)
    n = len(counts)
    out = np.empty((n_draw, n), float)

    S = np.rint(np.clip(counts, 0, None)).astype(np.int64)
    bright = S > max_terms
    for s in np.unique(S[~bright]):
        idx = np.flatnonzero((S == s) & ~bright)
        if not len(idx):
            continue
        k = np.arange(s + 1)
        # log w_k = log C(s,k) + (s-k) log b + log k!   (b varies per source)
        from scipy.special import gammaln
        logC = gammaln(s + 1) - gammaln(k + 1) - gammaln(s - k + 1)
        logw = (logC[None, :]
                + (s - k)[None, :] * np.log(bkg[idx])[:, None]
                + gammaln(k + 1)[None, :])
        logw -= logw.max(axis=1, keepdims=True)
        w = np.exp(logw); w /= w.sum(axis=1, keepdims=True)
        # pick a mixture component per draw, then Gamma(k+1, 1)
        cum = np.cumsum(w, axis=1)
        u = rng.random((n_draw, len(idx)))
        comp = (u[..., None] > cum[None, :, :]).sum(axis=-1)
        out[:, idx] = rng.gamma(shape=comp + 1.0, scale=1.0)
    if bright.any():
        idx = np.flatnonzero(bright)
        # Gaussian limit: lambda ~ N(S - b, sqrt(S)), truncated at 0
        m = counts[idx] - bkg[idx]
        s_ = np.sqrt(np.clip(counts[idx], 1.0, None))
        out[:, idx] = np.clip(rng.normal(m[None, :], s_[None, :], (n_draw, len(idx))), 0.0, None)
    return out / np.clip(exp, 1e-9, None)[None, :]


def hr_posterior(cs, bs, es, ch, bh, eh, n_draw=2000, seed=0):
    """(mean, sd, q16, q50, q84, q02.5, q97.5) of HR = (rH - rS)/(rH + rS)."""
    rng = np.random.default_rng(seed)
    rs = rate_draws(cs, bs, es, n_draw, rng)
    rh = rate_draws(ch, bh, eh, n_draw, rng)
    tot = rs + rh
    hr = np.where(tot > 0, (rh - rs) / np.where(tot > 0, tot, 1.0), 0.0)
    q = np.percentile(hr, [2.5, 16, 50, 84, 97.5], axis=0)
    return hr.mean(axis=0), hr.std(axis=0), q


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fits", type=Path, required=True)
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--pairs", nargs="+", default=["1,2", "2,3", "3,4"],
                    help="soft,hard band index pairs")
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import pandas as pd
    from astropy.io import fits

    tg = pd.read_csv(args.targets, usecols=["targetid", "ero_detuid"])
    with fits.open(args.fits, memmap=True) as h:
        d = h[1].data
        order = {u: i for i, u in enumerate(np.asarray(d["DETUID"]))}
        row = np.array([order.get(u, -1) for u in tg.ero_detuid])
        keep = row >= 0
        r = row[keep]
        out = pd.DataFrame({"targetid": tg.targetid.to_numpy()[keep]})
        for pair in args.pairs:
            s, hb = (int(x) for x in pair.split(","))
            get = lambda pre, b: np.asarray(d[f"{pre}_P{b}"], float)[r]
            mean, sd, q = hr_posterior(get("APE_CTS", s), get("APE_BKG", s), get("APE_EXP", s),
                                       get("APE_CTS", hb), get("APE_BKG", hb), get("APE_EXP", hb),
                                       n_draw=args.draws)
            n = f"hr{s}{hb}"
            out[f"{n}_mean"], out[f"{n}_sd"] = mean, sd
            for lbl, v in zip(("q025", "q16", "q50", "q84", "q975"), q):
                out[f"{n}_{lbl}"] = v
            w68 = q[3] - q[1]
            print(f"HR P{s}/P{hb}: mean {mean.mean():+.3f}  median sd {np.median(sd):.3f}  "
                  f"median 68% width {np.median(w68):.3f}  "
                  f"frac width<0.5 {np.mean(w68 < 0.5):.1%}", flush=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(out):,} rows, {out.shape[1]} cols)")


if __name__ == "__main__":
    main()
