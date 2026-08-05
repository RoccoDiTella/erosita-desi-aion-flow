#!/usr/bin/env python
"""Is the joint posterior's correlation structure different from the data's?

FOUR correlation matrices, which answer four different questions:

  1. EMPIRICAL      corr of the true targets across sources.
                    "How do these quantities co-vary in the population?"
  2. PREDICTED      corr of the posterior MEANS across sources.
                    "How does the model think they co-vary in the population?"
                    Should approximate 1; a large gap means the model has
                    collapsed real population structure.
  3. CONDITIONAL    corr WITHIN p(y|x), averaged over sources.
                    "Given everything we know about one object, how do our
                    remaining uncertainties co-vary?"
  4. RESIDUAL       corr of (truth - posterior mean) across sources.
                    The EMPIRICAL CHECK on 3. The flow can assert any
                    conditional covariance it likes; the residuals are what the
                    data actually did. If 3 and 4 disagree in sign, the
                    conditional structure is a modelling artefact, not physics.

1 and 3 are not the same quantity and have no reason to agree. A SIGN FLIP
between them is the interesting case and is physically expected for
(M*, SFR): unconditionally they are positively correlated through the
star-forming main sequence, but at fixed observed light an SED fit trades mass
against star formation -- the classic mass/SFR degeneracy -- so the residual
correlation can be NEGATIVE. That is structure only a joint head can express;
independent marginals assert conditional independence by construction.

CAVEAT to carry into any claim: 3 and 4 agree exactly only if the posterior
width is similar across sources. Where widths vary a lot, 4 is a
spread-weighted average of 3. Treat agreement in SIGN as the test, not
agreement to three decimals.

Everything is computed pairwise-complete and BOOTSTRAPPED over sources, so a
reported flip comes with an interval rather than a point estimate.

    python scripts/posterior_structure.py --checkpoint best.pt --staged-dir ... \
        --clean-split-csv ... --extra-targets-csv ... --out posterior_structure.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shareable_aion_flow"))

import multitarget as mt                                              # noqa: E402
from data_to_aion_embeddings import AIONTokenEncoder, build_dataloaders  # noqa: E402
from normalizing_flow import TargetStandardizer                       # noqa: E402

ALL_COMBO = ("spectra", "z", "wise", "image")


def pair_corr(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """Pearson r on pairwise-complete rows, plus the count used."""
    ok = np.isfinite(a) & np.isfinite(b)
    n = int(ok.sum())
    if n < 10:
        return float("nan"), n
    x, y = a[ok], b[ok]
    if x.std() == 0 or y.std() == 0:
        return float("nan"), n
    return float(np.corrcoef(x, y)[0, 1]), n


def corr_matrix(x: np.ndarray) -> np.ndarray:
    d = x.shape[1]
    m = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            m[i, j] = m[j, i] = pair_corr(x[:, i], x[:, j])[0]
    return m


def boot_ci(a: np.ndarray, b: np.ndarray, n_boot: int, rng) -> tuple[float, float]:
    """Percentile CI on the correlation of two aligned columns."""
    ok = np.isfinite(a) & np.isfinite(b)
    x, y = a[ok], b[ok]
    if len(x) < 10:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    xs, ys = x[idx], y[idx]
    xc = xs - xs.mean(1, keepdims=True)
    yc = ys - ys.mean(1, keepdims=True)
    num = (xc * yc).sum(1)
    den = np.sqrt((xc ** 2).sum(1) * (yc ** 2).sum(1))
    r = np.where(den > 0, num / np.maximum(den, 1e-30), np.nan)
    return float(np.nanpercentile(r, 2.5)), float(np.nanpercentile(r, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--staged-dir", type=Path, required=True)
    ap.add_argument("--clean-split-csv", type=Path, required=True)
    ap.add_argument("--extra-targets-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=512,
                    help="posterior draws per source; the per-source correlation "
                         "estimate is noisy below a few hundred")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--eval-batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mt.configure_heads_from_config(ck.get("config", {}))
    names = [t["name"] for t in mt.MULTI_TARGETS]
    jidx = list(mt.JOINT_IDX)              # flow-column order, authoritative
    dims = [names[j] for j in jidx]
    print(f"[post] joint dims (flow order): {dims}")

    encoder = AIONTokenEncoder(freeze=False, cls_mode=True, cls_variant="readonly",
                               num_cls=mt.N_HEADS).to(device)
    encoder.load_state_dict(ck["encoder_trainable_state_dict"], strict=False)
    head = mt.SharedCLSHead().to(device)
    head.load_state_dict(ck["head_state_dict"])
    flows = mt.MultiTargetFlows().to(device)
    flows.load_state_dict(ck["flows_state_dict"])
    encoder.eval(); head.eval(); flows.eval()
    stds = [TargetStandardizer.from_state_dict(s) for s in ck["standardizers"]]

    lookup = mt.MultiTargetLookup(args.staged_dir, args.extra_targets_csv)
    _, val_loader, _ = build_dataloaders(
        staged_dir=args.staged_dir, target_name="log_ml_flux_1",
        batch_size=args.eval_batch_size, eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers, seed=42, clean_split_csv=args.clean_split_csv)

    truths, means, sds, per_source = [], [], [], []
    with torch.no_grad():
        for b in val_loader:
            b = tuple(t.to(device, non_blocking=True) for t in b)
            y, _, _ = lookup.batch(b[7], device)
            cls_seq, _ = encoder.encode_tokens(b, ALL_COMBO)
            ctx = head(cls_seq)[:, mt.N_TARGETS]                # the joint's slot
            s = flows.joint.sample(ctx, num_samples=args.samples)   # [S, B, D]
            if s.dim() == 2:
                s = s.unsqueeze(-1)
            s = s.permute(1, 0, 2)                              # [B, S, D]
            # back to physical units so correlations are of the real quantities
            phys = torch.stack([s[:, :, k] * stds[j].std + stds[j].mean
                                for k, j in enumerate(jidx)], dim=-1)
            means.append(phys.mean(dim=1).cpu().numpy())
            sds.append(phys.std(dim=1).cpu().numpy())
            truths.append(y[:, jidx].cpu().numpy())
            pn = phys.cpu().numpy()
            for i in range(pn.shape[0]):
                per_source.append(corr_matrix(pn[i]))

    truth = np.concatenate(truths)
    mean = np.concatenate(means)
    spread = np.concatenate(sds)
    resid = truth - mean                    # NaN wherever the label is missing
    stack = np.stack(per_source)
    emp, pred = corr_matrix(truth), corr_matrix(mean)
    res = corr_matrix(resid)
    cond = np.nanmean(stack, axis=0)
    print(f"[post] sources: {len(truth):,}   draws each: {args.samples}")
    print("[post] median posterior sd per dim: " +
          "  ".join(f"{d}={np.nanmedian(spread[:, k]):.3f}" for k, d in enumerate(dims)))

    def show(title, m):
        print(f"\n{title}")
        print("            " + "".join(f"{d[:11]:>13s}" for d in dims))
        for i, d in enumerate(dims):
            print(f"{d[:11]:>11s} " + "".join(f"{m[i, j]:>13.3f}" for j in range(len(dims))))

    show("1. EMPIRICAL   (true targets, across sources)", emp)
    show("2. PREDICTED   (posterior means, across sources)", pred)
    show("3. CONDITIONAL (within p(y|x), averaged over sources)", cond)
    show("4. RESIDUAL    (truth - posterior mean, across sources)", res)

    rng = np.random.default_rng(0)
    print("\n=== per pair: population vs conditional ===")
    print(f"{'pair':>30s} {'empirical [95% CI]':>24s} {'conditional':>12s} "
          f"{'residual [95% CI]':>24s} {'flip?':>6s} {'frac<0':>7s} {'n':>6s}")
    out_pairs = []
    for a in range(len(dims)):
        for b in range(a + 1, len(dims)):
            e, n_e = pair_corr(truth[:, a], truth[:, b])
            r, n_r = pair_corr(resid[:, a], resid[:, b])
            c = float(np.nanmean(stack[:, a, b]))
            c_lo, c_hi = np.nanpercentile(stack[:, a, b], [2.5, 97.5])
            e_lo, e_hi = boot_ci(truth[:, a], truth[:, b], args.n_boot, rng)
            r_lo, r_hi = boot_ci(resid[:, a], resid[:, b], args.n_boot, rng)
            frac_neg = float(np.nanmean(stack[:, a, b] < 0))
            # A flip counts only if BOTH the model's conditional and the data's
            # residual oppose the population sign, and the residual CI excludes 0.
            flip = (np.isfinite(e) and np.isfinite(c) and np.isfinite(r)
                    and e * c < 0 and e * r < 0 and r_lo * r_hi > 0)
            tag = "FLIP" if flip else ("cond" if np.isfinite(e * c) and e * c < 0 else "")
            print(f"{dims[a][:13]+' x '+dims[b][:13]:>30s} "
                  f"{e:>7.3f} [{e_lo:>6.3f},{e_hi:>6.3f}] {c:>12.3f} "
                  f"{r:>7.3f} [{r_lo:>6.3f},{r_hi:>6.3f}] {tag:>6s} "
                  f"{frac_neg:>6.1%} {n_r:>6d}")
            out_pairs.append({
                "a": dims[a], "b": dims[b],
                "empirical": e, "empirical_ci": [e_lo, e_hi], "n_empirical": n_e,
                "conditional_mean": c, "conditional_spread_ci": [float(c_lo), float(c_hi)],
                "residual": r, "residual_ci": [r_lo, r_hi], "n_residual": n_r,
                "frac_sources_negative": frac_neg,
                "sign_flip_confirmed": bool(flip)})

    print("\n  FLIP  = population sign, model conditional sign, and DATA residual")
    print("          sign all agree that conditioning reverses it, with the")
    print("          residual CI excluding zero. This is the defensible claim.")
    print("  cond  = only the model's posterior flips; the residuals do not")
    print("          confirm it, so it may be a modelling artefact.")
    print("  frac<0 = fraction of SOURCES whose posterior correlation is")
    print("          negative. A mean near zero with frac<0 near 50% means the")
    print("          population splits in two, not that the correlation is absent.")

    args.out.write_text(json.dumps(
        {"dims": dims, "n_sources": int(len(truth)), "samples": args.samples,
         "checkpoint": str(args.checkpoint),
         "median_posterior_sd": {d: float(np.nanmedian(spread[:, k]))
                                 for k, d in enumerate(dims)},
         "empirical": emp.tolist(), "predicted": pred.tolist(),
         "conditional": cond.tolist(), "residual": res.tolist(),
         "pairs": out_pairs}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
