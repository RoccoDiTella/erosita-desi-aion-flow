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

COMPARING ACROSS --combo IS NOT FREE. Dropping a modality removes information,
which widens the posterior, and widening ONE dimension attenuates every
correlation involving it by sqrt(V / (V + dV)) with no change in the underlying
structure. Before reading a drop as "the artefact went away", predict the
mechanical attenuation from how much that target's R^2 fell and subtract it.
Dropping WISE costs log M* about 0.17 in R^2 and the X-ray heads almost
nothing, so M* pairs need the correction and X-ray pairs essentially do not.

    python scripts/posterior_structure.py --checkpoint best.pt --staged-dir ... \
        --clean-split-csv ... --extra-targets-csv ... --out posterior_structure.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shareable_aion_flow"))

import multitarget as mt                                              # noqa: E402
from data_to_aion_embeddings import build_dataloaders                 # noqa: E402
from normalizing_flow import TargetStandardizer                       # noqa: E402
from stub_encoder import build_encoder                                # noqa: E402

ALL_COMBO = ("spectra", "z", "wise", "image")


def parse_derived(spec: str, dims: list[str]) -> tuple[str, np.ndarray]:
    """'log_ssfr=log_sfr-logmstar_cigale' -> ('log_ssfr', weight vector over dims).

    Derived quantities are the reason a JOINT head is not optional. sSFR is a
    DIFFERENCE of two dimensions, so its posterior width depends on their
    covariance: independent marginals would have to assume zero correlation and
    would get both the width and every correlation involving it wrong.
    """
    name, _, rhs = spec.partition("=")
    name, rhs = name.strip(), rhs.strip()
    if not name or not rhs:
        raise ValueError(f"--derived wants NAME=EXPR, got {spec!r}")
    w = np.zeros(len(dims))
    tok, sign = "", 1.0
    for ch in rhs.replace(" ", "") + "+":
        if ch in "+-":
            if tok:
                coef, _, var = tok.rpartition("*")
                if var not in dims:
                    raise ValueError(f"{var!r} is not a joint dim; have {dims}")
                w[dims.index(var)] += sign * (float(coef) if coef else 1.0)
            tok, sign = "", 1.0 if ch == "+" else -1.0
        else:
            tok += ch
    return name, w


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
    ap.add_argument("--splits", nargs="+", default=["val"],
                    choices=["train", "val", "test"],
                    help="which splits to pool. Held-out only by default: the "
                         "RESIDUAL check needs unseen data, because a memorised "
                         "training residual is not a fair test of the posterior.")
    ap.add_argument("--group-csv", type=Path, default=None,
                    help="CSV with targetid + a class column, to report the "
                         "correlations separately per class")
    ap.add_argument("--group-col", default="cigale_spectype")
    ap.add_argument("--combo", nargs="+", default=list(ALL_COMBO),
                    choices=list(ALL_COMBO),
                    help="input modalities to condition on. The model was trained "
                         "with modality dropout, so any subset is valid without "
                         "retraining. Dropping WISE is the CIGALE-artefact test: "
                         "CIGALE fits M* and SFR from grz+WISE photometry, so "
                         "conditioning on WISE offers a route to reproducing that "
                         "fit rather than the physics. Read the caveat in the "
                         "docstring before comparing correlations across combos.")
    ap.add_argument("--derived", nargs="*", default=[],
                    help="extra columns as linear combinations of the joint dims, "
                         "e.g. log_ssfr=log_sfr-logmstar_cigale. Propagated through "
                         "the DRAWS, so the posterior covariance is carried exactly.")
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
    combo = tuple(args.combo)
    print(f"[post] joint dims (flow order): {dims}")
    print(f"[post] conditioning on: {combo}")
    derived = [parse_derived(s, dims) for s in args.derived]
    base_dims = list(dims)
    dims = dims + [n for n, _ in derived]
    W = (torch.tensor(np.stack([w for _, w in derived]), dtype=torch.float32)
         if derived else None)
    for n, w in derived:
        print(f"[post] derived {n} = " + " ".join(
            f"{c:+g}*{d}" for c, d in zip(w, base_dims) if c))

    encoder = build_encoder(num_cls=mt.N_HEADS, device=device, tag="post")
    encoder.load_state_dict(ck["encoder_trainable_state_dict"], strict=False)
    head = mt.SharedCLSHead().to(device)
    head.load_state_dict(ck["head_state_dict"])
    flows = mt.MultiTargetFlows().to(device)
    flows.load_state_dict(ck["flows_state_dict"])
    encoder.eval(); head.eval(); flows.eval()
    stds = [TargetStandardizer.from_state_dict(s) for s in ck["standardizers"]]

    lookup = mt.MultiTargetLookup(args.staged_dir, args.extra_targets_csv)
    loaders = dict(zip(("train", "val", "test"), build_dataloaders(
        staged_dir=args.staged_dir, target_name=None,
        batch_size=args.eval_batch_size, eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers, seed=42, clean_split_csv=args.clean_split_csv)))
    if "train" in args.splits:
        print("[post] WARNING: pooling TRAIN. Residuals there are shrunk by "
              "memorisation, so the residual check is no longer a fair test.")
    batches = [b for s in args.splits for b in loaders[s]]

    truths, means, sds, per_source, tids = [], [], [], [], []
    with torch.no_grad():
        for b in batches:
            b = tuple(t.to(device, non_blocking=True) for t in b)
            y, _, _ = lookup.batch(b[7], device)
            cls_seq, _ = encoder.encode_tokens(b, combo)
            ctx = head(cls_seq)[:, mt.N_TARGETS]                # the joint's slot
            s = flows.joint.sample(ctx, num_samples=args.samples)   # [S, B, D]
            if s.dim() == 2:
                s = s.unsqueeze(-1)
            s = s.permute(1, 0, 2)                              # [B, S, D]
            # back to physical units so correlations are of the real quantities
            phys = torch.stack([s[:, :, k] * stds[j].std + stds[j].mean
                                for k, j in enumerate(jidx)], dim=-1)
            yt = y[:, jidx]
            if W is not None:
                # propagate through the DRAWS, not through summary statistics:
                # this is what carries the posterior covariance into the
                # derived quantity's width and correlations.
                Wd = W.to(phys.device, phys.dtype)
                phys = torch.cat([phys, phys @ Wd.T], dim=-1)
                yt = torch.cat([yt, yt @ Wd.T], dim=-1)
            means.append(phys.mean(dim=1).cpu().numpy())
            sds.append(phys.std(dim=1).cpu().numpy())
            truths.append(yt.cpu().numpy())
            tids.append(b[7].cpu().numpy().astype(np.int64))
            pn = phys.cpu().numpy()
            for i in range(pn.shape[0]):
                per_source.append(corr_matrix(pn[i]))

    truth_all = np.concatenate(truths)
    mean_all = np.concatenate(means)
    spread_all = np.concatenate(sds)
    stack_all = np.stack(per_source)
    tid_all = np.concatenate(tids)

    groups: list[tuple[str, np.ndarray]] = [("ALL", np.ones(len(tid_all), bool))]
    if args.group_csv:
        gdf = (pd.read_csv(args.group_csv, usecols=["targetid", args.group_col])
               .drop_duplicates("targetid"))
        gmap = dict(zip(gdf.targetid.astype(np.int64), gdf[args.group_col]))
        labels = np.array([str(gmap.get(int(t), "MISSING")) for t in tid_all])
        for lab in sorted(set(labels) - {"MISSING"}):
            groups.append((lab, labels == lab))

    # Which base dims feed each column, so pairs that SHARE one can be flagged:
    # corr(sSFR, M*) is partly mechanical because M* appears in sSFR, whereas
    # corr(Lx, sSFR) shares nothing and is a clean comparison.
    parts = [{d} for d in base_dims] + [
        {b for c, b in zip(w, base_dims) if c} for _, w in derived]
    rng = np.random.default_rng(0)
    report = {}

  # ---- one full analysis per group -----------------------------------------
    for gname, gmask in groups:
        truth = truth_all[gmask]
        mean = mean_all[gmask]
        spread = spread_all[gmask]
        stack = stack_all[gmask]
        resid = truth - mean               # NaN wherever the label is missing
        emp, pred = corr_matrix(truth), corr_matrix(mean)
        res = corr_matrix(resid)
        cond = np.nanmean(stack, axis=0)
        print(f"\n{'='*100}\n### GROUP {gname}: {len(truth):,} sources"
              f"   draws each: {args.samples}\n{'='*100}")
        print("[post] median posterior sd per dim: " + "  ".join(
            f"{d}={np.nanmedian(spread[:, k]):.3f}" for k, d in enumerate(dims)))

        def show(title, m):
            print(f"\n{title}")
            print("            " + "".join(f"{d[:11]:>13s}" for d in dims))
            for i, d in enumerate(dims):
                print(f"{d[:11]:>11s} " + "".join(
                    f"{m[i, j]:>13.3f}" for j in range(len(dims))))

        show("1. EMPIRICAL   (true targets, across sources)", emp)
        show("2. PREDICTED   (posterior means, across sources)", pred)
        show("3. CONDITIONAL (within p(y|x), averaged over sources)", cond)
        show("4. RESIDUAL    (truth - posterior mean, across sources)", res)

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
                # A flip counts only if BOTH the model's conditional and the
                # data's residual oppose the population sign, and the residual
                # CI excludes 0.
                flip = (np.isfinite(e) and np.isfinite(c) and np.isfinite(r)
                        and e * c < 0 and e * r < 0 and r_lo * r_hi > 0)
                shared = parts[a] & parts[b]
                tag = "FLIP" if flip else (
                    "cond" if np.isfinite(e * c) and e * c < 0 else "")
                if shared:
                    tag = (tag + "*").strip()
                print(f"{dims[a][:13]+' x '+dims[b][:13]:>30s} "
                      f"{e:>7.3f} [{e_lo:>6.3f},{e_hi:>6.3f}] {c:>12.3f} "
                      f"{r:>7.3f} [{r_lo:>6.3f},{r_hi:>6.3f}] {tag:>6s} "
                      f"{frac_neg:>6.1%} {n_r:>6d}")
                out_pairs.append({
                    "a": dims[a], "b": dims[b],
                    "empirical": e, "empirical_ci": [e_lo, e_hi], "n_empirical": n_e,
                    "conditional_mean": c,
                    "conditional_spread_ci": [float(c_lo), float(c_hi)],
                    "residual": r, "residual_ci": [r_lo, r_hi], "n_residual": n_r,
                    "frac_sources_negative": frac_neg,
                    "shares_constituent": sorted(shared),
                    "sign_flip_confirmed": bool(flip)})

        report[gname] = {
            "n_sources": int(len(truth)),
            "median_posterior_sd": {d: float(np.nanmedian(spread[:, k]))
                                    for k, d in enumerate(dims)},
            "empirical": emp.tolist(), "predicted": pred.tolist(),
            "conditional": cond.tolist(), "residual": res.tolist(),
            "pairs": out_pairs}

    print("\n  FLIP  = population sign, model conditional sign, and DATA residual")
    print("          sign all agree that conditioning reverses it, with the")
    print("          residual CI excluding zero. This is the defensible claim.")
    print("  cond  = only the model's posterior flips; the residuals do not")
    print("          confirm it, so it may be a modelling artefact.")
    print("  *     = the two columns SHARE a constituent (e.g. M* appears in")
    print("          sSFR), so part of the correlation is mechanical, not physical.")
    print("  frac<0 = fraction of SOURCES whose posterior correlation is")
    print("          negative. A mean near zero with frac<0 near 50% means the")
    print("          population splits in two, not that the correlation is absent.")

    args.out.write_text(json.dumps(
        {"dims": dims, "samples": args.samples, "splits": args.splits,
         "combo": list(combo), "checkpoint": str(args.checkpoint),
         "groups": report}, indent=2))
    print(f"\nwrote {args.out}")

    # Per-source correlations, so the DISTRIBUTION can be inspected rather than
    # only its mean. A mean near zero is ambiguous between "no correlation" and
    # "two populations of opposite sign", and only the histogram separates them.
    npz = args.out.with_suffix(".npz")
    labels = np.array(["ALL"] * len(tid_all), dtype=object)
    for gname, gmask in groups[1:]:
        labels[gmask] = gname
    np.savez_compressed(
        npz, per_source=stack_all.astype(np.float32), targetid=tid_all,
        group=labels.astype(str), dims=np.array(dims), truth=truth_all.astype(np.float32),
        mean=mean_all.astype(np.float32), sd=spread_all.astype(np.float32))
    print(f"wrote {npz}  (per-source correlations, {stack_all.shape})")


if __name__ == "__main__":
    main()
