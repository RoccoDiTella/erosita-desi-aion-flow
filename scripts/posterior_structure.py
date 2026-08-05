#!/usr/bin/env python
"""Is the joint posterior's correlation structure different from the data's?

Three correlation matrices, which answer three different questions:

  1. EMPIRICAL      corr of the true targets across sources.
                    "How do these quantities co-vary in the population?"
  2. PREDICTED      corr of the posterior MEANS across sources.
                    "How does the model think they co-vary in the population?"
                    It should approximate 1; a large gap means the model has
                    collapsed real population structure.
  3. CONDITIONAL    corr WITHIN p(y|x), averaged over sources.
                    "Given everything we know about one object, how do our
                    remaining uncertainties co-vary?"

1 and 3 are not the same quantity and have no reason to agree. A SIGN FLIP
between them is the interesting case and is physically expected for
(M*, SFR): unconditionally they are positively correlated through the
star-forming main sequence, but at fixed observed light an SED fit trades mass
against star formation -- the classic mass/SFR degeneracy -- so the residual
correlation can be NEGATIVE. That is structure only a joint head can express;
independent marginals assert conditional independence by construction.

Reported per source as well as on average, because a mean correlation can hide
a population that splits into two signs.

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


def corr(x: np.ndarray) -> np.ndarray:
    """Correlation matrix of columns, NaN-safe."""
    ok = np.isfinite(x).all(axis=1)
    if ok.sum() < 10:
        return np.full((x.shape[1], x.shape[1]), np.nan)
    return np.corrcoef(x[ok], rowvar=False)


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
    ap.add_argument("--eval-batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mt.configure_heads_from_config(ck.get("config", {}))
    dims = [n for n in mt.JOINT_PAIR]
    names = [t["name"] for t in mt.MULTI_TARGETS]
    jidx = [names.index(n) for n in dims]
    print(f"[post] joint dims: {dims}")

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

    truths, means, per_source = [], [], []
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
            truths.append(y[:, jidx].cpu().numpy())
            for i in range(phys.shape[0]):
                per_source.append(corr(phys[i].cpu().numpy()))

    truth = np.concatenate(truths); mean = np.concatenate(means)
    cond = np.nanmean(np.stack(per_source), axis=0)
    emp, pred = corr(truth), corr(mean)
    print(f"[post] sources: {len(truth):,}   draws each: {args.samples}")

    def show(title, m):
        print(f"\n{title}")
        print("            " + "".join(f"{d[:11]:>13s}" for d in dims))
        for i, d in enumerate(dims):
            print(f"{d[:11]:>11s} " + "".join(f"{m[i, j]:>13.3f}" for j in range(len(dims))))

    show("1. EMPIRICAL  (true targets, across sources)", emp)
    show("2. PREDICTED  (posterior means, across sources)", pred)
    show("3. CONDITIONAL (within p(y|x), averaged over sources)", cond)

    print("\n=== pairs where CONDITIONAL differs from EMPIRICAL ===")
    print(f"{'pair':>28s} {'empirical':>10s} {'conditional':>12s} {'flip?':>7s} "
          f"{'frac<0':>8s}")
    stack = np.stack(per_source)
    out_pairs = []
    for a in range(len(dims)):
        for b in range(a + 1, len(dims)):
            e, c = emp[a, b], cond[a, b]
            frac_neg = float(np.nanmean(stack[:, a, b] < 0))
            flip = "SIGN" if np.isfinite(e) and np.isfinite(c) and e * c < 0 else ""
            print(f"{dims[a][:13]+' x '+dims[b][:13]:>28s} {e:>10.3f} {c:>12.3f} "
                  f"{flip:>7s} {frac_neg:>8.1%}")
            out_pairs.append({"a": dims[a], "b": dims[b], "empirical": float(e),
                              "conditional": float(c), "sign_flip": bool(flip),
                              "frac_sources_negative": frac_neg})
    print("\n  frac<0 is the fraction of SOURCES whose posterior correlation is")
    print("  negative -- a mean near zero with frac<0 near 50% means the population")
    print("  splits, not that the correlation is absent.")
    args.out.write_text(json.dumps(
        {"dims": dims, "n_sources": int(len(truth)), "samples": args.samples,
         "empirical": emp.tolist(), "predicted": pred.tolist(),
         "conditional": cond.tolist(), "pairs": out_pairs}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
