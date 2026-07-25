#!/usr/bin/env python
"""HR32 as an IMPLIED target: exact marginalization of the joint (P2,P3) flow.

The band rates are the fluxes divided by EXACTLY constant per-band ECFs, so

    rho = R3/R2 = 10^d,  d = logF3 - logF2 + (C2 - C3)
    HR  = (rho-1)/(rho+1) = tanh(d ln10 / 2)

i.e. hardness is a monotone function of the DIFFERENCE of the two log fluxes
alone. The hardness posterior is therefore the joint density marginalized along
lines of constant difference (a shear, unit Jacobian) times the analytic
transform Jacobian:

    p(d|x)  = int p_joint(u, u + d - Delta | x) du
    p(HR|x) = p(d|x) * 2 / (ln10 * (1 - HR^2))

Computed by direct quadrature over u -- no sampling, no KDE bandwidth bias
(a sampled+KDE estimate of the same quantity is strictly noisier). All grid
points evaluate under ONE context-conditioned distribution per source.

    python scripts/hr_from_joint.py --checkpoint .../best.pt --staged-dir ... \
        --clean-split-csv ... --extra-targets-csv ... --hr-ref-csv ...
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shareable_aion_flow.data_to_aion_embeddings import AIONTokenEncoder, build_dataloaders  # noqa: E402
from shareable_aion_flow.multitarget import (  # noqa: E402
    JOINT_IDX, N_HEADS, N_TARGETS, MultiTargetFlows, MultiTargetLookup, SharedCLSHead,
)
from shareable_aion_flow.normalizing_flow import TargetStandardizer  # noqa: E402

C_P2, C_P3 = -12.1332, -12.0060       # exact log10(FLUX/RATE) per band
DELTA = C_P2 - C_P3                   # d = v - u + DELTA
LN10 = math.log(10.0)


def hr_to_d(hr: np.ndarray | float) -> np.ndarray:
    return (2.0 / LN10) * np.arctanh(np.clip(hr, -0.999999, 0.999999))


def d_to_hr(d: np.ndarray) -> np.ndarray:
    return np.tanh(d * LN10 / 2.0)


@torch.no_grad()
def line_log_density(
    flow, context: torch.Tensor, d_phys: torch.Tensor, std2: TargetStandardizer,
    std3: TargetStandardizer, u_nodes: torch.Tensor, chunk: int = 64
) -> torch.Tensor:
    """log p(d|x) for one physical difference per source, by quadrature over u.

    ``d_phys`` [B]: the difference v-u+DELTA to evaluate per source.
    ``u_nodes`` [K]: integration nodes in PHYSICAL logF2 units (shared grid).
    Returns [B] log-densities in physical d units.
    """
    B = context.shape[0]
    K = u_nodes.shape[0]
    du = float(u_nodes[1] - u_nodes[0])
    parts = []
    for lo in range(0, K, chunk):
        u = u_nodes[lo : lo + chunk]                        # [k]
        uu = u.view(-1, 1).expand(-1, B)                    # [k, B] physical logF2
        vv = uu + d_phys.view(1, -1) - DELTA                # [k, B] physical logF3
        pair = torch.stack([(uu - std2.mean) / std2.std,
                            (vv - std3.mean) / std3.std], dim=-1)   # [k, B, 2]
        lp = flow.log_prob_draws(pair, context)             # [k, B] (standardized units)
        parts.append(lp)
    lp_all = torch.cat(parts, dim=0)
    # standardized -> physical density: divide by the two scale factors
    lp_all = lp_all - math.log(std2.std) - math.log(std3.std)
    # integrate over u (log-sum-exp trapezoid)
    return torch.logsumexp(lp_all, dim=0) + math.log(du)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--staged-dir", type=Path, required=True)
    ap.add_argument("--clean-split-csv", type=Path, required=True)
    ap.add_argument("--extra-targets-csv", type=Path, required=True)
    ap.add_argument("--hr-ref-csv", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=224)
    ap.add_argument("--u-nodes", type=int, default=96)
    ap.add_argument("--hr-grid", type=int, default=81)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    encoder = AIONTokenEncoder(freeze=False, cls_mode=True, cls_variant="readonly",
                               num_cls=N_HEADS).to(device)
    encoder.load_state_dict(ckpt["encoder_trainable_state_dict"], strict=False)
    head = SharedCLSHead().to(device); head.load_state_dict(ckpt["head_state_dict"])
    flows = MultiTargetFlows().to(device); flows.load_state_dict(ckpt["flows_state_dict"])
    stds = [TargetStandardizer.from_state_dict(s) for s in ckpt["standardizers"]]
    encoder.eval(); head.eval(); flows.eval()
    j2, j3 = JOINT_IDX
    std2, std3 = stds[j2], stds[j3]

    _, _, test_loader = build_dataloaders(
        staged_dir=args.staged_dir, target_name="log_ml_flux_1",
        batch_size=args.batch_size, eval_batch_size=args.batch_size,
        num_workers=8, clean_split_csv=args.clean_split_csv,
    )
    lookup = MultiTargetLookup(args.staged_dir, args.extra_targets_csv)

    # integration nodes: physical logF2 spanning the marginal generously
    u_lo = std2.mean - 6.0 * std2.std
    u_hi = std2.mean + 6.0 * std2.std
    u_nodes = torch.linspace(u_lo, u_hi, args.u_nodes, device=device)
    # HR grid for posterior summaries (uniform in d, the natural variable)
    hr_grid = np.linspace(-0.97, 0.97, args.hr_grid)
    d_grid = hr_to_d(hr_grid)

    ref = pd.read_csv(args.hr_ref_csv).drop_duplicates("targetid").set_index("targetid")
    assign = pd.read_csv(args.clean_split_csv)
    train_tids = assign.loc[assign.split == "train", "targetid"].to_numpy(np.int64)
    hr_train = ref.reindex(train_tids).hr32.to_numpy(np.float64)
    hr_train = hr_train[np.isfinite(hr_train)]
    prior_kde = gaussian_kde(hr_train, bw_method="scott")
    print(f"[prior] {len(hr_train)} train hardness values, mean {hr_train.mean():+.3f} "
          f"sd {hr_train.std():.3f}", flush=True)
    log_jac_grid = np.log(2.0 / (LN10 * (1.0 - hr_grid**2)))

    recs = []
    with torch.no_grad():
        for batch in test_loader:
            batch = tuple(t.to(device, non_blocking=True) for t in batch)
            y, _, _ = lookup.batch(batch[7], device)
            mask = torch.isfinite(y[:, j2]) & torch.isfinite(y[:, j3])
            if not bool(mask.any()):
                continue
            cls_seq, _ = encoder.encode_tokens(batch, ("spectra", "z", "wise", "image"))
            ctx = head(cls_seq)[mask, N_TARGETS]
            tids = batch[7][mask].cpu().numpy()
            meas = ref.reindex(tids).hr32.to_numpy(np.float64)
            good = np.isfinite(meas)
            if not good.any():
                continue

            # (a) density AT the measured hardness -- one line integral per source
            d_meas = torch.tensor(hr_to_d(np.nan_to_num(meas)), dtype=torch.float32, device=device)
            lp_d = line_log_density(flows.joint, ctx, d_meas, std2, std3, u_nodes).cpu().numpy()
            log_jac = np.log(2.0 / (LN10 * (1.0 - np.clip(meas, -0.999, 0.999) ** 2)))
            log_post = lp_d + log_jac

            # (b) full HR posterior on the grid -> median, 16/84, R2
            grid_lp = np.zeros((len(hr_grid), int(mask.sum())))
            for gi, dv in enumerate(d_grid):
                dvec = torch.full((int(mask.sum()),), float(dv), dtype=torch.float32, device=device)
                grid_lp[gi] = line_log_density(flows.joint, ctx, dvec, std2, std3, u_nodes).cpu().numpy()
            grid_lp = grid_lp + log_jac_grid[:, None]
            dens = np.exp(grid_lp - grid_lp.max(axis=0, keepdims=True))
            cdf = np.cumsum(dens, axis=0)
            cdf = cdf / np.maximum(cdf[-1:], 1e-300)
            q = lambda p: np.array([np.interp(p, cdf[:, k], hr_grid) for k in range(cdf.shape[1])])
            p16, p50, p84 = q(0.16), q(0.50), q(0.84)

            for k, tid in enumerate(tids):
                if not good[k]:
                    continue
                lp0 = float(np.log(max(prior_kde.evaluate([meas[k]])[0], 1e-300)))
                recs.append({"targetid": int(tid), "hr_meas": float(meas[k]),
                             "hr_p16": float(p16[k]), "hr_p50": float(p50[k]),
                             "hr_p84": float(p84[k]), "log_post": float(log_post[k]),
                             "log_prior": lp0, "info_gain": float(log_post[k]) - lp0,
                             "ok": bool(ref.hr32_ok.get(int(tid), False))})
            print(f"[hr] {len(recs)} sources done", flush=True)

    df = pd.DataFrame(recs)
    out = args.checkpoint.parent / "eval" / "hr_implied_target.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n[HR as implied target — exact quadrature] n={len(df)}")
    for label, m in [("all finite", np.ones(len(df), bool)), ("hr32_ok", df.ok.to_numpy())]:
        d = df[m]
        if len(d) < 30:
            continue
        ig = d.info_gain.mean()
        r2 = 1 - np.sum((d.hr_meas - d.hr_p50) ** 2) / np.sum((d.hr_meas - d.hr_meas.mean()) ** 2)
        corr = np.corrcoef(d.hr_p50, d.hr_meas)[0, 1]
        width = np.median(0.5 * (d.hr_p84 - d.hr_p16))
        print(f"  {label:11s} n={len(d):5d}  IG={ig:+.4f} nats  exp(IG)={np.exp(ig):.3f}  "
              f"R2={r2:+.3f}  corr={corr:+.3f}  median 68% half-width {width:.3f}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
