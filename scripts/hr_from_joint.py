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

This is an EXACTLY 2-D construction: the shear integrates one axis against the
other and there is no third axis to hold fixed or integrate as well. A joint of
any other width is refused at startup rather than silently reinterpreted, which
is what `j2, j3 = JOINT_IDX` did while the joint was (M*, SFR, Lx, P3).

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
from shareable_aion_flow.attention_pooling_head import MODALITIES  # noqa: E402
from shareable_aion_flow.data_to_aion_embeddings import build_dataloaders  # noqa: E402
from shareable_aion_flow.eval_core import C_P2, C_P3, HR_BANDS, assert_joint_matches_flow  # noqa: E402
from shareable_aion_flow.multitarget import (  # noqa: E402
    MultiTargetFlows, MultiTargetLookup, SharedCLSHead,
)
from shareable_aion_flow.normalizing_flow import TargetStandardizer  # noqa: E402
from shareable_aion_flow.stub_encoder import build_encoder  # noqa: E402

DELTA = C_P2 - C_P3                   # d = v - u + DELTA
LN10 = math.log(10.0)

# Exit code for "this checkpoint's joint is not a 2-D (P2,P3) one", as opposed to
# an ordinary failure. The distinction is not cosmetic: sbatch/eval_multi.sbatch
# runs under `set -e`, so an undifferentiated exit 1 here marks the whole eval
# job FAILED after eval_multitarget.py has already written every table plus
# hr_joint_summary.csv, whose SAMPLED hardness works for any joint containing
# both bands. Not-applicable is a property of the checkpoint, not an error, and
# the launcher reports it as a skip.
NOT_APPLICABLE = 3


def hr_to_d(hr: np.ndarray | float) -> np.ndarray:
    return (2.0 / LN10) * np.arctanh(np.clip(hr, -0.999999, 0.999999))


def d_to_hr(d: np.ndarray) -> np.ndarray:
    return np.tanh(d * LN10 / 2.0)


@torch.no_grad()
def line_log_density(
    flow, context: torch.Tensor, d_phys: torch.Tensor, std2: TargetStandardizer,
    std3: TargetStandardizer, u_nodes: torch.Tensor, col2: int = 0, col3: int = 1,
    chunk: int = 64,
) -> torch.Tensor:
    """log p(d|x) for one physical difference per source, by quadrature over u.

    ``d_phys`` [B]: the difference v-u+DELTA to evaluate per source.
    ``u_nodes`` [K]: integration nodes in PHYSICAL logF2 units (shared grid).
    ``col2``/``col3``: which FLOW COLUMNS carry P2 and P3. They are passed in
    from joint_col() rather than assumed to be 0 and 1 -- the flow's column
    order is JOINT_PAIR declaration order, which need not put the bands first
    or even in band order.
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
        cols: list[torch.Tensor | None] = [None, None]
        cols[col2] = (uu - std2.mean) / std2.std
        cols[col3] = (vv - std3.mean) / std3.std
        pair = torch.stack(cols, dim=-1)                    # [k, B, 2]
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
    ap.add_argument("--device", default=None,
                    help="torch device; default cuda when available. cpu + "
                         "AIONFLOW_STUB_ENCODER=1 runs this off the cluster.")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.checkpoint, map_location=device)
    import shareable_aion_flow.multitarget as _mt
    _mt.configure_heads_from_config(ckpt.get("config", {}))
    dims = _mt.joint_dims()
    # The shear quadrature is a 2-D construction. Refuse anything else loudly:
    # a wider joint has axes this integral does not know how to hold fixed, and
    # the failure mode of guessing is a finite, plausible-looking hardness.
    if len(dims) != 2 or set(dims) != set(HR_BANDS):
        print(
            f"[hr] NOT APPLICABLE (exit {NOT_APPLICABLE}): hr_from_joint needs an exactly 2-D "
            f"joint over {HR_BANDS}; this checkpoint's joint is {dims} ({len(dims)}-D). The "
            "sampled HR block in scripts/eval_multitarget.py handles a wider joint and has "
            "already written hr_joint_summary.csv; the exact quadrature applies to a P2xP3 "
            "joint only. Nothing failed -- this checkpoint cannot answer this question.",
            flush=True)
        raise SystemExit(NOT_APPLICABLE)
    col2, col3 = _mt.joint_col(HR_BANDS[0]), _mt.joint_col(HR_BANDS[1])

    encoder = build_encoder(num_cls=_mt.N_HEADS, device=device, tag="hr")
    encoder.load_state_dict(ckpt["encoder_trainable_state_dict"], strict=False)
    head = SharedCLSHead().to(device); head.load_state_dict(ckpt["head_state_dict"])
    flows = MultiTargetFlows().to(device); flows.load_state_dict(ckpt["flows_state_dict"])
    assert_joint_matches_flow(flows)
    stds = [TargetStandardizer.from_state_dict(s) for s in ckpt["standardizers"]]
    encoder.eval(); head.eval(); flows.eval()
    # Two different indexing spaces: target_col addresses the [B, N_TARGETS]
    # target matrix, joint_col the flow's feature vector. Never one for the other.
    std2, std3 = stds[_mt.target_col(HR_BANDS[0])], stds[_mt.target_col(HR_BANDS[1])]

    _, _, test_loader = build_dataloaders(
        staged_dir=args.staged_dir, target_name=None,
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
    if "hr32" not in ref.columns:
        raise SystemExit(f"{args.hr_ref_csv} has no hr32 column; columns are "
                         f"{sorted(ref.columns)[:12]}")
    # hr32_ok is the DR1 reference's own quality flag. A DR2-depth reference need
    # not carry one, and a missing quality flag must not take the whole run down
    # on an AttributeError three minutes in -- it just means one fewer subset.
    has_ok = "hr32_ok" in ref.columns
    if not has_ok:
        print(f"[hr] {args.hr_ref_csv} has no hr32_ok column: reporting the "
              "'all finite' subset only", flush=True)
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
            # One availability rule, shared with the trainer. For this 2-D joint
            # nothing is marginalisable, so have_req and have_all coincide.
            _, mask = _mt.joint_availability(y)
            if not bool(mask.any()):
                continue
            cls_seq, _ = encoder.encode_tokens(batch, MODALITIES)
            ctx = head(cls_seq)[mask, _mt.N_TARGETS]
            tids = batch[7][mask].cpu().numpy()
            meas = ref.reindex(tids).hr32.to_numpy(np.float64)
            good = np.isfinite(meas)
            if not good.any():
                continue

            # (a) density AT the measured hardness -- one line integral per source
            d_meas = torch.tensor(hr_to_d(np.nan_to_num(meas)), dtype=torch.float32, device=device)
            lp_d = line_log_density(flows.joint, ctx, d_meas, std2, std3, u_nodes,
                                    col2, col3).cpu().numpy()
            log_jac = np.log(2.0 / (LN10 * (1.0 - np.clip(meas, -0.999, 0.999) ** 2)))
            log_post = lp_d + log_jac

            # (b) full HR posterior on the grid -> median, 16/84, R2
            grid_lp = np.zeros((len(hr_grid), int(mask.sum())))
            for gi, dv in enumerate(d_grid):
                dvec = torch.full((int(mask.sum()),), float(dv), dtype=torch.float32, device=device)
                grid_lp[gi] = line_log_density(flows.joint, ctx, dvec, std2, std3, u_nodes,
                                               col2, col3).cpu().numpy()
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
                             "ok": bool(ref.hr32_ok.get(int(tid), False)) if has_ok else False})
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
