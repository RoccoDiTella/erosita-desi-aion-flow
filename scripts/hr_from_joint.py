#!/usr/bin/env python
"""HR32 as an IMPLIED target: marginalize the joint (P2,P3) flow onto hardness.

HR32 = (R3-R2)/(R3+R2) is a deterministic function of the two band fluxes
(rates = fluxes / exact per-band ECFs), so the hardness posterior is the
push-forward of the joint flux posterior:

    p(HR|x) = int int p(F2,F3|x) delta(HR - f(F2,F3)) dF2 dF3

estimated by sampling the joint flow and kernel-density-estimating the
transformed samples. That gives a proper per-source log-likelihood at the
MEASURED hardness, hence an information gain against a KDE prior fit on the
training-set hardness -- the one way HR gets an IG number without ever being a
trained target.

    python scripts/hr_from_joint.py --checkpoint .../best.pt --staged-dir ... \
        --clean-split-csv ... --extra-targets-csv ... --hr-ref-csv ...
"""

from __future__ import annotations

import argparse
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

C_P2, C_P3 = -12.1332, -12.0060  # exact log10(FLUX/RATE) per band


def hr_from_log_fluxes(lf2: np.ndarray, lf3: np.ndarray) -> np.ndarray:
    rho = 10.0 ** ((lf3 - C_P3) - (lf2 - C_P2))
    return (rho - 1.0) / (rho + 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--staged-dir", type=Path, required=True)
    ap.add_argument("--clean-split-csv", type=Path, required=True)
    ap.add_argument("--extra-targets-csv", type=Path, required=True)
    ap.add_argument("--hr-ref-csv", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=224)
    ap.add_argument("--samples", type=int, default=2048)
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

    _, _, test_loader = build_dataloaders(
        staged_dir=args.staged_dir, target_name="log_ml_flux_1",
        batch_size=args.batch_size, eval_batch_size=args.batch_size,
        num_workers=8, clean_split_csv=args.clean_split_csv,
    )
    lookup = MultiTargetLookup(args.staged_dir, args.extra_targets_csv)

    # prior: KDE over TRAIN-split measured hardness
    ref = pd.read_csv(args.hr_ref_csv).drop_duplicates("targetid").set_index("targetid")
    assign = pd.read_csv(args.clean_split_csv)
    train_tids = assign.loc[assign.split == "train", "targetid"].to_numpy(np.int64)
    tr = ref.reindex(train_tids)
    hr_train = tr.hr32.to_numpy(np.float64)
    hr_train = hr_train[np.isfinite(hr_train)]
    prior_kde = gaussian_kde(hr_train, bw_method="scott")
    print(f"[prior] {len(hr_train)} train hardness values, "
          f"mean {hr_train.mean():+.3f} sd {hr_train.std():.3f}", flush=True)

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
            s = flows.joint.distribution(ctx).sample((args.samples,))
            lf2 = s[..., 0].cpu().numpy() * stds[j2].std + stds[j2].mean
            lf3 = s[..., 1].cpu().numpy() * stds[j3].std + stds[j3].mean
            hr = hr_from_log_fluxes(lf2, lf3)                      # [S, m]
            tids = batch[7][mask].cpu().numpy()
            for k, tid in enumerate(tids):
                col = hr[:, k]
                col = col[np.isfinite(col)]
                if len(col) < 50 or col.std() < 1e-6:
                    continue
                kde = gaussian_kde(col, bw_method="scott")
                meas = ref.hr32.get(int(tid), np.nan)
                if not np.isfinite(meas):
                    continue
                lp = float(np.log(max(kde.evaluate([meas])[0], 1e-300)))
                lp0 = float(np.log(max(prior_kde.evaluate([meas])[0], 1e-300)))
                p16, p50, p84 = np.percentile(col, [16, 50, 84])
                recs.append({"targetid": int(tid), "hr_meas": float(meas),
                             "hr_p16": p16, "hr_p50": p50, "hr_p84": p84,
                             "log_post": lp, "log_prior": lp0, "info_gain": lp - lp0,
                             "ok": bool(ref.hr32_ok.get(int(tid), False))})
    df = pd.DataFrame(recs)
    out = args.checkpoint.parent / "eval" / "hr_implied_target.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\n[HR as implied target] n={len(df)}")
    for label, m in [("all finite", np.ones(len(df), bool)), ("hr32_ok", df.ok.to_numpy())]:
        d = df[m]
        if len(d) < 30:
            continue
        ig = d.info_gain.mean()
        r2 = 1 - np.sum((d.hr_meas - d.hr_p50) ** 2) / np.sum((d.hr_meas - d.hr_meas.mean()) ** 2)
        corr = np.corrcoef(d.hr_p50, d.hr_meas)[0, 1]
        print(f"  {label:11s} n={len(d):5d}  IG={ig:+.4f} nats  exp(IG)={np.exp(ig):.3f}  "
              f"R2={r2:+.3f}  corr={corr:+.3f}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
