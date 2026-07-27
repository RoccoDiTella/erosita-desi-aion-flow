#!/usr/bin/env python
"""Test-set evaluation of a train-multi checkpoint: per-head tables + joint HR.

Per modality combo (all 15): one encoder pass serves all 8 heads. Scalar heads
get plain-LL NLL, posterior-mean R2/RMSE, and IG vs a per-target KDE prior fit
on the train view. The joint (P2,P3) head is sampled to produce per-source HR
posteriors via the exact per-band flux/rate constants; summary compares against
measured HR32 (and the hr32_ok subset) and the independent-bands baseline.

    python scripts/eval_multitarget.py --checkpoint .../best.pt \
        --staged-dir ... --clean-split-csv ... --extra-targets-csv ... \
        [--hr-ref-csv targets_extra.csv] [--batch-size 224]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shareable_aion_flow.attention_pooling_head import MODALITIES, all_nonempty_modality_combos  # noqa: E402
from shareable_aion_flow.data_to_aion_embeddings import AIONTokenEncoder, build_dataloaders  # noqa: E402
from shareable_aion_flow.multitarget import (  # noqa: E402
    HEAD_NAMES, JOINT_IDX, MULTI_TARGETS, N_HEADS, N_TARGETS,
    MultiTargetFlows, MultiTargetLookup, SharedCLSHead,
)
from shareable_aion_flow.normalizing_flow import KDEPrior, TargetStandardizer  # noqa: E402

C_P2, C_P3 = -12.1332, -12.0060  # exact log10(FLUX/RATE) global ECFs per band


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--staged-dir", type=Path, required=True)
    ap.add_argument("--clean-split-csv", type=Path, required=True)
    ap.add_argument("--extra-targets-csv", type=Path, required=True)
    ap.add_argument("--hr-ref-csv", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=224)
    ap.add_argument("--num-samples", type=int, default=128)
    ap.add_argument("--joint-samples", type=int, default=256)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    # The checkpoint may have been trained with heads dropped; rebind the head
    # constants before building anything, since module-level imports captured
    # the default set at import time.
    import shareable_aion_flow.multitarget as _mt
    _mt.configure_heads(tuple(ckpt.get("config", {}).get("drop_heads", ()) or ()))
    HEAD_NAMES, N_HEADS, N_TARGETS, JOINT_IDX = _mt.HEAD_NAMES, _mt.N_HEADS, _mt.N_TARGETS, _mt.JOINT_IDX
    print(f"[eval] heads: {HEAD_NAMES}", flush=True)
    encoder = AIONTokenEncoder(freeze=False, cls_mode=True, cls_variant="readonly",
                               num_cls=N_HEADS).to(device)
    missing, unexpected = encoder.load_state_dict(ckpt["encoder_trainable_state_dict"], strict=False)
    assert not unexpected, f"unexpected encoder keys: {unexpected[:4]}"
    head = SharedCLSHead().to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    flows = MultiTargetFlows().to(device)
    flows.load_state_dict(ckpt["flows_state_dict"])
    standardizers = [TargetStandardizer.from_state_dict(s) for s in ckpt["standardizers"]]
    encoder.eval(); head.eval(); flows.eval()

    _, _, test_loader = build_dataloaders(
        staged_dir=args.staged_dir, target_name="log_ml_flux_1",
        batch_size=args.batch_size, eval_batch_size=args.batch_size,
        num_workers=8, clean_split_csv=args.clean_split_csv,
    )
    lookup = MultiTargetLookup(args.staged_dir, args.extra_targets_csv)
    assign = pd.read_csv(args.clean_split_csv)
    train_tids = assign.loc[assign.split == "train", "targetid"].to_numpy(np.int64)
    train_y = lookup.values_for(train_tids)
    priors = []
    for j in range(N_TARGETS):
        vals = train_y[:, j][np.isfinite(train_y[:, j])]
        priors.append(KDEPrior(standardizers[j].transform_numpy(vals), bw_method="scott"))

    out_dir = args.checkpoint.parent / "eval"
    out_dir.mkdir(exist_ok=True)
    combos = all_nonempty_modality_combos()
    rows = []
    hr_records = []

    with torch.no_grad():
        for combo in combos:
            name = "+".join(m for m in MODALITIES if m in combo)
            acc = {j: {"nll": 0.0, "plp": 0.0, "n": 0, "pred": [], "true": []} for j in range(N_TARGETS)}
            jacc = {"nll": 0.0, "n": 0}
            for batch in test_loader:
                batch = tuple(t.to(device, non_blocking=True) for t in batch)
                y, _, _ = lookup.batch(batch[7], device)
                cls_seq, _ = encoder.encode_tokens(batch, combo)
                ctx = head(cls_seq)
                for j in range(N_TARGETS):
                    mask = torch.isfinite(y[:, j])
                    if not bool(mask.any()):
                        continue
                    ys = standardizers[j].transform_tensor(y[mask, j])
                    lp = flows.flows[j].log_prob(ys, ctx[mask, j])
                    plp = priors[j].log_prob_tensor(ys)
                    samp = flows.flows[j].sample(ctx[mask, j], num_samples=args.num_samples)
                    m = int(mask.sum())
                    acc[j]["nll"] += float(-lp.mean()) * m
                    acc[j]["plp"] += float(plp.mean()) * m
                    acc[j]["n"] += m
                    acc[j]["pred"].append(samp.mean(dim=0).cpu().numpy())
                    acc[j]["true"].append(ys.cpu().numpy())
                # joint head
                j2, j3 = JOINT_IDX
                jm = torch.isfinite(y[:, j2]) & torch.isfinite(y[:, j3])
                if bool(jm.any()):
                    pair = torch.stack([standardizers[j2].transform_tensor(y[jm, j2]),
                                        standardizers[j3].transform_tensor(y[jm, j3])], dim=-1)
                    jlp = flows.joint.distribution(ctx[jm, N_TARGETS]).log_prob(pair)
                    jacc["nll"] += float(-jlp.mean()) * int(jm.sum())
                    jacc["n"] += int(jm.sum())
                    if name == "spectra+z+wise+image":
                        js = flows.joint.distribution(ctx[jm, N_TARGETS]).sample((args.joint_samples,))
                        lf2 = js[..., 0].cpu().numpy() * standardizers[j2].std + standardizers[j2].mean
                        lf3 = js[..., 1].cpu().numpy() * standardizers[j3].std + standardizers[j3].mean
                        rho = 10.0 ** ((lf3 - C_P3) - (lf2 - C_P2))
                        hr = (rho - 1.0) / (rho + 1.0)
                        p16, p50, p84 = np.percentile(hr, [16, 50, 84], axis=0)
                        tids = batch[7][jm].cpu().numpy()
                        for t, a, b, c in zip(tids, p16, p50, p84):
                            hr_records.append({"targetid": int(t), "hr_p16": a, "hr_p50": b, "hr_p84": c})
            for j in range(N_TARGETS):
                n = acc[j]["n"]
                if n == 0:
                    continue
                yp = np.concatenate(acc[j]["pred"]); yt = np.concatenate(acc[j]["true"])
                ss = np.sum((yt - yt.mean()) ** 2)
                rows.append({
                    "head": HEAD_NAMES[j], "input_group": name, "n_test": n,
                    "nll": acc[j]["nll"] / n,
                    "r2": 1.0 - float(np.sum((yt - yp) ** 2)) / ss if ss > 0 else np.nan,
                    "rmse_dex": float(np.sqrt(np.mean((yt - yp) ** 2))) * standardizers[j].std,
                    "info_gain_nats": -(acc[j]["nll"] / n) - (acc[j]["plp"] / n),
                })
            if jacc["n"]:
                rows.append({"head": "p2xp3_joint", "input_group": name, "n_test": jacc["n"],
                             "nll": jacc["nll"] / jacc["n"], "r2": np.nan, "rmse_dex": np.nan,
                             "info_gain_nats": np.nan})
            print(f"[eval] combo {name} done", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "multi_test_metrics.csv", index=False)
    allin = table[table.input_group == "spectra+z+wise+image"]
    print(allin.to_string(index=False))

    hr_df = pd.DataFrame(hr_records)
    hr_df.to_csv(out_dir / "hr_joint_posteriors.csv", index=False)
    if args.hr_ref_csv is not None and len(hr_df):
        ref = pd.read_csv(args.hr_ref_csv).drop_duplicates("targetid").set_index("targetid")
        ref = ref.reindex(hr_df.targetid.values)
        meas, ok = ref.hr32.values, ref.hr32_ok.astype(bool).values
        pred = hr_df.hr_p50.values
        width = 0.5 * (hr_df.hr_p84.values - hr_df.hr_p16.values)
        print(f"\n[HR-from-JOINT] n={len(hr_df)}  median 68% half-width {np.median(width):.3f} "
              f"(independent-bands baseline: 0.551)")
        for label, m in [("all finite", np.isfinite(meas)), ("hr32_ok", ok & np.isfinite(meas))]:
            if m.sum() < 30:
                continue
            r = np.corrcoef(pred[m], meas[m])[0, 1]
            r2 = 1 - np.sum((meas[m] - pred[m]) ** 2) / np.sum((meas[m] - meas[m].mean()) ** 2)
            print(f"  {label:12s} n={m.sum():5d} corr={r:+.3f} (baseline +0.10)  R2={r2:+.3f}")
        qs = np.percentile(pred, [10, 25, 50, 75, 90])
        print("  population HR quantiles p10..p90: " + " ".join(f"{q:+.3f}" for q in qs))
    print(f"written to {out_dir}")


if __name__ == "__main__":
    main()
