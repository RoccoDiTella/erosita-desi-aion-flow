#!/usr/bin/env python
"""V3 (CLS + LoRA) smoke grid: pick rank, depth K, and batch size before the real run.

For each (lora_rank, top-K fraction) config: build the cls-mode model, run
~50 training steps on the real clean-view loader with the standard combo mix,
and record median step time, peak GPU memory, loss slope (first-10 vs last-10
mean NLL), and per-group gradient norms. Falls back to batch 256 on OOM.
Writes cls_smoke_results.csv next to the log.

    python scripts/cls_smoke.py --staged-dir <staged_paper> --clean-split-csv <csv> [--steps 50]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shareable_aion_flow.attention_pooling_head import ComboSampler  # noqa: E402
from shareable_aion_flow.data_to_aion_embeddings import build_dataloaders, read_view_target_values  # noqa: E402
from shareable_aion_flow.main import batch_nll, build_model  # noqa: E402
from shareable_aion_flow.normalizing_flow import TargetStandardizer  # noqa: E402

RANKS = (0, 4, 8, 16)  # 0 = CLS + head + flow only (no LoRA), the baseline
K_FRACS = (0.25, 0.5, 1.0)


def run_config(rank, k_frac, loader, standardizer, device, steps, lr, encoder_lr, llrd_gamma):
    encoder, context_encoder, flow = build_model(
        device, dropout=0.05, head={"num_queries": 1, "num_layers": 1, "context_hidden": [128]},
        head_type="cls", lora_rank=rank,
        lora_blocks=0, grad_checkpoint=True,
    )
    depth = len(encoder.backbone.encoder)
    k = max(1, int(round(k_frac * depth))) if rank > 0 else 0
    if rank > 0:
        # rebuild with the resolved absolute K
        del encoder, context_encoder, flow
        torch.cuda.empty_cache()
        encoder, context_encoder, flow = build_model(
            device, dropout=0.05, head={"num_queries": 1, "num_layers": 1, "context_hidden": [128]},
            head_type="cls", lora_rank=rank, lora_blocks=k, grad_checkpoint=True,
        )
    groups = [{"params": list(context_encoder.parameters()) + list(flow.parameters()), "lr": lr},
              {"params": [encoder.cls_token], "lr": lr}]
    groups += encoder.encoder_param_groups(encoder_lr, llrd_gamma)
    optimizer = torch.optim.AdamW(groups, lr=lr, weight_decay=1e-4)
    trainable = [p for g in groups for p in g["params"]]
    n_lora = sum(p.numel() for g in groups[2:] for p in g["params"])

    sampler = ComboSampler.default()
    generator = torch.Generator(); generator.manual_seed(0)
    encoder.eval(); context_encoder.train(); flow.train()
    torch.cuda.reset_peak_memory_stats()
    losses, times = [], []
    it = iter(loader)
    for step in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)
        batch = tuple(t.to(device, non_blocking=True) for t in batch)
        t0 = time.monotonic()
        loss = batch_nll(
            encoder=encoder, context_encoder=context_encoder, flow=flow, batch=batch,
            combo=sampler.sample(generator), standardizer=standardizer,
            error_mode="inject", inject_samples=8,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 5.0)
        optimizer.step()
        torch.cuda.synchronize()
        times.append(time.monotonic() - t0)
        losses.append(float(loss.item()))
    result = {
        "rank": rank, "k_blocks": k, "depth": depth, "n_lora_params": n_lora,
        "step_time_s": float(np.median(times[3:])),
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
        "loss_first10": float(np.mean(losses[:10])),
        "loss_last10": float(np.mean(losses[-10:])),
    }
    del encoder, context_encoder, flow, optimizer
    torch.cuda.empty_cache()
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--staged-dir", type=Path, required=True)
    ap.add_argument("--clean-split-csv", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=448)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--encoder-lr", type=float, default=2e-5)
    ap.add_argument("--llrd-gamma", type=float, default=0.8)
    ap.add_argument("--out", type=Path, default=Path("cls_smoke_results.csv"))
    args = ap.parse_args()

    device = torch.device("cuda")
    rows = []
    bs = args.batch_size
    train_loader, _, _ = build_dataloaders(
        staged_dir=args.staged_dir, target_name="log_ml_flux_1", batch_size=bs,
        num_workers=8, seed=0, clean_split_csv=args.clean_split_csv,
    )
    tv = read_view_target_values(args.staged_dir, "log_ml_flux_1", args.clean_split_csv, "train")
    standardizer = TargetStandardizer.fit(tv)
    configs = [(r, kf) for r in RANKS for kf in (K_FRACS if r > 0 else (0.0,))]
    for rank, k_frac in configs:
        try:
            row = run_config(rank, k_frac, train_loader, standardizer, device,
                             args.steps, args.lr, args.encoder_lr, args.llrd_gamma)
            row["batch_size"] = bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"rank={rank} k_frac={k_frac}: OOM at bs {bs}, retrying at 256", flush=True)
            small_loader, _, _ = build_dataloaders(
                staged_dir=args.staged_dir, target_name="log_ml_flux_1", batch_size=256,
                num_workers=8, seed=0, clean_split_csv=args.clean_split_csv,
            )
            row = run_config(rank, k_frac, small_loader, standardizer, device,
                             args.steps, args.lr, args.encoder_lr, args.llrd_gamma)
            row["batch_size"] = 256
        print(row, flush=True)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
