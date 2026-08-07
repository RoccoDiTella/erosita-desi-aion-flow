#!/usr/bin/env python
"""Throughput vs batch size probe: a few timed train steps per config, no epochs.

Loads ONE large batch into GPU memory, then for each (architecture, combo,
batch-size) slice runs warmup + timed forward/backward/optimizer steps and
reports samples/sec, sec/step, and peak VRAM. The codec encode and inject
draws run inside each timed step, so the numbers are end-to-end minus
dataloader I/O (which overlaps compute in real training). OOM at a config is
recorded and the sweep continues.

    python scripts/throughput_probe.py --staged-dir ... --clean-split-csv ... \
        [--batch-sizes 56 112 224 448 896 1792] [--timed-steps 5]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shareable_aion_flow.attention_pooling_head import MODALITIES  # noqa: E402
from shareable_aion_flow.data_to_aion_embeddings import build_dataloaders  # noqa: E402
from shareable_aion_flow.main import _OOM_ERROR, batch_nll, build_model  # noqa: E402
from shareable_aion_flow.normalizing_flow import KDEPrior, TargetStandardizer  # noqa: E402


def probe_config(*, encoder, context_encoder, flow, standardizer, batch, combo,
                 bs, optimizer, warmup, timed, error_mode="inject") -> dict:
    sub = tuple(t[:bs] for t in batch)
    torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(warmup + timed):
            if step == warmup:
                torch.cuda.synchronize()
                t0 = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss = batch_nll(
                encoder=encoder, context_encoder=context_encoder, flow=flow,
                batch=sub, combo=combo, standardizer=standardizer,
                error_mode=error_mode, inject_samples=8,
            )
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        dt = (time.monotonic() - t0) / timed
        return {"sec_per_step": dt, "samples_per_sec": bs / dt,
                "peak_gb": torch.cuda.max_memory_allocated() / 2**30, "status": "ok"}
    except _OOM_ERROR:
        torch.cuda.empty_cache()
        return {"sec_per_step": float("nan"), "samples_per_sec": float("nan"),
                "peak_gb": float("nan"), "status": "OOM"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--staged-dir", type=Path, required=True)
    ap.add_argument("--clean-split-csv", type=Path, required=True)
    ap.add_argument("--batch-sizes", type=int, nargs="+",
                    default=[56, 112, 224, 448, 896, 1792])
    ap.add_argument("--warmup-steps", type=int, default=2)
    ap.add_argument("--timed-steps", type=int, default=5)
    ap.add_argument("--v3b-max-bs", type=int, default=896)
    args = ap.parse_args()

    device = torch.device("cuda")
    max_bs = max(args.batch_sizes)
    train_loader, _, _ = build_dataloaders(
        staged_dir=args.staged_dir, target_name=None,
        batch_size=max_bs, num_workers=8, seed=0,
        clean_split_csv=args.clean_split_csv,
    )
    batch = next(iter(train_loader))
    batch = tuple(t.to(device) for t in batch)
    print(f"loaded probe batch: {batch[6].shape[0]} sources", flush=True)
    standardizer = TargetStandardizer.fit(batch[6].detach().cpu().numpy())

    header = f"{'arch':10s} {'combo':16s} {'bs':>5s} {'s/step':>8s} {'samp/s':>8s} {'peakGB':>7s} {'status':>6s}"
    print(header, flush=True)

    # ---- V_simple (frozen encoder, minimal attention head)
    encoder, head, flow = build_model(
        device, dropout=0.05,
        head={"num_queries": 1, "num_layers": 1, "context_hidden": [128], "context_dim": 256},
    )
    opt = torch.optim.AdamW(list(head.parameters()) + list(flow.parameters()), lr=1e-4)
    for combo_name, combo in [("all_inputs", tuple(MODALITIES)),
                              ("spectra", ("spectra",)),
                              ("z+wise", ("z", "wise"))]:
        for bs in args.batch_sizes:
            r = probe_config(encoder=encoder, context_encoder=head, flow=flow,
                             standardizer=standardizer, batch=batch, combo=combo,
                             bs=bs, optimizer=opt, warmup=args.warmup_steps,
                             timed=args.timed_steps)
            print(f"{'V_simple':10s} {combo_name:16s} {bs:5d} {r['sec_per_step']:8.3f} "
                  f"{r['samples_per_sec']:8.1f} {r['peak_gb']:7.1f} {r['status']:>6s}", flush=True)
    del encoder, head, flow, opt
    torch.cuda.empty_cache()

    # ---- V3b (read-only CLS, current implementation)
    encoder, head, flow = build_model(
        device, dropout=0.05, head=None, head_type="cls", cls_variant="readonly",
        lora_rank=8, grad_checkpoint=True,
    )
    params = ([encoder.cls_token] + list(encoder.cls_read_adapters.parameters())
              + list(head.parameters()) + list(flow.parameters()))
    opt = torch.optim.AdamW(params, lr=1e-4)
    for bs in [b for b in args.batch_sizes if b <= args.v3b_max_bs]:
        r = probe_config(encoder=encoder, context_encoder=head, flow=flow,
                         standardizer=standardizer, batch=batch,
                         combo=tuple(MODALITIES), bs=bs, optimizer=opt,
                         warmup=args.warmup_steps, timed=args.timed_steps)
        print(f"{'V3b_cls':10s} {'all_inputs':16s} {bs:5d} {r['sec_per_step']:8.3f} "
              f"{r['samples_per_sec']:8.1f} {r['peak_gb']:7.1f} {r['status']:>6s}", flush=True)
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
