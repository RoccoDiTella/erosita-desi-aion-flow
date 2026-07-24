"""Multi-target read-only-CLS training: 8 heads, one frozen forward.

Heads: flux, Lx, logmstar, p1..p4 band fluxes (runtime sidecar) as 1-D flows,
plus a JOINT 2-D flow over (P2, P3) whose samples give per-source hardness
posteriors with correlated band errors (HR itself is never a direct target). Architecture: per-target CLS
vectors + SHARED per-block Q/V read adapters (data stream frozen, no_grad),
one SHARED 768->512->256 MLP over the stacked CLS states, one small NSF flow
per target. Joint loss: per-target NLL on standardized values, split-normal
noise injection where sigma exists, per-source availability masks, and
detached-EMA loss normalization so harder targets do not dominate gradients.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

try:
    from .data_to_aion_embeddings import read_dataset
    from .normalizing_flow import ConditionalNSFFlow, TargetStandardizer, sample_split_normal
except ImportError:
    from data_to_aion_embeddings import read_dataset
    from normalizing_flow import ConditionalNSFFlow, TargetStandardizer, sample_split_normal

# name, sigma columns (None = no error model), availability sigma gate
MULTI_TARGETS: list[dict] = [
    {"name": "log_ml_flux_1", "sig": ("flux_sig_lo", "flux_sig_hi"), "max_sigma": None, "sidecar": False},
    {"name": "log_lx", "sig": ("flux_sig_lo", "flux_sig_hi"), "max_sigma": None, "sidecar": False},
    {"name": "logmstar", "sig": None, "max_sigma": None, "sidecar": False},
    {"name": "log_flux_p1", "sig": ("log_flux_p1_sig_lo", "log_flux_p1_sig_hi"), "max_sigma": 1.0, "sidecar": True},
    {"name": "log_flux_p2", "sig": ("log_flux_p2_sig_lo", "log_flux_p2_sig_hi"), "max_sigma": 1.0, "sidecar": True},
    {"name": "log_flux_p3", "sig": ("log_flux_p3_sig_lo", "log_flux_p3_sig_hi"), "max_sigma": 1.0, "sidecar": True},
    {"name": "log_flux_p4", "sig": ("log_flux_p4_sig_lo", "log_flux_p4_sig_hi"), "max_sigma": 1.0, "sidecar": True},
]
N_TARGETS = len(MULTI_TARGETS)          # 7 scalar heads
JOINT_PAIR = ("log_flux_p2", "log_flux_p3")  # 8th head: joint 2-D flow for HR
JOINT_IDX = tuple(next(j for j, t in enumerate(MULTI_TARGETS) if t["name"] == n) for n in JOINT_PAIR)
N_HEADS = N_TARGETS + 1
HEAD_NAMES = [t["name"] for t in MULTI_TARGETS] + ["p2xp3_joint"]


def load_multi_target_matrix(
    staged_path: Path, extra_targets_csv: Path | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(targets [n,7], sig_lo [n,7], sig_hi [n,7]) aligned to a staged split file.

    Unavailable entries are NaN in ``targets`` (masked in the loss); missing
    sigmas are 0 (no injection). Sidecar targets join by targetid.
    """
    import pandas as pd

    with h5py.File(staged_path, "r") as handle:
        n = int(handle["desi_targetid"].shape[0])
        tids = handle["desi_targetid"][:].astype(np.int64)
        store_cols = {}
        for spec in MULTI_TARGETS:
            cols = [spec["name"]] + (list(spec["sig"]) if spec["sig"] else [])
            for c in cols:
                if not spec["sidecar"] and c in handle:
                    store_cols[c] = read_dataset(handle, c).astype(np.float64)
    side = None
    if extra_targets_csv is not None:
        side = pd.read_csv(extra_targets_csv).drop_duplicates("targetid").set_index("targetid")
        side = side.reindex(tids)

    y = np.full((n, N_TARGETS), np.nan)
    slo = np.zeros((n, N_TARGETS))
    shi = np.zeros((n, N_TARGETS))
    for j, spec in enumerate(MULTI_TARGETS):
        if spec["sidecar"]:
            if side is None:
                continue
            def col(c, _side=side):
                return _side[c].to_numpy(np.float64) if c in _side.columns else None
        else:
            def col(c, _store=store_cols):
                return _store.get(c)
        vals = col(spec["name"])
        if vals is None:
            continue
        y[:, j] = vals
        if spec["sig"]:
            lo, hi = col(spec["sig"][0]), col(spec["sig"][1])
            if lo is not None and hi is not None:
                slo[:, j] = np.nan_to_num(np.abs(lo))
                shi[:, j] = np.nan_to_num(np.abs(hi))
        if spec["max_sigma"] is not None:
            mean_sig = 0.5 * (slo[:, j] + shi[:, j])
            bad = ~np.isfinite(mean_sig) | (mean_sig > spec["max_sigma"])
            y[bad, j] = np.nan
    return y, slo, shi


class SharedCLSHead(nn.Module):
    """One shared 768->512->256 MLP applied across all stacked CLS streams."""

    def __init__(self, embed_dim: int = 768, hidden: int = 512, context_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, context_dim),
            nn.LayerNorm(context_dim),
        )
        self.context_dim = context_dim

    def forward(self, cls_states: torch.Tensor) -> torch.Tensor:  # [B, K, 768] -> [B, K, 256]
        return self.net(cls_states)


class MultiTargetFlows(nn.Module):
    """7 scalar 1-D flows + one joint 2-D flow over (P2, P3) for hardness."""

    def __init__(self, context_dim: int = 256, n_targets: int = N_TARGETS) -> None:
        super().__init__()
        self.flows = nn.ModuleList(ConditionalNSFFlow(context_dim=context_dim) for _ in range(n_targets))
        self.joint = ConditionalNSFFlow(context_dim=context_dim, features=2)


class EMALossWeights:
    """Detached EMA of each target's mean loss; weight = 1/EMA (unit-scale grads)."""

    def __init__(self, n_targets: int = N_HEADS, beta: float = 0.98) -> None:
        self.ema = np.ones(n_targets)
        self.beta = float(beta)
        self.seen = np.zeros(n_targets, dtype=bool)

    def update_and_weights(self, losses: list[float | None]) -> np.ndarray:
        for j, val in enumerate(losses):
            if val is None or not np.isfinite(val):
                continue
            if not self.seen[j]:
                self.ema[j] = val
                self.seen[j] = True
            else:
                self.ema[j] = self.beta * self.ema[j] + (1 - self.beta) * val
        return 1.0 / np.clip(np.abs(self.ema), 0.2, None)

    def state_dict(self) -> dict:
        return {"ema": self.ema.tolist(), "seen": self.seen.tolist(), "beta": self.beta}


def multi_target_nll(
    *,
    contexts: torch.Tensor,            # [B, K, 256]
    flows: MultiTargetFlows,
    targets: torch.Tensor,             # [B, K] (NaN = unavailable)
    sig_lo: torch.Tensor,              # [B, K]
    sig_hi: torch.Tensor,
    standardizers: list[TargetStandardizer],
    weights: np.ndarray,               # [K] detached loss weights
    inject: bool = True,
) -> tuple[torch.Tensor, list[float | None]]:
    """Weighted joint NLL over available (source, target) pairs.

    Returns (total_loss, per-target UNweighted mean losses for the EMA update).
    """
    total = contexts.new_zeros(())
    raw: list[float | None] = []

    def _std_inject(j: int, mask: torch.Tensor) -> torch.Tensor:
        std = standardizers[j]
        y = std.transform_tensor(targets[mask, j])
        if inject:
            lo = (sig_lo[mask, j].abs() / std.std).clamp_min(0.0)
            hi = (sig_hi[mask, j].abs() / std.std).clamp_min(0.0)
            has = (lo + hi) > 1e-8
            if bool(has.any()):
                eps = sample_split_normal(lo.clamp_min(1e-6), hi.clamp_min(1e-6))
                y = torch.where(has, y + eps, y)
        return y

    for j, flow in enumerate(flows.flows):
        mask = torch.isfinite(targets[:, j])
        if not bool(mask.any()):
            raw.append(None)
            continue
        nll = -flow.log_prob(_std_inject(j, mask), contexts[mask, j]).mean()
        raw.append(float(nll.item()))
        total = total + float(weights[j]) * nll

    # Joint (P2, P3) head: only sources with BOTH bands available; per-band
    # injection is independent (band counts are Poisson-independent).
    j2, j3 = JOINT_IDX
    mask = torch.isfinite(targets[:, j2]) & torch.isfinite(targets[:, j3])
    if bool(mask.any()):
        pair = torch.stack([_std_inject(j2, mask), _std_inject(j3, mask)], dim=-1)
        nll = -flows.joint.log_prob(pair, contexts[mask, N_TARGETS]).mean()
        raw.append(float(nll.item()))
        total = total + float(weights[N_TARGETS]) * nll
    else:
        raw.append(None)
    return total, raw


class MultiTargetLookup:
    """targetid -> (targets [7], sig_lo [7], sig_hi [7]) across all staged splits."""

    def __init__(self, staged_dir: Path, extra_targets_csv: Path | None) -> None:
        ys, slos, shis, tids = [], [], [], []
        for split in ("train", "val", "test"):
            path = Path(staged_dir) / f"desi_{split}.hdf5"
            y, lo, hi = load_multi_target_matrix(path, extra_targets_csv)
            with h5py.File(path, "r") as handle:
                tids.append(handle["desi_targetid"][:].astype(np.int64))
            ys.append(y); slos.append(lo); shis.append(hi)
        tid = np.concatenate(tids)
        y = np.concatenate(ys); lo = np.concatenate(slos); hi = np.concatenate(shis)
        _, first = np.unique(tid, return_index=True)
        self.index = {int(t): int(i) for i, t in zip(first, tid[first])}
        self.y = torch.from_numpy(y.astype(np.float32))
        self.slo = torch.from_numpy(lo.astype(np.float32))
        self.shi = torch.from_numpy(hi.astype(np.float32))

    def batch(self, targetids: torch.Tensor, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = torch.tensor([self.index[int(t)] for t in targetids.cpu().tolist()], dtype=torch.long)
        return (self.y[rows].to(device), self.slo[rows].to(device), self.shi[rows].to(device))

    def values_for(self, targetids: np.ndarray) -> np.ndarray:
        rows = [self.index[int(t)] for t in targetids if int(t) in self.index]
        return self.y[rows].numpy()


def run_train_multi(args) -> None:
    """Train the 8-head read-only-CLS multi-target model."""
    import json
    import time

    try:
        from .attention_pooling_head import ComboSampler
        from .data_to_aion_embeddings import AIONTokenEncoder, build_dataloaders, write_json
        from .main import _OOM_ERROR  # noqa: F401  (import kept for parity)
        from .tracking import init_tracking
    except ImportError:
        from attention_pooling_head import ComboSampler
        from data_to_aion_embeddings import AIONTokenEncoder, build_dataloaders, write_json
        from tracking import init_tracking

    import pandas as pd

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = Path(args.output_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = build_dataloaders(
        staged_dir=Path(args.staged_dir), target_name="log_ml_flux_1",
        batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed,
        clean_split_csv=Path(args.clean_split_csv) if args.clean_split_csv else None,
    )
    lookup = MultiTargetLookup(Path(args.staged_dir), Path(args.extra_targets_csv) if args.extra_targets_csv else None)
    assign = pd.read_csv(args.clean_split_csv)
    train_tids = assign.loc[assign.split == "train", "targetid"].to_numpy(np.int64)
    train_y = lookup.values_for(train_tids)
    standardizers = []
    for j in range(N_TARGETS):
        vals = train_y[:, j][np.isfinite(train_y[:, j])]
        standardizers.append(TargetStandardizer.fit(vals))
        print(f"[multi] {MULTI_TARGETS[j]['name']}: {len(vals)} train values, "
              f"mean {standardizers[j].mean:.3f} std {standardizers[j].std:.3f}", flush=True)

    encoder = AIONTokenEncoder(
        freeze=False, cls_mode=True, cls_variant="readonly", num_cls=N_HEADS,
        grad_checkpoint=args.grad_checkpoint,
    ).to(device)
    head = SharedCLSHead().to(device)
    flows = MultiTargetFlows().to(device)
    param_groups = [
        {"params": list(head.parameters()) + list(flows.parameters()), "lr": args.lr,
         "weight_decay": args.weight_decay},
        {"params": [encoder.cls_token], "lr": args.lr, "weight_decay": 0.0},
        {"params": list(encoder.cls_read_adapters.parameters()), "lr": args.lr,
         "weight_decay": args.adapter_wd},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    trainable = [p for g in param_groups for p in g["params"]]
    steps_per_epoch = max(1, len(train_loader)) * (len(LENGTH_BUCKETS) if args.bucketed else 1)
    scheduler = None
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=steps_per_epoch * args.epochs)
    sampler = ComboSampler.default()
    generator = torch.Generator(); generator.manual_seed(args.seed)
    ema = EMALossWeights()

    config = {
        "mode": "train-multi", "heads": HEAD_NAMES, "epochs": args.epochs,
        "batch_size": args.batch_size, "lr": args.lr, "lr_schedule": args.lr_schedule,
        "weight_decay": args.weight_decay, "adapter_wd": args.adapter_wd,
        "inject": not args.no_inject, "grad_checkpoint": args.grad_checkpoint,
        "bucketed": bool(args.bucketed),
        "cls_variant": "readonly", "num_cls": N_HEADS, "adapter": "full-rank",
        "standardizers": [s.state_dict() for s in standardizers],
        "clean_split_csv": str(args.clean_split_csv), "extra_targets_csv": str(args.extra_targets_csv),
    }
    write_json(run_dir / "config.json", config)
    tracker = init_tracking(
        enabled=args.wandb, project=args.wandb_project, entity=args.wandb_entity,
        run_name=args.run_id, mode="online", config=config, run_dir=run_dir,
        tags=["multi-target", "v3b"],
    )

    def save(path, epoch, val_sum):
        torch.save({
            "epoch": epoch, "config": config,
            "encoder_trainable_state_dict": encoder.encoder_trainable_state(),
            "head_state_dict": head.state_dict(),
            "flows_state_dict": flows.state_dict(),
            "standardizers": [s.state_dict() for s in standardizers],
            "ema": ema.state_dict(), "val_multi_nll_sum": val_sum,
        }, path)

    best = float("inf"); global_step = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.monotonic()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        encoder.eval(); head.train(); flows.train()
        weights = ema.update_and_weights([None] * N_HEADS)
        n_seen = 0
        for batch in train_loader:
            batch = tuple(t.to(device, non_blocking=True) for t in batch)
            y_all, slo_all, shi_all = lookup.batch(batch[7], device)
            if args.bucketed:
                # Length-bucketed packing: per-source combos, one forward AND
                # one optimizer step per bucket -- each bucket lands near the
                # calibrated per-forward size while the step count stays high.
                B = int(batch[6].shape[0])
                combos_ps = [sampler.sample(generator) for _ in range(B)]
                steps = []
                for bucket, idx in bucket_assignments(combos_ps):
                    rows = torch.from_numpy(idx).to(device)
                    sub = tuple(t[rows] for t in batch)
                    drop = {k: v.to(device) for k, v in
                            bucket_modality_dropout(bucket, combos_ps, idx).items()}
                    steps.append((bucket["union"], sub, drop, rows))
            else:
                steps = [(sampler.sample(generator), batch, None, None)]
            for combo, sub, drop, rows in steps:
                if rows is None:
                    y, slo, shi = y_all, slo_all, shi_all
                else:
                    y, slo, shi = y_all[rows], slo_all[rows], shi_all[rows]
                cls_seq, _ = encoder.encode_tokens(sub, tuple(combo), modality_dropout=drop)
                contexts = head(cls_seq)
                loss, raw = multi_target_nll(
                    contexts=contexts, flows=flows, targets=y, sig_lo=slo, sig_hi=shi,
                    standardizers=standardizers, weights=weights, inject=not args.no_inject,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 5.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                weights = ema.update_and_weights(raw)
                n_seen += int(sub[6].shape[0]); global_step += 1
                if tracker.enabled and global_step % args.log_every == 0:
                    payload = {"train/weighted_loss": float(loss.item()),
                               "train/grad_norm": float(grad_norm),
                               "train/lr": float(optimizer.param_groups[0]["lr"]), "epoch": epoch}
                    for name, val in zip(HEAD_NAMES, raw):
                        if val is not None:
                            payload[f"train/nll_{name}"] = val
                    tracker.log(payload, step=global_step)

        # validation: all-inputs combo, plain NLL (no injection) + posterior-mean
        # R^2 per scalar head so the review does not need a separate eval pass
        encoder.eval(); head.eval(); flows.eval()
        sums = np.zeros(N_HEADS); counts = np.zeros(N_HEADS)
        preds = [[] for _ in range(N_TARGETS)]; trues = [[] for _ in range(N_TARGETS)]
        with torch.no_grad():
            for batch in val_loader:
                batch = tuple(t.to(device, non_blocking=True) for t in batch)
                y, slo, shi = lookup.batch(batch[7], device)
                cls_seq, _ = encoder.encode_tokens(batch, ("spectra", "z", "wise", "image"))
                contexts = head(cls_seq)
                _, raw = multi_target_nll(
                    contexts=contexts, flows=flows, targets=y, sig_lo=slo, sig_hi=shi,
                    standardizers=standardizers, weights=np.ones(N_HEADS), inject=False,
                )
                n = int(batch[6].shape[0])
                for j, val in enumerate(raw):
                    if val is not None:
                        sums[j] += val * n; counts[j] += n
                for j in range(N_TARGETS):
                    mask = torch.isfinite(y[:, j])
                    if bool(mask.any()):
                        samp = flows.flows[j].sample(contexts[mask, j], num_samples=64)
                        preds[j].append(samp.mean(dim=0).cpu().numpy())
                        trues[j].append(standardizers[j].transform_tensor(y[mask, j]).cpu().numpy())
        val_r2 = np.full(N_TARGETS, np.nan)
        for j in range(N_TARGETS):
            if preds[j]:
                yp, yt = np.concatenate(preds[j]), np.concatenate(trues[j])
                ss = np.sum((yt - yt.mean()) ** 2)
                val_r2[j] = 1.0 - np.sum((yt - yp) ** 2) / ss if ss > 0 else np.nan
        val_nll = sums / np.maximum(counts, 1)
        val_sum = float(val_nll.sum())
        dt = time.monotonic() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        print(f"epoch={epoch} val_sum={val_sum:.3f} " +
              " ".join(f"{n}={v:.3f}" for n, v in zip(HEAD_NAMES, val_nll)) +
              f" ({dt:.0f}s, {n_seen/dt:.1f}/s, {peak:.1f}GB)", flush=True)
        payload = {"epoch": epoch, "val/multi_nll_sum": val_sum, "epoch_seconds": dt,
                   "throughput/samples_per_second": n_seen / dt, "vram/peak_gb": peak}
        payload.update({f"val/nll_{n}": float(v) for n, v in zip(HEAD_NAMES, val_nll)})
        payload.update({f"val/r2_{n}": float(v) for n, v in zip(HEAD_NAMES[:N_TARGETS], val_r2) if np.isfinite(v)})
        payload.update({f"ema/weight_{n}": float(w) for n, w in zip(HEAD_NAMES, weights)})
        tracker.log(payload, step=global_step)
        save(run_dir / "last.pt", epoch, val_sum)
        if val_sum < best:
            best = val_sum
            save(run_dir / "best.pt", epoch, val_sum)
    tracker.summary({"val/best_multi_nll_sum": best})
    tracker.finish()


# Length-bucketed combo packing (user design): combos grouped by padded token
# length, not identity. Each bucket encodes the UNION of its combos' modalities
# once; per-source masks drop the rest (padding waste <=1.5%). One optimizer
# step per bucket forward: with a large loader batch, each bucket lands at the
# calibrated per-forward size while keeping the step count of small batches.
LENGTH_BUCKETS: list[dict] = [
    {"name": "tiny", "union": ("z", "wise"),
     "combos": {("z",), ("wise",), ("z", "wise")}},
    {"name": "spectra", "union": ("spectra", "z", "wise"),
     "combos": {("spectra",), ("spectra", "z"), ("spectra", "wise"), ("spectra", "z", "wise")}},
    {"name": "image", "union": ("z", "wise", "image"),
     "combos": {("image",), ("z", "image"), ("wise", "image"), ("z", "wise", "image")}},
    {"name": "heavy", "union": ("spectra", "z", "wise", "image"),
     "combos": {("spectra", "image"), ("spectra", "z", "image"), ("spectra", "wise", "image"),
                ("spectra", "z", "wise", "image")}},
]


def bucket_assignments(
    combos_per_source: list[tuple[str, ...]],
) -> list[tuple[dict, np.ndarray]]:
    """Partition per-source combos into length buckets.

    Returns (bucket_spec, source_indices) for each non-empty bucket; every
    source lands in exactly one bucket.
    """
    out = []
    assigned = np.zeros(len(combos_per_source), dtype=bool)
    for bucket in LENGTH_BUCKETS:
        idx = np.array([i for i, c in enumerate(combos_per_source) if tuple(c) in bucket["combos"]],
                       dtype=np.int64)
        if len(idx):
            out.append((bucket, idx))
            assigned[idx] = True
    if not bool(assigned.all()):
        missing = [combos_per_source[i] for i in np.flatnonzero(~assigned)]
        raise ValueError(f"Combos not covered by any length bucket: {set(missing)}")
    return out


def bucket_modality_dropout(
    bucket: dict, combos: list[tuple[str, ...]], idx: np.ndarray
) -> dict[str, torch.Tensor]:
    """Per-source modality-dropout masks WITHIN a bucket (True = drop)."""
    return {
        group: torch.tensor([group not in combos[i] for i in idx], dtype=torch.bool)
        for group in bucket["union"]
    }
