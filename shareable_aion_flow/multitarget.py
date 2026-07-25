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
_ALL_TARGETS = list(MULTI_TARGETS)
N_TARGETS = len(MULTI_TARGETS)          # scalar heads (P4 droppable)
JOINT_PAIR = ("log_flux_p2", "log_flux_p3")  # 8th head: joint 2-D flow for HR
JOINT_IDX = tuple(next(j for j, t in enumerate(MULTI_TARGETS) if t["name"] == n) for n in JOINT_PAIR)
N_HEADS = N_TARGETS + 1
HEAD_NAMES = [t["name"] for t in MULTI_TARGETS] + ["p2xp3_joint"]


def configure_heads(drop: tuple[str, ...] = ()) -> None:
    """Drop scalar heads (e.g. P4, which never learns) before building the model.

    Rebinds the module-level head configuration; call once at startup, before
    anything reads N_TARGETS / N_HEADS / HEAD_NAMES.
    """
    global MULTI_TARGETS, N_TARGETS, JOINT_IDX, N_HEADS, HEAD_NAMES
    MULTI_TARGETS = [t for t in _ALL_TARGETS if t["name"] not in set(drop)]
    N_TARGETS = len(MULTI_TARGETS)
    missing = [n for n in JOINT_PAIR if n not in {t["name"] for t in MULTI_TARGETS}]
    if missing:
        raise ValueError(f"cannot drop {missing}: the joint head needs both bands")
    JOINT_IDX = tuple(next(j for j, t in enumerate(MULTI_TARGETS) if t["name"] == n)
                      for n in JOINT_PAIR)
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
    inject_samples: int = 50,
    return_terms: bool = False,
) -> tuple[torch.Tensor, list[float | None]]:
    """Weighted joint NLL over available (source, target) pairs.

    Returns (total_loss, per-target UNweighted mean losses for the EMA update).
    """
    total = contexts.new_zeros(())
    raw: list[float | None] = []
    terms: list[torch.Tensor | None] = []

    def _std_inject(j: int, mask: torch.Tensor) -> torch.Tensor:
        """Standardized targets as draw stacks: [k, m] with k=1 when not injecting.

        Broadcast injection (user design): k independent split-normal draws per
        source, all evaluated under one context-conditioned distribution.
        Sources without sigma get zero perturbation in every draw.
        """
        std = standardizers[j]
        y = std.transform_tensor(targets[mask, j])
        if not inject:
            return y.unsqueeze(0)
        k = max(1, int(inject_samples))
        lo = (sig_lo[mask, j].abs() / std.std).clamp_min(0.0)
        hi = (sig_hi[mask, j].abs() / std.std).clamp_min(0.0)
        has = ((lo + hi) > 1e-8).unsqueeze(0)
        lo_k = lo.clamp_min(1e-6).unsqueeze(0).expand(k, -1)
        hi_k = hi.clamp_min(1e-6).unsqueeze(0).expand(k, -1)
        eps = sample_split_normal(lo_k, hi_k) * has
        return y.unsqueeze(0) + eps

    B = targets.shape[0]
    for j, flow in enumerate(flows.flows):
        mask = torch.isfinite(targets[:, j])
        if not bool(mask.any()):
            raw.append(None)
            terms.append(None)
            continue
        nll = -flow.log_prob_draws(_std_inject(j, mask), contexts[mask, j]).mean()
        raw.append(float(nll.item()))
        terms.append(nll)
        # availability weighting (user): sources without this target do not
        # count, so the head's effective weight in the batch is its coverage.
        total = total + float(weights[j]) * float(mask.float().mean()) * nll

    # Joint (P2, P3) head: only sources with BOTH bands available; per-band
    # injection is independent (band counts are Poisson-independent).
    j2, j3 = JOINT_IDX
    mask = torch.isfinite(targets[:, j2]) & torch.isfinite(targets[:, j3])
    if bool(mask.any()):
        pair = torch.stack([_std_inject(j2, mask), _std_inject(j3, mask)], dim=-1)
        nll = -flows.joint.log_prob_draws(pair, contexts[mask, N_TARGETS]).mean()
        raw.append(float(nll.item()))
        terms.append(nll)
        total = total + float(weights[N_TARGETS]) * float(mask.float().mean()) * nll
    else:
        raw.append(None)
        terms.append(None)
    return (total, raw, terms) if return_terms else (total, raw)


# ---------------------------------------------------------------- diagnostics
def param_diagnostic_groups(encoder, head, flows) -> dict[str, list[torch.Tensor]]:
    """Named parameter groups for LR / weight-decay diagnostics.

    Adapters are split by encoder depth so a per-depth learning-rate decision
    has evidence behind it.
    """
    groups: dict[str, list[torch.Tensor]] = {"cls_tokens": [encoder.cls_token]}
    adapters = list(encoder.cls_read_adapters)
    n = len(adapters)
    for label, lo, hi in (("adapters_low", 0, n // 3),
                          ("adapters_mid", n // 3, 2 * n // 3),
                          ("adapters_high", 2 * n // 3, n)):
        params = [p for a in adapters[lo:hi] for p in a.parameters()]
        if params:
            groups[label] = params
    groups["shared_mlp"] = list(head.parameters())
    for j, f in enumerate(flows.flows):
        groups[f"flow_{HEAD_NAMES[j]}"] = list(f.parameters())
    groups["flow_p2xp3_joint"] = list(flows.joint.parameters())
    return groups


@torch.no_grad()
def group_snapshot(groups: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.cat([p.detach().reshape(-1) for p in v]).clone() for k, v in groups.items()}


@torch.no_grad()
def group_metrics(groups, prev: dict[str, torch.Tensor] | None) -> dict[str, float]:
    """Weight norm, grad norm, and relative movement since the last snapshot.

    ``move`` (|w_t - w_{t-k}| / |w_t|) is the update-to-weight ratio aggregated
    over k steps: groups moving orders of magnitude faster or slower than the
    rest are the candidates for a different learning rate; ``wnorm`` drifting
    up means weight decay is not binding.
    """
    out: dict[str, float] = {}
    for name, params in groups.items():
        flat = torch.cat([p.detach().reshape(-1) for p in params])
        wn = float(flat.norm())
        out[f"wnorm/{name}"] = wn
        gs = [p.grad.detach().reshape(-1) for p in params if p.grad is not None]
        if gs:
            out[f"gnorm/{name}"] = float(torch.cat(gs).norm())
        if prev is not None and name in prev and wn > 0:
            out[f"move/{name}"] = float((flat - prev[name]).norm() / wn)
    return out


def head_influence(loss_terms, contexts) -> dict[str, float]:
    """How hard each head pulls on the SHARED representation.

    d(head loss)/d(context) at the shared/private boundary: one small backward
    per head through its flow only. This is the honest measure of a head's
    influence on the shared trunk -- loss weight is not it, since a converged
    head has small gradients whatever its weight.
    """
    out: dict[str, float] = {}
    for j, term in enumerate(loss_terms):
        if term is None:
            continue
        g = torch.autograd.grad(term, contexts, retain_graph=True, allow_unused=True)[0]
        if g is None:
            continue
        out[f"influence/{HEAD_NAMES[j]}"] = float(g.detach().norm())
    tot = sum(out.values())
    if tot > 0:
        out.update({k.replace("influence/", "influence_share/"): v / tot for k, v in list(out.items())})
    return out


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

    configure_heads(tuple(args.drop_heads or ()))
    if args.drop_heads:
        print(f"[multi] dropped heads: {list(args.drop_heads)} -> {N_HEADS} heads", flush=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = Path(args.output_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Validation forwards are all-inputs at full length; cap their batch so the
    # attention transient (B x H x N^2) stays bounded regardless of loader size.
    eval_bs = min(args.batch_size, args.bucket_chunk if args.bucketed else 224)
    train_loader, val_loader, _ = build_dataloaders(
        staged_dir=Path(args.staged_dir), target_name="log_ml_flux_1",
        batch_size=args.batch_size, eval_batch_size=eval_bs,
        num_workers=args.num_workers, seed=args.seed,
        clean_split_csv=Path(args.clean_split_csv) if args.clean_split_csv else None,
    )
    # Train probe: a FIXED slice of the training split, scored with the exact
    # validation protocol, so train and val curves measure the same estimand
    # and their gap is the generalization gap.
    from torch.utils.data import DataLoader, Subset
    probe_n = min(int(args.train_probe_size), len(train_loader.dataset))
    probe_loader = DataLoader(
        Subset(train_loader.dataset, list(range(probe_n))),
        batch_size=eval_bs, shuffle=False, num_workers=args.num_workers, pin_memory=True,
    ) if probe_n > 0 else None
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
        "inject": not args.no_inject, "inject_samples": int(args.inject_samples),
        "grad_checkpoint": args.grad_checkpoint,
        "bucketed": bool(args.bucketed),
        "accumulate_buckets": bool(args.accumulate_buckets),
        "drop_heads": list(args.drop_heads or []),
        "heads": HEAD_NAMES,
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

    diag_groups = param_diagnostic_groups(encoder, head, flows)
    prev_snapshot = None
    best = float("inf"); best_epoch = 0; global_step = 0
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
                bucket_names = []
                for bucket, idx in bucket_assignments(combos_ps):
                    # Chunk cap: bucket shares are uneven (the heavy bucket is
                    # ~46% of the size-stratified mix), so cap each forward at
                    # the calibrated size instead of trusting the bucket split.
                    for lo in range(0, len(idx), args.bucket_chunk):
                        part = idx[lo : lo + args.bucket_chunk]
                        rows = torch.from_numpy(part).to(device)
                        sub = tuple(t[rows] for t in batch)
                        drop = {k: v.to(device) for k, v in
                                bucket_modality_dropout(bucket, combos_ps, part).items()}
                        steps.append((bucket["union"], sub, drop, rows))
                        bucket_names.append(bucket["name"])
            else:
                steps = [(sampler.sample(generator), batch, None, None)]
                bucket_names = ["all"]
            # With --accumulate-buckets every optimizer step sees the FULL combo
            # mix (gradients summed over all buckets) instead of alternating
            # between combo families: fewer, much lower-variance updates.
            accumulate = bool(args.accumulate_buckets)
            if accumulate:
                optimizer.zero_grad(set_to_none=True)
            n_steps = len(steps)
            want_diag = (global_step % args.diag_every == 0)
            for si, (combo, sub, drop, rows) in enumerate(steps):
                if rows is None:
                    y, slo, shi = y_all, slo_all, shi_all
                else:
                    y, slo, shi = y_all[rows], slo_all[rows], shi_all[rows]
                cls_seq, _ = encoder.encode_tokens(sub, tuple(combo), modality_dropout=drop)
                contexts = head(cls_seq)
                loss, raw, terms = multi_target_nll(
                    contexts=contexts, flows=flows, targets=y, sig_lo=slo, sig_hi=shi,
                    standardizers=standardizers, weights=weights, inject=not args.no_inject,
                    inject_samples=args.inject_samples, return_terms=True,
                )
                if want_diag and si == 0:
                    diag_extra.update(head_influence(terms, contexts))
                if not accumulate:
                    optimizer.zero_grad(set_to_none=True)
                (loss / n_steps if accumulate else loss).backward()
                nb = int(sub[6].shape[0])
                bname = bucket_names[si] if si < len(bucket_names) else "mixed"
                for name, val in zip(HEAD_NAMES, raw):
                    if val is not None:
                        ep_sum[name] += val * nb; ep_cnt[name] += nb
                        bk_sum[(bname, name)] = bk_sum.get((bname, name), 0.0) + val * nb
                        bk_cnt[(bname, name)] = bk_cnt.get((bname, name), 0) + nb
                n_seen += nb
                if not accumulate:
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 5.0)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    weights = ema.update_and_weights(raw)
                    global_step += 1
                last_raw, last_loss = raw, float(loss.item())
            if accumulate:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 5.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                weights = ema.update_and_weights(last_raw)
                global_step += 1
            if tracker.enabled and want_diag:
                payload = {"train/weighted_loss": last_loss,
                           "train/grad_norm": float(grad_norm),
                           "train/lr": float(optimizer.param_groups[0]["lr"]), "epoch": epoch}
                payload.update(diag_extra)
                payload.update(group_metrics(diag_groups, prev_snapshot))
                prev_snapshot = group_snapshot(diag_groups)
                payload.update({f"weight/{n}": float(w) for n, w in zip(HEAD_NAMES, weights)})
                tracker.log(payload, step=global_step)
                diag_extra = {}

        def evaluate(loader):
            """Plain-LL NLL per head on the all-inputs combo (the val protocol)."""
            sums_ = np.zeros(N_HEADS); counts_ = np.zeros(N_HEADS)
            with torch.no_grad():
                for b in loader:
                    b = tuple(t.to(device, non_blocking=True) for t in b)
                    yv, slov, shiv = lookup.batch(b[7], device)
                    cs, _ = encoder.encode_tokens(b, ("spectra", "z", "wise", "image"))
                    _, raw_ = multi_target_nll(
                        contexts=head(cs), flows=flows, targets=yv, sig_lo=slov, sig_hi=shiv,
                        standardizers=standardizers, weights=np.ones(N_HEADS), inject=False,
                    )
                    nb = int(b[6].shape[0])
                    for jj, vv in enumerate(raw_):
                        if vv is not None:
                            sums_[jj] += vv * nb; counts_[jj] += nb
            return sums_ / np.maximum(counts_, 1)

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
        val_pair_mean = float(sums.sum() / max(counts.sum(), 1))
        dt = time.monotonic() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        print(f"epoch={epoch} val_sum={val_sum:.3f} " +
              " ".join(f"{n}={v:.3f}" for n, v in zip(HEAD_NAMES, val_nll)) +
              f" ({dt:.0f}s, {n_seen/dt:.1f}/s, {peak:.1f}GB)", flush=True)
        payload = {"epoch": epoch, "val/multi_nll_sum": val_sum,
                   "val/pair_mean_nll": val_pair_mean, "epoch_seconds": dt,
                   "throughput/samples_per_second": n_seen / dt, "vram/peak_gb": peak}
        payload.update({f"val/nll_{n}": float(v) for n, v in zip(HEAD_NAMES, val_nll)})
        # per-epoch train means (count-weighted): smooth convergence signal,
        # still on injected targets so NOT comparable to val
        payload.update({f"trainmean/nll_{n}": ep_sum[n] / max(ep_cnt[n], 1)
                        for n in HEAD_NAMES if ep_cnt[n]})
        # per-bucket series: makes the combo-composition effect explicit
        # instead of aliasing it into the step-level curve
        for (bname, hname), tot in bk_sum.items():
            c = bk_cnt[(bname, hname)]
            if c:
                payload[f"bucket/{bname}/nll_{hname}"] = tot / c
        # train probe: SAME protocol as val, so the gap is the generalization gap
        if probe_loader is not None:
            probe_nll = evaluate(probe_loader)
            payload.update({f"probe/nll_{n}": float(v) for n, v in zip(HEAD_NAMES, probe_nll)})
            payload.update({f"gap/nll_{n}": float(v - p_) for n, v, p_
                            in zip(HEAD_NAMES, val_nll, probe_nll)})
            payload["gap/pair_mean"] = float(np.mean(val_nll - probe_nll))
        payload.update({f"val/r2_{n}": float(v) for n, v in zip(HEAD_NAMES[:N_TARGETS], val_r2) if np.isfinite(v)})
        payload.update({f"ema/weight_{n}": float(w) for n, w in zip(HEAD_NAMES, weights)})
        tracker.log(payload, step=global_step)
        save(run_dir / "last.pt", epoch, val_pair_mean)
        if val_pair_mean < best:
            best = val_pair_mean
            best_epoch = epoch
            save(run_dir / "best.pt", epoch, val_pair_mean)
        if epoch - best_epoch >= args.early_stop_patience:
            print(f"[early-stop] no improvement for {args.early_stop_patience} epochs "
                  f"(best epoch {best_epoch})", flush=True)
            break
    tracker.summary({"val/best_pair_mean_nll": best, "best_epoch": best_epoch})
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
