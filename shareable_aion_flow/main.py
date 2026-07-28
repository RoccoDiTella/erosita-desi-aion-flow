"""Command-line entry point: prepare-data, train, eval, make-table.

Trains and evaluates the clean AION attention-flow model on the eROSITA/DESI
matched sample. Run ``python -m shareable_aion_flow.main --help`` for usage.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

try:
    from .attention_pooling_head import (
        MODALITIES,
        AIONAttentionContext,
        CLSContext,
        ComboSampler,
        all_nonempty_modality_combos,
    )
    from .data_to_aion_embeddings import (
        FITS_POOL_DIR,
        SOURCE_HDF5,
        SPLIT_MANIFEST,
        STAGED_DIR,
        AIONTokenEncoder,
        build_dataloaders,
        prepare_staged_data,
        read_target_values,
        read_view_target_values,
        write_json,
    )
    from .evals import build_results_table, evaluate_all_combos, save_results_table_pdf
    from .normalizing_flow import ConditionalNSFFlow, KDEPrior, TargetStandardizer, sample_split_normal
    from .tracking import init_tracking
except ImportError:
    from attention_pooling_head import (
        MODALITIES,
        AIONAttentionContext,
        CLSContext,
        ComboSampler,
        all_nonempty_modality_combos,
    )
    from data_to_aion_embeddings import (
        FITS_POOL_DIR,
        SOURCE_HDF5,
        SPLIT_MANIFEST,
        STAGED_DIR,
        AIONTokenEncoder,
        build_dataloaders,
        prepare_staged_data,
        read_target_values,
        read_view_target_values,
        write_json,
    )
    from evals import build_results_table, evaluate_all_combos, save_results_table_pdf
    from normalizing_flow import ConditionalNSFFlow, KDEPrior, TargetStandardizer, sample_split_normal
    from tracking import init_tracking


PACKAGE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = PACKAGE_DIR / "outputs"


def timestamp_run_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{prefix}"


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def set_global_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


PAPER_HEAD: dict[str, Any] = {"num_queries": 4, "num_layers": 2, "context_hidden": [512, 512], "context_dim": 256}


def build_model(
    device: torch.device,
    dropout: float,
    head: dict[str, Any] | None = None,
    head_type: str = "attention",
    cls_variant: str = "input",
    lora_rank: int = 0,
    lora_blocks: int = -1,
    lora_modules: str = "all",
    grad_checkpoint: bool = False,
) -> tuple[AIONTokenEncoder, torch.nn.Module, ConditionalNSFFlow]:
    """Build the AION encoder, context head, and conditional flow.

    ``head_type="attention"`` (default): frozen encoder + attention-pooling
    head. ``head_type="cls"`` (V3): trainable CLS token at the encoder input,
    LoRA on the last ``lora_blocks`` blocks, CLS hidden projected to the SAME
    256-dim context feeding an identical flow (equivalent-flow comparison).
    """
    head = {**PAPER_HEAD, **(head or {})}
    if head_type == "cls":
        encoder = AIONTokenEncoder(
            freeze=False, cls_mode=True, cls_variant=cls_variant, lora_rank=lora_rank,
            lora_blocks=lora_blocks, lora_modules=lora_modules,
            grad_checkpoint=grad_checkpoint,
        ).to(device)
        context_encoder = CLSContext(context_dim=int(head["context_dim"])).to(device)
    else:
        encoder = AIONTokenEncoder(freeze=True).to(device)
        context_encoder = AIONAttentionContext(
            dropout=dropout,
            num_queries=int(head["num_queries"]),
            num_layers=int(head["num_layers"]),
            context_hidden=tuple(head["context_hidden"]),
            context_dim=int(head["context_dim"]),
        ).to(device)
    flow = ConditionalNSFFlow(context_dim=context_encoder.context_dim).to(device)
    return encoder, context_encoder, flow


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    context_encoder: AIONAttentionContext,
    flow: ConditionalNSFFlow,
    optimizer: torch.optim.Optimizer | None,
    standardizer: TargetStandardizer,
    config: dict[str, object],
    val_all_inputs_nll: float,
    encoder_trainable: dict[str, torch.Tensor] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "context_encoder_state_dict": context_encoder.state_dict(),
        "flow_state_dict": flow.state_dict(),
        "encoder_trainable_state_dict": encoder_trainable,
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "standardizer": standardizer.state_dict(),
        "config": config,
        "val_all_inputs_nll": float(val_all_inputs_nll),
    }
    torch.save(payload, path)


def load_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
    dropout: float,
) -> tuple[AIONTokenEncoder, AIONAttentionContext, ConditionalNSFFlow, TargetStandardizer, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get("config", {})
    encoder, context_encoder, flow = build_model(
        device, dropout=dropout, head=cfg.get("head"),
        head_type=str(cfg.get("head_type", "attention")),
        cls_variant=str(cfg.get("cls_variant", "input")),
        lora_rank=int(cfg.get("lora_rank", 0) or 0),
        lora_blocks=int(cfg.get("lora_blocks", -1)),
        lora_modules=str(cfg.get("lora_modules", "all")),
    )
    context_encoder.load_state_dict(checkpoint["context_encoder_state_dict"])
    flow.load_state_dict(checkpoint["flow_state_dict"])
    encoder_trainable = checkpoint.get("encoder_trainable_state_dict")
    if encoder_trainable:
        missing, unexpected = encoder.load_state_dict(encoder_trainable, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected encoder-trainable keys: {unexpected[:5]}")
    standardizer = TargetStandardizer.from_state_dict(checkpoint["standardizer"])
    return encoder, context_encoder, flow, standardizer, checkpoint.get("config", {})


def batch_nll(
    *,
    encoder: AIONTokenEncoder,
    context_encoder: AIONAttentionContext,
    flow: ConditionalNSFFlow,
    batch: tuple[torch.Tensor, ...],
    combo: tuple[str, ...],
    standardizer: TargetStandardizer,
    error_mode: str = "none",
    inject_samples: int = 1,
    spectrum_token_mask: torch.Tensor | None = None,
    mask_mode: str = "drop",
    combos_per_source: list[tuple[str, ...]] | None = None,
    _loss_scale: float = 1.0,
) -> torch.Tensor:
    """Negative log-likelihood of one batch under the flow for a given modality combo.

    ``error_mode`` folds in the per-source split-normal measurement error carried
    as ``batch[8]=sig_lo, batch[9]=sig_hi`` (in target units): ``convolve``
    deconvolves it via the convolution likelihood, ``inject`` adds it as noise,
    ``none`` ignores it. Falls back to plain log-prob when errors are absent/zero.
    ``mask_mode`` governs how ``spectrum_token_mask`` removes tokens (drop|replace).
    """
    target = batch[6]
    if not torch.isfinite(target).all():
        bad = int((~torch.isfinite(target)).sum().item())
        raise FloatingPointError(f"Encountered {bad} non-finite target values in a training/eval batch.")
    target_std = standardizer.transform_tensor(target)
    modality_dropout = None
    if combos_per_source is not None:
        # Stratified fixed-composition batch: encode once at the full all-inputs
        # sequence length; each source excludes modalities via the native input
        # mask. Constant per-batch memory, one codec pass, one backbone forward.
        combo = tuple(MODALITIES)
        modality_dropout = {
            group: torch.tensor(
                [group not in c for c in combos_per_source], dtype=torch.bool
            )
            for group in MODALITIES
        }
    tokens, group_ids = encoder.encode_tokens(
        batch, combo, spectrum_token_mask=spectrum_token_mask, mask_mode=mask_mode,
        modality_dropout=modality_dropout,
    )
    context = context_encoder(tokens, group_ids)
    log_prob = _log_prob_with_error(flow, target_std, context, batch, standardizer, error_mode, inject_samples)
    loss = -log_prob.mean() * float(_loss_scale)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite NLL for combo={'+'.join(combo)}.")
    return loss


def _log_prob_with_error(
    flow: ConditionalNSFFlow,
    target_std: torch.Tensor,
    context: torch.Tensor,
    batch: tuple[torch.Tensor, ...],
    standardizer: TargetStandardizer,
    error_mode: str,
    inject_samples: int = 1,
) -> torch.Tensor:
    """Flow log-prob under the run's error mode, given a precomputed context."""
    has_err = error_mode != "none" and len(batch) >= 10 and float((batch[8].abs() + batch[9].abs()).max()) > 1e-8
    if has_err:
        sig_lo = (batch[8].abs() / standardizer.std).clamp_min(1e-6)
        sig_hi = (batch[9].abs() / standardizer.std).clamp_min(1e-6)
        if error_mode == "convolve":
            return flow.log_prob_convolved(target_std, sig_lo, sig_hi, context)
        if error_mode == "inject":
            # Multi-draw injection, broadcast form: the context-conditioned
            # distribution is built once and all k perturbed targets evaluate
            # under it in a single call (the old per-draw loop re-ran the
            # conditioner k times for identical output).
            k = max(1, int(inject_samples))
            lo = sig_lo.unsqueeze(0).expand(k, -1)
            hi = sig_hi.unsqueeze(0).expand(k, -1)
            draws = target_std.unsqueeze(0) + sample_split_normal(lo, hi)
            return flow.log_prob_draws(draws, context).mean(dim=0)
        raise ValueError(f"Unknown error_mode {error_mode!r}.")
    return flow.log_prob(target_std, context)




_OOM_ERROR = getattr(torch, "OutOfMemoryError", getattr(torch.cuda, "OutOfMemoryError", RuntimeError))


def oom_resilient_step(
    *, batch, combos_per_source, token_mask, optimizer, nll_kwargs, depth: int = 0
) -> tuple[float, int]:
    """Forward+backward with CUDA-OOM fallback: on OOM, free the cache and rerun
    the SAME batch as two half-batches with size-weighted gradient accumulation
    (mathematically the identical update). CUDA OOM is a catchable exception,
    not a process kill -- recovery is safe as long as partial references are
    dropped before ``empty_cache``. Recurses at most twice (quarter batches).
    Returns ``(loss_value, n_chunks_used)``.
    """
    try:
        loss = batch_nll(
            batch=batch,
            combos_per_source=combos_per_source,
            spectrum_token_mask=token_mask,
            **nll_kwargs,
        )
        loss.backward()
        return float(loss.item()), 1
    except _OOM_ERROR:
        if depth >= 2 or int(batch[6].shape[0]) < 2:
            raise
        optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        n = int(batch[6].shape[0])
        half = n // 2
        print(f"[oom] step fell back to 2x{half} micro-batches (depth {depth})", flush=True)
        total, chunks = 0.0, 0
        for sl in (slice(0, half), slice(half, n)):
            sub_batch = tuple(t[sl] for t in batch)
            sub_combos = combos_per_source[sl] if combos_per_source is not None else None
            sub_mask = token_mask[sl] if token_mask is not None else None
            weight = (sl.stop - sl.start) / n
            value, sub_chunks = oom_resilient_step(
                batch=sub_batch, combos_per_source=sub_combos, token_mask=sub_mask,
                optimizer=optimizer,
                nll_kwargs={**nll_kwargs, "_loss_scale": weight},
                depth=depth + 1,
            )
            total += value
            chunks += sub_chunks
        return total, chunks


@torch.no_grad()
def validation_metrics(
    *,
    encoder: AIONTokenEncoder,
    context_encoder: AIONAttentionContext,
    flow: ConditionalNSFFlow,
    val_loader,
    standardizer: TargetStandardizer,
    prior: KDEPrior | None = None,
    device: torch.device,
    error_mode: str = "none",
    inject_samples: int = 1,
    r2_samples: int = 64,
) -> dict[str, float]:
    """Per-epoch validation panel (RunPod-era standard): per-combo NLL, R², IG.

    For all-inputs and each single modality: one AION encode per combo per batch,
    from which NLL (error-mode consistent), posterior-mean R²/RMSE (affine-
    invariant, computed in standardized space), and info gain vs the KDE prior
    all follow. ``val/all_inputs_nll`` stays the checkpoint criterion.
    """
    encoder.eval()
    context_encoder.eval()
    flow.eval()
    combos: dict[str, tuple[str, ...]] = {"all_inputs": tuple(MODALITIES)}
    combos.update({modality: (modality,) for modality in MODALITIES})
    nll_sums = {name: 0.0 for name in combos}
    preds: dict[str, list[np.ndarray]] = {name: [] for name in combos}
    trues: list[np.ndarray] = []
    prior_lp_sum = 0.0
    total = 0
    for batch in val_loader:
        batch = tuple(item.to(device, non_blocking=True) for item in batch)
        target_std = standardizer.transform_tensor(batch[6])
        n = int(target_std.shape[0])
        total += n
        trues.append(target_std.detach().cpu().numpy().ravel())
        if prior is not None:
            prior_lp_sum += float(prior.log_prob_tensor(target_std).sum().item())
        for name, combo in combos.items():
            tokens, group_ids = encoder.encode_tokens(batch, combo)
            context = context_encoder(tokens, group_ids)
            log_prob = _log_prob_with_error(flow, target_std, context, batch, standardizer, error_mode, inject_samples)
            nll_sums[name] += float(-log_prob.mean().item()) * n
            samples = flow.sample(context, num_samples=r2_samples)
            preds[name].append(samples.mean(dim=0).detach().cpu().numpy().ravel())

    y_true = np.concatenate(trues)
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    prior_mean_lp = prior_lp_sum / total if prior is not None else None
    metrics: dict[str, float] = {}
    for name in combos:
        nll = nll_sums[name] / total
        y_pred = np.concatenate(preds[name])
        r2 = 1.0 - float(np.sum((y_true - y_pred) ** 2)) / ss_tot if ss_tot > 0 else float("nan")
        metrics[f"val/{name}_nll"] = nll
        metrics[f"val/{name}_r2"] = r2
        metrics[f"val/{name}_rmse"] = float(np.sqrt(np.mean((y_true - y_pred) ** 2))) * standardizer.std
        if prior_mean_lp is not None:
            # IG = E[log p_post] - E[log p_prior] = (-nll) - prior_mean_lp
            metrics[f"val/{name}_info_gain"] = -nll - prior_mean_lp
    return metrics


@torch.no_grad()
def validation_all_inputs_nll(
    *,
    encoder: AIONTokenEncoder,
    context_encoder: AIONAttentionContext,
    flow: ConditionalNSFFlow,
    val_loader,
    standardizer: TargetStandardizer,
    device: torch.device,
    error_mode: str = "none",
) -> float:
    """Mean all-inputs validation NLL -- the early-stopping and checkpoint criterion."""
    encoder.eval()
    context_encoder.eval()
    flow.eval()
    losses: list[float] = []
    counts: list[int] = []
    combo = tuple(MODALITIES)
    for batch in val_loader:
        batch = tuple(item.to(device, non_blocking=True) for item in batch)
        loss = batch_nll(
            encoder=encoder,
            context_encoder=context_encoder,
            flow=flow,
            batch=batch,
            combo=combo,
            standardizer=standardizer,
            error_mode=error_mode,
        )
        losses.append(float(loss.item()))
        counts.append(int(batch[6].shape[0]))
    return float(np.average(losses, weights=counts))


def train(args: argparse.Namespace) -> None:
    """Train the attention head and flow (AION frozen) on a mix of modality combos.

    Samples a modality combination per batch, minimizes the flow NLL, and
    checkpoints the best all-inputs validation NLL to ``best.pt``.
    """
    set_global_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = Path(args.output_dir) / (args.run_id or timestamp_run_id("aion-flow"))
    run_dir.mkdir(parents=True, exist_ok=True)

    clean_split_csv = Path(args.clean_split_csv) if args.clean_split_csv else None
    train_loader, val_loader, test_loader = build_dataloaders(
        staged_dir=Path(args.staged_dir),
        target_name=args.target,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        clean_split_csv=clean_split_csv,
        max_target_sigma=args.max_target_sigma,
        extra_targets_csv=args.extra_targets_csv,
    )
    # Standardizer + prior must be fit on the same rows the model trains on.
    if clean_split_csv is not None:
        target_values = read_view_target_values(
            Path(args.staged_dir), args.target, clean_split_csv, split="train",
            extra_targets_csv=args.extra_targets_csv,
        )
    else:
        target_values = read_target_values(Path(args.staged_dir) / "desi_train.hdf5", args.target)
    standardizer = TargetStandardizer.fit(target_values)
    prior = KDEPrior(standardizer.transform_numpy(target_values), bw_method="scott")
    prior_metadata = {
        "target": args.target,
        "standardizer": standardizer.state_dict(),
        "staged_dir": str(args.staged_dir),
        "train_count": int(np.isfinite(target_values).sum()),
    }
    prior.save(run_dir / "kde_prior.npz", metadata=prior_metadata)

    head = {
        "num_queries": args.num_queries,
        "num_layers": args.num_layers,
        "context_hidden": list(args.context_hidden),
        "context_dim": args.context_dim,
    }
    is_paper_head = head == PAPER_HEAD and args.head_type == "attention"
    encoder, context_encoder, flow = build_model(
        device, dropout=args.dropout, head=head, head_type=args.head_type,
        cls_variant=args.cls_variant,
        lora_rank=args.lora_rank, lora_blocks=args.lora_blocks,
        lora_modules=args.lora_modules, grad_checkpoint=args.grad_checkpoint,
    )
    param_groups: list[dict] = [
        {"params": list(context_encoder.parameters()) + list(flow.parameters()), "lr": args.lr}
    ]
    if args.head_type == "cls":
        param_groups.append({"params": [encoder.cls_token], "lr": args.lr})
        param_groups.extend(encoder.encoder_param_groups(args.encoder_lr, args.llrd_gamma))
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    trainable_params = [p for g in param_groups for p in g["params"]]
    sampler = ComboSampler.default()
    config = {
        "target": args.target,
        "staged_dir": str(args.staged_dir),
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "head": head,
        "error_mode": args.error_mode,
        "inject_samples": int(args.inject_samples),
        "mask_mode": args.mask_mode,
        "stratified_combos": bool(args.stratified_combos),
        "head_type": args.head_type,
        "cls_variant": args.cls_variant,
        "lora_rank": int(args.lora_rank),
        "lora_blocks": int(args.lora_blocks),
        "lora_modules": args.lora_modules,
        "encoder_lr": float(args.encoder_lr),
        "llrd_gamma": float(args.llrd_gamma),
        "encoder_warmup_epochs": int(args.encoder_warmup_epochs),
        "grad_checkpoint": bool(args.grad_checkpoint),
        "clean_split_csv": str(clean_split_csv) if clean_split_csv else None,
        "max_target_sigma": args.max_target_sigma,
        "extra_targets_csv": str(args.extra_targets_csv) if args.extra_targets_csv else None,
        "architecture": (
            "v3_cls_lora_flow_clean" if args.head_type == "cls"
            else "paper_q4_l2_attention_flow_clean" if is_paper_head
            else "aion_attention_flow_clean_custom_head"
        ),
        "training_mix": "25% singles, 25% pairs, 25% triples, 25% all-input",
        "modalities": list(MODALITIES),
        "standardizer": standardizer.state_dict(),
        "prior_path": str(run_dir / "kde_prior.npz"),
    }
    write_json(run_dir / "config.json", config)

    tracker = init_tracking(
        enabled=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name or run_dir.name,
        mode=args.wandb_mode,
        config={**config, "run_dir": str(run_dir), "seed": args.seed, "device": str(device)},
        run_dir=run_dir,
        tags=[args.target, args.error_mode, "V1" if not is_paper_head else "paper"],
    )

    fixed_combo = tuple(args.fixed_combo.split("+")) if args.fixed_combo else None
    shapley_players = None
    mask_rng = None
    if args.token_mask_augment:
        from .line_shapley import player_catalog, sample_training_mask  # noqa: F401
        shapley_players = player_catalog()
        mask_rng = np.random.default_rng(args.seed + 7)

    best_val = float("inf")
    metrics_path = run_dir / "epoch_metrics.jsonl"
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.monotonic()
        encoder.eval()
        context_encoder.train()
        flow.train()
        if args.head_type == "cls" and args.encoder_warmup_epochs > 0:
            # Warmup: CLS + head + flow adapt to the frozen encoder before the
            # LoRA adapters move (their LR is 0 during warmup epochs).
            scale = 0.0 if epoch <= args.encoder_warmup_epochs else 1.0
            for group in optimizer.param_groups:
                if group.get("is_encoder"):
                    group["lr"] = group["base_lr"] * scale
        train_losses: list[float] = []
        train_counts: list[int] = []
        size_losses: dict[int, list[float]] = {1: [], 2: [], 3: [], 4: []}

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        oom_fallback_steps = 0
        for batch in train_loader:
            batch = tuple(item.to(device, non_blocking=True) for item in batch)
            combos_per_source = None
            if args.stratified_combos and not fixed_combo:
                combo = tuple(MODALITIES)
                combos_per_source = sampler.sample_per_source(int(batch[6].shape[0]), generator)
            else:
                combo = fixed_combo if fixed_combo else sampler.sample(generator)
            token_mask = None
            if args.token_mask_augment:
                mask_np = sample_training_mask(
                    batch[3].detach().cpu().numpy().ravel(), shapley_players, mask_rng
                )
                token_mask = torch.from_numpy(mask_np).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss_value, n_chunks = oom_resilient_step(
                batch=batch,
                combos_per_source=combos_per_source,
                token_mask=token_mask,
                optimizer=optimizer,
                nll_kwargs=dict(
                    encoder=encoder,
                    context_encoder=context_encoder,
                    flow=flow,
                    combo=combo,
                    standardizer=standardizer,
                    error_mode=args.error_mode,
                    inject_samples=args.inject_samples,
                    mask_mode=args.mask_mode,
                ),
            )
            if n_chunks > 1:
                oom_fallback_steps += 1
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()

            train_losses.append(loss_value)
            train_counts.append(int(batch[6].shape[0]))
            if combos_per_source is None:
                size_losses[len(combo)].append(loss_value)
            global_step += 1
            if tracker.enabled and global_step % args.log_every == 0:
                tracker.log(
                    {
                        "train/batch_nll": loss_value,
                        "train/combo_size": 0 if combos_per_source is not None else len(combo),
                        "train/grad_norm": float(grad_norm),
                        "train/lr": float(optimizer.param_groups[0]["lr"]),
                        "epoch": epoch,
                    },
                    step=global_step,
                )

        val_metrics = validation_metrics(
            encoder=encoder,
            context_encoder=context_encoder,
            flow=flow,
            val_loader=val_loader,
            standardizer=standardizer,
            prior=prior,
            device=device,
            error_mode=args.error_mode,
            inject_samples=args.inject_samples,
        )
        val_nll = val_metrics["val/all_inputs_nll"]
        train_nll = float(np.average(train_losses, weights=train_counts))
        epoch_seconds = time.monotonic() - epoch_start
        row = {
            "epoch": epoch,
            "train_nll": train_nll,
            "val_all_inputs_nll": val_nll,
            "val_r2_all_inputs": val_metrics["val/all_inputs_r2"],
            "epoch_seconds": round(epoch_seconds, 1),
        }
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(
            f"epoch={epoch} train_nll={train_nll:.4f} val_all_inputs_nll={val_nll:.4f} "
            f"val_r2={val_metrics['val/all_inputs_r2']:.4f} ig={val_metrics.get('val/all_inputs_info_gain', float('nan')):.3f} ({epoch_seconds:.0f}s)",
            flush=True,
        )
        # train/nll mixes modality combos (mostly harder, fewer-input ones), so it
        # is NOT comparable to the all-inputs val NLL; the per-size curves and
        # train/nll_all_inputs are the comparable views.
        log_payload = {
            "epoch": epoch,
            "train/nll": train_nll,
            "val/best_all_inputs_nll": min(best_val, val_nll),
            "epoch_seconds": epoch_seconds,
            "throughput/samples_per_second": float(sum(train_counts) / max(epoch_seconds, 1e-6)),
            "vram/peak_gb": float(torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else 0.0,
            "train/oom_fallback_steps": oom_fallback_steps,
            **val_metrics,
        }
        for size, losses in size_losses.items():
            if losses:
                key = "train/nll_all_inputs" if size == 4 else f"train/nll_combo_size_{size}"
                log_payload[key] = float(np.mean(losses))
        tracker.log(log_payload, step=global_step)

        encoder_trainable = (
            encoder.encoder_trainable_state() if args.head_type == "cls" else None
        )
        save_checkpoint(
            run_dir / "last.pt",
            epoch=epoch,
            context_encoder=context_encoder,
            flow=flow,
            optimizer=optimizer,
            standardizer=standardizer,
            config=config,
            val_all_inputs_nll=val_nll,
            encoder_trainable=encoder_trainable,
        )
        if val_nll < best_val:
            best_val = val_nll
            save_checkpoint(
                run_dir / "best.pt",
                epoch=epoch,
                context_encoder=context_encoder,
                flow=flow,
                optimizer=optimizer,
                standardizer=standardizer,
                config=config,
                val_all_inputs_nll=val_nll,
                encoder_trainable=encoder_trainable,
            )

    tracker.summary({"val/best_all_inputs_nll": best_val, "epochs_run": args.epochs})

    if args.eval_after_train:
        eval_args = argparse.Namespace(
            checkpoint=run_dir / "best.pt",
            staged_dir=args.staged_dir,
            target=args.target,
            output_dir=run_dir,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            device=args.device,
            dropout=args.dropout,
            num_samples=args.eval_samples,
            allow_target_override=False,
            clean_split_csv=clean_split_csv,
            max_target_sigma=args.max_target_sigma,
            extra_targets_csv=args.extra_targets_csv,
        )
        evaluate(eval_args)
        log_eval_metrics(tracker, run_dir / "test_flow_metrics.csv")

    tracker.finish()


def log_eval_metrics(tracker: Any, metrics_csv: Path) -> None:
    """Push the per-combination test metrics table + all-inputs headline to wandb."""
    if not tracker.enabled or not metrics_csv.exists():
        return
    table = pd.read_csv(metrics_csv)
    tracker.log_table("test/metrics", table)
    combo_col = next((c for c in ("combo", "combination", "modalities") if c in table.columns), None)
    if combo_col is None:
        return
    # The all-inputs row is the headline number the paper reports.
    all_inputs = table.loc[table[combo_col].astype(str).str.count(r"[+,]") == len(MODALITIES) - 1]
    if all_inputs.empty:
        return
    summary = {
        f"test/all_inputs_{col}": float(all_inputs.iloc[0][col])
        for col in table.columns
        if col != combo_col and pd.api.types.is_numeric_dtype(table[col])
    }
    tracker.summary(summary)


def evaluate(args: argparse.Namespace) -> None:
    """Evaluate a checkpoint over all 15 modality combinations.

    Writes ``test_flow_metrics.csv``, ``test_predictions.csv`` and the readable
    results table (CSV + PDF) to ``--output-dir``.
    """
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder, context_encoder, flow, standardizer, _config = load_checkpoint(
        Path(args.checkpoint),
        device=device,
        dropout=args.dropout,
    )
    checkpoint_target = _config.get("target")
    if args.target is None:
        if checkpoint_target is None:
            raise ValueError("Checkpoint config has no target; pass --target explicitly.")
        resolved_target = str(checkpoint_target)
    else:
        resolved_target = str(args.target)
        if (
            checkpoint_target is not None
            and resolved_target != str(checkpoint_target)
            and not args.allow_target_override
        ):
            raise ValueError(
                f"Checkpoint target is {checkpoint_target!r}, but eval requested {resolved_target!r}. "
                "Pass --allow-target-override only for an intentional diagnostic override."
            )
    eval_clean_split_csv = Path(args.clean_split_csv) if getattr(args, "clean_split_csv", None) else None
    checkpoint_prior = _config.get("prior_path")
    prior_path = (
        Path(checkpoint_prior) if checkpoint_prior else Path(args.checkpoint).parent / "kde_prior.npz"
    )
    if not prior_path.exists():
        if eval_clean_split_csv is not None:
            target_values = read_view_target_values(
                Path(args.staged_dir), resolved_target, eval_clean_split_csv, split="train",
                extra_targets_csv=Path(eval_extra_csv) if eval_extra_csv else None,
            )
        else:
            target_values = read_target_values(Path(args.staged_dir) / "desi_train.hdf5", resolved_target)
        KDEPrior(standardizer.transform_numpy(target_values), bw_method="scott").save(
            output_dir / "kde_prior.npz",
            metadata={
                "target": resolved_target,
                "standardizer": standardizer.state_dict(),
                "staged_dir": str(args.staged_dir),
                "train_count": int(np.isfinite(target_values).sum()),
            },
        )
        prior_path = output_dir / "kde_prior.npz"
    prior = KDEPrior.load(prior_path)
    prior_target = prior.metadata.get("target")
    if prior_target is not None and str(prior_target) != resolved_target:
        raise ValueError(f"Prior target {prior_target!r} does not match eval target {resolved_target!r}.")
    prior_standardizer = prior.metadata.get("standardizer")
    if isinstance(prior_standardizer, dict):
        if (
            abs(float(prior_standardizer["mean"]) - standardizer.mean) > 1e-6
            or abs(float(prior_standardizer["std"]) - standardizer.std) > 1e-6
        ):
            raise ValueError("Prior standardizer metadata does not match checkpoint standardizer.")

    eval_max_sigma = getattr(args, "max_target_sigma", None)
    if eval_max_sigma is None:
        eval_max_sigma = _config.get("max_target_sigma")
    eval_extra_csv = getattr(args, "extra_targets_csv", None) or _config.get("extra_targets_csv")
    _train_loader, _val_loader, test_loader = build_dataloaders(
        staged_dir=Path(args.staged_dir),
        target_name=resolved_target,
        clean_split_csv=eval_clean_split_csv,
        max_target_sigma=eval_max_sigma,
        extra_targets_csv=Path(eval_extra_csv) if eval_extra_csv else None,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    # Score with the same likelihood the checkpoint was trained with: a
    # convolve-trained flow models the deconvolved density and must be evaluated
    # through the measurement kernel, or its info gain is understated.
    eval_error_mode = str(_config.get("error_mode", "none"))
    metrics, predictions = evaluate_all_combos(
        encoder=encoder,
        context_encoder=context_encoder,
        flow=flow,
        loader=test_loader,
        standardizer=standardizer,
        prior=prior,
        device=device,
        num_samples=args.num_samples,
        combos=all_nonempty_modality_combos(),
        error_mode=eval_error_mode,
    )
    metrics.to_csv(output_dir / "test_flow_metrics.csv", index=False)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    table = build_results_table(metrics)
    table.to_csv(output_dir / "results_table.csv", index=False)
    save_results_table_pdf(table, output_dir / "results_table.pdf")
    write_json(
        output_dir / "metrics.json",
        {
            "checkpoint": str(args.checkpoint),
            "target": resolved_target,
            "prior_path": str(prior_path),
            "error_mode": eval_error_mode,
        },
    )
    print(table.drop(columns=["input_group"]).to_string(index=False), flush=True)


def make_table(args: argparse.Namespace) -> None:
    """Render the readable results table (CSV + PDF) from a metrics CSV."""
    metrics = pd.read_csv(args.metrics_csv)
    table = build_results_table(metrics)
    table.to_csv(args.output_csv, index=False)
    save_results_table_pdf(table, args.output_pdf)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean shareable AION attention-flow package.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare-data", help="Build image-backed staged HDF5 files from raw inputs."
    )
    prepare.add_argument("--source-hdf5", type=Path, default=None)
    prepare.add_argument("--split-manifest-csv", type=Path, default=None)
    prepare.add_argument("--fits-pool-dir", type=Path, default=None)
    prepare.add_argument("--output-dir", type=Path, default=STAGED_DIR)
    prepare.add_argument("--limit", type=int, default=None)
    prepare.add_argument("--overwrite", action="store_true")
    prepare.add_argument("--image-compression", choices=["none", "gzip", "lzf"], default="none")
    prepare.add_argument("--clean", action="store_true",
                         help="Apply NWAY clean filter + dedup + re-split on the cleaned sample.")
    prepare.add_argument("--match-quality-csv", type=Path, default=None, help="Per-targetid NWAY keep filter.")
    prepare.add_argument("--targets-extra-csv", type=Path, default=None,
                         help="Extra target + error columns (logmstar, hr32_u, flux_sig_lo/hi).")

    train_parser = subparsers.add_parser("train", help="Train the clean paper AION attention-flow model.")
    train_parser.add_argument("--staged-dir", type=Path, default=STAGED_DIR)
    train_parser.add_argument(
        "--target",
        choices=["log_ml_flux_1", "log_lx", "logmstar", "hr32_u",
                 "log_flux_p1", "log_flux_p2", "log_flux_p3", "log_flux_p4"],
        default="log_ml_flux_1",
    )
    train_parser.add_argument("--extra-targets-csv", type=Path, default=None,
                              help="Runtime sidecar with <target>[,_sig_lo,_sig_hi] columns by targetid "
                              "(e.g. per-band fluxes) -- no re-staging needed.")
    train_parser.add_argument(
        "--error-mode", choices=["none", "inject", "convolve"], default="none",
        help="How per-source split-normal errors enter the loss. Default 'none' is the primary "
             "estimand p(y|x): plain likelihood at the observed values, matching how eval scores. "
             "'inject' (the adopted training mode, with --inject-samples 8) smears labels by their "
             "own errors; 'convolve' deconvolves to a latent p(t|x) and is kept for latent analysis "
             "only. See docs/decisions.md.",
    )
    train_parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    train_parser.add_argument("--run-id", default=None)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch-size", type=int, default=448)
    train_parser.add_argument("--eval-batch-size", type=int, default=448)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--dropout", type=float, default=0.05)
    train_parser.add_argument("--grad-clip", type=float, default=5.0)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", default=None)
    train_parser.add_argument("--eval-after-train", action="store_true")
    train_parser.add_argument("--eval-samples", type=int, default=256)
    # Attention-head size (defaults = paper q4/l2; e.g. V1 minimal head:
    #   --num-queries 1 --num-layers 1 --context-hidden 128)
    train_parser.add_argument("--num-queries", type=int, default=PAPER_HEAD["num_queries"])
    train_parser.add_argument("--num-layers", type=int, default=PAPER_HEAD["num_layers"])
    train_parser.add_argument(
        "--context-hidden", type=int, nargs="+", default=list(PAPER_HEAD["context_hidden"])
    )
    train_parser.add_argument("--context-dim", type=int, default=PAPER_HEAD["context_dim"])
    train_parser.add_argument(
        "--stratified-combos", action="store_true",
        help="Fixed-composition batches: one combo PER SOURCE (same size-stratified "
        "marginal), encoded in a single all-inputs-length masked forward. Constant "
        "per-batch memory and full combo mix in every gradient.",
    )
    train_parser.add_argument("--fixed-combo", default=None,
                              help="Train on one fixed modality combo, e.g. 'spectra'.")
    train_parser.add_argument("--token-mask-augment", action="store_true",
                              help="Random rest-frame coalition masking of spectrum tokens (Shapley prep).")
    train_parser.add_argument(
        "--head-type", choices=["attention", "cls"], default="attention",
        help="cls = V3: trainable CLS token at the encoder input + LoRA fine-tune, "
        "projected to the same 256-dim context and identical flow.",
    )
    train_parser.add_argument(
        "--cls-variant", choices=["input", "readonly"], default="input",
        help="input = V3a (CLS in the sequence, encoder LoRA-tuned). readonly = V3b: "
        "no token attends to the CLS, data stream stays exactly frozen; the CLS "
        "cross-reads each block with adapter deltas on its query and consumed values.",
    )
    train_parser.add_argument("--lora-rank", type=int, default=8,
                              help="LoRA rank for --head-type cls (0 = CLS+head only).")
    train_parser.add_argument("--lora-blocks", type=int, default=-1,
                              help="Apply LoRA to the LAST k encoder blocks (-1 = all, 0 = none).")
    train_parser.add_argument("--lora-modules", choices=["attn", "all"], default="all",
                              help="Adapt attention q/k/v/proj only, or every transformer linear (QLoRA-style).")
    train_parser.add_argument("--encoder-lr", type=float, default=2e-5,
                              help="Base LR for encoder LoRA groups (decayed by --llrd-gamma per block).")
    train_parser.add_argument("--llrd-gamma", type=float, default=0.8,
                              help="Layer-wise LR decay per block from the top, cls mode.")
    train_parser.add_argument("--encoder-warmup-epochs", type=int, default=1,
                              help="Epochs with encoder LoRA LR at 0 (CLS/head/flow adapt first).")
    train_parser.add_argument("--grad-checkpoint", action="store_true",
                              help="Gradient-checkpoint encoder blocks (needed for cls mode at bs 448).")
    train_parser.add_argument(
        "--mask-mode", choices=["drop", "replace"], default="drop",
        help="Token removal semantics for masking: drop = AION-native input mask "
        "with pixel sanitization (default); replace = legacy continuum-code swap.",
    )
    train_parser.add_argument(
        "--max-target-sigma",
        type=float,
        default=None,
        help="S/N gate: exclude rows whose mean target sigma exceeds this (hr32_u: use 1.0).",
    )
    train_parser.add_argument(
        "--inject-samples",
        type=int,
        default=8,
        help="Noise draws per source per step in --error-mode inject (context is reused, so cheap).",
    )
    train_parser.add_argument(
        "--clean-split-csv",
        type=Path,
        default=None,
        help="targetid->split sidecar (scripts/make_clean_split.py): train on the cleaned "
        "re-split VIEW of the staged superset instead of the files' native splits.",
    )
    train_parser.add_argument("--wandb", action="store_true", help="Track this run with Weights & Biases.")
    train_parser.add_argument("--wandb-project", default="erosita-desi-aion-flow")
    train_parser.add_argument("--wandb-entity", default=None)
    train_parser.add_argument("--wandb-run-name", default=None, help="Defaults to the run id.")
    train_parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
        help="Falls back to offline automatically if the compute node has no internet.",
    )
    train_parser.add_argument("--log-every", type=int, default=10, help="Log batch NLL every N steps.")

    mt = subparsers.add_parser("train-multi", help="Train the 9-head read-only-CLS multi-target model.")
    mt.add_argument("--staged-dir", type=Path, required=True)
    mt.add_argument("--clean-split-csv", type=Path, required=True)
    mt.add_argument("--extra-targets-csv", type=Path, required=True,
                    help="Runtime sidecar joined by targetid: band targets and log_sfr "
                         "(targets_sidecar.csv, built by scripts/make_sfr_sidecar.py).")
    mt.add_argument("--epochs", type=int, default=30)
    mt.add_argument("--batch-size", type=int, default=448)
    mt.add_argument("--lr", type=float, default=1e-4)
    mt.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant")
    mt.add_argument("--weight-decay", type=float, default=1e-4, help="Head + flows.")
    mt.add_argument("--adapter-wd", type=float, default=1e-2, help="Full-rank read adapters (capacity control).")
    mt.add_argument("--adapter-lr", type=float, default=None,
                    help="Separate LR for the read adapters (default: --lr). They are zero-init, so "
                    "a shared LR moves them ~30x faster in update/weight terms than the flows.")
    mt.add_argument("--beta1", type=float, default=0.9,
                    help="Adam beta1. Higher averages gradients over more steps, which implicitly "
                    "damps heads whose gradient has gone incoherent (saturated).")
    mt.add_argument("--grad-checkpoint", action="store_true")
    mt.add_argument("--bucketed", action="store_true",
                    help="Length-bucketed combo packing: per-source combos grouped into 4 "
                    "token-length buckets; one forward + one step per bucket. Use a LARGE "
                    "--batch-size (e.g. 896) so each bucket lands near the calibrated 224.")
    mt.add_argument("--no-inject", action="store_true")
    mt.add_argument("--inject-samples", type=int, default=50,
                    help="Broadcast noise draws per source per step (one conditioner pass).")
    mt.add_argument("--accumulate-buckets", action="store_true",
                    help="Accumulate gradients over all buckets into ONE optimizer step, so every "
                    "update sees the full combo mix (lower gradient variance, fewer steps).")
    mt.add_argument("--drop-heads", nargs="*", default=None,
                    help="Scalar heads to omit, e.g. log_flux_p4 (never learns; P2/P3 are required "
                    "by the joint head).")
    mt.add_argument("--train-probe-size", type=int, default=2520,
                    help="Fixed training-split subset scored with the VAL protocol each epoch "
                    "(0 disables). Makes train and val directly comparable.")
    mt.add_argument("--diag-every", type=int, default=50,
                    help="Steps between diagnostic logs (head influence, group norms/movement).")
    mt.add_argument("--bucket-chunk", type=int, default=224,
                    help="Max sources per bucket forward (buckets larger than this are chunked).")
    mt.add_argument("--early-stop-patience", type=int, default=6,
                    help="Stop when the val pair-mean NLL has not improved for this many epochs.")
    mt.add_argument("--num-workers", type=int, default=8)
    mt.add_argument("--seed", type=int, default=42)
    mt.add_argument("--device", default=None)
    mt.add_argument("--run-id", required=True)
    mt.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    mt.add_argument("--log-every", type=int, default=10)
    mt.add_argument("--wandb", action="store_true")
    mt.add_argument("--wandb-project", default="erosita-desi-aion-flow")
    mt.add_argument("--wandb-entity", default=None)

    eval_parser = subparsers.add_parser(
        "eval", help="Evaluate all 15 modality combinations from a checkpoint."
    )
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--staged-dir", type=Path, default=STAGED_DIR)
    eval_parser.add_argument(
        "--target",
        choices=["log_ml_flux_1", "log_lx", "logmstar", "hr32_u",
                 "log_flux_p1", "log_flux_p2", "log_flux_p3", "log_flux_p4"],
        default=None,
    )
    eval_parser.add_argument("--extra-targets-csv", type=Path, default=None,
                             help="Defaults to the checkpoint config's value.")
    eval_parser.add_argument("--allow-target-override", action="store_true")
    eval_parser.add_argument("--max-target-sigma", type=float, default=None,
                             help="Defaults to the checkpoint config's value.")
    eval_parser.add_argument(
        "--clean-split-csv",
        type=Path,
        default=None,
        help="Evaluate on the cleaned re-split VIEW of the staged superset (must match training).",
    )
    eval_parser.add_argument("--output-dir", type=Path, required=True)
    eval_parser.add_argument("--batch-size", type=int, default=448)
    eval_parser.add_argument("--num-workers", type=int, default=0)
    eval_parser.add_argument("--num-samples", type=int, default=256)
    eval_parser.add_argument("--dropout", type=float, default=0.05)
    eval_parser.add_argument("--device", default=None)

    table_parser = subparsers.add_parser("make-table", help="Create CSV/PDF readable table from metrics CSV.")
    table_parser.add_argument("--metrics-csv", type=Path, required=True)
    table_parser.add_argument("--output-csv", type=Path, required=True)
    table_parser.add_argument("--output-pdf", type=Path, required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare-data":
        summary = prepare_staged_data(
            source_hdf5=args.source_hdf5 or SOURCE_HDF5,
            split_manifest_csv=args.split_manifest_csv or SPLIT_MANIFEST,
            fits_pool_dir=args.fits_pool_dir or FITS_POOL_DIR,
            output_dir=args.output_dir,
            limit=args.limit,
            overwrite=args.overwrite,
            image_compression=args.image_compression,
            clean=args.clean,
            match_quality_csv=args.match_quality_csv,
            targets_extra_csv=args.targets_extra_csv,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif args.command == "train":
        train(args)
    elif args.command == "train-multi":
        from .multitarget import run_train_multi
        run_train_multi(args)
    elif args.command == "eval":
        evaluate(args)
    elif args.command == "make-table":
        make_table(args)


if __name__ == "__main__":
    main()
