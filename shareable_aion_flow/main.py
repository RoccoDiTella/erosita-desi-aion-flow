"""Command-line entry point: prepare-data, train, eval, make-table.

Trains and evaluates the clean AION attention-flow model on the eROSITA/DESI
matched sample. Run ``python -m shareable_aion_flow.main --help`` for usage.
"""

from __future__ import annotations

import argparse
import json
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
        write_json,
    )
    from .evals import build_results_table, evaluate_all_combos, save_results_table_pdf
    from .normalizing_flow import ConditionalNSFFlow, KDEPrior, TargetStandardizer, sample_split_normal
    from .tracking import init_tracking
except ImportError:
    from attention_pooling_head import (
        MODALITIES,
        AIONAttentionContext,
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
) -> tuple[AIONTokenEncoder, AIONAttentionContext, ConditionalNSFFlow]:
    """Build the frozen AION encoder, attention-context head, and conditional flow.

    ``head`` overrides the attention-head sizes (num_queries/num_layers/
    context_hidden/context_dim); ``None`` uses the paper q4/l2 defaults.
    """
    head = {**PAPER_HEAD, **(head or {})}
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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "context_encoder_state_dict": context_encoder.state_dict(),
        "flow_state_dict": flow.state_dict(),
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
    head = checkpoint.get("config", {}).get("head")
    encoder, context_encoder, flow = build_model(device, dropout=dropout, head=head)
    context_encoder.load_state_dict(checkpoint["context_encoder_state_dict"])
    flow.load_state_dict(checkpoint["flow_state_dict"])
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
) -> torch.Tensor:
    """Negative log-likelihood of one batch under the flow for a given modality combo.

    ``error_mode`` folds in the per-source split-normal measurement error carried
    as ``batch[8]=sig_lo, batch[9]=sig_hi`` (in target units): ``convolve``
    deconvolves it via the convolution likelihood, ``inject`` adds it as noise,
    ``none`` ignores it. Falls back to plain log-prob when errors are absent/zero.
    """
    target = batch[6]
    if not torch.isfinite(target).all():
        bad = int((~torch.isfinite(target)).sum().item())
        raise FloatingPointError(f"Encountered {bad} non-finite target values in a training/eval batch.")
    target_std = standardizer.transform_tensor(target)
    tokens, group_ids = encoder.encode_tokens(batch, combo)
    context = context_encoder(tokens, group_ids)

    has_err = error_mode != "none" and len(batch) >= 10 and float((batch[8].abs() + batch[9].abs()).max()) > 1e-8
    if has_err:
        sig_lo = (batch[8].abs() / standardizer.std).clamp_min(1e-6)
        sig_hi = (batch[9].abs() / standardizer.std).clamp_min(1e-6)
        if error_mode == "convolve":
            log_prob = flow.log_prob_convolved(target_std, sig_lo, sig_hi, context)
        elif error_mode == "inject":
            log_prob = flow.log_prob(target_std + sample_split_normal(sig_lo, sig_hi), context)
        else:
            raise ValueError(f"Unknown error_mode {error_mode!r}.")
    else:
        log_prob = flow.log_prob(target_std, context)

    loss = -log_prob.mean()
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite NLL for combo={'+'.join(combo)}.")
    return loss


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

    train_loader, val_loader, test_loader = build_dataloaders(
        staged_dir=Path(args.staged_dir),
        target_name=args.target,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
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
    is_paper_head = head == PAPER_HEAD
    encoder, context_encoder, flow = build_model(device, dropout=args.dropout, head=head)
    optimizer = torch.optim.AdamW(
        list(context_encoder.parameters()) + list(flow.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
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
        "architecture": "paper_q4_l2_attention_flow_clean" if is_paper_head else "aion_attention_flow_clean_custom_head",
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

    best_val = float("inf")
    metrics_path = run_dir / "epoch_metrics.jsonl"
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        encoder.eval()
        context_encoder.train()
        flow.train()
        train_losses: list[float] = []
        train_counts: list[int] = []

        for batch in train_loader:
            batch = tuple(item.to(device, non_blocking=True) for item in batch)
            combo = sampler.sample(generator)
            loss = batch_nll(
                encoder=encoder,
                context_encoder=context_encoder,
                flow=flow,
                batch=batch,
                combo=combo,
                standardizer=standardizer,
                error_mode=args.error_mode,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(context_encoder.parameters()) + list(flow.parameters()), args.grad_clip
            )
            optimizer.step()

            train_losses.append(float(loss.item()))
            train_counts.append(int(batch[6].shape[0]))
            global_step += 1
            if tracker.enabled and global_step % args.log_every == 0:
                tracker.log(
                    {
                        "train/batch_nll": float(loss.item()),
                        "train/combo_size": len(combo),
                        "epoch": epoch,
                    },
                    step=global_step,
                )

        val_nll = validation_all_inputs_nll(
            encoder=encoder,
            context_encoder=context_encoder,
            flow=flow,
            val_loader=val_loader,
            standardizer=standardizer,
            device=device,
            error_mode=args.error_mode,
        )
        train_nll = float(np.average(train_losses, weights=train_counts))
        row = {"epoch": epoch, "train_nll": train_nll, "val_all_inputs_nll": val_nll}
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"epoch={epoch} train_nll={train_nll:.4f} val_all_inputs_nll={val_nll:.4f}", flush=True)
        tracker.log(
            {
                "epoch": epoch,
                "train/nll": train_nll,
                "val/all_inputs_nll": val_nll,
                "val/best_all_inputs_nll": min(best_val, val_nll),
            },
            step=global_step,
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
    checkpoint_prior = _config.get("prior_path")
    prior_path = (
        Path(checkpoint_prior) if checkpoint_prior else Path(args.checkpoint).parent / "kde_prior.npz"
    )
    if not prior_path.exists():
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

    _train_loader, _val_loader, test_loader = build_dataloaders(
        staged_dir=Path(args.staged_dir),
        target_name=resolved_target,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
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
        "--target", choices=["log_ml_flux_1", "log_lx", "logmstar", "hr32_u"], default="log_ml_flux_1"
    )
    train_parser.add_argument(
        "--error-mode", choices=["none", "inject", "convolve"], default="convolve",
        help="Fold per-source split-normal errors into the likelihood (convolve=deconvolve).",
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

    eval_parser = subparsers.add_parser(
        "eval", help="Evaluate all 15 modality combinations from a checkpoint."
    )
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--staged-dir", type=Path, default=STAGED_DIR)
    eval_parser.add_argument(
        "--target", choices=["log_ml_flux_1", "log_lx", "logmstar", "hr32_u"], default=None
    )
    eval_parser.add_argument("--allow-target-override", action="store_true")
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
    elif args.command == "eval":
        evaluate(args)
    elif args.command == "make-table":
        make_table(args)


if __name__ == "__main__":
    main()
