"""Evaluation: posterior metrics over modality combinations and the results table.

Runs a trained attention-flow over every non-empty modality combination and
reports R squared, RMSE, information gain (nats) and exp(mean information gain),
plus a human-readable results table (CSV/PDF) matching the paper.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .attention_pooling_head import MODALITIES, all_nonempty_modality_combos, combo_name
from .data_to_aion_embeddings import move_batch_to_device
from .normalizing_flow import KDEPrior, TargetStandardizer

# Canonical results schema. A fresh `eval` writes exactly these columns, in this
# order, matching results/test_flow_metrics.csv so the shipped table is fully
# regenerable by this code (and so `make-table` consumes it without surprises).
METRIC_COLUMNS = [
    "input_group",
    "n_test",
    "r2",
    "r2_trimmed_0.05",
    "rmse",
    "mae",
    "mean_posterior_log_prob_nats",
    "mean_prior_log_prob_nats",
    "info_gain_nats",
    "exp_info_gain",
    "delta_ll_frac_negative",
    "delta_ll_frac_near_zero",
    "delta_ll_frac_positive",
]

# Single source of truth for the per-modality column markers used in the table.
MODALITY_MARKERS = {"spectra": "S", "z": "Z", "wise": "W", "image": "I"}


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    if denom <= 0.0:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def _trim_mask(residual: np.ndarray, trim_fraction: float) -> np.ndarray:
    """Boolean mask keeping all but the ``trim_fraction`` largest-|residual| points."""
    n = residual.shape[0]
    n_drop = int(np.floor(trim_fraction * n))
    keep = np.ones(n, dtype=bool)
    if n_drop > 0:
        keep[np.argsort(np.abs(residual))[n - n_drop :]] = False
    return keep


def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, trim_fraction: float = 0.05
) -> dict[str, float]:
    """R squared, outlier-trimmed R squared, RMSE, and MAE for predictions.

    The trimmed R squared drops the ``trim_fraction`` of points with the largest
    absolute residual before recomputing, as a robustness check against a handful
    of catastrophic outliers dominating the score.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual = y_true - y_pred
    keep = _trim_mask(residual, trim_fraction)
    return {
        "r2": _r_squared(y_true, y_pred),
        "r2_trimmed_0.05": _r_squared(y_true[keep], y_pred[keep]),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
    }


@torch.no_grad()
def evaluate_all_combos(
    *,
    encoder,
    context_encoder,
    flow,
    loader,
    standardizer: TargetStandardizer,
    prior: KDEPrior,
    device: torch.device,
    num_samples: int = 256,
    near_zero_threshold: float = 0.05,
    combos: list[tuple[str, ...]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate posterior metrics for every requested modality combination.

    Returns ``(metrics, predictions)``. ``metrics`` carries one row per combination
    in the canonical ``METRIC_COLUMNS`` schema; ``predictions`` is the per-source
    posterior summary (mean, 16/50/84th percentiles, log-probs, info gain).
    """

    combos = combos or all_nonempty_modality_combos()
    encoder.eval()
    context_encoder.eval()
    flow.eval()

    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []

    for combo in combos:
        y_true_parts: list[np.ndarray] = []
        y_pred_parts: list[np.ndarray] = []
        targetid_parts: list[np.ndarray] = []
        posterior_ll_parts: list[np.ndarray] = []
        prior_ll_parts: list[np.ndarray] = []
        p16_parts: list[np.ndarray] = []
        p50_parts: list[np.ndarray] = []
        p84_parts: list[np.ndarray] = []

        for batch_index, batch in enumerate(loader):
            batch = move_batch_to_device(batch, device)
            target = batch[6]
            targetids = batch[7]
            input_group = combo_name(combo)
            if not torch.isfinite(target).all():
                bad_targetids = targetids[~torch.isfinite(target)][:5].detach().cpu().tolist()
                raise FloatingPointError(
                    f"Non-finite targets during eval for combo={input_group}, batch={batch_index}, "
                    f"example_targetids={bad_targetids}."
                )
            target_std = standardizer.transform_tensor(target)

            tokens, group_ids = encoder.encode_tokens(batch, combo)
            context = context_encoder(tokens, group_ids)
            posterior_ll = flow.log_prob(target_std, context)
            prior_ll = prior.log_prob_tensor(target_std)
            if not torch.isfinite(posterior_ll).all():
                raise FloatingPointError(
                    f"Non-finite posterior log-prob for combo={input_group}, batch={batch_index}."
                )
            if not torch.isfinite(prior_ll).all():
                raise FloatingPointError(
                    f"Non-finite prior log-prob for combo={input_group}, batch={batch_index}."
                )

            samples_std = flow.sample(context, num_samples=num_samples)
            if not torch.isfinite(samples_std).all():
                raise FloatingPointError(
                    f"Non-finite posterior samples for combo={input_group}, batch={batch_index}."
                )
            samples = standardizer.inverse_tensor(samples_std)
            pred = samples.mean(dim=0)
            p16 = torch.quantile(samples, 0.16, dim=0)
            p50 = torch.quantile(samples, 0.50, dim=0)
            p84 = torch.quantile(samples, 0.84, dim=0)

            y_true_parts.append(target.detach().cpu().numpy())
            y_pred_parts.append(pred.detach().cpu().numpy())
            targetid_parts.append(targetids.detach().cpu().numpy())
            posterior_ll_parts.append(posterior_ll.detach().cpu().numpy())
            prior_ll_parts.append(prior_ll.detach().cpu().numpy())
            p16_parts.append(p16.detach().cpu().numpy())
            p50_parts.append(p50.detach().cpu().numpy())
            p84_parts.append(p84.detach().cpu().numpy())

        y_true = np.concatenate(y_true_parts)
        y_pred = np.concatenate(y_pred_parts)
        targetids = np.concatenate(targetid_parts)
        posterior_ll = np.concatenate(posterior_ll_parts)
        prior_ll = np.concatenate(prior_ll_parts)
        info_gain = posterior_ll - prior_ll
        metrics = regression_metrics(y_true, y_pred)
        input_group = combo_name(combo)
        # One row in the canonical METRIC_COLUMNS schema (see module top).
        metric_rows.append(
            {
                "input_group": input_group,
                "n_test": int(y_true.shape[0]),
                "r2": metrics["r2"],
                "r2_trimmed_0.05": metrics["r2_trimmed_0.05"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "mean_posterior_log_prob_nats": float(np.mean(posterior_ll)),
                "mean_prior_log_prob_nats": float(np.mean(prior_ll)),
                "info_gain_nats": float(np.mean(info_gain)),
                "exp_info_gain": float(math.exp(np.mean(info_gain))),
                "delta_ll_frac_negative": float(np.mean(info_gain < -near_zero_threshold)),
                "delta_ll_frac_near_zero": float(np.mean(np.abs(info_gain) <= near_zero_threshold)),
                "delta_ll_frac_positive": float(np.mean(info_gain > near_zero_threshold)),
            }
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "input_group": input_group,
                    "targetid": targetids,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "posterior_p16": np.concatenate(p16_parts),
                    "posterior_p50": np.concatenate(p50_parts),
                    "posterior_p84": np.concatenate(p84_parts),
                    "posterior_log_prob": posterior_ll,
                    "prior_log_prob": prior_ll,
                    "info_gain": info_gain,
                }
            )
        )

    return pd.DataFrame(metric_rows, columns=METRIC_COLUMNS), pd.concat(prediction_rows, ignore_index=True)


def _indicator(combo: str, modality: str) -> str:
    return MODALITY_MARKERS[modality] if modality in combo.split("+") else ""


def build_results_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Order combinations by R squared and add S/Z/W/I modality indicator columns."""
    table = metrics.copy()
    for modality in MODALITIES:
        table[modality] = table["input_group"].map(lambda combo: _indicator(str(combo), modality))
    table = table.sort_values("r2", ascending=True).reset_index(drop=True)
    return table[
        [
            "z",
            "spectra",
            "wise",
            "image",
            "r2",
            "info_gain_nats",
            "exp_info_gain",
            "rmse",
            "input_group",
        ]
    ]


def save_results_table_pdf(table: pd.DataFrame, path: Path) -> None:
    """Save a compact readable PDF table. NLL is intentionally not displayed."""

    display = table.rename(
        columns={
            "z": "Redshift",
            "spectra": "Spectra",
            "wise": "WISE",
            "image": "Images",
            "r2": r"$R^2$",
            "info_gain_nats": "Info Gain",
            "exp_info_gain": r"$\exp(\mathrm{Info\ Gain})$",
            "rmse": "RMSE",
        }
    ).drop(columns=["input_group"])

    text = display.copy()
    for column in (r"$R^2$", "Info Gain", r"$\exp(\mathrm{Info\ Gain})$", "RMSE"):
        text[column] = text[column].map(lambda value: f"{float(value):.3f}")

    numeric_columns = [r"$R^2$", "Info Gain", r"$\exp(\mathrm{Info\ Gain})$", "RMSE"]
    colors = [["white" for _ in text.columns] for _ in range(len(text))]
    cmaps = {
        r"$R^2$": plt.cm.Blues,
        "Info Gain": plt.cm.Purples,
        r"$\exp(\mathrm{Info\ Gain})$": plt.cm.Greens,
        "RMSE": plt.cm.Oranges_r,
    }
    for column in numeric_columns:
        values = display[column].to_numpy(dtype=float)
        vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
        denom = max(vmax - vmin, 1e-8)
        col_idx = list(text.columns).index(column)
        for row_idx, value in enumerate(values):
            colors[row_idx][col_idx] = cmaps[column]((float(value) - vmin) / denom)

    fig_height = max(3.0, 0.34 * len(text) + 0.8)
    fig, ax = plt.subplots(figsize=(7.2, fig_height))
    ax.axis("off")
    table_artist = ax.table(
        cellText=text.values,
        colLabels=text.columns,
        cellColours=colors,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(9)
    table_artist.scale(1.0, 1.25)
    for (row, col), cell in table_artist.get_celld().items():
        cell.set_linewidth(0.4)
        if row == 0:
            cell.set_facecolor("#222222")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif col < 4 and cell.get_text().get_text():
            cell.set_facecolor("#e9edf3")
            cell.get_text().set_weight("bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
