from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

try:
    from .attention_pooling_head import MODALITIES, all_nonempty_modality_combos, combo_name
    from .data_to_aion_embeddings import move_batch_to_device
    from .normalizing_flow import KDEPrior, TargetStandardizer
except ImportError:
    from attention_pooling_head import MODALITIES, all_nonempty_modality_combos, combo_name
    from data_to_aion_embeddings import move_batch_to_device
    from normalizing_flow import KDEPrior, TargetStandardizer


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual = y_true - y_pred
    denom = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "r2": float(1.0 - np.sum(residual**2) / denom) if denom > 0 else float("nan"),
        "rmse": float(np.sqrt(np.mean(residual**2))),
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
    combos: list[tuple[str, ...]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate posterior metrics for every requested modality combination."""

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
                raise FloatingPointError(f"Non-finite posterior log-prob for combo={input_group}, batch={batch_index}.")
            if not torch.isfinite(prior_ll).all():
                raise FloatingPointError(f"Non-finite prior log-prob for combo={input_group}, batch={batch_index}.")

            samples_std = flow.sample(context, num_samples=num_samples)
            if not torch.isfinite(samples_std).all():
                raise FloatingPointError(f"Non-finite posterior samples for combo={input_group}, batch={batch_index}.")
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
        metric_rows.append(
            {
                "input_group": input_group,
                "r2": metrics["r2"],
                "rmse": metrics["rmse"],
                "mean_info_gain": float(np.mean(info_gain)),
                "exp_mean_info_gain": float(math.exp(np.mean(info_gain))),
                "mean_nll": float(-np.mean(posterior_ll)),
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

    return pd.DataFrame(metric_rows), pd.concat(prediction_rows, ignore_index=True)


def _indicator(combo: str, modality: str) -> str:
    marker = {"spectra": "S", "z": "Z", "wise": "W", "image": "I"}[modality]
    return marker if modality in combo.split("+") else ""


def build_results_table(metrics: pd.DataFrame) -> pd.DataFrame:
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
            "mean_info_gain",
            "exp_mean_info_gain",
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
            "mean_info_gain": "Info Gain",
            "exp_mean_info_gain": r"$\exp(\mathrm{Info\ Gain})$",
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
