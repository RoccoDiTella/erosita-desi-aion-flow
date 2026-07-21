#!/usr/bin/env python
"""Build the evaluation packet for a finished run (port of the RunPod-era packet).

Consumes a run directory containing ``test_predictions.csv`` + ``test_flow_metrics.csv``
(written by ``aion-flow eval`` / ``--eval-after-train``) and the staged data (for
redshift + spectype joins). Emits into ``<run_dir>/packet/``:

- ``test_diagnostics.pdf``  — scatter grid (pred vs true w/ 16-84 errorbars, R2+IG
  annotations), info-gain histograms, and interval-coverage calibration.
- ``upset_metrics.pdf``     — combos sorted by R2 with an upset-style modality
  membership matrix; bars for R2 and exp(IG).
- ``by_spectype.csv``       — per spectype x combo: n, R2, IG (QSO vs GALAXY).
- ``by_redshift.csv`` + ``performance_by_redshift.pdf`` — z-binned R2 / exp(IG).

    python scripts/make_run_packet.py --run-dir <outputs/run-id> \
        --staged-dir <staged_paper> [--clean-split-csv <clean_split.csv>]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

MODALITIES = ("z", "spectra", "wise", "image")


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def attach_metadata(predictions: pd.DataFrame, staged_dir: Path) -> pd.DataFrame:
    """Join redshift + spectype by targetid from the staged files (dedup'd)."""
    frames = []
    for split in ("train", "val", "test"):
        with h5py.File(staged_dir / f"desi_{split}.hdf5", "r") as handle:
            data = {
                "targetid": handle["desi_targetid"][:].astype(np.int64),
                "redshift": handle["redshift"][:].astype(np.float64),
            }
            if "spectype" in handle:
                data["spectype"] = [s.decode() for s in handle["spectype"][:]]
            frames.append(pd.DataFrame(data))
    meta = pd.concat(frames, ignore_index=True).drop_duplicates("targetid")
    return predictions.merge(meta, on="targetid", how="left")


def combo_sort_order(metrics: pd.DataFrame) -> list[str]:
    return metrics.sort_values("r2")["input_group"].tolist()


def grid_axes(n: int, panel: tuple[float, float] = (3.0, 2.8)):
    cols = 4
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(panel[0] * cols, panel[1] * rows))
    return fig, axes.ravel()


def page_scatter(pdf: PdfPages, predictions: pd.DataFrame, order: list[str], target: str) -> None:
    fig, axes = grid_axes(len(order))
    for ax, group in zip(axes, order):
        frame = predictions.loc[predictions["input_group"].eq(group)]
        if frame.empty:
            ax.axis("off")
            continue
        lo = (frame["y_pred"] - frame["posterior_p16"]).clip(lower=0)
        hi = (frame["posterior_p84"] - frame["y_pred"]).clip(lower=0)
        ax.errorbar(
            frame["y_true"], frame["y_pred"], yerr=np.vstack([lo, hi]),
            fmt="none", ecolor="0.6", alpha=0.18, elinewidth=0.3, rasterized=True,
        )
        ax.scatter(frame["y_true"], frame["y_pred"], s=2.5, alpha=0.35, rasterized=True, zorder=2)
        bounds = [
            min(frame["y_true"].min(), frame["y_pred"].min()),
            max(frame["y_true"].max(), frame["y_pred"].max()),
        ]
        ax.plot(bounds, bounds, "--", color="black", linewidth=0.8)
        r2 = r_squared(frame["y_true"].to_numpy(), frame["y_pred"].to_numpy())
        ig = float(frame["info_gain"].mean())
        ax.text(0.04, 0.96, f"$R^2$={r2:.3f}\nIG={ig:.3f}", transform=ax.transAxes, va="top", fontsize=8)
        ax.set_title(group, fontsize=9)
        ax.set_xlabel(f"true {target}", fontsize=8)
        ax.set_ylabel(f"predicted {target}", fontsize=8)
    for ax in axes[len(order):]:
        ax.axis("off")
    fig.suptitle("Predicted vs. true — all modality combinations", y=1.0)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_info_gain_hist(pdf: PdfPages, predictions: pd.DataFrame, order: list[str]) -> None:
    fig, axes = grid_axes(len(order))
    for ax, group in zip(axes, order):
        frame = predictions.loc[predictions["input_group"].eq(group)]
        if frame.empty:
            ax.axis("off")
            continue
        values = frame["info_gain"].clip(-3, 3)
        ax.hist(values, bins=60)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        frac_pos = float((frame["info_gain"] > 0).mean())
        ax.text(0.04, 0.95, f"mean={frame['info_gain'].mean():.3f}\nP(IG>0)={frac_pos:.2f}",
                transform=ax.transAxes, va="top", fontsize=8)
        ax.set_title(group, fontsize=9)
        ax.set_xlabel("per-source info gain (nats)", fontsize=8)
    for ax in axes[len(order):]:
        ax.axis("off")
    fig.suptitle("Per-source information gain vs. KDE prior (clipped to ±3)", y=1.0)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_calibration(pdf: PdfPages, predictions: pd.DataFrame, order: list[str]) -> None:
    rows = []
    for group in order:
        frame = predictions.loc[predictions["input_group"].eq(group)]
        if frame.empty:
            continue
        rows.append({
            "input_group": group,
            "cov68": float(((frame["y_true"] >= frame["posterior_p16"]) & (frame["y_true"] <= frame["posterior_p84"])).mean()),
            "below_p50": float((frame["y_true"] <= frame["posterior_p50"]).mean()),
        })
    table = pd.DataFrame(rows)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 0.35 * len(table) + 2))
    y = np.arange(len(table))
    ax1.barh(y, table["cov68"])
    ax1.axvline(0.68, color="black", linestyle="--", linewidth=1)
    ax1.set_yticks(y, table["input_group"], fontsize=8)
    ax1.set_xlabel("coverage of [p16, p84] (nominal 0.68)")
    ax2.barh(y, table["below_p50"])
    ax2.axvline(0.50, color="black", linestyle="--", linewidth=1)
    ax2.set_yticks(y, table["input_group"], fontsize=8)
    ax2.set_xlabel("fraction below median (nominal 0.50)")
    fig.suptitle("Posterior calibration")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def upset_plot(metrics: pd.DataFrame, output_path: Path) -> None:
    table = metrics.sort_values("r2").reset_index(drop=True)
    n = len(table)
    fig, (ax_r2, ax_ig, ax_dots) = plt.subplots(
        3, 1, figsize=(max(7, 0.62 * n), 7.5),
        gridspec_kw={"height_ratios": [3, 2, 1.4]}, sharex=True,
    )
    x = np.arange(n)
    ax_r2.bar(x, table["r2"])
    ax_r2.set_ylabel("$R^2$")
    for i, value in enumerate(table["r2"]):
        ax_r2.text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax_ig.bar(x, table["exp_info_gain"], color="#888")
    ax_ig.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax_ig.set_ylabel("exp(IG)")
    for row_index, modality in enumerate(MODALITIES):
        member = [modality in g.split("+") for g in table["input_group"]]
        ax_dots.scatter(x[member], [row_index] * sum(member), s=48, color="black")
        ax_dots.scatter(x[[not m for m in member]], [row_index] * (n - sum(member)),
                        s=48, facecolors="none", edgecolors="0.7")
    ax_dots.set_yticks(range(len(MODALITIES)), MODALITIES, fontsize=9)
    ax_dots.set_ylim(-0.6, len(MODALITIES) - 0.4)
    ax_dots.invert_yaxis()
    ax_dots.set_xticks([])
    fig.suptitle("Performance by modality combination")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def slice_tables(predictions: pd.DataFrame, order: list[str], packet_dir: Path, n_z_bins: int = 8) -> None:
    # per-spectype
    rows = []
    if "spectype" in predictions.columns:
        for group in order:
            frame = predictions.loc[predictions["input_group"].eq(group)]
            for spectype, sub in frame.groupby(predictions["spectype"].fillna("UNKNOWN")):
                if len(sub) < 10:
                    continue
                rows.append({
                    "input_group": group, "spectype": spectype, "n": len(sub),
                    "r2": r_squared(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy()),
                    "info_gain_nats": float(sub["info_gain"].mean()),
                })
    pd.DataFrame(rows).to_csv(packet_dir / "by_spectype.csv", index=False)

    # per-redshift (quantile bins over the full test set)
    z = predictions["redshift"]
    edges = np.nanquantile(z.unique(), np.linspace(0, 1, n_z_bins + 1))
    rows = []
    for group in order:
        frame = predictions.loc[predictions["input_group"].eq(group)]
        for k in range(n_z_bins):
            sub = frame.loc[(frame["redshift"] >= edges[k]) & (frame["redshift"] <= edges[k + 1])]
            if len(sub) < 20:
                continue
            rows.append({
                "input_group": group, "z_lo": edges[k], "z_hi": edges[k + 1],
                "z_mid": float(sub["redshift"].median()), "n": len(sub),
                "r2": r_squared(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy()),
                "info_gain_nats": float(sub["info_gain"].mean()),
                "exp_info_gain": float(np.exp(sub["info_gain"].mean())),
            })
    z_table = pd.DataFrame(rows)
    z_table.to_csv(packet_dir / "by_redshift.csv", index=False)
    if z_table.empty:
        print("[packet] redshift slices skipped (too few rows per bin)")
        return

    show = [g for g in ("z", "image", "spectra", "z+spectra+wise+image") if g in set(z_table["input_group"])]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for group in show:
        sub = z_table.loc[z_table["input_group"].eq(group)]
        ax1.plot(sub["z_mid"], sub["r2"], marker="o", label=group)
        ax2.plot(sub["z_mid"], sub["exp_info_gain"], marker="o", label=group)
    ax1.set_xlabel("redshift"); ax1.set_ylabel("$R^2$"); ax1.legend(fontsize=8)
    ax2.set_xlabel("redshift"); ax2.set_ylabel("exp(IG)"); ax2.axhline(1, color="black", ls="--", lw=0.8)
    fig.suptitle("Performance by redshift")
    fig.tight_layout()
    fig.savefig(packet_dir / "performance_by_redshift.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--staged-dir", type=Path, required=True)
    parser.add_argument("--target-label", default=None, help="Defaults to config.json's target.")
    args = parser.parse_args()

    run_dir = args.run_dir
    predictions = pd.read_csv(run_dir / "test_predictions.csv")
    metrics = pd.read_csv(run_dir / "test_flow_metrics.csv")
    target = args.target_label
    if target is None:
        import json
        config_path = run_dir / "config.json"
        target = json.loads(config_path.read_text()).get("target", "target") if config_path.exists() else "target"

    packet_dir = run_dir / "packet"
    packet_dir.mkdir(exist_ok=True)
    predictions = attach_metadata(predictions, args.staged_dir)
    order = combo_sort_order(metrics)

    with PdfPages(packet_dir / "test_diagnostics.pdf") as pdf:
        page_scatter(pdf, predictions, order, target)
        page_info_gain_hist(pdf, predictions, order)
        page_calibration(pdf, predictions, order)
    upset_plot(metrics, packet_dir / "upset_metrics.pdf")
    slice_tables(predictions, order, packet_dir)
    print(f"packet written to {packet_dir}: test_diagnostics.pdf, upset_metrics.pdf, "
          f"by_spectype.csv, by_redshift.csv, performance_by_redshift.pdf")


if __name__ == "__main__":
    main()
