#!/usr/bin/env python
"""Render the project summary as a 16:9 slide deck -> docs/slides.pdf.

Each slide is one matplotlib figure page. Content pulls from the living docs
(pipeline.md / decisions.md are the sources of truth), the generated figures in
docs/figures/, and per-run metrics CSVs when present. Regenerate with:

    python docs/make_slides.py [--v1-metrics <test_flow_metrics.csv>]
                               [--paperhead-metrics <test_flow_metrics.csv>]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

DOCS = Path(__file__).resolve().parent
FIGS = DOCS / "figures"
SLIDE = (13.333, 7.5)  # 16:9 inches

INK = "#1a1a1a"
ACCENT = "#0072B2"  # Okabe-Ito blue, matching the figure palette
MUTED = "#6a6a6a"


def new_slide(title: str, subtitle: str = "") -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=SLIDE)
    fig.patch.set_facecolor("white")
    fig.text(0.045, 0.935, title, fontsize=24, fontweight="bold", color=INK, va="top")
    if subtitle:
        fig.text(0.045, 0.858, subtitle, fontsize=13, color=MUTED, va="top")
    fig.add_artist(plt.Line2D([0.045, 0.955], [0.88, 0.88], color=ACCENT, lw=2.5, transform=fig.transFigure))
    ax = fig.add_axes([0.045, 0.05, 0.91, 0.78])
    ax.axis("off")
    return fig, ax


def bullets(ax: plt.Axes, items: list[str], *, x: float = 0.0, y: float = 0.97, dy: float = 0.088, fontsize: int = 15) -> None:
    for i, item in enumerate(items):
        indent = item.startswith("  ")
        ax.text(
            x + (0.03 if indent else 0.0), y - i * dy,
            ("◦  " if indent else "•  ") + item.strip(),
            fontsize=fontsize - (1 if indent else 0), color=INK, va="top", transform=ax.transAxes, wrap=True,
        )


def image_panel(fig: plt.Figure, path: Path, rect: tuple[float, float, float, float]) -> None:
    if not path.exists():
        return
    ax = fig.add_axes(rect)
    ax.imshow(mpimg.imread(path))
    ax.axis("off")


def metric_table(ax: plt.Axes, frame: pd.DataFrame, *, rect: tuple[float, float, float, float], fontsize: int = 11) -> None:
    table = ax.table(
        cellText=frame.values.tolist(), colLabels=frame.columns.tolist(),
        cellLoc="center", bbox=rect,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row == 0:
            cell.set_facecolor(ACCENT)
            cell.set_text_props(color="white", fontweight="bold")


def load_headline(metrics_csv: Path | None) -> pd.DataFrame | None:
    if metrics_csv is None or not Path(metrics_csv).exists():
        return None
    t = pd.read_csv(metrics_csv)
    keep = ["z", "wise", "image", "spectra", "spectra+z+image", "spectra+z+wise+image"]
    t = t[t.input_group.isin(keep)].copy()
    t["order"] = t.input_group.map({k: i for i, k in enumerate(keep)})
    t = t.sort_values("order")
    out = pd.DataFrame({
        "inputs": t.input_group,
        "R²": t.r2.round(3),
        "exp(IG)": t.exp_info_gain.round(2),
        "RMSE (dex)": t.rmse.round(3),
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-metrics", type=Path, default=None)
    parser.add_argument("--paperhead-metrics", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DOCS / "slides.pdf")
    args = parser.parse_args()

    with PdfPages(args.output) as pdf:
        # ---- 1. Title
        fig = plt.figure(figsize=SLIDE)
        fig.text(0.5, 0.62, "Predicting X-ray Properties from\nOptical/IR with AION + Normalizing Flows",
                 fontsize=30, fontweight="bold", ha="center", color=INK)
        fig.text(0.5, 0.42, "Cleaned crossmatch · uncertainty-aware likelihoods · per-target flows",
                 fontsize=16, ha="center", color=ACCENT)
        fig.text(0.5, 0.30, "eROSITA eRASS1 × DESI DR1 × Legacy Survey  |  FASRC  |  extends the PAI26 paper",
                 fontsize=12, ha="center", color=MUTED)
        pdf.savefig(fig); plt.close(fig)

        # ---- 2. Architecture
        fig, ax = new_slide("Architecture", "Frozen AION-1 base (318M + codecs) → attention pooling → conditional NSF flow")
        image_panel(fig, FIGS / "fig_architecture.png", (0.06, 0.10, 0.60, 0.66))
        ax2 = fig.add_axes([0.68, 0.10, 0.29, 0.66]); ax2.axis("off")
        bullets(ax2, [
            "Per batch: one modality combo (25% singles / pairs / triples / all)",
            "Paper head: 4 queries, 2 layers (~16.8M)",
            "V1 head: 1 query, 1 layer (~7.8M)",
            "Flow: 1-D neural spline, KDE prior anchors info gain",
            "Per-target runs: flux, Lx, logM*, HR",
        ], fontsize=13, dy=0.14)
        pdf.savefig(fig); plt.close(fig)

        # ---- 3. Data & cleaning
        fig, ax = new_slide("Data & NWAY cleaning", "32,092 paper rows → keep NWAY-correct only → dedup → re-split (25,200)")
        image_panel(fig, FIGS / "fig_cleanup.png", (0.05, 0.10, 0.62, 0.66))
        ax2 = fig.add_axes([0.69, 0.10, 0.28, 0.66]); ax2.axis("off")
        bullets(ax2, [
            "Naive 5″ match: 12.5% not confirmed",
            "correct 26,632 · wrong 1,017 ·",
            "  ambiguous 1,033 · spurious 1,514",
            "Paper model 2× worse on rejects",
            "Impossible values (log Lx=48.1)",
            "  are all NWAY rejects",
            "One staged copy + runtime view",
        ], fontsize=13, dy=0.105)
        pdf.savefig(fig); plt.close(fig)

        # ---- 4. Errors
        fig, ax = new_slide("Measurement-error-aware training", "Split-normal kernel; flow learns the deconvolved density")
        image_panel(fig, FIGS / "fig_errors.png", (0.05, 0.10, 0.62, 0.66))
        ax2 = fig.add_axes([0.69, 0.10, 0.28, 0.66]); ax2.axis("off")
        bullets(ax2, [
            "log p(y|x) = log ∫ p(t|x) K(y|t) dt",
            "41-node quadrature, ±5σ",
            "flux σ: −log₁₀(1−LO/F), log₁₀(1+UP/F)",
            "HR: u = arctanh(HR), σᵤ = σ/(1−HR²)",
            "logM*: spectype floor 0.2/0.3 dex",
            "Eval scores through the same kernel",
        ], fontsize=13, dy=0.105)
        pdf.savefig(fig); plt.close(fig)

        # ---- 5. Validation & lessons
        fig, ax = new_slide("Validation gates (and what they caught)", "No GPU-hour runs on unvalidated data")
        bullets(ax, [
            "validate_staged.py: schema · leakage · clean filter · σ>0 · physical ranges · derived-column consistency · fits coverage · model contract — hard gate in train.sbatch",
            "Caught: interrupted unzip had silently dropped 62% of the sample (10,355/27,373 cutouts on disk; no error anywhere)",
            "Caught: 4 logmstar=0 sentinel rows; out-of-range Lx/logM* rows all trace to NWAY-rejected matches",
            "Caught (by audit): convolve-trained flows were scored un-convolved — IG was structurally understated",
            "Break-one-thing tests: every validator check has a negative test that proves it fires",
            "Ops: count-based idempotency · sbatch-only compute · one canonical data copy · measured timings before submits",
        ], fontsize=14, dy=0.145)
        pdf.savefig(fig); plt.close(fig)

        # ---- 6. Results
        headline = load_headline(args.v1_metrics)
        paperhead = load_headline(args.paperhead_metrics)
        fig, ax = new_slide("Results — log flux (0.2–2.3 keV), cleaned + convolve",
                            "Anchors: paper published 0.549 (noisy test) · paper model on clean rows 0.567")
        if headline is not None:
            fig.text(0.075, 0.80, "V1 head (7.8M, 15 epochs)", fontsize=14, fontweight="bold", color=INK)
            metric_table(ax, headline, rect=(0.03, 0.28 if paperhead is None else 0.02, 0.44, 0.60), fontsize=12)
        if paperhead is not None:
            fig.text(0.555, 0.80, "Paper head (16.8M, same data)", fontsize=14, fontweight="bold", color=INK)
            metric_table(ax, paperhead, rect=(0.52, 0.02, 0.44, 0.60), fontsize=12)
        bullets(ax, [
            "Modality ordering reproduces the paper: z < WISE < image < spectra < combos",
            "QSO R² 0.60 (n=2,195) vs GALAXY 0.40 (n=325); signal concentrated at z < 0.7",
        ], y=0.14 if paperhead is None else -0.06, fontsize=12, dy=0.08)
        pdf.savefig(fig); plt.close(fig)

        # ---- 7. Next steps
        fig, ax = new_slide("Next steps")
        bullets(ax, [
            "Finish V1 vs paper-head A/B on identical clean data (in flight) → pick the sweep head",
            "Paper-table reproduction (paper head, error_mode=none, native split, 50 epochs)",
            "Per-target runs: log Lx · logM* · HR (arctanh, IG-primary, S/N gate)",
            "AION token cache (~185 GB per-combo): 5–15× epoch speedup for the sweep",
            "siag_gpu (32×A100) once membership lands; V2 self-trained encoder; upper-limit censoring",
        ], fontsize=15, dy=0.13)
        pdf.savefig(fig); plt.close(fig)

    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
