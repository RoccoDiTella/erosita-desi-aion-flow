#!/usr/bin/env python
"""Render the project summary deck -> docs/slides.pdf (16:9, minimal text).

    python docs/make_slides.py [--mt-metrics multi_test_metrics.csv]

Narrative: setup -> data quality -> what is measurable -> V3b design ->
V3b results -> training diagnostics -> vs the paper architecture ->
attribution (modality, then spectral) -> efficiency -> next.

Full per-combination tables live in the interactive HTML deck
(docs/make_html_deck.py), not here.
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
ASSETS = DOCS.parent / "assets"
SLIDE = (13.333, 7.5)

INK = "#1a1a1a"
ACCENT = "#0072B2"
MUTED = "#6a6a6a"


def new_slide(title: str = "", subtitle: str = ""):
    fig = plt.figure(figsize=SLIDE)
    fig.patch.set_facecolor("white")
    if title:
        fig.text(0.045, 0.935, title, fontsize=24, fontweight="bold", color=INK, va="top")
        fig.add_artist(plt.Line2D([0.045, 0.955], [0.875, 0.875], color=ACCENT, lw=2.5,
                                  transform=fig.transFigure))
    if subtitle:
        fig.text(0.045, 0.855, subtitle, fontsize=12, color=MUTED, va="top")
    ax = fig.add_axes([0.045, 0.05, 0.91, 0.77])
    ax.axis("off")
    return fig, ax


def bullets(ax, items, *, x=0.0, y=0.97, dy=0.11, fontsize=14):
    for i, item in enumerate(items):
        ax.text(x, y - i * dy, "•  " + item, fontsize=fontsize, color=INK, va="top",
                transform=ax.transAxes, wrap=True)


def image_panel(fig, path: Path, rect):
    if not path.exists():
        return False
    ax = fig.add_axes(rect)
    ax.imshow(mpimg.imread(path))
    ax.axis("off")
    return True


def metric_table(ax, frame: pd.DataFrame, *, rect, fontsize=12):
    table = ax.table(cellText=frame.values.tolist(), colLabels=frame.columns.tolist(),
                     cellLoc="center", bbox=rect)
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row == 0:
            cell.set_facecolor(ACCENT)
            cell.set_text_props(color="white", fontweight="bold")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mt-metrics", type=Path, default=None,
                        help="multi_test_metrics.csv from the V3b eval")
    parser.add_argument("--output", type=Path, default=DOCS / "slides.pdf")
    args = parser.parse_args()

    with PdfPages(args.output) as pdf:
        # ---- 1 Title
        fig = plt.figure(figsize=SLIDE)
        fig.text(0.5, 0.60, "Predicting X-ray Properties from Optical/IR\nwith AION + Normalizing Flows",
                 fontsize=30, fontweight="bold", ha="center", color=INK)
        fig.text(0.5, 0.40, "eROSITA eRASS1 × DESI DR1 × Legacy Survey", fontsize=15,
                 ha="center", color=ACCENT)
        pdf.savefig(fig); plt.close(fig)

        # ---- 2 Architecture
        fig, ax = new_slide("Architecture")
        image_panel(fig, FIGS / "fig_architecture_poster.png", (0.03, 0.08, 0.70, 0.72)) or \
            image_panel(fig, ASSETS / "architecture.png", (0.03, 0.08, 0.70, 0.72))
        ax2 = fig.add_axes([0.75, 0.15, 0.23, 0.60]); ax2.axis("off")
        bullets(ax2, [
            "frozen AION-1 base",
            "pooled context → NSF flow",
            "V_PAI: the paper head",
            "V3b: read-only CLS,\n     one head per target",
        ], fontsize=14, dy=0.16)
        pdf.savefig(fig); plt.close(fig)

        # ---- 3 Buchner comment (screenshot only)
        fig = plt.figure(figsize=SLIDE)
        fig.patch.set_facecolor("white")
        image_panel(fig, FIGS / "johannes_buchner_comment.png", (0.05, 0.28, 0.90, 0.44))
        pdf.savefig(fig); plt.close(fig)

        # ---- 4 NWAY cleaning
        fig, ax = new_slide("Crossmatch cleaning with NWAY",
                            "Salvato+2025 (A&A 704 A344): Bayesian matching of eRASS1 X-ray sources to optical counterparts")
        image_panel(fig, FIGS / "fig_dz_log.png", (0.04, 0.08, 0.52, 0.62))
        ax2 = fig.add_axes([0.58, 0.08, 0.40, 0.66]); ax2.axis("off")
        bullets(ax2, [
            "in NWAY, ~95% of our sources have a\n     confirmed optical match",
            "3.3%: z mismatch > 0.01",
            "3.4%: no z, so no way to check",
            "we keep only NWAY-confirmed sources",
            "PAI performance: R² 0.549",
            "PAI on mismatched objects: R² 0.29",
        ], fontsize=13, dy=0.15)
        pdf.savefig(fig); plt.close(fig)

        # ---- 5 Errors
        fig, ax = new_slide("Measurement errors: per-source split normal")
        image_panel(fig, FIGS / "fig_split_normal.png", (0.04, 0.08, 0.55, 0.66))
        ax2 = fig.add_axes([0.62, 0.15, 0.36, 0.55]); ax2.axis("off")
        bullets(ax2, [
            "the catalog reports the error borders;\n     in log space these are the ±1σ\n     points of the split normal",
            "draws truncated at 1.5σ per side;\n     50 draws/step, broadcast under one\n     conditioner pass (free)",
            "logM★ has no catalog σ →\n     class floor: 0.2 (GALAXY) / 0.3 (QSO) dex",
            "eval: plain likelihood at observed y,\n     no σ involved",
        ], fontsize=12.5, dy=0.17)
        pdf.savefig(fig); plt.close(fig)

        # ---- 6 Band coverage (what is measurable at all)
        fig, ax = new_slide("X-ray band coverage", "cleaned sample, n = 26,632; selection is in the broad band")
        band_table = pd.DataFrame([
            {"band": "broad 0.2–2.3", "measured": "100%", "detected": "100%"},
            {"band": "P1  0.2–0.6", "measured": "79%", "detected": "31%"},
            {"band": "P2  0.6–2.3", "measured": "94%", "detected": "56%"},
            {"band": "P3  2.3–5.0", "measured": "92%", "detected": "51%"},
            {"band": "P4  5.0–8.0", "measured": "48%", "detected": "5%"},
        ])
        metric_table(ax, band_table, rect=(0.05, 0.30, 0.48, 0.52), fontsize=13)
        ax2 = fig.add_axes([0.60, 0.30, 0.37, 0.44]); ax2.axis("off")
        bullets(ax2, [
            "HR32 = (P3−P2)/(P3+P2), from rates",
            "both HR bands measured: 86%;\n     both detected: 30%",
            "training gate: σ ≤ 1.0 per band:\n     precision, not detection",
        ], fontsize=13, dy=0.24)
        fig.text(0.05, 0.155,
                 "measured: forced photometry gives a positive rate in this band\n"
                 "detected: DET_LIKE ≥ 6, the catalog threshold (≈14% spurious at 6, 4% at ≥8; Seppi+2022)",
                 fontsize=10.5, color=MUTED)
        pdf.savefig(fig); plt.close(fig)

        # ---- 7 V3b design
        fig, ax = new_slide("V3b: read-only CLS, one frozen forward, 8 heads",
                            "no token attends to a CLS: the data stream is bit-identical to frozen AION and runs without gradients")
        bullets(ax, [
            "8 CLS tokens read every block with the block's own frozen attention",
            "shared full-rank deltas on the CLS query + consumed-value projections; capacity control = weight decay",
            "shared MLP 768 $\\to$ 512 $\\to$ 256; sharing ends at the conditioning vectors; one flow per head",
            "joint 2-D flow over (P2, P3) $\\to$ hardness by exact marginalization, never a trained target",
            "losses: per-source availability masks + per-head EMA normalization",
            "cost of 8 heads $\\approx$ 1.2$\\times$ one head: the K/V reads are shared",
        ], fontsize=13.5, dy=0.13)
        pdf.savefig(fig); plt.close(fig)

        # ---- 8 MAIN RESULT
        fig, ax = new_slide("V3b results: every target from one 3-hour run",
                            "test set, all inputs · HR32 is implied: marginalized out of the joint (P2,P3) posterior, never trained")
        image_panel(fig, FIGS / "fig_v3b_results.png", (0.03, 0.20, 0.94, 0.60))
        fig.text(0.05, 0.155,
                 "log flux 0.603 against V_PAI's 0.565 on the same test split.\n"
                 "Hardness was never trained, yet still gains information: 1.13$\\times$ and correlation +0.25 on\n"
                 "well-measured sources, purely by marginalizing the joint (P2,P3) posterior.",
                 fontsize=12, color=INK, va="top")
        pdf.savefig(fig); plt.close(fig)

        # ---- 9 Training diagnostics
        fig, ax = new_slide("Training diagnostics",
                            "accumulated-bucket run: every update sees the full input mix · train probe = training data scored with the validation protocol")
        image_panel(fig, FIGS / "fig_v3b_training.png", (0.015, 0.10, 0.97, 0.72))
        fig.text(0.045, 0.082,
                 "Convergence is uneven: P1/P2 by epoch 5, logM$_*$ still improving at 20.  "
                 "Overfitting grows linearly, 0.009 $\\to$ 0.062 nats.\n"
                 "The old spiky curve was the 0.35-nat spread between input buckets, not optimizer noise.  "
                 "No head dominates the shared trunk;\nthe adapters move 10-20$\\times$ more than the CLS "
                 "tokens and the shared MLP.",
                 fontsize=11.5, color=INK, va="top")
        pdf.savefig(fig); plt.close(fig)

        # ---- 10 vs the paper architecture
        fig, ax = new_slide("Against the paper architecture",
                            "V_PAI trained on the same cleaned data and scored on the same test split · the published 0.549 used the noisy split and is not row-comparable")
        cmp = pd.DataFrame([
            {"": "V_PAI (paper head)", "targets": "1 (flux)", "runs": "1", "flux R²": "0.565"},
            {"": "V3b multi-head", "targets": "7 + hardness", "runs": "1", "flux R²": "0.603"},
        ])
        metric_table(ax, cmp, rect=(0.04, 0.46, 0.62, 0.34), fontsize=13)
        ax2 = fig.add_axes([0.70, 0.42, 0.28, 0.38]); ax2.axis("off")
        bullets(ax2, [
            "better flux, and every\n     other target for free",
            "one run replaces six",
            "the encoder never moves;\n     only the readers learn",
        ], fontsize=13, dy=0.28)
        pdf.savefig(fig); plt.close(fig)

        # ---- 11 Modality Shapley
        if (FIGS / "fig_modality_shapley.png").exists():
            fig, ax = new_slide("Information by input type",
                                "exact Shapley over the 4 inputs, from the 16-coalition test tables")
            image_panel(fig, FIGS / "fig_modality_shapley.png", (0.08, 0.08, 0.84, 0.72))
            pdf.savefig(fig); plt.close(fig)

        # ---- 12 Pairwise interactions
        if (FIGS / "fig_modality_interactions.png").exists():
            fig, ax = new_slide("Input-type interactions",
                                "exact pairwise Shapley interaction index of info gain, same 16 coalitions")
            image_panel(fig, FIGS / "fig_modality_interactions.png", (0.16, 0.05, 0.60, 0.75))
            ax2 = fig.add_axes([0.76, 0.15, 0.23, 0.55]); ax2.axis("off")
            bullets(ax2, [
                "L$_X$: spectra + z redundant\n   (spectrum carries z)",
                "logM$_*$: spectra + WISE\n   synergistic",
                "flux: mild redundancy\n   everywhere",
            ], fontsize=12, dy=0.28)
            pdf.savefig(fig); plt.close(fig)

        # ---- 13 Line coverage
        fig, ax = new_slide("Emission-line coverage",
                            "each spectrum covers 3600-9824 A observed; in rest frame that window slides with z")
        image_panel(fig, FIGS / "fig_line_coverage.png", (0.05, 0.06, 0.90, 0.74))
        pdf.savefig(fig); plt.close(fig)

        # ---- 14 Line/continuum Shapley
        if (FIGS / "fig_shapley_heatmap.png").exists():
            fig, ax = new_slide("Where in the spectrum is the flux information?",
                                "line/continuum Shapley, AION-native token dropping (guard band from measured codec receptive field)")
            image_panel(fig, FIGS / "fig_shapley_heatmap.png", (0.03, 0.42, 0.94, 0.40))
            image_panel(fig, FIGS / "fig_shapley_lines.png", (0.16, 0.01, 0.48, 0.40))
            ax2 = fig.add_axes([0.68, 0.06, 0.30, 0.32]); ax2.axis("off")
            bullets(ax2, [
                "Balmer lines dominate",
                "Hbeta+[OIII] merged: one player\n   (4959 doublet inseparable)",
                "line vs continuum split under\n   re-measurement (guard audit)",
            ], fontsize=12, dy=0.30)
            pdf.savefig(fig); plt.close(fig)

        # ---- 15 Efficiency (concise)
        fig, ax = new_slide("Efficiency", "GPU wall-time is the entire fairshare cost (A100 = 836 CPU-core-equivalents)")
        bullets(ax, [
            "batches sized at the measured throughput knee, not at the memory limit",
            "combos packed by token length: 4 buckets, padding $\\leq$ 1.5%",
            "50 noise draws per step, broadcast under one conditioner pass: free",
            "8 targets in 3 GPU-hours, where 6 separate runs cost ~9",
        ], fontsize=15, dy=0.15)
        pdf.savefig(fig); plt.close(fig)

        # ---- 16 Next
        fig, ax = new_slide("Next")
        bullets(ax, [
            "training regime: per-head schedules, shorter run (see diagnostics)",
            "hardness IG by exact marginalization of the joint posterior",
            "band-σ inflation probes; Shapley guard audit",
            "predictions for X-ray non-detections (Buchner's test)",
        ], fontsize=17, dy=0.14)
        pdf.savefig(fig); plt.close(fig)

    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
