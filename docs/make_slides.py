#!/usr/bin/env python
"""Render the project summary deck -> docs/slides.pdf (16:9, minimal text).

    python docs/make_slides.py [--v1-metrics …] [--paperhead-metrics …]
                               [--v1-inject-metrics …] [--paperhead-inject-metrics …]

Naming: V_PAI = paper head (4 queries, 2 layers); V_simple = minimal head
(1 query, 1 layer).
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

MODALITY_ORDER = [("spectra", "S"), ("z", "Z"), ("wise", "W"), ("image", "I")]


def new_slide(title: str = "", subtitle: str = "") -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=SLIDE)
    fig.patch.set_facecolor("white")
    if title:
        fig.text(0.045, 0.935, title, fontsize=24, fontweight="bold", color=INK, va="top")
        fig.add_artist(plt.Line2D([0.045, 0.955], [0.875, 0.875], color=ACCENT, lw=2.5, transform=fig.transFigure))
    if subtitle:
        fig.text(0.045, 0.855, subtitle, fontsize=12, color=MUTED, va="top")
    ax = fig.add_axes([0.045, 0.05, 0.91, 0.77])
    ax.axis("off")
    return fig, ax


def bullets(ax: plt.Axes, items: list[str], *, x: float = 0.0, y: float = 0.97, dy: float = 0.11, fontsize: int = 14) -> None:
    for i, item in enumerate(items):
        ax.text(x, y - i * dy, "•  " + item, fontsize=fontsize, color=INK, va="top",
                transform=ax.transAxes, wrap=True)


def image_panel(fig: plt.Figure, path: Path, rect: tuple[float, float, float, float]) -> bool:
    if not path.exists():
        return False
    ax = fig.add_axes(rect)
    ax.imshow(mpimg.imread(path))
    ax.axis("off")
    return True


def metric_table(ax: plt.Axes, frame: pd.DataFrame, *, rect: tuple[float, float, float, float], fontsize: int = 12) -> None:
    table = ax.table(cellText=frame.values.tolist(), colLabels=frame.columns.tolist(), cellLoc="center", bbox=rect)
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    mono_cols = [i for i, c in enumerate(frame.columns) if c == "inputs"]
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row == 0:
            cell.set_facecolor(ACCENT)
            cell.set_text_props(color="white", fontweight="bold")
        elif col in mono_cols:
            cell.set_text_props(family="monospace")


def load_marker_table(metrics_csv: Path | None) -> pd.DataFrame | None:
    """Single aligned 'S Z W I' column (spaces when absent), R^2 + exp(IG)."""
    if metrics_csv is None or not Path(metrics_csv).exists():
        return None
    t = pd.read_csv(metrics_csv).sort_values("r2")
    rows = []
    for _, r in t.iterrows():
        parts = set(str(r.input_group).split("+"))
        combo = " ".join(short if name in parts else " " for name, short in MODALITY_ORDER)
        rows.append({"inputs": combo, "R²": f"{r.r2:.3f}", "exp(IG)": f"{r.exp_info_gain:.2f}"})
    return pd.DataFrame(rows)


def allinputs_numbers(metrics_csv: Path | None) -> tuple[float, float] | None:
    if metrics_csv is None or not Path(metrics_csv).exists():
        return None
    t = pd.read_csv(metrics_csv)
    r = t[t.input_group == "spectra+z+wise+image"].iloc[0]
    return float(r.r2), float(r.exp_info_gain)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-metrics", type=Path, default=None, help="V_simple + convolve")
    parser.add_argument("--paperhead-metrics", type=Path, default=None, help="V_PAI + convolve")
    parser.add_argument("--v1-inject-metrics", type=Path, default=None, help="V_simple + inject")
    parser.add_argument("--paperhead-inject-metrics", type=Path, default=None, help="V_PAI + inject")
    parser.add_argument("--output", type=Path, default=DOCS / "slides.pdf")
    args = parser.parse_args()

    with PdfPages(args.output) as pdf:
        # ---- 1 Title
        fig = plt.figure(figsize=SLIDE)
        fig.text(0.5, 0.60, "Predicting X-ray Properties from Optical/IR\nwith AION + Normalizing Flows",
                 fontsize=30, fontweight="bold", ha="center", color=INK)
        fig.text(0.5, 0.40, "eROSITA eRASS1 × DESI DR1 × Legacy Survey", fontsize=15, ha="center", color=ACCENT)
        pdf.savefig(fig); plt.close(fig)

        # ---- 2 Architecture
        fig, ax = new_slide("Architecture")
        arch = FIGS / "fig_architecture_poster.png"
        placed = image_panel(fig, arch, (0.03, 0.08, 0.70, 0.72)) or image_panel(fig, ASSETS / "architecture.png", (0.03, 0.08, 0.70, 0.72))
        ax2 = fig.add_axes([0.75, 0.15, 0.23, 0.60]); ax2.axis("off")
        bullets(ax2, [
            "frozen AION-1 base",
            "attention pooling → NSF flow",
            "V_PAI: 2 layers, 4 queries",
            "V_simple: 1 layer, 1 query",
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
            "3.4%: no z — no way to check",
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
            "catalog gives the error borders F−LOWERR,\n     F+UPERR → they ARE the ±1σ borders\n     of the split normal (in log space)",
            "draws truncated at 1.5σ per side",
            "logM★ has no catalog σ →\n     class floor: 0.2 (GALAXY) / 0.3 (QSO) dex",
            "eval: plain likelihood at observed y — no σ",
        ], fontsize=12.5, dy=0.17)
        pdf.savefig(fig); plt.close(fig)

        # ---- 5b Band coverage
        fig, ax = new_slide("X-ray band coverage", "cleaned sample, n = 26,632; selection is in the broad band")
        band_table = pd.DataFrame([
            {"band": "broad 0.2–2.3", "measured": "100%", "detected": "100%"},
            {"band": "P1  0.2–0.6",   "measured": "79%",  "detected": "31%"},
            {"band": "P2  0.6–2.3",   "measured": "94%",  "detected": "56%"},
            {"band": "P3  2.3–5.0",   "measured": "92%",  "detected": "51%"},
            {"band": "P4  5.0–8.0",   "measured": "48%",  "detected": "5%"},
            {"band": "P5",             "measured": "22%",  "detected": "0.2%"},
        ])
        metric_table(ax, band_table, rect=(0.05, 0.20, 0.48, 0.62), fontsize=13)
        ax2 = fig.add_axes([0.60, 0.25, 0.37, 0.40]); ax2.axis("off")
        bullets(ax2, [
            "HR32 = (P3−P2) / (P3+P2), from rates",
            "both HR bands measured: 86%;\n     both detected: 30%",
        ], fontsize=14, dy=0.30)
        pdf.savefig(fig); plt.close(fig)

        # ---- 6 Results
        runs = [
            ("V_simple · convolve", args.v1_metrics),
            ("V_PAI · convolve", args.paperhead_metrics),
            ("V_simple · inject", args.v1_inject_metrics),
            ("V_PAI · inject", args.paperhead_inject_metrics),
        ]
        available = [(label, path) for label, path in runs if load_marker_table(path) is not None]
        detail = [r for r in available if "inject" in r[0]] or available
        fig, ax = new_slide("Results — log flux, cleaned data",
                            "paper anchors: 0.549 published · 0.567 on clean rows")
        if len(detail) >= 2:
            (label_a, path_a), (label_b, path_b) = detail[0], detail[1]
            ta = pd.read_csv(path_a).sort_values("r2").reset_index(drop=True)
            tb = pd.read_csv(path_b).set_index("input_group").loc[ta.input_group].reset_index()
            rows = []
            for (_, ra), (_, rb) in zip(ta.iterrows(), tb.iterrows()):
                parts = set(str(ra.input_group).split("+"))
                combo = " ".join(short if name in parts else " " for name, short in MODALITY_ORDER)
                rows.append({"inputs": combo,
                             "R²": f"{ra.r2:.3f}", "exp(IG)": f"{ra.exp_info_gain:.2f}",
                             "R² ": f"{rb.r2:.3f}", "exp(IG) ": f"{rb.exp_info_gain:.2f}"})
            merged = pd.DataFrame(rows)
            metric_table(ax, merged, rect=(0.14, 0.02, 0.72, 0.86), fontsize=11)
            # supertitles over columns 2-3 and 4-5 (table spans x 0.14-0.86 of the axes,
            # inputs column ~ first fifth)
            fig.text(0.415, 0.795, label_a.split(" · ")[0], fontsize=14, fontweight="bold", ha="center", color=INK)
            fig.text(0.675, 0.795, label_b.split(" · ")[0], fontsize=14, fontweight="bold", ha="center", color=INK)
        if len(available) > 2:
            summary = pd.DataFrame([
                {"run": label, "R² (all)": f"{allinputs_numbers(path)[0]:.3f}", "exp(IG)": f"{allinputs_numbers(path)[1]:.2f}"}
                for label, path in available
            ])
            fig2, ax_s = new_slide("Results — cross-run summary (all inputs)")
            metric_table(ax_s, summary, rect=(0.15, 0.30, 0.60, 0.45), fontsize=14)
            pdf.savefig(fig); plt.close(fig)
            pdf.savefig(fig2); plt.close(fig2)
        else:
            pdf.savefig(fig); plt.close(fig)

        # ---- 7 Next
        fig, ax = new_slide("Next")
        bullets(ax, [
            "per-target sweep: log Lx · logM★ · HR",
            "50-epoch paper reproduction",
            "token cache → 5–15× faster epochs",
            "V_3 head; self-trained encoder",
        ], fontsize=17, dy=0.14)
        pdf.savefig(fig); plt.close(fig)

    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
