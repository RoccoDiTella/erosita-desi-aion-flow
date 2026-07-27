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
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

DOCS = Path(__file__).resolve().parent
FIGS = DOCS / "figures"
ASSETS = DOCS.parent / "assets"
SLIDE = (13.333, 7.5)

APPENDIX_HEADS = [
    ("log_ml_flux_1", "log flux, 0.2-2.3 keV"),
    ("log_lx", "log L$_X$"),
    ("logmstar", "log M$_*$"),
    ("log_flux_p1", "P1 flux, 0.2-0.6 keV"),
    ("log_flux_p2", "P2 flux, 0.6-2.3 keV"),
    ("log_flux_p3", "P3 flux, 2.3-5.0 keV"),
    ("p2xp3_joint", "P2 $\\times$ P3 joint"),
]
MODS = [("spectra", "S"), ("z", "Z"), ("wise", "W"), ("image", "I")]


def marker(group: str) -> str:
    parts = set(str(group).split("+"))
    return " ".join(short if name in parts else "·" for name, short in MODS)


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


def _ramp(cmap, lo: float = 0.05, hi: float = 0.62):
    """Use only the light end of a colormap, so cell text stays legible."""
    return lambda t: cmap(lo + (hi - lo) * float(np.clip(t, 0.0, 1.0)))


def paper_table(ax, frame: pd.DataFrame, *, indicator_cols: int, numeric_cmaps: dict,
                rect=(0.08, 0.02, 0.84, 0.90), fontsize: int = 10) -> None:
    """The paper's results-table styling.

    Modality indicators as separate blank-or-letter columns, each numeric column
    shaded by its own colour ramp, dark header, NLL deliberately absent.
    """
    text = frame.astype(str)
    colors = [["white"] * len(frame.columns) for _ in range(len(frame))]
    for col, cmap in numeric_cmaps.items():
        if col not in frame.columns:
            continue
        vals = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        if not len(finite):
            continue
        vmin, vmax = float(finite.min()), float(finite.max())
        denom = max(vmax - vmin, 1e-8)
        ci = list(frame.columns).index(col)
        for ri, v in enumerate(vals):
            if np.isfinite(v):
                colors[ri][ci] = cmap((v - vmin) / denom)
    artist = ax.table(cellText=text.values, colLabels=list(frame.columns),
                      cellColours=colors, cellLoc="center", colLoc="center", bbox=rect)
    artist.auto_set_font_size(False)
    artist.set_fontsize(fontsize)
    for (row, col), cell in artist.get_celld().items():
        cell.set_linewidth(0.4)
        if row == 0:
            cell.set_facecolor("#222222")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
            continue
        if col < indicator_cols and cell.get_text().get_text().strip():
            cell.set_facecolor("#e9edf3")
            cell.get_text().set_weight("bold")
        # keep numbers readable at the dark end of each colour ramp
        r, g, b = cell.get_facecolor()[:3]
        if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5:
            cell.get_text().set_color("white")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mt-metrics", type=Path, default=None,
                        help="multi_test_metrics.csv from the V3b eval")
    parser.add_argument("--compare-metrics", type=Path, default=None,
                        help="V_PAI single-target flux table, for the appendix comparison column")
    parser.add_argument("--baseline", action="append", default=None, metavar="LABEL=PATH",
                        help="Extra single-target baseline table to render in the appendix; "
                        "repeatable, e.g. --baseline 'V_simple=v1_inject_metrics.csv'")
    parser.add_argument("--hr-csv", type=Path, default=None,
                        help="hr_implied_target.csv, for the appendix hardness slide")
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
            "one CLS token per target reads every block with the block's own frozen attention",
            "shared full-rank deltas on the CLS query + consumed-value projections; capacity control = weight decay",
            "shared MLP 768 $\\to$ 512 $\\to$ 256; sharing ends at the conditioning vectors; one flow per head",
            "joint 2-D flow over (P2, P3) $\\to$ hardness by exact marginalization, never a trained target",
            "losses: per-source availability masks + per-head EMA normalization",
            "cost of 8 heads $\\approx$ 1.2$\\times$ one head: the K/V reads are shared",
        ], fontsize=13.5, dy=0.13)
        pdf.savefig(fig); plt.close(fig)

        # ---- 8 MAIN RESULT
        fig, ax = new_slide("V3b results: every target from one run",
                            "test set, all inputs · P4 dropped (no signal) · HR32 is implied: marginalized out of the joint (P2,P3) posterior, never trained")
        image_panel(fig, FIGS / "fig_v3b_results.png", (0.03, 0.20, 0.94, 0.60))
        fig.text(0.05, 0.155,
                 "log flux 0.604 against V_PAI's 0.565 on the same test split.\n"
                 "Hardness was never trained, yet still gains information: 1.12$\\times$ on well-measured "
                 "sources,\npurely by marginalizing the joint (P2,P3) posterior.",
                 fontsize=12, color=INK, va="top")
        pdf.savefig(fig); plt.close(fig)

        # ---- 9 Training diagnostics
        fig, ax = new_slide("Training diagnostics",
                            "accumulated buckets, separate adapter learning rate · train probe = training data scored with the validation protocol")
        image_panel(fig, FIGS / "fig_v3b_training.png", (0.015, 0.10, 0.97, 0.72))
        fig.text(0.045, 0.082,
                 "Separating the adapter learning rate moved the flows into the healthy update band "
                 "(1.2e-4 $\\to$ 7.5e-4 per step), and logM$_*$ gained the most.\n"
                 "The bucket panel shows the old spiky curve was input composition, not optimizer noise. "
                 "No head dominates the shared trunk.\nOverfitting is now the binding constraint: the gap "
                 "reaches 0.16 nats, so early stopping fires at epoch 22.",
                 fontsize=11.5, color=INK, va="top")
        pdf.savefig(fig); plt.close(fig)

        # ---- 10 vs the paper architecture
        fig, ax = new_slide("Against the paper architecture",
                            "V_PAI trained on the same cleaned data and scored on the same test split · the published 0.549 used the noisy split and is not row-comparable")
        cmp = pd.DataFrame([
            {"": "V_PAI (paper head)", "targets": "1 (flux)", "runs": "1", "flux R²": "0.565"},
            {"": "V3b multi-head", "targets": "6 + hardness", "runs": "1", "flux R²": "0.604"},
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

        # ---- Appendix: full per-target tables, every input combination
        if args.mt_metrics and Path(args.mt_metrics).exists():
            mt = pd.read_csv(args.mt_metrics)
            cmp_r2 = {}
            if args.compare_metrics and Path(args.compare_metrics).exists():
                c = pd.read_csv(args.compare_metrics)
                cmp_r2 = {marker(r.input_group): float(r.r2) for r in c.itertuples()}
            fig, ax = new_slide("Appendix", "every target, every input combination")
            bullets(ax, [
                "one slide per target; rows are the 15 input combinations",
                "inputs: S spectra, Z redshift, W WISE, I image",
                "V_PAI is the paper head trained on flux alone, same data and test split",
            ], fontsize=15, dy=0.15)
            pdf.savefig(fig); plt.close(fig)

            for head, label in APPENDIX_HEADS:
                rows = mt[mt["head"] == head]
                if not len(rows):
                    continue
                rows = rows.sort_values("r2", ascending=True, na_position="first")
                recs = []
                for r in rows.itertuples():
                    parts = set(str(r.input_group).split("+"))
                    rec = {"Redshift": "Z" if "z" in parts else "",
                           "Spectra": "S" if "spectra" in parts else "",
                           "WISE": "W" if "wise" in parts else "",
                           "Images": "I" if "image" in parts else ""}
                    rec[r"$R^2$"] = "—" if pd.isna(r.r2) else f"{r.r2:.3f}"
                    rec["Info Gain"] = "—" if pd.isna(r.info_gain_nats) else f"{r.info_gain_nats:.3f}"
                    rec["exp(Info Gain)"] = ("—" if pd.isna(r.info_gain_nats)
                                             else f"{np.exp(r.info_gain_nats):.3f}")
                    recs.append(rec)
                sub = f"n = {int(rows.iloc[0].n_test):,} test sources"
                if head == "p2xp3_joint":
                    sub += " · 2-D flow: R² is not defined for a joint density"
                fig, ax = new_slide(f"Appendix: {label}", sub)
                paper_table(ax, pd.DataFrame(recs), indicator_cols=4,
                            numeric_cmaps={r"$R^2$": _ramp(plt.cm.Blues),
                                           "Info Gain": _ramp(plt.cm.Purples),
                                           "exp(Info Gain)": _ramp(plt.cm.Greens, 0.05, 0.5)},
                            rect=(0.06, 0.02, 0.88, 0.88), fontsize=10)
                pdf.savefig(fig); plt.close(fig)

            baselines = []
            if args.compare_metrics and Path(args.compare_metrics).exists():
                baselines.append(("V_PAI", Path(args.compare_metrics)))
            for spec in (args.baseline or []):
                lbl, _, path = spec.partition("=")
                if path and Path(path).exists():
                    baselines.append((lbl, Path(path)))
            for lbl, path in baselines:
                c = pd.read_csv(path).sort_values("r2", ascending=True)
                recs = []
                for r in c.itertuples():
                    parts = set(str(r.input_group).split("+"))
                    recs.append({
                        "Redshift": "Z" if "z" in parts else "",
                        "Spectra": "S" if "spectra" in parts else "",
                        "WISE": "W" if "wise" in parts else "",
                        "Images": "I" if "image" in parts else "",
                        r"$R^2$": f"{r.r2:.3f}",
                        "Info Gain": f"{r.info_gain_nats:.3f}",
                        "exp(Info Gain)": f"{r.exp_info_gain:.3f}"})
                head_note = ("paper head" if lbl == "V_PAI" else
                             "minimal head, 1 query and 1 layer" if lbl == "V_simple" else lbl)
                fig, ax = new_slide(f"Appendix: {lbl}, log flux (single-target)",
                                    f"{head_note} trained on flux alone · n = {int(c.iloc[0].n_test):,} · "
                                    "same cleaned data and test split")
                paper_table(ax, pd.DataFrame(recs), indicator_cols=4,
                            numeric_cmaps={r"$R^2$": _ramp(plt.cm.Blues),
                                           "Info Gain": _ramp(plt.cm.Purples),
                                           "exp(Info Gain)": _ramp(plt.cm.Greens, 0.05, 0.5)},
                            rect=(0.10, 0.02, 0.80, 0.88), fontsize=10)
                pdf.savefig(fig); plt.close(fig)

            if args.hr_csv and Path(args.hr_csv).exists():
                h = pd.read_csv(args.hr_csv)
                ok = h[h["ok"]] if "ok" in h.columns else h
                rows = []
                for name, d in (("all measured", h), ("well measured", ok)):
                    if len(d) < 30:
                        continue
                    r2 = 1 - np.sum((d.hr_meas - d.hr_p50) ** 2) / np.sum((d.hr_meas - d.hr_meas.mean()) ** 2)
                    rows.append({"subset": name, "n": f"{len(d):,}",
                                 "R²": f"{r2:+.3f}",
                                 "corr": f"{np.corrcoef(d.hr_p50, d.hr_meas)[0,1]:+.3f}",
                                 "IG (nats)": f"{d.info_gain.mean():+.3f}",
                                 "exp(IG)": f"{np.exp(d.info_gain.mean()):.2f}",
                                 "68% half-width": f"{np.median(0.5*(d.hr_p84-d.hr_p16)):.3f}"})
                if rows:
                    fig, ax = new_slide("Appendix: HR32, implied",
                                        "marginalized from the joint (P2,P3) posterior by quadrature · never a trained target")
                    metric_table(ax, pd.DataFrame(rows), rect=(0.06, 0.42, 0.88, 0.34), fontsize=12)
                    bullets(ax, [
                        "R² stays near zero: the posterior is narrower than the noisy measured values",
                        "the information gain is real, and it comes from a head that was never trained on hardness",
                    ], y=0.30, fontsize=12.5, dy=0.10)
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
