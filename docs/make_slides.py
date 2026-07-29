#!/usr/bin/env python
"""Render the project summary deck -> docs/slides.pdf (16:9, minimal text).

    python docs/make_slides.py [--mt-metrics multi_test_metrics.csv]

Narrative: framing -> the data (provenance, what it looks like, the empirical
joint, counterpart validation, band coverage) -> training (train vs validation,
total then per target) -> per-target tables for the three headline targets ->
results against the emission-line baseline -> what each input modality is worth
-> emission-line coverage.

Full per-combination tables for every head live in the interactive HTML deck
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

# Cut to the three headline targets on request; every other head is still in
# the results figure and in the HTML deck.
TABLE_HEADS = [
    ("log_ml_flux_1", "log flux, 0.2-2.3 keV"),
    ("log_lx", "log L$_X$"),
    ("log_sfr", "log SFR"),
]
MODS = [("spectra", "S"), ("z", "Z"), ("wise", "W"), ("image", "I")]

INK = "#1a1a1a"
ACCENT = "#0072B2"
MUTED = "#6a6a6a"


def marker(group: str) -> str:
    parts = set(str(group).split("+"))
    return " ".join(short if name in parts else "·" for name, short in MODS)


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


def metric_table(ax, frame: pd.DataFrame, *, rect, fontsize=12, left_align_first=False):
    table = ax.table(cellText=frame.values.tolist(), colLabels=frame.columns.tolist(),
                     cellLoc="center", bbox=rect)
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row == 0:
            cell.set_facecolor(ACCENT)
            cell.set_text_props(color="white", fontweight="bold")
        elif left_align_first and col == 0:
            cell.get_text().set_ha("left")
            cell._text.set_x(0.02)


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


def combo_table(rows: pd.DataFrame) -> pd.DataFrame:
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
    return pd.DataFrame(recs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mt-metrics", type=Path, default=None,
                        help="multi_test_metrics.csv from the multi-target eval")
    parser.add_argument("--counts-csv", type=Path, default=None,
                        help="per-variable usable row counts for the data slide")
    parser.add_argument("--output", type=Path, default=DOCS / "slides.pdf")
    args = parser.parse_args()

    with PdfPages(args.output) as pdf:
        # ---- Title
        fig = plt.figure(figsize=SLIDE)
        fig.text(0.5, 0.60, "Predicting X-ray Properties from Optical/IR\nwith AION + Normalizing Flows",
                 fontsize=30, fontweight="bold", ha="center", color=INK)
        fig.text(0.5, 0.40, "eROSITA eRASS1 × DESI DR1 × Legacy Survey", fontsize=15,
                 ha="center", color=ACCENT)
        pdf.savefig(fig); plt.close(fig)

        # ---- Architecture
        fig, ax = new_slide("Architecture")
        image_panel(fig, FIGS / "fig_architecture_poster.png", (0.03, 0.08, 0.70, 0.72)) or \
            image_panel(fig, ASSETS / "architecture.png", (0.03, 0.08, 0.70, 0.72))
        ax2 = fig.add_axes([0.75, 0.15, 0.23, 0.60]); ax2.axis("off")
        bullets(ax2, [
            "frozen AION-1 base",
            "pooled context → NSF flow",
        ], fontsize=15, dy=0.16)
        pdf.savefig(fig); plt.close(fig)

        # ---- Buchner comment (screenshot only)
        fig = plt.figure(figsize=SLIDE)
        fig.patch.set_facecolor("white")
        image_panel(fig, FIGS / "johannes_buchner_comment.png", (0.05, 0.28, 0.90, 0.44))
        pdf.savefig(fig); plt.close(fig)

        # ---- DATA 1: where everything comes from, and how much of it is usable
        counts = {}
        if args.counts_csv and Path(args.counts_csv).exists():
            c = pd.read_csv(args.counts_csv)
            counts = dict(zip(c.variable.astype(str), c.n_usable.astype(int)))

        def n(key, default):
            return f"{counts.get(key, default):,}"

        fig, ax = new_slide("The data", "one row per source after cleaning · n = 25,200")
        inputs = pd.DataFrame([
            {"input": "optical spectrum", "from": "DESI DR1"},
            {"input": "redshift", "from": "DESI DR1 (Redrock)"},
            {"input": "image cutout, grz", "from": "Legacy Survey DR10"},
            {"input": "WISE W1-W3", "from": "Legacy Survey DR10"},
        ])
        metric_table(ax, inputs, rect=(0.02, 0.70, 0.42, 0.26), fontsize=12,
                     left_align_first=True)
        # How much of the sample carries several targets at once. Heads are masked
        # per source, so a source missing one target still trains the others --
        # which is why "at least one" matters more than "all four".
        overlap = pd.DataFrame([
            {"joint availability": "M★ and SFR", "n": "17,294", "of sample": "69%"},
            {"joint availability": "M$_{BH}$ and SFR", "n": "6,533", "of sample": "26%"},
            {"joint availability": "M$_{BH}$, M★, SFR, P3", "n": "5,997", "of sample": "24%"},
            {"joint availability": "at least one of them", "n": "24,893", "of sample": "99%"},
        ])
        metric_table(ax, overlap, rect=(0.50, 0.70, 0.48, 0.26), fontsize=11.5,
                     left_align_first=True)
        tgt = pd.DataFrame([
            {"target": "log flux", "catalogue": "eROSITA eRASS1",
             "column / formula": "log₁₀ ML_FLUX_1", "n": n("log_ml_flux_1", 25200), "per-object error": "yes"},
            {"target": "log L$_X$", "catalogue": "eRASS1 + DESI z",
             "column / formula": "log₁₀(4π D$_L$(z)² · F)", "n": n("log_lx", 25200), "per-object error": "yes"},
            {"target": "P2, P3 flux", "catalogue": "eROSITA eRASS1",
             "column / formula": "log₁₀ ML_FLUX_P2, _P3",
             "n": f"{n('log_flux_p2', 23803)} / {n('log_flux_p3', 23151)}", "per-object error": "yes"},
            {"target": "HR32", "catalogue": "eROSITA eRASS1",
             "column / formula": "from the joint (P2,P3) flow", "n": "2,169", "per-object error": "—"},
            {"target": "log M★", "catalogue": "CIGALE (Siudek+2024)",
             "column / formula": "LOGM  (direct)", "n": n("logmstar_cigale", 19444), "per-object error": "yes"},
            {"target": "log SFR", "catalogue": "CIGALE (Siudek+2024)",
             "column / formula": "LOGSFR  (direct)", "n": n("log_sfr", 17294), "per-object error": "yes"},
            {"target": "log M$_{BH}$", "catalogue": "qmassiron BH VAC",
             "column / formula": "LOGMASS_DAS_PAN25  (direct)", "n": n("log_mbh_pan25", 9373), "per-object error": "yes"},
            {"target": "log M$_{BH}$ (VO09)", "catalogue": "qmassiron BH VAC",
             "column / formula": "LOGMASS_DAS_VO09  (direct)", "n": "9,366", "per-object error": "yes"},
            {"target": "log sSFR", "catalogue": "CIGALE (Siudek+2024)",
             "column / formula": "implied from the (M★,SFR) joint", "n": "17,294", "per-object error": "from the joint"},
        ])
        metric_table(ax, tgt, rect=(0.02, 0.05, 0.96, 0.58), fontsize=10.5,
                     left_align_first=True)
        pdf.savefig(fig); plt.close(fig)

        # ---- DATA 2: what the inputs look like
        fig, ax = new_slide("The data",
                            "four sources spanning the redshift range, same sources in both rows")
        image_panel(fig, FIGS / "fig_examples_spectra.png", (0.03, 0.34, 0.94, 0.48))
        image_panel(fig, FIGS / "fig_examples_images.png", (0.13, 0.02, 0.74, 0.30))
        pdf.savefig(fig); plt.close(fig)

        # ---- DATA 3: the empirical joint, drawn NATIVELY so it stays sharp when
        # zoomed. This is the one figure people zoom into, and a pasted raster
        # stops being useful the moment they do.
        fig, ax = new_slide("The data",
                            "empirical joint over the clean sample")
        npz = FIGS / "corner_data.npz"
        if npz.exists():
            import numpy as _np
            from make_data_figures import VARS as _VARS, draw_corner
            d = _np.load(npz)
            data = {n: d[n].astype(float) for n, _ in _VARS}
            k = len(_VARS)
            gs = fig.add_gridspec(k, k, left=0.20, right=0.82, bottom=0.05, top=0.79,
                                  wspace=0.13, hspace=0.13)
            axes = _np.empty((k, k), dtype=object)
            for i in range(k):
                for j in range(k):
                    axes[i, j] = fig.add_subplot(gs[i, j])
            draw_corner(fig, data, axes=axes, label_size=8.0, tick_size=5.6)
        else:
            image_panel(fig, FIGS / "fig_corner.png", (0.16, 0.02, 0.68, 0.80))
        pdf.savefig(fig); plt.close(fig)

        # ---- DATA 4: NWAY counterpart validation
        fig, ax = new_slide("The data", "counterparts validated against NWAY / Salvato+2025")
        nway = pd.DataFrame([
            {"outcome": "confirmed correct", "n": "26,632", "share": "87.5%"},
            {"outcome": "confirmed wrong", "n": "1,017", "share": "3.3%"},
            {"outcome": "unverifiable", "n": "2,792", "share": "9.2%"},
        ])
        metric_table(ax, nway, rect=(0.02, 0.42, 0.50, 0.34), fontsize=13,
                     left_align_first=True)
        fig.text(0.54, 0.72,
                 "Their LS10 object IDs could not be reconciled with ours,\n"
                 "so the same-object test is spectroscopic redshift agreement:\n\n"
                 "        |z − z$_{sp}$| < 0.01\n\n"
                 "That threshold sits in the valley of a strongly bimodal Δz\n"
                 "distribution (88.7% below 1e-4, ~3% above 0.1).\n\n"
                 "Caveat: z agreement is an identity proxy — two distinct\n"
                 "objects at the same redshift would pass as correct.",
                 fontsize=12.5, color=INK, va="top")
        fig.text(0.045, 0.30,
                 "We keep only the confirmed-correct rows. 'Unverifiable' is mostly sources NWAY gives no "
                 "counterpart at all\n(consistent with a spurious X-ray detection), plus counterparts with no "
                 "spectroscopic redshift to compare against.",
                 fontsize=12, color=MUTED, va="top")
        pdf.savefig(fig); plt.close(fig)

        # ---- DATA 5: band coverage
        fig, ax = new_slide("X-ray band coverage", "cleaned sample, n = 26,632")
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
            "HR32 = (P3−P2)/(P3+P2)",
            "both bands measured 86%, detected 30%",
            "we gate on σ, not on detection",
        ], fontsize=13.5, dy=0.22)
        fig.text(0.05, 0.145,
                 "measured: forced photometry gives a positive rate.   "
                 "detected: DET_LIKE ≥ 6, the catalog threshold.",
                 fontsize=11, color=MUTED)
        pdf.savefig(fig); plt.close(fig)

        # ---- TRAINING 1: train vs validation, whole model
        fig, ax = new_slide("Train vs validation",
                            "the train curve is a fixed training slice scored with the validation "
                            "protocol, so the two are directly comparable")
        image_panel(fig, FIGS / "fig_loss_total.png", (0.16, 0.04, 0.68, 0.76))
        pdf.savefig(fig); plt.close(fig)

        # ---- TRAINING 2: same, per target
        fig, ax = new_slide("Train vs validation, per target", "independent y-scale per head")
        image_panel(fig, FIGS / "fig_loss_by_target.png", (0.03, 0.04, 0.94, 0.78))
        pdf.savefig(fig); plt.close(fig)

        # ---- Per-target tables, every input combination (three headline targets)
        if args.mt_metrics and Path(args.mt_metrics).exists():
            mt = pd.read_csv(args.mt_metrics)
            for head, label in TABLE_HEADS:
                rows = mt[mt["head"] == head]
                if not len(rows):
                    continue
                rows = rows.sort_values("r2", ascending=True, na_position="first")
                fig, ax = new_slide(label, f"n = {int(rows.iloc[0].n_test):,} test sources")
                paper_table(ax, combo_table(rows), indicator_cols=4,
                            numeric_cmaps={r"$R^2$": _ramp(plt.cm.Blues),
                                           "Info Gain": _ramp(plt.cm.Purples),
                                           "exp(Info Gain)": _ramp(plt.cm.Greens, 0.05, 0.5)},
                            rect=(0.06, 0.02, 0.88, 0.88), fontsize=10)
                pdf.savefig(fig); plt.close(fig)

        # ---- MAIN RESULT
        fig, ax = new_slide("Every target from one run",
                            "test set, all inputs · reference marks are a flow given only "
                            "emission-line fluxes")
        image_panel(fig, FIGS / "fig_results_v3.png", (0.03, 0.04, 0.94, 0.77))
        pdf.savefig(fig); plt.close(fig)

        # ---- Modality UpSet, one slide per target
        for head, lbl in (("log_ml_flux_1", "log flux"), ("log_lx", "log $L_X$"),
                          ("log_sfr", "log SFR")):
            f = FIGS / f"fig_upset_{head}.png"
            if not f.exists():
                continue
            fig, ax = new_slide("Results",
                                "information gain for every combination of the four inputs · "
                                "a bar below a line it contains is redundancy")
            image_panel(fig, f, (0.01, 0.02, 0.98, 0.80))
            pdf.savefig(fig); plt.close(fig)

        # ---- Line coverage
        fig, ax = new_slide("Emission-line coverage", "the observed window slides in rest frame as z grows")
        image_panel(fig, FIGS / "fig_line_coverage.png", (0.05, 0.06, 0.90, 0.74))
        pdf.savefig(fig); plt.close(fig)

    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
