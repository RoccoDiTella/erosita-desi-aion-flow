#!/usr/bin/env python
"""Data-description figures: the empirical joint, and what the inputs look like.

Produces
  fig_corner.png           corner plot of the empirical joint over the targets
  fig_examples_spectra.png a few DESI spectra spanning redshift and spectype
  fig_examples_images.png  a few Legacy Survey cutouts, same sources

All inputs are local raw products, so this runs without the cluster.

    python docs/make_data_figures.py --all-properties ... --spectra-hdf5 ... \
        --sidecar ... --clean-split-csv ... --fits-pool ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

INK, MUTED, GRID = "#1a1a1a", "#6a6a6a", "#d8d8d8"
VARS = [
    ("log_ml_flux_1",   r"log $F_X$"),
    ("log_lx",          r"log $L_X$"),
    ("logmstar_cigale", r"log $M_*$"),
    ("log_sfr",         r"log SFR"),
    ("log_mbh_pan25",   r"log $M_{BH}$"),
    ("log_flux_p2",     r"log $F_{P2}$"),
    ("z",               r"$z$"),
]


def load(args) -> pd.DataFrame:
    props = pd.read_csv(args.all_properties, low_memory=False,
                        usecols=["targetid", "z", "ls_id", "spectype"]).drop_duplicates("targetid")
    with h5py.File(args.spectra_hdf5, "r") as h:
        hd = pd.DataFrame({"targetid": h["desi_targetid"][:].astype(np.int64),
                           "ml_flux_1": h["ml_flux_1"][:].astype(np.float64)})
    side = pd.read_csv(args.sidecar).drop_duplicates("targetid")
    split = pd.read_csv(args.clean_split_csv).drop_duplicates("targetid")
    df = (props.merge(hd.drop_duplicates("targetid"), on="targetid")
               .merge(side, on="targetid", how="left")
               .merge(split, on="targetid", how="inner"))
    with np.errstate(divide="ignore", invalid="ignore"):
        df["log_ml_flux_1"] = np.log10(np.where(df.ml_flux_1 > 0, df.ml_flux_1, np.nan))
        from astropy.cosmology import Planck18
        import astropy.units as u
        dl = Planck18.luminosity_distance(np.clip(df.z.to_numpy(float), 1e-4, None)).to(u.cm).value
        df["log_lx"] = df.log_ml_flux_1 + np.log10(4 * np.pi * dl ** 2)
    return df


def draw_corner(fig, data: dict, axes=None, label_size: float = 9.5,
                tick_size: float = 7.0) -> None:
    """Draw the corner panels onto an existing figure.

    Factored out so the slide deck can render this NATIVELY as vector art
    instead of pasting a raster: a corner plot is the one figure people zoom
    into, and a PNG stops being useful the moment they do.
    """
    k = len(VARS)
    if axes is None:
        axes = fig.subplots(k, k)
    for i, (ni, li) in enumerate(VARS):
        for j, (nj, lj) in enumerate(VARS):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                v = data[ni][np.isfinite(data[ni])]
                ax.hist(v, bins=50, color="#0072B2", alpha=0.85, edgecolor="none")
                ax.set_yticks([])
                ax.margins(y=0.22)          # headroom so the count never sits on the peak
                ax.text(0.95, 0.88, f"n={len(v):,}", transform=ax.transAxes, ha="right",
                        fontsize=7.2, color=MUTED)
            else:
                x, y = data[nj], data[ni]
                m = np.isfinite(x) & np.isfinite(y)
                if m.sum() > 10:
                    ax.hexbin(x[m], y[m], gridsize=38, cmap="Blues", norm=LogNorm(),
                              linewidths=0, mincnt=1)
            ax.tick_params(labelsize=tick_size, length=2)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            # every panel is labelled on both axes, not just the outer edge, so
            # each one can be read on its own without counting rows and columns
            ax.set_xlabel(lj, fontsize=label_size, labelpad=1.5)
            ax.set_ylabel(li if i != j else "count", fontsize=label_size, labelpad=1.5)
            # tick NUMBERS stay on the outside only, or the grid is unreadable
            if i != k - 1:
                ax.set_xticklabels([])
            if j != 0 or i == 0:
                ax.set_yticklabels([])


def corner(df: pd.DataFrame, out: Path) -> None:
    data = {n: df[n].to_numpy(float) for n, _ in VARS}
    fig, axes = plt.subplots(len(VARS), len(VARS), figsize=(12.4, 10.6))
    draw_corner(fig, data, axes=axes)
    fig.suptitle("Empirical joint of the targets (clean sample)", fontsize=12.5, color=INK, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=0.7, w_pad=0.7)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    # The deck redraws this natively as vector; ship the columns so it does not
    # need the 2 GB spectra file or the raw catalogs to do so.
    npz = out.with_name("corner_data.npz")
    np.savez_compressed(npz, **{n: data[n].astype(np.float32) for n, _ in VARS})
    print(f"wrote {npz}")


def pick_examples(df: pd.DataFrame, n: int = 4, pool=None) -> pd.DataFrame:
    """Spread the examples over redshift, and prefer sources we can also show an
    image for, so the two figures line up source-for-source."""
    d = df[np.isfinite(df.z) & df.targetid.notna()].copy()
    if pool is not None:
        have = {int(p.stem) for p in Path(pool).glob("*.fits")}
        d = d[d.targetid.astype(np.int64).isin(have)]
    qs = np.linspace(0.12, 0.88, n)
    picks = []
    for q in qs:
        target = d.z.quantile(q)
        cand = d.loc[(d.z - target).abs().sort_values().index[:40]]
        cand = cand[~cand.targetid.isin([p.targetid for p in picks])] if picks else cand
        if len(cand):
            picks.append(cand.iloc[0])
    return pd.DataFrame(picks)


def spectra_figure(df: pd.DataFrame, args, out: Path) -> pd.DataFrame:
    ex = pick_examples(df, pool=args.fits_pool)
    with h5py.File(args.spectra_hdf5, "r") as h:
        tid = h["desi_targetid"][:].astype(np.int64)
        lam = h["spectra_lambda"][:].astype(float)
        idx = {int(t): i for i, t in enumerate(tid)}
        rows = [(r, h["spectra_flux"][idx[int(r.targetid)]].astype(float))
                for _, r in ex.iterrows() if int(r.targetid) in idx]
    fig, axes = plt.subplots(len(rows), 1, figsize=(11.0, 6.2), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (r, fl) in zip(axes, rows):
        good = np.isfinite(fl)
        # light smoothing purely for legibility at slide size
        w = 9
        sm = np.convolve(np.where(good, fl, 0.0), np.ones(w) / w, mode="same")
        ax.plot(lam, sm, lw=0.6, color="#0072B2")
        ax.set_ylabel(r"$f_\lambda$", fontsize=9)
        ax.text(0.995, 0.86, f"{str(r.spectype).strip()},  z = {r.z:.3f}",
                transform=ax.transAxes, ha="right", fontsize=9, color=INK)
        ax.tick_params(labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        lo, hi = np.nanpercentile(sm[good], [1, 99.5]) if good.any() else (0, 1)
        ax.set_ylim(lo - 0.1 * abs(hi - lo), hi + 0.25 * abs(hi - lo))
    axes[-1].set_xlabel("observed wavelength [Å]", fontsize=10)
    axes[0].set_title("DESI spectra, four sources spanning the redshift range",
                      fontsize=12.5, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return ex


def images_figure(ex: pd.DataFrame, args, out: Path) -> None:
    from astropy.io import fits

    panels = []
    for _, r in ex.iterrows():
        # the pool is keyed by DESI targetid, NOT ls_id
        p = Path(args.fits_pool) / f"{int(r.targetid)}.fits"
        if not p.exists():
            continue
        with fits.open(p) as h:
            cube = np.asarray(h[0].data, dtype=float)
            bands = str(h[0].header.get("BANDS", "")).strip()
        panels.append((r, cube, bands))
    if not panels:
        print("no cutouts found, skipping image figure")
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(11.0, 3.3))
    axes = np.atleast_1d(axes)
    for ax, (r, cube, bands) in zip(axes, panels):
        # asinh stretch on three bands -> RGB, standard for Legacy Survey cutouts
        chans = cube[:3][::-1] if cube.shape[0] >= 3 else np.repeat(cube[:1], 3, axis=0)
        rgb = np.stack([np.arcsinh(np.clip(c, 0, None) * 8.0) for c in chans], axis=-1)
        hi = np.nanpercentile(rgb, 99.5)
        ax.imshow(np.clip(rgb / (hi if hi > 0 else 1.0), 0, 1), origin="lower")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"z = {r.z:.3f}", fontsize=9.5, color=INK)
    fig.suptitle(f"Legacy Survey cutouts, same four sources  ({panels[0][2] or '4 bands'}, 160 px)",
                 fontsize=12.5, color=INK, y=1.04)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all-properties", type=Path, required=True)
    ap.add_argument("--spectra-hdf5", type=Path, required=True)
    ap.add_argument("--sidecar", type=Path, required=True)
    ap.add_argument("--clean-split-csv", type=Path, required=True)
    ap.add_argument("--fits-pool", type=Path, required=True)
    ap.add_argument("--figdir", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()

    args.figdir.mkdir(parents=True, exist_ok=True)
    df = load(args)
    print(f"[data] {len(df):,} clean-split rows")
    corner(df, args.figdir / "fig_corner.png")
    ex = spectra_figure(df, args, args.figdir / "fig_examples_spectra.png")
    images_figure(ex, args, args.figdir / "fig_examples_images.png")

    # row counts that feed the provenance table on the data slide
    counts = pd.DataFrame([{"variable": n, "n_usable": int(np.isfinite(df[n]).sum())}
                           for n, _ in VARS])
    counts_path = args.figdir / "data_counts.csv"
    counts.to_csv(counts_path, index=False)
    print("\n[counts] usable rows per variable (clean split):")
    print(counts.to_string(index=False))
    print(f"wrote {counts_path}")


if __name__ == "__main__":
    main()
