#!/usr/bin/env python
"""Build the DR2 target table: X-ray labels plus everything a later cut needs.

One row per matched DESI targetid. This is the UNIVERSE file: every downstream
script (`make_targets_sidecar.py --universe`, `make_split.py --targets`,
`build_manifest.py --targets`) takes its row set from here, so the sample is
defined in exactly one place.

What this file does NOT do is apply a reliability cut. `nway_p_any` and friends
are carried as COLUMNS. A threshold is a property of a run, not of the
catalogue, and baking one in means the only way to change it is to re-derive the
X-ray labels. `make_split.py --min-p-any` applies it, and records it.

Errors follow the existing convention: a flux F with asymmetric linear errors
becomes log10(F) with
    sig_lo = -log10(1 - LOWERR/F)      (capped where LOWERR >= F)
    sig_hi =  log10(1 + UPERR/F)
so a source whose lower error swallows its flux is treated as unmeasured rather
than given an infinite bar.

Counts (`ape_cts_*`, `ape_bkg_*`, `ape_exp_*`, `ml_cts_*`, ...) are carried too,
raw and untransformed, because a Poisson likelihood on them is the only way a
non-detection stays in the sample: eSASS reports aperture photometry for every
catalogued source, so `N = 0` is a measurement and `log10(0)` is not. See the
block comment on `APE_COLUMNS` for the three traps in those columns.

DESI-side labels (CIGALE M*, SFR, black-hole mass) are deliberately NOT carried
here. They used to be, read out of the previous sidecar via --old-sidecar, which
made the DR2 table depend on the DR1 table it was meant to replace and left the
CIGALE labels un-rebuildable for any target the DR1 sidecar never held. They now
come from the VACs in `make_targets_sidecar.py`, which reads this file as its
universe.

    python scripts/make_dr2_targets.py \
        --counterparts data/dr2/eRASSc3_Main_LS10.fits \
        --main data/dr2/eRASS3_Main_v1.3.fits \
        --desi data/erosita_desi_dr1_matches_all_properties.csv \
        --out data/dr2/dr2_targets.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u

BANDS = (1, 2, 3, 4)
SIG_CAP = 1.5          # dex; beyond this the measurement carries no information

# Counts, for a Poisson likelihood. eSASS reports these for 100% of catalogued
# sources in every band (measured 2026-08-07 on all 1,975,540 rows of
# eRASS3_Main_v1.3.fits: every APE_* and every ML_CTS/RATE/EXP/EEF column is
# finite on 100.0000% of rows), so a non-detection is an exact ZERO COUNT and
# not a missing value. That is the whole reason to carry them: log(0) = -inf
# destroys those rows, N = 0 is ordinary data.
#
#   APE_*  aperture photometry in an APE_RADIUS circle (units are PIXELS of 4
#          arcsec, per the column's own TCOMM, so the ~7-10 pix radii are
#          ~30-40 arcsec, set by eROSITA's ~30 arcsec survey PSF). This is what
#          `N ~ Poisson(lambda * t + B)` is written against: N = APE_CTS,
#          t = APE_EXP, B = APE_BKG, all per-band and per-source. The catalogue
#          defines APE_CTS as "Total counts extracted within the aperture" --
#          SOURCE PLUS BACKGROUND, which is why B is an additive term in the
#          likelihood and must not also be subtracted from N.
#   ML_*   the PSF-fitted alternative, carried so aperture-vs-PSF can be
#          compared rather than assumed (RUN_PLAN §3(d)). ML_CTS is "Source NET
#          counts", already background-subtracted and NOT an integer, so it is
#          the wrong quantity for a Poisson likelihood and the right one for a
#          cross-check. ML_EXP is the VIGNETTED exposure at the source position;
#          APE_EXP is the exposure-map value (they agree to a median ratio of
#          1.00000 in every band on the current sample).
#
# Bands, from the header: 1 = 0.2-2.3 keV (the broad band behind log_lx),
# P1 = 0.2-0.5, P2 = 0.5-1.0, P3 = 1.0-2.0, P4 = 2.0-5.0.
#
# Carried RAW, in catalogue units, with no threshold, no repair and no
# transform, per this file's stated principle. Three things a consumer must know
# and cannot see from the column name, all measured 2026-08-07 on the full
# parent catalogue:
#
# 1. APE_CTS IS int16 IN THE FITS ('I') AND IT OVERFLOWS. Rows with a negative
#    APE_CTS: 20 in band 1, 2 in P1, 5 in P2, 8 in P3, 4 in P4 (of 1,975,540),
#    always on sources whose ML_CTS is 5e4-9e5, i.e. the count wrapped past
#    32,767. It is NOT repaired here: +65536 is a transform, it is not
#    invertible (a source past 98,303 counts wraps twice, and a wrap can land
#    back on a plausible positive number -- 7 band-1 rows already sit above
#    |30,000|), and the catalogue value is what we promised to carry. The build
#    PRINTS the count instead, so a rebuild that pulls in a bright source says
#    so; a run that uses these columns must drop or repair them itself.
# 2. APE_BKG can be slightly NEGATIVE: 20 rows in band 1 (min -13.5 counts),
#    0 rows in P1-P4. Background-map subtraction, not a count. A Poisson
#    likelihood needs B >= 0 and must clip or drop those.
# 3. APE_POIS is the probability that the aperture counts are a background
#    fluctuation alone -- verified numerically as poisson.sf(APE_CTS - 1,
#    APE_BKG), matching to float32 on a 40,000-row slab -- with -9.99 as the
#    SENTINEL wherever that statistic is undefined. It is a probability, not a
#    log, so -9.99 is not a small number, it is "not computed". On the 25,454-row
#    current sample the sentinel lands on EXACTLY the rows with no counts
#    (ape_cts == 0: 1,744 in p1, 176 in p2, 332 in p3, 3,469 in p4, all four
#    matching row for row) plus, in band 1 where nothing has zero counts, the 2
#    rows with a negative ape_bkg.
#
# ML_EEF is a per-band CONSTANT, verified: exactly one distinct value per band
# over all 1,975,540 rows -- 0.8836025 (band 1 and P3, the same number),
# 0.89230239 (P1), 0.88694167 (P2), 0.85624605 (P4). It is carried anyway
# because it is a catalogue value and the aperture/EEF systematic on a hardness
# ratio (RUN_PLAN §3(d)) should read it from the table rather than from a
# hardcode that goes stale at the next data release.
#
# NOT carried: ML_CTS_ERR/LOWERR/UPERR. A Poisson likelihood on the counts needs
# no error bar, and one of the three is a units trap: ML_CTS_UPERR_Pn is bit for
# bit ML_RATE_UPERR_Pn (100.0000% of rows in P2/P3/P4, 1,975,537 of 1,975,540 in
# P1), i.e. ct/s, despite its own TUNIT saying "count" -- while ML_CTS_ERR and
# ML_CTS_LOWERR really are counts, and the broad band _1 is exempt in all three.
# RUN_PLAN §3.1 has the full measurement; the short version is that "multiply
# them all by ML_EXP" inflates two of the three by the exposure.
APE_COLUMNS = {"APE_CTS": "ape_cts", "APE_BKG": "ape_bkg", "APE_EXP": "ape_exp",
               "APE_RADIUS": "ape_radius", "APE_POIS": "ape_pois"}
ML_COUNT_COLUMNS = {"ML_CTS": "ml_cts", "ML_RATE": "ml_rate", "ML_EXP": "ml_exp",
                    "ML_EEF": "ml_eef"}
# FITS band suffix -> ours. "1" is the broad band, already the source of
# log_ml_flux_1 and log_lx; P1-P4 are the sub-bands behind log_flux_p1..p4.
BAND_SUFFIX = [("1", "1")] + [(f"P{b}", f"p{b}") for b in BANDS]

# Carried verbatim off the matched counterpart row, FITS name -> our name.
COUNTERPART_COLUMNS = {
    # NWAY reliability. p_any is P(this X-ray source has ANY LS10 counterpart);
    # it says nothing about whether the DESI fibre sits on that counterpart, so
    # it does not replace the spec-z match audit in match_quality.csv (336 of
    # the 345 sources that audit calls "wrong" have p_any > 0.5, median 0.9994).
    # Both are needed, and both are carried rather than applied.
    "NWAY_p_any": "nway_p_any",
    "NWAY_p_i": "nway_p_i",
    "NWAY_p_single": "nway_p_single",
    "NWAY_match_flag": "nway_match_flag",
    # arcsec: no TUNIT in the header, verified against SkyCoord.separation on
    # (RA, DEC) vs (LS10_RA, LS10_DEC) over 2,000 rows, ratio 1.0000000001.
    "NWAY_Separation_LS10_ERO": "nway_sep_arcsec",
    # The catalogue's OWN per-row reliability recommendation. Carried so a run
    # can use the publishers' cut instead of a flat one. Every count below names
    # the row set it was measured on, because this column is NaN for rows whose
    # counterpart row carries no threshold and a bare "25,136 of 25,218" reads
    # as a fraction of the sample when it is not. Measured 2026-08-06:
    #   rebuilt current sample (25,454 rows): 25,218 carry both nway_p_any and
    #     nway_threshold6, of which 25,136 clear their own row's threshold;
    #     median threshold6 0.0449.
    #   rebuilt expansion (104,032 rows): 101,837 carry both, 101,832 clear;
    #     median threshold6 0.0452.
    # So the publishers' own cut is far looser than any flat p_any threshold.
    "NWAY_threshold6": "nway_threshold6",
    # Counterpart identity: the join key against the DR1 match audit, which is
    # only transferable where the counterpart did not change between releases.
    "LS10_RELEASE": "ls10_release",
    "LS10_BRICKID": "ls10_brickid",
    "LS10_OBJID": "ls10_objid",
    # LS10 forced photometry in the W bands, nanomaggy. Not model input (the
    # model reads the DESI-side flux_w*), but the manifest needs a per-band
    # presence flag and this is the deeper of the two measurements.
    "LS10_flux_w1": "ls10_flux_w1",
    "LS10_flux_w2": "ls10_flux_w2",
    "LS10_flux_w3": "ls10_flux_w3",
    "LS10_flux_ivar_w1": "ls10_flux_ivar_w1",
    "LS10_flux_ivar_w2": "ls10_flux_ivar_w2",
    "LS10_flux_ivar_w3": "ls10_flux_ivar_w3",
    # Star/extragalactic separation. STAR counts by row set, measured 2026-08-06
    # (the row set matters: the secondary-candidate and no-DR2-detection drops
    # below remove proportionally more STARs than QSOs):
    #   rebuilt current sample, 25,454 rows: 6 STAR (22,244 QSO, 3,204 GALAXY)
    #   raw new_targets_nway.csv, 104,945 rows: 1,909 STAR
    #   rebuilt expansion, 104,032 rows: 1,858 STAR (88,609 QSO, 13,565 GALAXY)
    # A STAR's log_lx is meaningless, and these three columns are how they get
    # identified without trusting spectype alone.
    "Exgal_prob_STAREX": "exgal_prob_starex",
    "class_gal_exgal": "class_gal_exgal",
    "simbad_known_galactic": "simbad_known_galactic",
}

# Optional DESI metadata. Absent columns are reported, not fatal: the expansion
# frame (new_targets_nway.csv) carries survey/program/healpix/spectype but no
# zwarn, and the split and manifest both need to say so rather than guess.
DESI_META = ("spectype", "zwarn", "survey", "program", "healpix")

# Which astrometric reference the match is made against. Mixing the two inside
# one 1-arcsec match puts two references in one radius, so it is an explicit
# argument and the choice is recorded per row in `desi_pos_ref`.
POSITION_COLUMNS = {"target": ("target_ra", "target_dec"),
                    "fiber": ("mean_fiber_ra", "mean_fiber_dec")}


def native(a):
    """FITS is big-endian; pandas refuses to hash or sort that on x86."""
    a = np.asarray(a)
    return a.astype(a.dtype.newbyteorder("=")) if a.dtype.byteorder == ">" else a


def log_with_asym_errors(flux, lowerr, uperr):
    """log10 flux plus split-normal sigmas, NaN where the flux is not a measurement."""
    f = np.asarray(flux, float)
    lo = np.asarray(lowerr, float)
    hi = np.asarray(uperr, float)
    good = np.isfinite(f) & (f > 0)
    out = np.full(f.shape, np.nan)
    slo = np.full(f.shape, np.nan)
    shi = np.full(f.shape, np.nan)
    out[good] = np.log10(f[good])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(good & np.isfinite(lo), lo / f, np.nan)
        # LOWERR >= F means the flux is consistent with zero: cap rather than -inf
        slo_all = np.where(ratio < 1.0, -np.log10(np.clip(1.0 - ratio, 1e-12, None)), SIG_CAP)
        shi_all = np.where(good & np.isfinite(hi), np.log10(1.0 + hi / f), np.nan)
    slo[good] = slo_all[good]
    shi[good] = shi_all[good]
    bad = ~np.isfinite(slo) | ~np.isfinite(shi) | (slo > SIG_CAP) | (shi > SIG_CAP)
    out[bad] = np.nan
    slo[bad] = np.nan
    shi[bad] = np.nan
    return out, slo, shi


def log_luminosity(log_flux, z):
    """log10 Lx from log10 flux, NaN at z <= 0 rather than a fabricated number.

    The previous version clipped z to 1e-4 before the distance call, which turns
    a negative or zero redshift into a luminosity roughly 7 dex too small and
    hands the flow a label that looks measured. Counts by row set, measured
    2026-08-06: the rebuilt current sample (25,454 rows) never hits it, min z is
    positive and this function reports 0; the raw new_targets_nway.csv (104,945
    rows) has 957 rows with z <= 0, min -0.001815, and 1,667 below 1e-4; the
    expansion rebuilt through this script (104,032 rows) has 937, which is the
    number the run actually prints. So the clip would have manufactured between
    937 and 957 labels depending on which expansion table you built from.
    """
    from astropy.cosmology import Planck18

    z = np.asarray(z, float)
    log_flux = np.asarray(log_flux, float)
    out = np.full(z.shape, np.nan)
    ok = np.isfinite(z) & (z > 0) & np.isfinite(log_flux)
    if ok.any():
        dl = Planck18.luminosity_distance(z[ok]).to(u.cm).value
        out[ok] = log_flux[ok] + np.log10(4 * np.pi * dl ** 2)
    return out, int((np.isfinite(z) & (z <= 0)).sum())


def read_desi(path: Path, position: str) -> pd.DataFrame:
    ra_col, dec_col = POSITION_COLUMNS[position]
    header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in ("targetid", ra_col, dec_col, "z") if c not in header]
    if missing:
        raise SystemExit(f"--desi {path} lacks {missing} (needed for --position {position})")
    meta = [c for c in DESI_META if c in header]
    absent = [c for c in DESI_META if c not in header]
    if absent:
        print(f"[desi] no {absent} in {path.name}; those columns will be NaN")
    frame = pd.read_csv(path, low_memory=False,
                        usecols=["targetid", ra_col, dec_col, "z"] + meta)
    frame = frame.rename(columns={ra_col: "target_ra", dec_col: "target_dec"})
    frame = frame.dropna(subset=["target_ra", "target_dec"]).drop_duplicates("targetid")
    for col in absent:
        frame[col] = np.nan
    frame["desi_pos_ref"] = position
    return frame.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counterparts", type=Path, required=True,
                    help="eRASSc3_Main_LS10.fits (X-ray to LS10 counterparts, NWAY)")
    ap.add_argument("--main", type=Path, required=True,
                    help="eRASS3_Main_v1.3.fits (per-detection fluxes)")
    ap.add_argument("--desi", type=Path, required=True,
                    help="CSV with targetid, the --position columns, z and DESI metadata")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--radius", type=float, default=1.0, help="match radius, arcsec")
    ap.add_argument("--position", choices=sorted(POSITION_COLUMNS), default="target",
                    help="astrometric reference for the DESI side of the match")
    ap.add_argument("--keep-secondary", action="store_true",
                    help="keep fibres whose nearest LS10 object is an NWAY secondary candidate")
    args = ap.parse_args()

    desi = read_desi(args.desi, args.position)
    cp = fits.open(args.counterparts, memmap=True)[1].data
    cra, cdec = np.asarray(cp["LS10_RA"], float), np.asarray(cp["LS10_DEC"], float)
    g = np.isfinite(cra) & np.isfinite(cdec)
    idx, sep, _ = SkyCoord(desi.target_ra.to_numpy() * u.deg,
                           desi.target_dec.to_numpy() * u.deg).match_to_catalog_sky(
                               SkyCoord(cra[g] * u.deg, cdec[g] * u.deg))
    m = sep.arcsec < args.radius
    print(f"DESI targets: {len(desi):,}   matched to a DR2 counterpart: {int(m.sum()):,}"
          f"  ({m.mean():.1%})  [{args.position} positions, r < {args.radius}\"]")

    row_cp = np.flatnonzero(g)[idx][m]                # row in the counterpart table
    out = desi[m].reset_index(drop=True)
    out["desi_ls10_sep_arcsec"] = sep.arcsec[m]
    for src, dst in COUNTERPART_COLUMNS.items():
        out[dst] = native(cp[src])[row_cp]
    det = native(cp["DETUID"])[row_cp]

    # NWAY match_flag 2 marks a counterpart candidate NWAY considered and did
    # not adopt. A fibre landing on one is not evidence for that association:
    # measured, the ADOPTED counterpart of those same detections sits a median
    # 5.4 arcsec from the fibre, so re-pointing them at the primary would put a
    # 5-arcsec match inside a 1-arcsec sample. Dropped, and the count depends on
    # the row set: 128 of the 25,582 current-sample fibres that matched a DR2
    # counterpart, and 913 of the 104,945 expansion fibres (both measured
    # 2026-08-06 by running this script).
    secondary = out.nway_match_flag.to_numpy(int) == 2
    if secondary.any() and not args.keep_secondary:
        print(f"dropping {int(secondary.sum()):,} fibres matched to an NWAY secondary candidate")
        keep = ~secondary
        out = out[keep].reset_index(drop=True)
        det = det[keep]

    mn = fits.open(args.main, memmap=True)[1].data
    order = {d: i for i, d in enumerate(np.asarray(mn["DETUID"]))}
    row = np.array([order.get(d, -1) for d in det])
    have = row >= 0
    r = row[have]
    out = out[have].reset_index(drop=True)
    print(f"of those, present in the DR2 main catalogue: {len(out):,}")

    col = lambda c: np.asarray(mn[c], float)[r]
    # broad band -> log flux and log Lx
    lf, lo, hi = log_with_asym_errors(col("ML_FLUX_1"), col("ML_FLUX_LOWERR_1"),
                                      col("ML_FLUX_UPERR_1"))
    out["log_ml_flux_1"] = lf
    out["flux_sig_lo"], out["flux_sig_hi"] = lo, hi
    out["log_lx"], n_bad_z = log_luminosity(lf, out.z.to_numpy(float))
    if n_bad_z:
        print(f"[log_lx] {n_bad_z:,} rows with z <= 0 left NaN (no distance is defined)")

    for b in BANDS:
        f, l, h = log_with_asym_errors(col(f"ML_FLUX_P{b}"), col(f"ML_FLUX_LOWERR_P{b}"),
                                       col(f"ML_FLUX_UPERR_P{b}"))
        out[f"log_flux_p{b}"] = f
        out[f"log_flux_p{b}_sig_lo"], out[f"log_flux_p{b}_sig_hi"] = l, h
    out["ero_detuid"] = det[have]
    out["ero_exp"] = col("ML_EXP_1")
    # Detection likelihood per band. Availability is gated on DETECTION rather
    # than on the error bar: a band can pass a sigma cut while being an upper
    # limit, which feeds the flow a non-measurement as though it were measured.
    out["det_like_0"] = col("DET_LIKE_0")
    for b in BANDS:
        out[f"det_like_p{b}"] = col(f"DET_LIKE_P{b}")

    # ---- counts, background and exposure per band (see APE_COLUMNS above) ----
    # int32, not float: these are counts, and a CSV column of "9.0" invites a
    # consumer to treat an exact integer as a measurement with a decimal part.
    # The cast widens int16 but does NOT undo the wrap, by design.
    icol = lambda c: np.asarray(mn[c]).astype(np.int32)[r]
    for fits_b, ours in BAND_SUFFIX:
        for src, dst in APE_COLUMNS.items():
            name = f"{src}_{fits_b}"
            out[f"{dst}_{ours}"] = icol(name) if src == "APE_CTS" else col(name)
        for src, dst in ML_COUNT_COLUMNS.items():
            if src == "ML_EXP" and fits_b == "1":
                continue                      # already carried, as ero_exp
            out[f"{dst}_{ours}"] = col(f"{src}_{fits_b}")

    # Loud, because both are unusable in a Poisson likelihood and both are rare
    # enough to slip through unnoticed. Silence here means the sample is clean.
    for _, ours in BAND_SUFFIX:
        n_wrap = int((out[f"ape_cts_{ours}"] < 0).sum())
        n_negb = int((out[f"ape_bkg_{ours}"] < 0).sum())
        if n_wrap:
            print(f"[counts] ape_cts_{ours}: {n_wrap:,} rows NEGATIVE -- int16 overflow in "
                  f"the parent catalogue, carried raw; a counts run must drop or repair them")
        if n_negb:
            print(f"[counts] ape_bkg_{ours}: {n_negb:,} rows negative (background subtraction); "
                  f"a Poisson likelihood needs B >= 0")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    n_group = out.ero_detuid.nunique()
    # Two different quantities, both about multi-fibre detections, and they are
    # routinely confused. On THIS table: `excess rows` is rows - detuids, i.e.
    # how many rows a targetid-keyed split could leak; `shared detuids` is how
    # many distinct detections carry more than one fibre. They are equal only if
    # no detection carries three. Every restatement downstream has to name the
    # row set as well, because the on-disk 25,582-row file gives different
    # values for both (measured 2026-08-06: 786 excess rows, 773 shared detuids,
    # against 768 and 756 on this 25,454-row rebuild).
    n_shared = int((out.ero_detuid.value_counts() > 1).sum())
    print(f"wrote {args.out}  ({len(out):,} rows, {len(out.columns)} columns, "
          f"{n_group:,} distinct ero_detuid)")
    print(f"  on this {len(out):,}-row table: {len(out) - n_group:,} EXCESS ROWS "
          f"(rows - detuids) over {n_shared:,} DETUIDS that carry more than one "
          f"DESI fibre; make_split.py groups on ero_detuid for exactly this reason")


if __name__ == "__main__":
    main()
