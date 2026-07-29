#!/usr/bin/env python
"""Build the runtime target sidecar: band fluxes + CIGALE + black-hole mass.

Everything here is joined by targetid and consumed at training time via
``--extra-targets-csv``; nothing is re-staged.

**Why CIGALE for stellar mass, SFR and sSFR.** All three must come from ONE fit
or sSFR is a mongrel of two catalogues. CIGALE is chosen over FastSpecFit for
this sample because it fits an explicit AGN component (that is where AGNFRAC
comes from), while FastSpecFit models a stellar continuum plus emission lines
and has no AGN component at all. On a sample that is 87% QSO the accretion-disk
continuum would otherwise be attributed to stars, biasing both mass and SFR in a
way the fit never models. Corroboration: the two masses agree at corr +0.76 for
GALAXY but only +0.30 for QSO.

**sSFR is computed, not catalogued.** No DESI VAC publishes an sSFR column
(checked CIGALE 69 cols, FastSpecFit 817, stellar-mass-emline 420). It is
LOGSFR - LOGM: an exact definition, no calibration, both terms from the same
fit. Its error is the one gap -- sigma(sSFR) needs Cov(LOGM, LOGSFR), which
CIGALE does not publish, and the sign is not knowable a priori (overall
normalisation correlates the two, the young/old split anti-correlates them). We
use the independent sum and flag it here; training the mass, log_sfr and
log_ssfr heads together lets the true correlation be measured from residuals.

**Black-hole mass** comes from the DR1 qmassiron VAC. PAN25 is primary: it is
iron-corrected (FeII blends with MgII and inflates FWHM, and M_BH scales as
FWHM^2), it is the reason that catalogue exists, and it is the only estimator
that differs from the pack -- VO09/LE20/SHEN11/YU23 intercorrelate at 0.99+, and
LE20 vs VO09 is exactly 1.0000, a pure rescaling. VO09 rides along as the
classic comparison target; the other three are carried as columns, not trained.

    python scripts/make_targets_sidecar.py --cigale IronPhysProp_v1.2.fits \
        --bh-vac VAC_BHmass_338_v1.7.fits --match-quality match_quality.csv \
        --all-properties ... --bands-csv targets_bands.csv --out targets_sidecar.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

CIGALE_KEEP = ["TARGETID", "SURVEY", "PROGRAM", "SPECTYPE", "Z", "CHI2",
               "LOGM", "LOGM_ERR", "LOGSFR", "LOGSFR_ERR",
               "AGNFRAC", "AGNLUM", "FLAG_MASSPDF", "FLAG_SFRPDF"]
# Siudek+2024 keep only fits whose best-fit and Bayesian estimates agree within
# a factor of 5; wider means a broad or multi-modal PDF. FLAG_* are that RATIO,
# not a boolean -- they run to ~1e11.
FLAG_LO, FLAG_HI = 0.2, 5.0

BH_CAL = ["PAN25", "VO09", "SHEN11", "LE20", "YU23"]
BH_TARGETS = ["PAN25", "VO09"]


def native(a):
    """FITS is big-endian; pandas refuses to hash or sort that on x86."""
    a = np.asarray(a)
    return a.astype(a.dtype.newbyteorder("=")) if a.dtype.byteorder == ">" else a


def load_cigale(path: Path, want: set) -> pd.DataFrame:
    with fits.open(path, memmap=True) as handle:
        data = handle[1].data
        tid = np.asarray(data["TARGETID"], dtype=np.int64)
        sel = np.flatnonzero(np.isin(tid, list(want)))
        cat = pd.DataFrame({c: native(data[c][sel]) for c in CIGALE_KEEP})
    for c in ("SURVEY", "PROGRAM", "SPECTYPE"):
        cat[c] = cat[c].astype(str).str.strip()
    return cat


def load_bh(path: Path, want: set) -> pd.DataFrame:
    cols = ["TARGETID", "FWHM_DAS", "FWHM_DAS_ERR", "L3000_DAS", "RFE_DAS"] + \
           [f"LOGMASS_DAS_{c}" for c in BH_CAL] + [f"LOGMASS_DAS_{c}_ERR" for c in BH_CAL]
    with fits.open(path, memmap=True) as handle:
        data = handle[1].data
        have = [c for c in cols if c in data.columns.names]
        tab = pd.DataFrame({c: native(data[c]) for c in have})
    tab = tab[tab.TARGETID.isin(want)]
    key = f"LOGMASS_DAS_{BH_TARGETS[0]}_ERR"
    if key in tab.columns:                       # keep the tightest if duplicated
        tab = tab.sort_values(["TARGETID", key])
    return tab.drop_duplicates("TARGETID")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cigale", type=Path, required=True)
    ap.add_argument("--bh-vac", type=Path, default=None,
                    help="VAC_BHmass_338_v1.7.fits; omit to skip the M_BH targets")
    ap.add_argument("--match-quality", type=Path, required=True)
    ap.add_argument("--all-properties", type=Path, required=True)
    ap.add_argument("--bands-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-sigma", type=float, default=1.0,
                    help="drop a label whose error exceeds this (dex)")
    args = ap.parse_args()

    props = pd.read_csv(args.all_properties, low_memory=False,
                        usecols=["targetid", "survey", "program", "spectype", "logmstar", "z"])
    mq = pd.read_csv(args.match_quality)
    keep = set(mq.loc[mq.keep.astype(bool), "targetid"].astype(np.int64))
    ours = props[props.targetid.isin(keep)].drop_duplicates("targetid").reset_index(drop=True)
    n_ours = len(ours)
    print(f"[ours] clean sample n={n_ours:,}")
    want = set(ours.targetid.astype(np.int64))

    # ---------------- CIGALE: mass, SFR, sSFR ----------------
    cat = load_cigale(args.cigale, want)
    print(f"[cigale] {len(cat):,} rows match")
    cat = cat.merge(ours[["targetid", "survey", "program"]].rename(columns={"targetid": "TARGETID"}),
                    on="TARGETID", how="left")
    cat["exact"] = ((cat.SURVEY == cat.survey) & (cat.PROGRAM == cat.program)).astype(int)
    cat["is_main"] = (cat.SURVEY == "main").astype(int)
    cat = (cat.sort_values(["TARGETID", "exact", "is_main", "CHI2"],
                           ascending=[True, False, False, True])
              .drop_duplicates("TARGETID", keep="first"))

    logm = cat.LOGM.to_numpy(np.float64)
    sfr = cat.LOGSFR.to_numpy(np.float64)
    e_m = np.abs(cat.LOGM_ERR.to_numpy(np.float64))
    e_s = np.abs(cat.LOGSFR_ERR.to_numpy(np.float64))
    fm = cat.FLAG_MASSPDF.to_numpy(np.float64)
    fs = cat.FLAG_SFRPDF.to_numpy(np.float64)

    # Failed fits are written as EXACTLY zero in BOTH columns and still carry a
    # small error, so an error gate alone lets them through as fake labels.
    failed = (logm == 0) & (sfr == 0)
    sentinel = (logm <= -90) | (sfr <= -90) | (e_m <= -90) | (e_s <= -90) | (fs <= -90)
    broad_pdf = ~((fs > FLAG_LO) & (fs < FLAG_HI) & (fm > FLAG_LO) & (fm < FLAG_HI))
    base_bad = failed | sentinel | broad_pdf
    print(f"[cigale cuts] failed {int(failed.sum()):,} | sentinel {int(sentinel.sum()):,} | "
          f"broad PDF {int(broad_pdf.sum()):,}")

    out = pd.DataFrame({"targetid": cat.TARGETID.astype(np.int64)})

    def add(name, value, err, bad):
        gated = (bad | ~np.isfinite(value) | ~np.isfinite(err) | (err <= 0)
                 | (err > args.max_sigma))
        out[name] = np.where(gated, np.nan, value)
        out[f"{name}_sig_lo"] = np.where(gated, 0.0, err)
        out[f"{name}_sig_hi"] = np.where(gated, 0.0, err)
        print(f"[target] {name:20s} {int((~gated).sum()):6,} usable "
              f"({(~gated).sum()/n_ours*100:5.1f}%)")

    add("logmstar_cigale", logm, e_m, base_bad)
    add("log_sfr", sfr, e_s, base_bad)

    # sSFR is CARRIED, NOT TRAINED. It is an exact function of the two targets
    # above, so a head on it would add no label information -- only a different
    # projection of the same data, and one whose posterior could contradict
    # p(SFR) - p(M*). Once the (M*, SFR) joint head exists, sSFR follows by the
    # same shear-marginalisation already validated for HR, with the error
    # correlation read off the joint instead of assumed. Kept here purely as a
    # reference to validate that implied version against. The sigma below
    # assumes independence, which is exactly the assumption the joint removes.
    ssfr_bad = (base_bad | ~np.isfinite(sfr - logm) | (e_s <= 0) | (e_m <= 0))
    out["ref_log_ssfr"] = np.where(ssfr_bad, np.nan, sfr - logm)
    out["ref_log_ssfr_sig_indep"] = np.where(ssfr_bad, np.nan, np.sqrt(e_s ** 2 + e_m ** 2))
    print(f"[carried] {'ref_log_ssfr':20s} {int((~ssfr_bad).sum()):6,} "
          f"({(~ssfr_bad).sum()/n_ours*100:5.1f}%)  reference only, not a target")

    for col, src in [("cigale_agnfrac", "AGNFRAC"), ("cigale_agnlum", "AGNLUM"),
                     ("cigale_chi2", "CHI2"), ("cigale_flag_sfrpdf", "FLAG_SFRPDF"),
                     ("cigale_flag_masspdf", "FLAG_MASSPDF")]:
        out[col] = cat[src].to_numpy(np.float64)
    out["cigale_spectype"] = cat.SPECTYPE.to_numpy()

    # ---------------- black-hole mass ----------------
    if args.bh_vac and args.bh_vac.exists():
        bh = load_bh(args.bh_vac, want)
        print(f"\n[bh] {len(bh):,} rows match")
        bh_out = pd.DataFrame({"targetid": bh.TARGETID.astype(np.int64)})
        for cal in BH_CAL:
            v = bh[f"LOGMASS_DAS_{cal}"].to_numpy(np.float64)
            e = np.abs(bh[f"LOGMASS_DAS_{cal}_ERR"].to_numpy(np.float64))
            if cal in BH_TARGETS:
                name = f"log_mbh_{cal.lower()}"
                gated = (~np.isfinite(v) | ~np.isfinite(e) | (e <= 0)
                         | (e > args.max_sigma) | (v <= 0))
                bh_out[name] = np.where(gated, np.nan, v)
                bh_out[f"{name}_sig_lo"] = np.where(gated, 0.0, e)
                bh_out[f"{name}_sig_hi"] = np.where(gated, 0.0, e)
                print(f"[target] {name:20s} {int((~gated).sum()):6,} usable "
                      f"({(~gated).sum()/n_ours*100:5.1f}%)")
            else:
                bh_out[f"mbh_{cal.lower()}"] = v      # carried, not trained
        for c in ("FWHM_DAS", "RFE_DAS", "L3000_DAS"):
            if c in bh.columns:
                bh_out[c.lower()] = bh[c].to_numpy(np.float64)
        out = out.merge(bh_out, on="targetid", how="outer")

        cal_cols = [f"log_mbh_{c.lower()}" for c in BH_TARGETS] + \
                   [f"mbh_{c.lower()}" for c in BH_CAL if c not in BH_TARGETS]
        M = out[cal_cols].to_numpy(np.float64)
        k = np.isfinite(M).all(axis=1)
        if k.sum() > 100:
            print(f"[bh] spread across all five calibrations: median "
                  f"{np.median(M[k].max(1) - M[k].min(1)):.3f} dex on {int(k.sum()):,} rows")

    # ---------------- merge into the band sidecar ----------------
    bands = pd.read_csv(args.bands_csv)
    merged = bands.merge(out, on="targetid", how="outer")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"\n[out] {args.out}  rows={len(merged):,}  cols={len(merged.columns)}  "
          f"({args.out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
