#!/usr/bin/env python
"""Build the SFR sidecar from the DESI DR1 CIGALE physical-properties VAC.

Why CIGALE and not FastSpecFit: our `logmstar` IS FastSpecFit's LOGMSTAR (it
arrives via the agngal VAC, whose column description is verbatim FastSpecFit's).
FastSpecFit's SFR comes out of the SAME stellar-continuum fit, so it shares that
fit's priors and degeneracies -- an SFR head trained on it could score well by
re-predicting stellar mass. CIGALE is an independent SED fit, it reports LOGSFR
and LOGSFR_ERR already in log space (no linear->log propagation, no SFR<=0
censoring), and it fits an explicit AGN component (AGNFRAC) -- which matters
because this sample is ~87% QSO.

    python scripts/make_sfr_sidecar.py --cigale IronPhysProp_v1.2.fits \
        --match-quality match_quality.csv --bands-csv targets_bands.csv \
        --out targets_sidecar.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

# CIGALE columns we keep. LOGM/LOGM_ERR are carried for cross-checking against
# our FastSpecFit logmstar, not (yet) as a training target.
KEEP = ["TARGETID", "SURVEY", "PROGRAM", "SPECTYPE", "Z", "CHI2",
        "LOGM", "LOGM_ERR", "LOGSFR", "LOGSFR_ERR",
        "AGNFRAC", "AGNLUM", "FLAG_MASSPDF", "FLAG_SFRPDF",
        "FLAGOPTICAL", "FLAGINFRARED"]

# Siudek+2024 (the VAC paper) recommend keeping only fits whose best-fit and
# Bayesian estimates agree to within a factor of 5; wider ratios mean a broad or
# multi-modal PDF. FLAG_* are that RATIO, not a boolean -- they run to ~1e11.
FLAG_LO, FLAG_HI = 0.2, 5.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cigale", type=Path, required=True)
    ap.add_argument("--match-quality", type=Path, required=True)
    ap.add_argument("--all-properties", type=Path, required=True,
                    help="erosita_desi_dr1_matches_all_properties.csv (for survey/program/logmstar)")
    ap.add_argument("--bands-csv", type=Path, required=True,
                    help="existing band sidecar; SFR columns are merged into a copy of it")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-sigma", type=float, default=1.0,
                    help="drop log SFR whose error exceeds this (dex); mirrors the band gate")
    args = ap.parse_args()

    # ---- our clean sample, with the survey/program needed to disambiguate ----
    props = pd.read_csv(args.all_properties, low_memory=False,
                        usecols=["targetid", "survey", "program", "spectype", "logmstar", "z"])
    mq = pd.read_csv(args.match_quality)
    keep = set(mq.loc[mq.keep.astype(bool), "targetid"].astype(np.int64))
    ours = props[props.targetid.isin(keep)].drop_duplicates("targetid").reset_index(drop=True)
    print(f"[ours] clean sample n={len(ours):,}")
    want = set(ours.targetid.astype(np.int64))

    # ---- stream the VAC, keeping only our rows ----
    with fits.open(args.cigale, memmap=True) as hdul:
        data = hdul[1].data
        tid = np.asarray(data["TARGETID"], dtype=np.int64)
        print(f"[cigale] {len(tid):,} rows")
        sel = np.flatnonzero(np.isin(tid, list(want)))
        print(f"[cigale] {len(sel):,} rows match our targetids")
        # FITS stores big-endian; pandas refuses to hash/sort those on a
        # little-endian host, so take a native-order copy up front.
        def native(a):
            a = np.asarray(a)
            return a.astype(a.dtype.newbyteorder("=")) if a.dtype.byteorder == ">" else a
        cat = pd.DataFrame({c: native(data[c][sel]) for c in KEEP})
    for c in ("SURVEY", "PROGRAM", "SPECTYPE"):
        cat[c] = cat[c].astype(str).str.strip()

    # A targetid can be observed in several survey/program combinations. Prefer
    # the row matching how OUR row was observed, then main, then best chi2.
    cat = cat.merge(ours[["targetid", "survey", "program"]].rename(columns={"targetid": "TARGETID"}),
                    on="TARGETID", how="left")
    cat["exact"] = ((cat.SURVEY == cat.survey) & (cat.PROGRAM == cat.program)).astype(int)
    cat["is_main"] = (cat.SURVEY == "main").astype(int)
    cat = (cat.sort_values(["TARGETID", "exact", "is_main", "CHI2"],
                           ascending=[True, False, False, True])
              .drop_duplicates("TARGETID", keep="first"))
    print(f"[cigale] {len(cat):,} unique after survey/program disambiguation "
          f"({int(cat.exact.sum()):,} exact survey+program matches)")

    # ---- build the sidecar columns ----
    out = pd.DataFrame({"targetid": cat.TARGETID.astype(np.int64)})
    sfr = cat.LOGSFR.to_numpy(np.float64)
    err = np.abs(cat.LOGSFR_ERR.to_numpy(np.float64))
    logm = cat.LOGM.to_numpy(np.float64)
    fm = cat.FLAG_MASSPDF.to_numpy(np.float64)
    fs = cat.FLAG_SFRPDF.to_numpy(np.float64)

    # Failed fits are written as EXACTLY zero in both LOGM and LOGSFR, and they
    # carry a small LOGSFR_ERR (0.02) -- so an error gate alone lets them through
    # as spurious "log SFR = 0" labels. They must be dropped explicitly.
    failed = (logm == 0) & (sfr == 0)
    sentinel = (sfr <= -90) | (err <= -90) | (logm <= -90) | (fs <= -90)
    nonfinite = ~np.isfinite(sfr) | ~np.isfinite(err) | (err <= 0)
    # Siudek+2024 PDF-quality cut (see FLAG_LO/FLAG_HI above)
    broad_pdf = ~((fs > FLAG_LO) & (fs < FLAG_HI) & (fm > FLAG_LO) & (fm < FLAG_HI))
    bad = failed | sentinel | nonfinite | broad_pdf
    gated = bad | (err > args.max_sigma)
    print(f"[cuts] failed(LOGM=LOGSFR=0) {int(failed.sum()):,} | sentinel {int(sentinel.sum()):,} | "
          f"non-finite {int(nonfinite.sum()):,} | broad PDF {int(broad_pdf.sum()):,} | "
          f"err>{args.max_sigma} {int((~bad & (err > args.max_sigma)).sum()):,}")
    sfr = np.where(gated, np.nan, sfr)
    out["log_sfr"] = sfr
    # symmetric error, but the loader wants a lo/hi pair like the other targets
    out["log_sfr_sig_lo"] = np.where(gated, 0.0, err)
    out["log_sfr_sig_hi"] = np.where(gated, 0.0, err)
    # carried for eval splits and label-quality gating, never as model inputs
    out["cigale_logm"] = cat.LOGM.to_numpy(np.float64)
    out["cigale_logm_err"] = cat.LOGM_ERR.to_numpy(np.float64)
    out["cigale_agnfrac"] = cat.AGNFRAC.to_numpy(np.float64)
    out["cigale_flag_sfrpdf"] = cat.FLAG_SFRPDF.to_numpy(np.float64)
    out["cigale_chi2"] = cat.CHI2.to_numpy(np.float64)
    out["cigale_spectype"] = cat.SPECTYPE.to_numpy()

    n_ok = int(np.isfinite(out.log_sfr).sum())
    print(f"[sfr] usable log_sfr: {n_ok:,} of {len(ours):,} clean sources "
          f"({n_ok/len(ours)*100:.1f}%)")
    fin = out.log_sfr[np.isfinite(out.log_sfr)]
    if len(fin):
        print(f"[sfr] log SFR p5/50/95 = {np.percentile(fin,[5,50,95]).round(2)}  "
              f"median err {np.nanmedian(err[~gated]):.3f} dex")
    # Per-spectype, because the SED decomposition is only as good as the AGN
    # model: CIGALE mass agrees with FastSpecFit at corr +0.76 for GALAXY but
    # only +0.30 for QSO, and this sample is ~87% QSO.
    st = cat.SPECTYPE.to_numpy()
    for cls in ("GALAXY", "QSO"):
        k = (st == cls)
        if k.sum():
            print(f"[sfr]   {cls:7s} {int(np.isfinite(out.log_sfr.to_numpy()[k]).sum()):6,} usable "
                  f"of {int(k.sum()):6,}")

    # ---- cross-check against our FastSpecFit logmstar ----
    # Cross-check the two INDEPENDENT stellar masses, on the rows that survive
    # the cuts (including the sentinel rows here would report a meaningless
    # near-zero correlation driven by the LOGM=0 failures).
    chk = out.merge(ours[["targetid", "logmstar", "spectype"]], on="targetid", how="left")
    keep_row = np.isfinite(chk.log_sfr.to_numpy(np.float64))
    base = (keep_row & np.isfinite(chk.cigale_logm.to_numpy(np.float64))
            & np.isfinite(chk.logmstar.to_numpy(np.float64)) & (chk.logmstar.to_numpy(np.float64) > 2))
    st = chk.spectype.astype(str).str.strip().str.upper().to_numpy()
    for lbl, m in [("all", base), ("GALAXY", base & (st == "GALAXY")), ("QSO", base & (st == "QSO"))]:
        if m.sum() < 100:
            continue
        a = chk.cigale_logm.to_numpy(np.float64)[m]
        b = chk.logmstar.to_numpy(np.float64)[m]
        d = a - b
        print(f"[check] mass CIGALE-FastSpecFit  {lbl:7s} n={int(m.sum()):6,}  "
              f"corr {np.corrcoef(a, b)[0,1]:+.3f}  median {np.median(d):+.3f} dex  "
              f"scatter {1.4826*np.median(np.abs(d-np.median(d))):.3f}")

    # ---- merge into the band sidecar so one CSV still feeds --extra-targets-csv ----
    bands = pd.read_csv(args.bands_csv)
    merged = bands.merge(out, on="targetid", how="outer")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"[out] {args.out}  rows={len(merged):,}  cols={len(merged.columns)}  "
          f"({args.out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
