#!/usr/bin/env python
"""One row per targetid: what exists for this source, and where it lives.

This is the file the modality experiment reads. Everything in it is a statement
about AVAILABILITY, not about quality of fit: which inputs a source actually
has, which split it landed in, which row of the source HDF5 holds it, and the
reliability columns needed to say what the sample is.

THE PRESENCE FLAGS
------------------
This script is the AUTHORITATIVE source of "this input exists for this row". It
emits exactly the four columns named by
`shareable_aion_flow.data_to_aion_embeddings.PRESENCE_FLAGS`, as numpy bool:

    has_spectrum  has_z  has_wise  has_image

Staging carries those columns into the batch, and
`data_to_aion_embeddings.modality_presence(batch, flags=...)` ANDs them with a
content heuristic. That heuristic is a FALLBACK, for staged files written before
these columns existed. It is strictly weaker than what is computed here, and
weaker in a way that is not recoverable at batch time: the batch carries neither
`zwarn` nor a per-band WISE inverse variance, so it physically cannot express
`has_z` or `has_wise` as this script defines them. The one thing the batch adds
is a check this file cannot make -- that the staged bytes are not an all-zero
frame -- which is why the two are ANDed rather than one replacing the other.

A flag this script cannot MEASURE is not written as False. It is omitted, and
the loader then falls back to the content heuristic for that modality. A False
would AND the modality off for the whole sample and silently delete an arm of
the information-gain table; a missing column degrades to the weaker rule and
says so. This is why `--fits-pool` and `--wise-source none` omit rather than
zero.

**has_wise is defined on flux > 0 and ivar > 0, never on finiteness.** LS10 and
DESI both write a finite number in every W band for every row (measured on the
25,454-row DR2 rebuild: 25,454 of 25,454 finite in W1, W2 and W3, from either
source), so a finiteness rule marks WISE universally present, the presence check
becomes inert, and an information-gain table then reports the WISE delta over a
sample that never had a missing W band to begin with. On flux > 0 and ivar > 0
the same rows give, DESI-side, W1 25,372, W2 25,368, W3 23,359, has_wise 25,386.
It is per band, and `has_wise` is their OR, because the tokenizer takes one
token per band and a missing W3 must not remove W1 and W2.

WHAT THE FLAGS ACTUALLY LOOK LIKE (all re-derived 2026-08-06 by running this
script; the two row sets are named because they differ):

| flag         | rebuilt 25,454-row universe | within the 22,800-row split |
|--------------|-----------------------------|-----------------------------|
| has_spectrum | 24,058                      | 22,800  (constant)          |
| has_z        | 25,102                      | 22,691                      |
| has_wise     | 25,386                      | 22,768                      |
| has_image    | 24,058                      | 22,800  (constant)          |

Two of the four are CONSTANT over the split, and that is the number that
matters, not the universe count. `has_spectrum` is constant by construction --
`make_split --require-spectrum` cuts on it. `has_image` is constant for a
subtler reason: on this sample the cutout pool was fetched for exactly the rows
the merged spectrum HDF5 holds, so has_image equals has_spectrum row for row
(0 rows carry one and not the other, measured), and requiring a spectrum
therefore also requires an image. Consequence for the modality experiment: on
the CURRENT sample an image ablation measures what happens when you MASK an
image, not what happens when a source never had one, and `--sample common` vs
`--sample native` are the same row set for every image combo. The expansion is
where that stops being true -- its pool is still filling, and there the
like-for-like spectrum-and-image sample came out 25,099 of 104,032 (24.1%) on
one run and 25,139 on a rerun about forty minutes later, both 2026-08-06,
because the cutout fetcher is still filling that pool. Treat every expansion
image count as a floor with a timestamp, never as a fixed property. This
script prints both the universe and the within-split counts on every run, and
says CONSTANT explicitly, so an arm that cannot be measured is visible at build
time rather than in a finished table.

WHICH WISE (--wise-source): the model's input tensor is the DESI-side
`flux_w1..w3`, so `--wise-source desi` (the default) is the only setting that
describes what the model actually reads for the current sample, and it requires
`flux_w*`/`flux_ivar_w*` via --desi. `ls10_flux_w*` is NOT silently accepted as
a substitute: it is a different measurement (measured, the two disagree on
has_wise for 72 of 25,454 rows). But the 104,945-row expansion has no DESI-side
photometry anywhere on this machine -- `new_targets_nway.csv` carries 11 columns
and none of them is a W band -- so for the expansion `desi` is not merely strict,
it is impossible, and has_wise would be undefinable. `--wise-source ls10` is
therefore an EXPLICIT escape hatch rather than an inference: it is correct only
if staging also sources the expansion's `flux_w1..w3` from LS10, it is recorded
per row in the `wise_source` column, and it prints what it did. `--wise-source
none` omits has_wise entirely for a manifest that genuinely cannot answer.

`has_z` needs `zwarn`: a redshift with ZWARN != 0 is a fit the survey itself
does not stand behind, and z is a model INPUT here, not only a label. On the
current sample --desi supplies it and 352 of 25,454 rows are flagged. The
expansion carries no zwarn anywhere on this machine, so it has exactly the
has_wise shape: `--allow-unvouched-z` is the explicit weakening to
`finite AND z > 0` for rows with no zwarn, it is refused by default with the
routes named, and `z_rule` records per row which of the two rules decided it.

    # the 104,945-row expansion: no DESI photometry, no zwarn, so both
    # weakenings must be asked for, and both are stamped in the output
    python scripts/build_manifest.py --targets data/dr2/exp_targets.csv \
        --wise-source ls10 --allow-unvouched-z \
        --spectra-shards data/dr2/spectra_dr2_new.shards \
        --fits-pool data/dr2/fits_pool_dr2 \
        --out data/dr2/manifest_expansion.csv

`has_spectrum` is measured against wherever the spectra live, and for a merged
sample that is two places: the current sample's are in the merged source HDF5
(`--source-hdf5`), the expansion's are in the shard directory
(`--spectra-shards`). The two share zero targetids. Pass both and the flag is
their OR; pass only the shards for a current-sample manifest and every row comes
out False, which then makes `make_split --require-spectrum` drop the entire
sample.

    python scripts/build_manifest.py --targets data/dr2/dr2_targets.csv \
        --desi data/erosita_desi_dr1_matches_all_properties.csv \
        --split data/dr2/clean_split_dr2.csv \
        --match-quality data/match_quality.csv \
        --fits-pool <raw>/legacysurvey/fits_pool \
        --source-hdf5 <raw>/erosita_desi/erosita_spectra_merged_32k.hdf5 \
        --out data/dr2/manifest_dr2.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_ls_cutouts import spectra_available  # noqa: E402  same on-disk convention

# The names the loader looks for. Imported rather than restated so they cannot
# drift apart; the literal is only a fallback for a checkout without torch, and
# it is cross-checked whenever the import succeeds.
PRESENCE_FLAGS = ("has_spectrum", "has_z", "has_wise", "has_image")
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from shareable_aion_flow.data_to_aion_embeddings import (  # noqa: E402
        PRESENCE_FLAGS as _LOADER_FLAGS)
except Exception as exc:                                  # pragma: no cover
    print(f"[presence] could not import the loader's PRESENCE_FLAGS ({exc}); "
          f"using the local copy {PRESENCE_FLAGS} UNVERIFIED")
else:
    if tuple(_LOADER_FLAGS) != PRESENCE_FLAGS:
        raise SystemExit(
            f"presence flag names have drifted: this script writes {PRESENCE_FLAGS}, "
            f"data_to_aion_embeddings reads {tuple(_LOADER_FLAGS)}. The manifest is "
            "the authoritative source, so fix the names here and re-stage.")

BANDS = ("w1", "w2", "w3")
# Which measurement defines has_wise. See the module docstring: `desi` is the
# tensor the model reads; `ls10` is the explicit fallback for the expansion,
# which has no DESI-side photometry at all; `none` omits the column.
WISE_SOURCES = {"desi": ("flux_{b}", "flux_ivar_{b}"),
                "ls10": ("ls10_flux_{b}", "ls10_flux_ivar_{b}"),
                "none": None}
# The cutout fetcher rejects any response under 10 kB before it renames the temp
# file into place, so a present file above that size is one it accepted.
MIN_IMAGE_BYTES = 10_000

CARRY = ["ero_detuid", "spectype", "zwarn", "z",
         "nway_p_any", "nway_p_i", "nway_p_single", "nway_match_flag",
         "nway_sep_arcsec", "nway_threshold6", "ls10_objid",
         "det_like_0", "det_like_p1", "det_like_p2", "det_like_p3", "det_like_p4"]


def wise_presence(frame: pd.DataFrame, source: str = "desi") -> dict[str, np.ndarray]:
    """Per-band WISE availability from ONE named measurement. Never a mix.

    `source` is explicit because the two available measurements are not
    interchangeable and only one of them is what the model reads:

      desi  `flux_w*`/`flux_ivar_w*`, the DESI-side photometry that is staged
            into the `flux_w1..w3` tensor the tokenizer consumes. Correct for
            the current sample. Absent for the 104,945-row expansion, whose
            `new_targets_nway.csv` carries no W band at all.
      ls10  `ls10_flux_w*`/`ls10_flux_ivar_w*`, LS10 forced photometry carried
            off the NWAY counterpart row. A DIFFERENT measurement of the same
            object -- measured on the 25,454-row rebuild the two disagree about
            has_wise for 72 rows -- so it is only correct when staging also
            sources WISE from LS10. That is a staging decision, not something
            this script can infer, which is why it must be asked for.

    Deliberately not offered: an `auto` that falls back on its own. The whole
    defect this replaces was a column that could not be computed for half the
    intended universe and would have been quietly filled by whatever was to
    hand.
    """
    if source not in WISE_SOURCES:
        raise SystemExit(f"--wise-source {source} is not one of {sorted(WISE_SOURCES)}")
    if source == "none":
        return {}
    flux_pat, ivar_pat = WISE_SOURCES[source]
    flux_cols = {b: flux_pat.format(b=b) for b in BANDS}
    ivar_cols = {b: ivar_pat.format(b=b) for b in BANDS}
    missing = [c for b in BANDS for c in (flux_cols[b], ivar_cols[b])
               if c not in frame.columns]
    if missing:
        raise SystemExit(
            f"--wise-source {source} needs {missing}, which --targets/--desi do not "
            "carry.\n"
            "  * for the current sample: pass --desi with the DESI photometry join; "
            "that is the measurement the model actually reads.\n"
            "  * for the 104,945-row expansion there is no DESI-side photometry on "
            "this machine, so pass --wise-source ls10 EXPLICITLY (and make staging "
            "source flux_w1..w3 from LS10 too), or --wise-source none to omit "
            "has_wise and let the loader fall back to its content heuristic.\n"
            "ls10_flux_w* is never substituted silently: it is a different "
            "measurement from the one staged into flux_w1..w3.")
    out = {}
    for b in BANDS:
        flux = frame[flux_cols[b]].to_numpy(float)
        ivar = frame[ivar_cols[b]].to_numpy(float)
        out[f"has_{b}"] = (np.isfinite(flux) & (flux > 0)
                           & np.isfinite(ivar) & (ivar > 0))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", type=Path, required=True,
                    help="DR2 target table or the sidecar built from it")
    ap.add_argument("--desi", type=Path, default=None,
                    help="DESI properties CSV; supplies flux_w*/flux_ivar_w*/zwarn/spectype")
    ap.add_argument("--split", type=Path, default=None, help="targetid,split from make_split.py")
    ap.add_argument("--match-quality", type=Path, default=None)
    ap.add_argument("--fits-pool", type=Path, default=None, help="directory of <targetid>.fits")
    ap.add_argument("--spectra-shards", type=Path, default=None, help="directory of *.npz shards")
    ap.add_argument("--source-hdf5", type=Path, default=None,
                    help="merged source file; supplies source_row and, if given, has_spectrum. "
                         "This is where the CURRENT sample's spectra live; the expansion's are "
                         "in --spectra-shards. Pass both for a merged manifest.")
    ap.add_argument("--allow-unvouched-z", action="store_true",
                    help="for rows with no zwarn, define has_z as `finite AND z > 0` instead "
                         "of refusing. Needed for the expansion, which has no zwarn anywhere; "
                         "it admits ZWARN != 0 redshifts, and which rule judged each row is "
                         "recorded in the z_rule column")
    ap.add_argument("--wise-source", choices=sorted(WISE_SOURCES), default="desi",
                    help="which measurement defines has_wise: 'desi' (default, the tensor the "
                         "model reads), 'ls10' (explicit fallback, the only option for the "
                         "expansion, valid only if staging also reads LS10), or 'none' "
                         "(omit has_wise rather than write a False nobody measured)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    frame = pd.read_csv(args.targets, low_memory=False)
    if "targetid" not in frame.columns:
        raise SystemExit(f"--targets {args.targets} has no targetid column")
    frame["targetid"] = frame.targetid.astype(np.int64)
    if frame.targetid.duplicated().any():
        raise SystemExit("--targets has duplicate targetids")
    print(f"[targets] {len(frame):,} rows from {args.targets.name}")

    if args.desi is not None:
        header = pd.read_csv(args.desi, nrows=0).columns
        wanted = [c for c in ["zwarn", "spectype"] + [f"flux_{b}" for b in BANDS]
                  + [f"flux_ivar_{b}" for b in BANDS] if c in header]
        # Columns already in the target table win: they came through the same
        # positional match as the X-ray labels, so they describe the same row.
        wanted = [c for c in wanted if c not in frame.columns]
        if wanted:
            desi = (pd.read_csv(args.desi, low_memory=False, usecols=["targetid"] + wanted)
                    .drop_duplicates("targetid"))
            desi["targetid"] = desi.targetid.astype(np.int64)
            frame = frame.merge(desi, on="targetid", how="left")
            print(f"[desi] joined {wanted}")

    out = pd.DataFrame({"targetid": frame.targetid.to_numpy()})

    # --- source_row and spectra ------------------------------------------
    out["source_row"] = np.full(len(out), -1, dtype=np.int64)
    if args.source_hdf5 is not None:
        import h5py

        with h5py.File(args.source_hdf5, "r") as handle:
            staged = np.asarray(handle["desi_targetid"][:], dtype=np.int64)
        row_of = pd.Series(np.arange(len(staged), dtype=np.int64), index=staged)
        row_of = row_of[~row_of.index.duplicated()]
        out["source_row"] = row_of.reindex(out.targetid).fillna(-1).to_numpy(np.int64)
        print(f"[source] {int((out.source_row >= 0).sum()):,} rows resolve to a source_row")
    # The two spectrum stores are disjoint by construction (the current sample
    # and the expansion share zero targetids), so this is a UNION and not a
    # preference. Making it an either/or is how a current-sample manifest ends
    # up with has_spectrum False everywhere.
    if args.spectra_shards is None and args.source_hdf5 is None:
        raise SystemExit("pass --spectra-shards and/or --source-hdf5 so has_spectrum is "
                         "measured, not assumed")
    has_spectrum = np.zeros(len(out), bool)
    if args.source_hdf5 is not None:
        from_source = (out.source_row >= 0).to_numpy()
        has_spectrum |= from_source
        print(f"[spectra] {int(from_source.sum()):,} of {len(out):,} from "
              f"{args.source_hdf5.name}")
    if args.spectra_shards is not None:
        have = spectra_available(args.spectra_shards)
        from_shards = out.targetid.isin(have).to_numpy()
        has_spectrum |= from_shards
        print(f"[spectra] {int(from_shards.sum()):,} of {len(out):,} from "
              f"{args.spectra_shards.name}")
    out["has_spectrum"] = has_spectrum
    print(f"[spectra] has_spectrum {int(has_spectrum.sum()):,} of {len(out):,}")
    if not has_spectrum.any():
        print("[spectra] WARNING: no row has a spectrum. Check that the store you passed "
              "covers the row set in --targets: the shard directory holds the expansion "
              "and shares zero targetids with the current sample. "
              "`make_split --require-spectrum` on this manifest will drop every row.")

    # --- images -----------------------------------------------------------
    if args.fits_pool is not None:
        pool = {p.stem for p in args.fits_pool.glob("*.fits")
                if p.stat().st_size >= MIN_IMAGE_BYTES}
        out["has_image"] = out.targetid.astype(str).isin(pool).to_numpy()
        print(f"[images] {int(out.has_image.sum()):,} of {len(out):,} have a cutout "
              f"of at least {MIN_IMAGE_BYTES:,} bytes in {args.fits_pool.name}")
    else:
        # OMITTED, not False. A False here would AND the image modality off for
        # every row once staging carries these flags, deleting the image arm of
        # the IG table without a word; a missing column degrades to the loader's
        # content heuristic, which at least reads the staged bytes.
        print("[images] no --fits-pool given; has_image is OMITTED from the manifest "
              "(not written False), so the loader falls back to its content check")

    # --- redshift and WISE ------------------------------------------------
    # has_z has exactly the shape of the has_wise problem: the strict rule needs
    # zwarn, and the 104,945-row expansion has no zwarn anywhere on this machine
    # (new_targets_nway.csv is 11 columns and none of them is zwarn). The strict
    # rule stays the default and still refuses rather than guessing, but the
    # weaker rule is reachable by asking for it, and rows that take it are
    # counted rather than blended into the same column unremarked.
    z = frame.z.to_numpy(float)
    if "zwarn" in frame.columns:
        zwarn = frame.zwarn.to_numpy(float)
    else:
        zwarn = np.full(len(frame), np.nan)
    unknown = ~np.isfinite(zwarn)
    if unknown.all() and not args.allow_unvouched_z:
        raise SystemExit(
            "zwarn is NaN for every row, so has_z cannot be defined: a redshift the "
            "survey flags is a model INPUT here, not just a label, and has_z would "
            "come out False for the whole sample.\n"
            "  * for the current sample: pass --desi with the DESI properties join, "
            "which carries zwarn.\n"
            "  * for the 104,945-row expansion there is no zwarn on this machine; "
            "join one from the DESI redshift catalogue, or pass --allow-unvouched-z "
            "to fall back to `finite AND z > 0` for the rows that have none. That "
            "fallback admits ZWARN != 0 redshifts, which on the current sample is "
            "352 of 25,454 rows, so it is a stated weakening and not a default.")
    strict = np.isfinite(z) & (z > 0) & (zwarn == 0)
    weak = np.isfinite(z) & (z > 0)
    use_weak = unknown & args.allow_unvouched_z
    out["has_z"] = np.where(use_weak, weak, strict)
    # Which rule decided each row, so a downstream reader can subset on it
    # instead of rediscovering that the two halves of a merged manifest were
    # judged by different standards.
    out["z_rule"] = np.where(use_weak, "unvouched", "zwarn")
    note = ("weak rule `finite AND z > 0` applied to those"
            if args.allow_unvouched_z
            else "counted as has_z False: an unvouched redshift is a model INPUT here")
    n_zwarn = int((np.isfinite(z) & (z > 0) & (zwarn != 0) & ~unknown).sum())
    n_badz = int((~np.isfinite(z) | (z <= 0)).sum())
    print(f"[z] has_z {int(out.has_z.sum()):,}   flagged by zwarn {n_zwarn:,}   "
          f"non-positive or missing z {n_badz:,}   zwarn unknown {int(unknown.sum()):,} "
          f"({note})")

    presence = wise_presence(frame, args.wise_source)
    out["wise_source"] = args.wise_source
    if presence:
        for name, value in presence.items():
            out[name] = value
        out["has_wise"] = np.logical_or.reduce(list(presence.values()))
        for name in list(presence) + ["has_wise"]:
            print(f"[wise] {name:9s} {int(out[name].sum()):,} of {len(out):,} "
                  f"[--wise-source {args.wise_source}]")
    else:
        print("[wise] --wise-source none: has_wise is OMITTED from the manifest "
              "(not written False), so the loader falls back to its content check")
    if args.wise_source == "ls10":
        # Loud, because this is only correct if staging agrees, and nothing in
        # this script can check that staging agrees.
        print("[wise] LS10 forced photometry defined has_wise, NOT the DESI-side "
              "flux_w*. This is correct ONLY IF staging also sources flux_w1..w3 "
              "from LS10 for these rows. Recorded per row in `wise_source`.")
        cross = wise_presence(frame, "desi") if all(
            c in frame.columns for b in BANDS
            for c in (f"flux_{b}", f"flux_ivar_{b}")) else None
        if cross:
            desi_any = np.logical_or.reduce(list(cross.values()))
            n_diff = int((desi_any != out.has_wise.to_numpy()).sum())
            print(f"[wise] both measurements are present here and disagree about "
                  f"has_wise for {n_diff:,} of {len(out):,} rows; the DESI-side "
                  f"answer would have been {int(desi_any.sum()):,}")

    # --- carried columns --------------------------------------------------
    for col in CARRY:
        out[col] = frame[col].to_numpy() if col in frame.columns else np.nan
    absent = [c for c in CARRY if c not in frame.columns]
    if absent:
        print(f"[carry] {absent} absent from --targets; written as NaN")

    if args.match_quality is not None:
        mq = (pd.read_csv(args.match_quality, usecols=["targetid", "keep", "match_class"])
              .drop_duplicates("targetid"))
        mq["targetid"] = mq.targetid.astype(np.int64)
        mq = mq.set_index("targetid").reindex(out.targetid)
        out["mq_keep"] = mq.keep.to_numpy()
        out["mq_match_class"] = mq.match_class.to_numpy()
        print(f"[mq] {int(mq.keep.notna().sum()):,} of {len(out):,} rows were audited")
    else:
        out["mq_keep"], out["mq_match_class"] = pd.NA, pd.NA

    if args.split is not None:
        split = pd.read_csv(args.split, usecols=["targetid", "split"]).drop_duplicates("targetid")
        split["targetid"] = split.targetid.astype(np.int64)
        out["split"] = split.set_index("targetid").split.reindex(out.targetid).to_numpy()
        held = out.split.notna()
        print(f"[split] {int(held.sum()):,} of {len(out):,} rows are in the split; "
              f"{int((~held).sum()):,} are outside it (filtered at split time)")
    else:
        out["split"] = pd.NA

    # THE CONTRACT. These four columns, under exactly these names and as bool,
    # are what staging carries into the batch and what
    # data_to_aion_embeddings.modality_presence ANDs its content heuristic with.
    # A flag we could not measure is absent from this list, never False.
    emitted = [f for f in PRESENCE_FLAGS if f in out.columns]
    for flag in emitted:
        column = out[flag]
        if column.isna().any():
            raise SystemExit(f"{flag} has NaN; a presence flag is a decision, not a "
                             "measurement that can be missing for one row")
        out[flag] = column.to_numpy(bool)
    if "has_spectrum" not in emitted:                # unreachable; asserted anyway
        raise SystemExit("has_spectrum was not written")
    print(f"[presence] emitting {emitted} as bool"
          + (f"; OMITTED {[f for f in PRESENCE_FLAGS if f not in emitted]} "
             "(loader falls back to its content check for those)"
             if len(emitted) < len(PRESENCE_FLAGS) else ""))

    # Presence over the whole target table is NOT the number an ablation runs
    # on. Only rows in the split are ever trained or scored, and a flag can be
    # informative over the universe while being constant over the split -- which
    # silently removes the negative arm of the modality experiment that flag
    # exists to define. So it is reported on the split too, at build time,
    # rather than discovered when an information-gain table is already in a deck.
    held = out.split.notna().to_numpy() if "split" in out.columns else np.zeros(len(out), bool)
    if held.any():
        print(f"[presence] within the {int(held.sum()):,}-row split "
              f"(the only rows that train or score):")
        for flag in emitted:
            value = out[flag].to_numpy()[held]
            n_false = int((~value).sum())
            note = ""
            if n_false == 0:
                note = ("   CONSTANT: no row in the split lacks this input, so it has "
                        "no natural negative here and every combo sees the same rows")
            print(f"[presence]   {flag:13s} True {int(value.sum()):7,}  "
                  f"False {n_false:6,}{note}")

    order = ([c for c in ["targetid", "ero_detuid", "source_row", "split",
                          "has_spectrum", "has_z", "has_wise"] if c in out.columns]
             + [f"has_{b}" for b in BANDS if f"has_{b}" in out.columns]
             + [c for c in ["has_image", "wise_source", "z_rule", "spectype", "zwarn", "z"]
                if c in out.columns]
             + [c for c in CARRY if c.startswith("nway_") or c == "ls10_objid"]
             + ["mq_keep", "mq_match_class"]
             + [c for c in CARRY if c.startswith("det_like")])
    out = out[order]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    # The image arm of the modality experiment can only be compared like for
    # like on rows that have BOTH, so that count is printed at build time
    # rather than discovered when a table is already in a deck.
    print(f"\n[out] {args.out}  rows={len(out):,}  cols={len(out.columns)}")
    if "has_image" in out.columns:
        both = int((out.has_spectrum & out.has_image).sum())
        print(f"  spectrum AND image: {both:,} ({both/len(out):.1%}) -- the largest sample on "
              f"which an image ablation is a like-for-like comparison")
        if held.any():
            in_split = int((out.has_spectrum & out.has_image).to_numpy()[held].sum())
            print(f"  spectrum AND image WITHIN the split: {in_split:,} of "
                  f"{int(held.sum()):,} ({in_split/int(held.sum()):.1%})")
        only_spec = int((out.has_spectrum & ~out.has_image).sum())
        only_img = int((~out.has_spectrum & out.has_image).sum())
        if only_spec == 0 and only_img == 0:
            print("  has_image is EXACTLY has_spectrum on this table (0 rows with one and "
                  "not the other). That is expected when the cutout pool was fetched for "
                  "whatever the spectrum store holds, and it means the image arm has no "
                  "naturally image-less source to contrast against: an image ablation here "
                  "measures masking, not availability.")
    else:
        print("  has_image not measured, so the like-for-like image sample is unknown "
              "from this manifest")


if __name__ == "__main__":
    main()
