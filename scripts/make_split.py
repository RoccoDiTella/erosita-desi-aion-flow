#!/usr/bin/env python
"""Assign train/val/test by a keyed hash of ero_detuid, and record how.

Two properties this buys that a shuffle does not:

**Grouping on the X-ray detection, not the DESI targetid.** Splitting on
targetid puts the same X-ray photons -- the same ML_FLUX_1, the same log_lx up
to a redshift -- in train and in test. Every split before this one was
targetid-grouped.

Two different quantities count that leak, and any number below is meaningless
without the row set it was measured on. All measured 2026-08-06:

| row set                                   | rows   | detuids | shared detuids | excess rows |
|-------------------------------------------|--------|---------|----------------|-------------|
| `dr2_targets.csv` rebuilt by make_dr2_targets | 25,454 | 24,686 | 756 | 768 |
| `targets_sidecar_dr2.csv` on disk (pre-rebuild) | 25,582 | 24,796 | 773 | 786 |

`shared detuids` = detections carrying more than one DESI fibre. `excess rows` =
rows minus detuids, the number of rows a targetid-keyed split could place apart
from a row sharing its photons. The docs' 773 is the shared-detuid count on the
25,582-row file; this docstring previously reported the rebuild's 768 EXCESS ROWS
as a count of detections, which it is not (that number is 756).

Detuids are also shared ACROSS samples, and here too a number means nothing
without BOTH row sets named -- the expansion has two of its own (the raw
`new_targets_nway.csv`, and that file rebuilt through make_dr2_targets, which
drops NWAY secondaries and rows with no DR2 detection). All measured 2026-08-06:

| this table     | vs raw 104,945       | vs rebuilt 104,032   |
|----------------|----------------------|----------------------|
| rebuilt 25,454 | 22 detuids, 22 rows  | 5 detuids, 5 rows    |
| on-disk 25,582 | 50 detuids, 51 rows  | 33 detuids, 34 rows  |

Targetid overlap is zero in all four cells, so merging the two samples under a
targetid split would put the same photons in two splits. Rebuilding both sides
shrinks the leak from 22 rows to 5; it does not remove it, and only grouping on
ero_detuid across the merged table does.

**No RNG.** The assignment is a pure function of (ero_detuid, salt, fractions),
so it is reproducible from the provenance file alone, is stable under row
reordering, and appending the expansion later moves no existing row. There is
deliberately no --seed: a seed makes the assignment a function of arrival order
as well, which is the thing that made the previous splits impossible to
reconstruct. The salt is the only knob, and it is written out.

Reliability cuts live here rather than in the target table, so the threshold is
a re-split away and the cost is recorded next to the counts it changed. p_any
and the DR1 spec-z audit are ORTHOGONAL: p_any asks whether the X-ray source has
any LS10 counterpart, never whether the DESI fibre sits on it (336 of the 345
sources the audit calls "wrong" have p_any > 0.5, median 0.9994). Both are
applied.

Order of the rebuild: make_dr2_targets -> make_targets_sidecar -> build_manifest
(no --split yet) -> make_split --require-spectrum manifest -> build_manifest
again, this time with --split. Two passes, because a split whose fractions
include sources that have no spectrum describes a sample that cannot train.

The manifest's `has_spectrum` for the CURRENT sample has to be measured against
the merged source HDF5 (`build_manifest --source-hdf5`). `spectra_dr2_new.shards`
holds the 104,945-row expansion and shares zero targetids with the current
sample, so pointing `--spectra-shards` at it yields has_spectrum=False for every
current-sample row and this script then exits with "every row was filtered out".

    python scripts/make_split.py --targets data/dr2/dr2_targets.csv \
        --match-quality data/match_quality.csv --min-p-any 0.5 \
        --require-spectrum data/dr2/manifest_dr2.csv \
        --previous data/dr2/clean_split_dr2.csv \
        --out data/dr2/clean_split_dr2_v2.csv

The output name is `_v2` and the `--previous` name is not. That is the point:
the rebuild is written to a NEW file so the old one stays readable for the
comparison `--previous` reports, and `sbatch/_dataset.sh` DATASET=dr2v2 resolves
the `_v2` names. Running this with `--out data/dr2/clean_split_dr2.csv` would
overwrite the artifact that reproduces the published number.

Measured 2026-08-06 on the rebuilt 25,454-row table with the command above:
p_any>0.5 keeps 24,792, the transferable DR1 audit rejects take it to 24,122,
requiring a spectrum takes it to 22,800, and the split is 18,275 / 2,274 / 2,251
(0.8015 / 0.0997 / 0.0987). 7,667 of those 22,800 change split against
`clean_split_dr2.csv`, which is expected: that one was targetid-grouped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")
GROUP_KEY = "ero_detuid"
# Changing this re-draws every split. It is recorded in the provenance file so a
# split can be rebuilt from that file alone; it is NOT derived from any seed.
DEFAULT_SALT = "erosita-desi-aion-flow/dr2/2026-08-06"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_unit(keys, salt: str) -> np.ndarray:
    """Map each key to a deterministic u in [0, 1) by keyed blake2b."""
    key = salt.encode()
    if len(key) > 64:
        raise SystemExit("--salt must encode to at most 64 bytes (blake2b key limit)")
    digest = [hashlib.blake2b(str(k).encode(), key=key, digest_size=8).digest() for k in keys]
    return np.array([int.from_bytes(d, "big") for d in digest], dtype=np.uint64) / 2.0 ** 64


_TRUTHY = {"true": True, "1": True, "1.0": True, "t": True, "yes": True,
           "false": False, "0": False, "0.0": False, "f": False, "no": False}


def require_bool(column: pd.Series, path) -> np.ndarray:
    """Read a presence flag as a boolean, or refuse. Never `astype(bool)`.

    `pd.Series.astype(bool)` maps the STRING "False" to True, because every
    non-empty string is truthy. A manifest whose has_spectrum column came back
    as object dtype -- one NaN anywhere, a lowercase writer, a hand-edited CSV --
    would therefore turn this cut into a silent no-op and admit sources that have
    no staged row at all. That failure is invisible until the loader drops them
    and the split fractions stop describing the sample that trained.
    """
    if column.dtype == bool:
        return column.to_numpy(bool)
    if pd.api.types.is_numeric_dtype(column) and column.notna().all():
        values = column.to_numpy()
        if np.isin(values, (0, 1)).all():
            return values.astype(bool)
    mapped = column.astype(str).str.strip().str.lower().map(_TRUTHY)
    if mapped.isna().any():
        bad = sorted({str(v) for v in column[mapped.isna()].unique()})[:5]
        raise SystemExit(
            f"--require-spectrum {path} has a has_spectrum column of dtype "
            f"{column.dtype} that is not boolean; unreadable values include {bad}.\n"
            "Refusing to coerce: `astype(bool)` reads the string 'False' as True, "
            "which would make this cut a no-op and put sources with no staged row "
            "into the split. Rebuild the manifest with scripts/build_manifest.py, "
            "which writes the presence flags as numpy bool and refuses to write NaN "
            "into one.")
    return mapped.to_numpy(bool)


def assign(groups: np.ndarray, salt: str, fracs) -> np.ndarray:
    unique, inverse = np.unique(groups, return_inverse=True)
    u = hash_unit(unique, salt)
    edges = np.cumsum(fracs)[:-1]
    return np.asarray(SPLITS)[np.searchsorted(edges, u, side="right")][inverse]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", type=Path, required=True,
                    help="DR2 target table or the sidecar built from it")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--provenance", type=Path, default=None,
                    help="default: <out stem>_provenance.json beside --out")
    ap.add_argument("--fracs", type=float, nargs=3, default=(0.80, 0.10, 0.10),
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--salt", default=DEFAULT_SALT)
    ap.add_argument("--tol", type=float, default=0.01,
                    help="allowed absolute drift of a split's ROW fraction from --fracs")
    # OPEN DECISION (rebuild plan D1): the threshold is not settled. 0.5 is a
    # placeholder default, not a recorded choice, and it is NOT free: measured
    # on the 25,454-row rebuild it costs GALAXY 12.5% against QSO 1.2%, and
    # GALAXY is the science arm rather than the negative control. docs/DATA.md
    # still states the selection function as four stacked cuts with p_any "not
    # yet wired in", so running this default makes the sample and that section
    # disagree. Pass --min-p-any 0 to reproduce the documented four-cut sample.
    ap.add_argument("--min-p-any", default=0.5,
                    help="NWAY reliability cut. A NUMBER applies one flat threshold to "
                         "every source; the word 'threshold6' applies NWAY's OWN "
                         "per-source cut from the nway_threshold6 column, calibrated at "
                         "the purity/completeness intersection by a randomised-catalogue "
                         "test. Prefer threshold6: a flat 0.5 is not class-neutral, and "
                         "on dr2v2 it cost GALAXY 12.5%% against QSO 1.2%% -- a 10x "
                         "asymmetry falling on the science arm. 0 keeps every row.")
    ap.add_argument("--missing-threshold", choices=("flat", "median", "drop"), default="flat",
                    help="What to do with rows that have no finite nway_threshold6. "
                         "They are missing metadata, not missing reliability: on the "
                         "merged table all 2,431 have a finite p_any with median 0.939. "
                         "'flat' applies --fallback-p-any and is the conservative choice; "
                         "'median' imputes the population threshold; 'drop' discards them.")
    ap.add_argument("--fallback-p-any", type=float, default=0.5,
                    help="Flat cut for rows with no calibrated threshold, under "
                         "--missing-threshold flat.")
    ap.add_argument("--match-quality", type=Path, default=None,
                    help="DR1 spec-z match audit; its keep=False verdict is applied "
                         "only where the LS10 counterpart is unchanged in DR2")
    ap.add_argument("--require-finite", nargs="*", default=(), metavar="COL",
                    help="restrict the split to rows with a finite value in these columns")
    ap.add_argument("--require-spectrum", type=Path, default=None,
                    help="manifest (or any CSV of targetids) restricting the split to sources "
                         "with a spectrum; a has_spectrum column, if present, is honoured")
    ap.add_argument("--previous", type=Path, default=None,
                    help="previous split CSV, for the n_changed report")
    args = ap.parse_args()

    fracs = np.asarray(args.fracs, float)
    if abs(fracs.sum() - 1.0) > 1e-9 or (fracs <= 0).any():
        raise SystemExit(f"--fracs must be positive and sum to 1, got {args.fracs}")

    frame = pd.read_csv(args.targets, low_memory=False)
    for col in ("targetid", GROUP_KEY):
        if col not in frame.columns:
            raise SystemExit(f"--targets {args.targets} has no {col} column")
    frame["targetid"] = frame.targetid.astype(np.int64)
    if frame.targetid.duplicated().any():
        raise SystemExit("--targets has duplicate targetids")

    # Name the row set once, up front, so every count printed below is
    # attributable to it. The two multi-fibre quantities are different numbers
    # and are reported as different numbers.
    n_input = len(frame)
    n_detuid = int(frame[GROUP_KEY].nunique())
    n_shared = int((frame[GROUP_KEY].value_counts() > 1).sum())
    print(f"[input] {args.targets}  {n_input:,} rows  {n_detuid:,} detuids  "
          f"{n_input - n_detuid:,} excess rows  {n_shared:,} detuids with >1 fibre")

    filters: list[dict] = []
    n = len(frame)

    def drop(name, keep_mask, note=None):
        nonlocal frame, n
        keep_mask = np.asarray(keep_mask, bool)
        frame = frame[keep_mask].reset_index(drop=True)
        row = {"filter": name, "kept": len(frame), "dropped": n - len(frame)}
        if note:
            row["note"] = note
        filters.append(row)
        print(f"[filter] {name:24s} kept {len(frame):7,}  dropped {n - len(frame):6,}")
        n = len(frame)

    drop("has_group_key", frame[GROUP_KEY].notna().to_numpy())

    # --- NWAY reliability -------------------------------------------------
    p_any_cost = {}
    use_thresh6 = str(args.min_p_any).lower() == "threshold6"
    min_p_any = 0.0 if use_thresh6 else float(args.min_p_any)
    if use_thresh6 or min_p_any > 0:
        if "nway_p_any" not in frame.columns:
            raise SystemExit("a reliability cut was asked for but --targets carries no "
                             "nway_p_any; rebuild the table with make_dr2_targets.py")
        if use_thresh6:
            # NWAY's OWN per-source cut, calibrated at the purity/completeness
            # intersection by a randomised-catalogue test (Salvato et al.). A flat
            # threshold is one number applied to sources with very different prior
            # densities of chance counterparts; theirs varies per source and is what
            # the catalogue authors intend the column to be compared against.
            #
            # It matters because a flat 0.5 is not class-neutral: on dr2v2 it cost
            # GALAXY 12.5% against QSO 1.2%, a 10x asymmetry falling entirely on the
            # science arm rather than on the negative control.
            if "nway_threshold6" not in frame.columns:
                raise SystemExit("--min-p-any threshold6 but --targets carries no "
                                 "nway_threshold6 column")
            thr = frame.nway_threshold6.to_numpy(float)
            bad = ~np.isfinite(thr)
            if bad.any():
                # Measured on the merged table: 2,431 of 129,486 rows (1.9%) have no
                # calibrated threshold, and they are NOT junk -- every one has a finite
                # p_any, median 0.939, against a typical threshold of 0.045. They are
                # missing METADATA, not missing reliability, and 911 of them are
                # GALAXY, the science arm. Dropping them silently would spend the
                # thing we are trying to buy.
                #
                # The fallback is a real choice, so it is named rather than assumed:
                #   flat   apply --fallback-p-any (default 0.5) to them. Cannot be
                #          laxer than the flat cut we would otherwise have used on
                #          EVERY row, so it can never inflate the sample relative to
                #          the status quo. Costs ~625 of the 2,431.
                #   median use the population median threshold. Statistically the
                #          better estimate of the cut they WOULD have had, but it is
                #          an imputation and is labelled as one.
                #   drop   discard them. Cleanest provenance, biggest cost.
                if args.missing_threshold == "drop":
                    thr = np.where(bad, np.inf, thr)
                elif args.missing_threshold == "median":
                    thr = np.where(bad, float(np.nanmedian(thr)), thr)
                else:
                    thr = np.where(bad, float(args.fallback_p_any), thr)
                print(f"[p_any>threshold6] {int(bad.sum()):,} rows have no calibrated "
                      f"threshold -> --missing-threshold {args.missing_threshold}"
                      + (f" (p_any > {args.fallback_p_any})"
                         if args.missing_threshold == "flat" else ""))
            keep = frame.nway_p_any.to_numpy(float) > thr
            print(f"[p_any>threshold6] per-source cut, median {np.median(thr):.4f}, "
                  f"range [{thr.min():.4f}, {thr.max():.4f}]")
        else:
            keep = (frame.nway_p_any.to_numpy(float) > min_p_any)
        # The cut is several times more expensive for galaxies than for quasars,
        # and the galaxy arm is the science arm rather than the negative control,
        # so the per-class cost is recorded rather than left to be rediscovered.
        if "spectype" in frame.columns:
            cls_of = frame.spectype.astype(str).to_numpy()
            for cls in sorted(set(cls_of)):
                mask = keep[cls_of == cls]
                p_any_cost[cls] = {"n": int(mask.size), "kept": int(mask.sum()),
                                   "lost_frac": float(1.0 - mask.mean())}
                label = "threshold6" if use_thresh6 else f">{min_p_any}"
                print(f"[p_any {label}] {cls:8s} {mask.sum():7,} of {mask.size:7,}"
                      f"   costs {1.0 - mask.mean():.1%}")
        drop("nway_p_any>threshold6" if use_thresh6 else f"nway_p_any>{min_p_any}", keep)

    # --- DR1 spec-z match audit ------------------------------------------
    n_void = 0
    if args.match_quality is not None:
        mq = pd.read_csv(args.match_quality, usecols=["targetid", "keep", "LS10objid"])
        mq["targetid"] = mq.targetid.astype(np.int64)
        mq = mq.drop_duplicates("targetid").set_index("targetid")
        aligned = mq.reindex(frame.targetid.to_numpy())
        # Rows the audit never saw are not rejects; only an explicit False is.
        rejected = ~aligned.keep.fillna(True).to_numpy(bool)
        # The audit was run against the DR1 counterpart. Where DR2 adopted a
        # different LS10 object the verdict describes a match we no longer make,
        # so it is void for that row rather than inherited.
        if "ls10_objid" in frame.columns:
            same = aligned.LS10objid.to_numpy() == frame.ls10_objid.to_numpy()
            n_void = int((rejected & ~same).sum())
            rejected = rejected & same
        else:
            print("[mq] no ls10_objid in --targets; applying the DR1 verdict to every row")
        drop("match_quality.keep", ~rejected,
             note=f"{n_void} DR1 rejects voided by a changed LS10 counterpart")

    # --- optional row requirements ---------------------------------------
    for col in args.require_finite:
        if col not in frame.columns:
            raise SystemExit(f"--require-finite {col} is not a column of {args.targets}")
        drop(f"finite:{col}", np.isfinite(frame[col].to_numpy(float)))
    if args.require_spectrum is not None:
        # Accepts the manifest, which is the file that measures this. Without the
        # cut the split's fractions describe a sample larger than the one that
        # can train: a source with no spectrum has no staged row at all.
        columns = pd.read_csv(args.require_spectrum, nrows=0).columns
        usecols = ["targetid"] + (["has_spectrum"] if "has_spectrum" in columns else [])
        listed = pd.read_csv(args.require_spectrum, usecols=usecols)
        if "has_spectrum" in usecols:
            listed = listed[require_bool(listed.has_spectrum, args.require_spectrum)]
        have = set(listed.targetid.astype(np.int64))
        drop("has_spectrum", frame.targetid.isin(have).to_numpy())

    if not len(frame):
        raise SystemExit("every row was filtered out")

    # --- assignment -------------------------------------------------------
    groups = frame[GROUP_KEY].astype(str).to_numpy()
    split = assign(groups, args.salt, fracs)
    out = pd.DataFrame({"targetid": frame.targetid.to_numpy(), "split": split})
    out = out.sort_values("targetid").reset_index(drop=True)

    counts = {s: int((split == s).sum()) for s in SPLITS}
    group_counts = {s: int(pd.unique(groups[split == s]).size) for s in SPLITS}
    drift = {s: counts[s] / len(out) - f for s, f in zip(SPLITS, fracs)}
    for s in SPLITS:
        print(f"[split] {s:5s} {counts[s]:7,} rows  {group_counts[s]:7,} detuids "
              f"({counts[s]/len(out):.4f}, drift {drift[s]:+.4f})")
    bad = [s for s in SPLITS if abs(drift[s]) > args.tol]
    if bad:
        raise SystemExit(f"row fractions for {bad} drift more than --tol {args.tol}; "
                         "a few detuids carry many fibres, so re-check --fracs or raise --tol")

    # --- what changed against the previous split --------------------------
    previous = None
    if args.previous is not None and args.previous.exists():
        prev = pd.read_csv(args.previous, usecols=["targetid", "split"])
        prev["targetid"] = prev.targetid.astype(np.int64)
        merged = out.merge(prev, on="targetid", how="outer", suffixes=("", "_prev"))
        common = merged.split.notna() & merged.split_prev.notna()
        previous = {
            "path": str(args.previous),
            "sha256": sha256(args.previous),
            "n_previous": int(len(prev)),
            "n_common": int(common.sum()),
            "n_changed": int((merged.split != merged.split_prev)[common].sum()),
            "n_added": int(merged.split_prev.isna().sum()),
            "n_removed": int(merged.split.isna().sum()),
        }
        print(f"[vs previous] {previous['n_changed']:,} of {previous['n_common']:,} shared "
              f"targetids change split; +{previous['n_added']:,} / -{previous['n_removed']:,}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    provenance = args.provenance or args.out.with_name(args.out.stem + "_provenance.json")
    record = {
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/make_split.py",
        "out": str(args.out),
        "group_key": GROUP_KEY,
        "hash": {"algorithm": "blake2b", "digest_size": 8, "keyed": True, "salt": args.salt},
        "fractions": dict(zip(SPLITS, [float(f) for f in fracs])),
        "tolerance": args.tol,
        "inputs": {"targets": {"path": str(args.targets), "sha256": sha256(args.targets),
                               "n_rows": n_input, "n_detuids": n_detuid,
                               "n_excess_rows": n_input - n_detuid,
                               "n_detuids_multi_fibre": n_shared}},
        "filters": filters,
        "min_p_any": "threshold6" if use_thresh6 else min_p_any,
        "missing_threshold": args.missing_threshold if use_thresh6 else None,
        "fallback_p_any": args.fallback_p_any if use_thresh6 else None,
        "p_any_cost_by_spectype": p_any_cost,
        "match_quality_rejects_voided": n_void,
        "require_finite": list(args.require_finite),
        "n_rows": len(out),
        "row_counts": counts,
        "group_counts": group_counts,
        "previous": previous,
    }
    if args.match_quality is not None:
        record["inputs"]["match_quality"] = {"path": str(args.match_quality),
                                             "sha256": sha256(args.match_quality)}
    if args.require_spectrum is not None:
        record["inputs"]["require_spectrum"] = {"path": str(args.require_spectrum),
                                                "sha256": sha256(args.require_spectrum)}
    provenance.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {args.out} ({len(out):,} rows) and {provenance}")


if __name__ == "__main__":
    main()
