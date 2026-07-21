#!/usr/bin/env python
"""Validate a staged train/val/test directory before spending GPU time on it.

Checks schema, row counts, split integrity (leakage/dedup/fractions), that the
NWAY clean filter was actually applied, per-source error columns, physical value
ranges, and that a batch matches what the model expects.

    python scripts/validate_staged.py --staged-dir <dir> \
        --match-quality-csv <csv> --expect-clean --expect-rows 25200

Exit code 0 = all checks pass. 1 = at least one FAIL (warnings do not fail).
Add --check-model to additionally push one real batch through AION + the head
(needs the aion package; slow on CPU, use a GPU node).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

SPLITS = ("train", "val", "test")

# Datasets every staged split must have for the model to run at all.
REQUIRED = (
    "source_row",
    "desi_targetid",
    "spectra",
    "spectra_ivar",
    "spectra_lambda",
    "redshift",
    "flux_w1",
    "flux_w2",
    "flux_w3",
    "ml_flux_1",
    "log_ml_flux_1",
    "log_lx",
    "image_flux",
)
# Extra target + error columns, present only when --targets-extra-csv was used.
EXTRA = ("logmstar", "logmstar_sig", "hr32_u", "hr32_u_sig", "flux_sig_lo", "flux_sig_hi")

# target -> (sig_lo dataset, sig_hi dataset); symmetric errors reuse one column.
TARGET_ERRORS = {
    "log_ml_flux_1": ("flux_sig_lo", "flux_sig_hi"),
    "log_lx": ("flux_sig_lo", "flux_sig_hi"),
    "logmstar": ("logmstar_sig", "logmstar_sig"),
    "hr32_u": ("hr32_u_sig", "hr32_u_sig"),
}

IMAGE_BANDS = 4
IMAGE_SIZE = 160

# Physically plausible ranges (generous — these catch unit mistakes, column
# swaps, and row misalignment, not marginal science outliers).
PHYSICAL_RANGES = {
    "log_ml_flux_1": (-17.0, -9.0),   # log10 erg/s/cm^2, eRASS1 broad band
    "log_lx": (38.0, 48.0),           # log10 erg/s
    "logmstar": (5.0, 14.0),          # log10 Msun
    "hr32_u": (-6.0, 6.0),            # arctanh(HR), clipped upstream
    "redshift": (0.0, 7.0),
}
# DESI spectra live on a fixed 3600–9824 A grid.
WAVELENGTH_RANGE = (3000.0, 11000.0)
# Range check fails above this outlier fraction (unit slips move MOST rows;
# rare extremes are astrophysics — empirically, NWAY-rejected wrong matches).
RANGE_FAIL_FRAC = 1e-3


class Report:
    """Collects PASS/WARN/FAIL lines and decides the exit code."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []
        self.failed = 0

    def ok(self, name: str, detail: str = "") -> None:
        self.rows.append(("PASS", name, detail))

    def warn(self, name: str, detail: str = "") -> None:
        self.rows.append(("WARN", name, detail))

    def fail(self, name: str, detail: str = "") -> None:
        self.rows.append(("FAIL", name, detail))
        self.failed += 1

    def check(self, condition: bool, name: str, detail: str = "") -> bool:
        (self.ok if condition else self.fail)(name, detail)
        return condition

    def render(self) -> int:
        width = max(len(name) for _, name, _ in self.rows) if self.rows else 0
        icons = {"PASS": "✓", "WARN": "!", "FAIL": "✗"}
        for status, name, detail in self.rows:
            print(f"  {icons[status]} {status:4}  {name:<{width}}  {detail}")
        total = len(self.rows)
        warns = sum(1 for status, _, _ in self.rows if status == "WARN")
        print(f"\n{total - self.failed - warns} passed, {warns} warnings, {self.failed} failed")
        return 1 if self.failed else 0


def finite_frac(values: np.ndarray) -> float:
    return float(np.isfinite(values).mean()) if values.size else 0.0


def check_schema(rep: Report, handles: dict[str, h5py.File]) -> None:
    for split, handle in handles.items():
        missing = [name for name in REQUIRED if name not in handle]
        rep.check(not missing, f"[{split}] required datasets", f"missing: {missing}" if missing else "all present")

        if "desi_targetid" not in handle:
            rep.fail(f"[{split}] row counts aligned", "no desi_targetid to align against")
            continue

        # Every per-source dataset must agree on N. spectra_lambda is the shared
        # wavelength grid and attrs are not datasets, so both are excluded.
        n = handle["desi_targetid"].shape[0]
        bad = {
            name: handle[name].shape
            for name in handle
            if name != "spectra_lambda" and handle[name].shape[0] != n
        }
        rep.check(not bad, f"[{split}] row counts aligned", f"N={n}" if not bad else f"mismatched: {bad}")

        if "image_flux" in handle:
            shape = tuple(handle["image_flux"].shape[1:])
            rep.check(
                shape == (IMAGE_BANDS, IMAGE_SIZE, IMAGE_SIZE),
                f"[{split}] image shape",
                f"{shape} (want {(IMAGE_BANDS, IMAGE_SIZE, IMAGE_SIZE)})",
            )
        if "spectra" in handle and "spectra_lambda" in handle:
            n_pix, n_grid = handle["spectra"].shape[1], handle["spectra_lambda"].shape[0]
            rep.check(n_pix == n_grid, f"[{split}] spectrum/grid length", f"spectra={n_pix} lambda={n_grid}")
            rep.check(
                handle["spectra_ivar"].shape == handle["spectra"].shape,
                f"[{split}] ivar matches spectra",
                str(handle["spectra_ivar"].shape),
            )


def check_splits(
    rep: Report,
    handles: dict[str, h5py.File],
    expect_rows: int | None,
    limited: bool,
    allow_duplicates: bool = False,
) -> None:
    ids = {split: handle["desi_targetid"][:].astype(np.int64) for split, handle in handles.items()}
    counts = {split: len(v) for split, v in ids.items()}
    total = sum(counts.values())
    print(f"\nrows: {counts} total={total}")

    for split, values in ids.items():
        dupes = len(values) - len(np.unique(values))
        if allow_duplicates:
            # The paper superset keeps the original manifest's duplicate rows by
            # design (mild oversampling); cross-split leakage is still fatal below.
            (rep.ok if dupes == 0 else rep.warn)(
                f"[{split}] targetids unique", f"{dupes} duplicates (allowed: paper superset)"
            )
        else:
            rep.check(dupes == 0, f"[{split}] targetids unique", f"{dupes} duplicates" if dupes else "no duplicates")

    # The load-bearing invariant: no object may appear in two splits.
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = np.intersect1d(ids[a], ids[b])
        rep.check(overlap.size == 0, f"leakage {a}/{b}", f"{overlap.size} shared targetids")

    if limited:
        rep.warn("split fractions", "skipped: --limit splits evenly by design")
    else:
        fracs = {split: counts[split] / total for split in SPLITS}
        want = {"train": 0.8, "val": 0.1, "test": 0.1}
        off = {s: round(fracs[s], 4) for s in SPLITS if abs(fracs[s] - want[s]) > 0.01}
        rep.check(
            not off,
            "split fractions 80/10/10",
            ", ".join(f"{s}={fracs[s]:.3f}" for s in SPLITS) + (f" OFF: {off}" if off else ""),
        )

    if expect_rows is not None:
        rep.check(
            abs(total - expect_rows) <= max(5, int(0.01 * expect_rows)),
            "total row count",
            f"{total} (expected ~{expect_rows})",
        )


def check_clean(rep: Report, handles: dict[str, h5py.File], match_quality_csv: Path | None, expect_clean: bool) -> None:
    if match_quality_csv is None:
        rep.warn("clean filter", "skipped: no --match-quality-csv")
        return
    import pandas as pd

    mq = pd.read_csv(match_quality_csv)
    keep_ids = set(mq.loc[mq["keep"].astype(bool), "targetid"].astype(np.int64))
    known = set(mq["targetid"].astype(np.int64))
    all_ids = np.concatenate([handle["desi_targetid"][:].astype(np.int64) for handle in handles.values()])

    unknown = [i for i in all_ids if i not in known]
    covered = 1.0 - len(unknown) / len(all_ids)
    rep.check(covered > 0.99, "NWAY coverage", f"{covered:.3%} of staged rows appear in match_quality.csv")

    bad = [i for i in all_ids if i in known and i not in keep_ids]
    if expect_clean:
        rep.check(not bad, "clean filter applied", f"{len(bad)} known-bad matches survived" if bad else "0 rejected matches present")
    else:
        rep.warn("clean filter", f"{len(bad)} rejected matches present (--expect-clean not set)")


def check_values(rep: Report, handles: dict[str, h5py.File]) -> None:
    for split, handle in handles.items():
        # Absent datasets are already reported by check_schema; skip them here so
        # one missing column cannot abort the remaining value checks.
        if "redshift" in handle:
            z = handle["redshift"][:]
            rep.check(
                bool(np.all(np.isfinite(z))) and float(z.min()) > 0,
                f"[{split}] redshift valid",
                f"min={z.min():.4f} max={z.max():.4f}",
            )

        for target in ("log_ml_flux_1", "log_lx"):
            if target not in handle:
                continue
            frac = finite_frac(handle[target][:])
            rep.check(frac == 1.0, f"[{split}] {target} all finite", f"finite={frac:.4%}")

        for target in ("logmstar", "hr32_u"):
            if target in handle:
                frac = finite_frac(handle[target][:])
                (rep.ok if frac > 0.5 else rep.warn)(f"[{split}] {target} coverage", f"finite={frac:.2%}")

        # Errors must be strictly positive wherever the target exists, or the
        # convolution likelihood divides by zero / takes log of a zero-width kernel.
        for target, (lo_name, hi_name) in TARGET_ERRORS.items():
            if target not in handle or lo_name not in handle:
                continue
            target_values = handle[target][:]
            lo, hi = handle[lo_name][:], handle[hi_name][:]
            usable = np.isfinite(target_values) & np.isfinite(lo) & np.isfinite(hi)
            if usable.sum() == 0:
                rep.fail(f"[{split}] {target} errors", "no rows have both target and finite sigma")
                continue
            nonpos = int(((lo[usable] <= 0) | (hi[usable] <= 0)).sum())
            rep.check(
                nonpos == 0,
                f"[{split}] {target} sigma > 0",
                f"{usable.sum()} usable rows, median sig_lo={np.median(lo[usable]):.4f} "
                f"sig_hi={np.median(hi[usable]):.4f}" + (f", {nonpos} NON-POSITIVE" if nonpos else ""),
            )
            with_target = int(np.isfinite(target_values).sum())
            cov = usable.sum() / with_target if with_target else 0.0
            (rep.ok if cov > 0.9 else rep.warn)(
                f"[{split}] {target} sigma coverage", f"{cov:.2%} of finite targets have sigma"
            )

        if "image_flux" in handle:
            images = handle["image_flux"]
            sample = images[: min(64, images.shape[0])]
            frac = finite_frac(sample)
            rep.check(frac > 0.99, f"[{split}] images finite", f"finite={frac:.4%} (first {len(sample)})")
            allzero = int((np.abs(sample).sum(axis=tuple(range(1, sample.ndim))) == 0).sum())
            (rep.ok if allzero == 0 else rep.warn)(
                f"[{split}] images non-empty", f"{allzero} all-zero cutouts in sample"
            )

        if "spectra" in handle:
            spec = handle["spectra"][: min(64, handle["spectra"].shape[0])]
            rep.check(finite_frac(spec) > 0.99, f"[{split}] spectra finite", f"finite={finite_frac(spec):.4%}")

        if "spectra_ivar" in handle:
            ivar = handle["spectra_ivar"][: min(64, handle["spectra_ivar"].shape[0])]
            neg = int((ivar < 0).sum())
            rep.check(neg == 0, f"[{split}] ivar non-negative", f"{neg} negative values" if neg else "all >= 0")

        if "spectra_lambda" in handle:
            grid = handle["spectra_lambda"][:]
            ascending = bool(np.all(np.diff(grid) > 0))
            in_range = WAVELENGTH_RANGE[0] < float(grid.min()) and float(grid.max()) < WAVELENGTH_RANGE[1]
            rep.check(
                ascending and in_range,
                f"[{split}] wavelength grid",
                f"{grid.min():.0f}-{grid.max():.0f} A, ascending={ascending}",
            )

        # Generous physical windows. This is a UNIT-MISTAKE detector, not an
        # outlier hunt: a unit slip or column swap throws most rows out of range,
        # while a handful of extremes are astrophysics (empirically: the staged
        # superset's out-of-range rows are NWAY-rejected wrong matches, e.g. a
        # z=4.7 mismatch at log_lx=48.07). Fail only above 0.1% outside; report
        # small counts as warnings.
        for column, (lo_bound, hi_bound) in PHYSICAL_RANGES.items():
            if column not in handle:
                continue
            values = handle[column][:]
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            n_out = int(((values < lo_bound) | (values > hi_bound)).sum())
            frac_out = n_out / values.size
            detail = (
                f"[{values.min():.2f}, {values.max():.2f}] within ({lo_bound}, {hi_bound})"
                + (f", {n_out} outside ({frac_out:.3%})" if n_out else "")
            )
            if frac_out > RANGE_FAIL_FRAC:
                rep.fail(f"[{split}] {column} in physical range", detail)
            elif n_out:
                rep.warn(f"[{split}] {column} in physical range", detail)
            else:
                rep.ok(f"[{split}] {column} in physical range", detail)


def check_consistency(rep: Report, handles: dict[str, h5py.File]) -> None:
    """Recompute derived targets from their inputs and compare to what's stored.

    log_ml_flux_1 must equal log10(ml_flux_1) and log_lx must equal
    log10(4 pi D_L(z)^2 ml_flux_1) row by row. A mismatch means the columns were
    written from misaligned rows — the silent failure mode of any indexed copy —
    which no schema or range check can see.
    """
    try:
        from astropy.cosmology import Planck18
    except ImportError:
        rep.warn("derived-column consistency", "skipped: astropy not available")
        return

    for split, handle in handles.items():
        needed = ("ml_flux_1", "redshift", "log_ml_flux_1", "log_lx")
        if any(name not in handle for name in needed):
            continue
        n = handle["ml_flux_1"].shape[0]
        idx = np.linspace(0, n - 1, min(256, n)).astype(int)
        flux = handle["ml_flux_1"][idx].astype(np.float64)
        z = handle["redshift"][idx].astype(np.float64)
        stored_logf = handle["log_ml_flux_1"][idx].astype(np.float64)
        stored_loglx = handle["log_lx"][idx].astype(np.float64)

        valid = np.isfinite(flux) & (flux > 0) & np.isfinite(z)
        expect_logf = np.log10(flux[valid])
        d_l_cm = Planck18.luminosity_distance(z[valid]).to("cm").value
        expect_loglx = np.log10(4.0 * np.pi * d_l_cm**2 * flux[valid])

        err_f = float(np.max(np.abs(expect_logf - stored_logf[valid]))) if valid.any() else 0.0
        err_lx = float(np.max(np.abs(expect_loglx - stored_loglx[valid]))) if valid.any() else 0.0
        # float32 storage: ~1e-6 relative; anything above 1e-3 dex is misalignment.
        rep.check(err_f < 1e-3, f"[{split}] log_flux consistent", f"max |Δ|={err_f:.2e} dex ({valid.sum()} rows)")
        rep.check(err_lx < 1e-3, f"[{split}] log_lx consistent w/ z+flux", f"max |Δ|={err_lx:.2e} dex")


def check_split_provenance(rep: Report, handles: dict[str, h5py.File], paper_manifest_csv: Path | None) -> None:
    """Report how our cleaned re-split relates to the paper's original split.

    Informational: after clean+dedup we re-derive the split, so our test set is
    NOT the paper's test set and published numbers are not row-for-row
    comparable. This surfaces the overlap so nobody is surprised later.
    """
    if paper_manifest_csv is None or not paper_manifest_csv.exists():
        rep.warn("paper-split provenance", "skipped: no --paper-manifest-csv")
        return
    import pandas as pd

    paper = pd.read_csv(paper_manifest_csv)
    paper_split = dict(zip(paper["targetid"].astype(np.int64), paper["split"].astype(str)))
    test_ids = handles["test"]["desi_targetid"][:].astype(np.int64)
    from_paper_train = sum(1 for i in test_ids if paper_split.get(int(i)) == "train")
    from_paper_test = sum(1 for i in test_ids if paper_split.get(int(i)) == "test")
    unknown = sum(1 for i in test_ids if int(i) not in paper_split)
    rep.warn(
        "paper-split provenance",
        f"new test set: {from_paper_test} were paper-test, {from_paper_train} were paper-train, "
        f"{unknown} not in paper manifest (n={len(test_ids)}) — paper numbers not row-comparable",
    )


def check_model_contract(rep: Report, staged_dir: Path, target: str, run_forward: bool) -> None:
    """Confirm the dataloader emits exactly what train()/batch_nll() destructure."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from shareable_aion_flow.data_to_aion_embeddings import build_dataloaders

    train_loader, _, _ = build_dataloaders(
        staged_dir=staged_dir, target_name=target, batch_size=4, eval_batch_size=4, num_workers=0, seed=0
    )
    batch = next(iter(train_loader))
    rep.check(len(batch) == 10, "dataloader tuple arity", f"{len(batch)} tensors (train() expects 10)")

    names = ["flux", "ivar", "wavelength", "redshift", "wise", "image", "target", "targetid", "sig_lo", "sig_hi"]
    shapes = {n: tuple(t.shape) for n, t in zip(names, batch)}
    print(f"\nbatch shapes: {shapes}")
    rep.check(shapes["image"][1:] == (IMAGE_BANDS, IMAGE_SIZE, IMAGE_SIZE), "batch image shape", str(shapes["image"]))
    rep.check(shapes["wise"][1] == 3, "batch wise shape", str(shapes["wise"]))
    rep.check(
        shapes["flux"] == shapes["ivar"] and shapes["flux"][1] == shapes["wavelength"][1],
        "batch spectrum shapes",
        f"flux={shapes['flux']} ivar={shapes['ivar']} lambda={shapes['wavelength']}",
    )
    rep.check(
        shapes["target"][0] == shapes["sig_lo"][0] == shapes["sig_hi"][0],
        "batch target/sigma aligned",
        f"target={shapes['target']} sig_lo={shapes['sig_lo']}",
    )

    if not run_forward:
        rep.warn("model forward", "skipped: pass --check-model to run AION + head on one batch")
        return

    import torch

    from shareable_aion_flow.attention_pooling_head import MODALITIES
    from shareable_aion_flow.main import batch_nll, build_model
    from shareable_aion_flow.normalizing_flow import TargetStandardizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, context_encoder, flow = build_model(
        device, dropout=0.0, head={"num_queries": 1, "num_layers": 1, "context_hidden": [128], "context_dim": 256}
    )
    batch = tuple(t.to(device) for t in batch)
    # Exercise the REAL training path (encode_tokens + head + flow NLL) so a
    # contract drift between loader and trainer fails here, not mid-job.
    with torch.no_grad():
        standardizer = TargetStandardizer.fit(batch[6].cpu().numpy())
        loss = batch_nll(
            encoder=encoder,
            context_encoder=context_encoder,
            flow=flow,
            batch=batch,
            combo=tuple(MODALITIES),
            standardizer=standardizer,
            error_mode="none",
        )
    rep.check(
        bool(torch.isfinite(loss)), "model forward", f"all-inputs batch NLL={float(loss):.4f}, device={device}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--staged-dir", type=Path, required=True)
    parser.add_argument("--match-quality-csv", type=Path, default=None)
    parser.add_argument("--expect-clean", action="store_true", help="Fail if NWAY-rejected matches are present.")
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Paper superset mode: duplicate targetids within a split warn instead of fail.",
    )
    parser.add_argument("--expect-rows", type=int, default=None, help="Expected total rows across splits (1%% tol).")
    parser.add_argument("--target", default="log_ml_flux_1", help="Target used for the dataloader contract check.")
    parser.add_argument(
        "--max-missing-fits",
        type=float,
        default=0.20,
        help="Fail if more than this fraction of manifest rows lacked a Legacy Survey cutout.",
    )
    parser.add_argument(
        "--paper-manifest-csv",
        type=Path,
        default=None,
        help="Original paper split manifest; reports (informationally) how the re-split relates to it.",
    )
    parser.add_argument("--check-model", action="store_true", help="Also run AION + head forward on one batch.")
    parser.add_argument("--skip-dataloader", action="store_true")
    args = parser.parse_args()

    staged_dir = args.staged_dir
    print(f"validating staged dir: {staged_dir}\n")
    rep = Report()

    paths = {split: staged_dir / f"desi_{split}.hdf5" for split in SPLITS}
    for split, path in paths.items():
        if not rep.check(path.exists(), f"[{split}] file exists", str(path)):
            return rep.render()

    summary_path = staged_dir / "summary.json"
    limited = False
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        limited = summary.get("limit") is not None
        rep.ok("summary.json", f"limit={summary.get('limit')} missing_fits={summary.get('missing_fits_count')}")
        dropped = summary.get("dropped_nonfinite_targets")
        if dropped:
            rep.warn("dropped rows", f"{dropped} rows had non-finite targets at staging")

        # Staging silently drops any object without a Legacy Survey cutout. An
        # interrupted unzip therefore shrinks the sample with no other symptom,
        # so make that loss loud rather than something you notice in a row count.
        missing = summary.get("missing_fits_count")
        original = summary.get("original_manifest_rows")
        if missing is not None and original:
            lost = missing / original
            rep.check(
                lost <= args.max_missing_fits,
                "fits coverage",
                f"{missing}/{original} manifest rows had no cutout ({lost:.1%} lost; "
                f"max {args.max_missing_fits:.0%}) — check the fits_pool unzip is complete",
            )
    else:
        rep.warn("summary.json", "absent")

    handles = {split: h5py.File(path, "r") for split, path in paths.items()}
    try:
        check_schema(rep, handles)
        check_splits(rep, handles, args.expect_rows, limited, allow_duplicates=args.allow_duplicates)
        check_clean(rep, handles, args.match_quality_csv, args.expect_clean)
        check_values(rep, handles)
        check_consistency(rep, handles)
        check_split_provenance(rep, handles, args.paper_manifest_csv)
    finally:
        for handle in handles.values():
            handle.close()

    if not args.skip_dataloader:
        try:
            check_model_contract(rep, staged_dir, args.target, args.check_model)
        except Exception as exc:  # a broken contract must show up as a FAIL, not a traceback
            rep.fail("dataloader contract", f"{type(exc).__name__}: {exc}")

    print()
    return rep.render()


if __name__ == "__main__":
    raise SystemExit(main())
