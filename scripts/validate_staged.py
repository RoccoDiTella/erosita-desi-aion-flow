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


def check_splits(rep: Report, handles: dict[str, h5py.File], expect_rows: int | None, limited: bool) -> None:
    ids = {split: handle["desi_targetid"][:].astype(np.int64) for split, handle in handles.items()}
    counts = {split: len(v) for split, v in ids.items()}
    total = sum(counts.values())
    print(f"\nrows: {counts} total={total}")

    for split, values in ids.items():
        dupes = len(values) - len(np.unique(values))
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

    from shareable_aion_flow.main import build_model
    from shareable_aion_flow.normalizing_flow import TargetStandardizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, context_encoder, flow = build_model(
        device, dropout=0.0, head={"num_queries": 1, "num_layers": 1, "context_hidden": [128], "context_dim": 256}
    )
    batch = tuple(t.to(device) for t in batch)
    with torch.no_grad():
        tokens = encoder.encode_tokens(*batch[:6])
        context = context_encoder(tokens, combo=tuple(sorted(tokens.keys())))
        standardizer = TargetStandardizer.fit(batch[6].cpu().numpy())
        y = torch.as_tensor(standardizer.transform_numpy(batch[6].cpu().numpy()), device=device).float()
        logp = flow.log_prob(y.unsqueeze(-1), context)
    rep.check(
        bool(torch.isfinite(logp).all()), "model forward", f"log_prob finite, shape={tuple(logp.shape)}, device={device}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--staged-dir", type=Path, required=True)
    parser.add_argument("--match-quality-csv", type=Path, default=None)
    parser.add_argument("--expect-clean", action="store_true", help="Fail if NWAY-rejected matches are present.")
    parser.add_argument("--expect-rows", type=int, default=None, help="Expected total rows across splits (1%% tol).")
    parser.add_argument("--target", default="log_ml_flux_1", help="Target used for the dataloader contract check.")
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
    else:
        rep.warn("summary.json", "absent")

    handles = {split: h5py.File(path, "r") for split, path in paths.items()}
    try:
        check_schema(rep, handles)
        check_splits(rep, handles, args.expect_rows, limited)
        check_clean(rep, handles, args.match_quality_csv, args.expect_clean)
        check_values(rep, handles)
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
