"""Tests for scripts/validate_staged.py.

Each test builds a small synthetic staged directory, breaks exactly one thing,
and asserts the corresponding check turns red. A validator whose checks never
fire is worse than no validator, so the negative cases are the point.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location("validate_staged", REPO_ROOT / "scripts" / "validate_staged.py")
assert _spec is not None and _spec.loader is not None
vs = importlib.util.module_from_spec(_spec)
sys.modules["validate_staged"] = vs
_spec.loader.exec_module(vs)


N_PIX = 32
SPLIT_SIZES = {"train": 16, "val": 2, "test": 2}


def _log_lx(z: np.ndarray, flux: np.ndarray) -> np.ndarray:
    from astropy.cosmology import Planck18

    d_l_cm = Planck18.luminosity_distance(z).to("cm").value
    return np.log10(4.0 * np.pi * d_l_cm**2 * flux)


def write_split(
    path: Path,
    targetids: np.ndarray,
    *,
    sig_lo: float = 0.1,
    extras: bool = True,
    labels: bool = True,
) -> None:
    """`labels=False` writes the inputs-only (dr2v2) contract: no X-ray labels."""
    n = len(targetids)
    rng = np.random.default_rng(len(targetids))
    with h5py.File(path, "w") as handle:
        handle.create_dataset("source_row", data=np.arange(n, dtype=np.int64))
        handle.create_dataset("desi_targetid", data=targetids.astype(np.int64))
        handle.create_dataset("spectra", data=rng.normal(size=(n, N_PIX)).astype(np.float32))
        handle.create_dataset("spectra_ivar", data=np.ones((n, N_PIX), dtype=np.float32))
        handle.create_dataset("spectra_lambda", data=np.linspace(3600, 9800, N_PIX).astype(np.float32))
        # Varying z and flux so derived-column consistency is a real constraint.
        z = rng.uniform(0.1, 2.0, size=n)
        flux = 10 ** rng.uniform(-14.0, -12.0, size=n)
        handle.create_dataset("redshift", data=z.astype(np.float64))
        for band in ("flux_w1", "flux_w2", "flux_w3"):
            handle.create_dataset(band, data=rng.normal(size=n).astype(np.float32))
        if labels:
            handle.create_dataset("ml_flux_1", data=flux.astype(np.float64))
            handle.create_dataset("log_ml_flux_1", data=np.log10(flux).astype(np.float64))
            handle.create_dataset("log_lx", data=_log_lx(z, flux).astype(np.float64))
        handle.create_dataset(
            "image_flux",
            data=rng.normal(size=(n, vs.IMAGE_BANDS, vs.IMAGE_SIZE, vs.IMAGE_SIZE)).astype(np.float32),
        )
        if extras:
            handle.create_dataset("flux_sig_lo", data=np.full(n, sig_lo, dtype=np.float32))
            handle.create_dataset("flux_sig_hi", data=np.full(n, 0.15, dtype=np.float32))
            handle.create_dataset("logmstar", data=np.full(n, 10.5, dtype=np.float32))
            handle.create_dataset("logmstar_sig", data=np.full(n, 0.2, dtype=np.float32))
            handle.create_dataset("hr32_u", data=np.full(n, 0.1, dtype=np.float32))
            handle.create_dataset("hr32_u_sig", data=np.full(n, 0.3, dtype=np.float32))


def build_staged(tmp_path: Path, *, ids: dict[str, np.ndarray] | None = None, **kwargs) -> Path:
    staged = tmp_path / "staged"
    staged.mkdir(exist_ok=True)
    if ids is None:
        cursor = 0
        ids = {}
        for split, size in SPLIT_SIZES.items():
            ids[split] = np.arange(cursor, cursor + size, dtype=np.int64)
            cursor += size
    for split, targetids in ids.items():
        write_split(staged / f"desi_{split}.hdf5", targetids, **kwargs)
    return staged


def run_checks(staged: Path, *, match_quality_csv: Path | None = None, expect_clean: bool = False) -> vs.Report:
    rep = vs.Report()
    handles = {split: h5py.File(staged / f"desi_{split}.hdf5", "r") for split in vs.SPLITS}
    try:
        vs.check_schema(rep, handles)
        vs.check_splits(rep, handles, expect_rows=None, limited=False)
        vs.check_clean(rep, handles, match_quality_csv, expect_clean)
        vs.check_values(rep, handles)
        vs.check_consistency(rep, handles)
    finally:
        for handle in handles.values():
            handle.close()
    return rep


def failed_names(rep: vs.Report) -> list[str]:
    return [name for status, name, _ in rep.rows if status == "FAIL"]


def test_wellformed_staged_dir_passes(tmp_path: Path) -> None:
    rep = run_checks(build_staged(tmp_path))
    assert rep.failed == 0, f"unexpected failures: {failed_names(rep)}"


def test_split_fraction_check_fires(tmp_path: Path) -> None:
    """The default 16/2/2 is exactly 80/10/10; a lopsided split must be caught."""
    ids = {
        "train": np.arange(0, 5, dtype=np.int64),
        "val": np.arange(5, 12, dtype=np.int64),
        "test": np.arange(12, 20, dtype=np.int64),
    }
    rep = run_checks(build_staged(tmp_path, ids=ids))
    assert "split fractions 80/10/10" in failed_names(rep)


def test_leakage_between_splits_is_caught(tmp_path: Path) -> None:
    ids = {
        "train": np.arange(0, 16, dtype=np.int64),
        "val": np.array([3, 100], dtype=np.int64),  # 3 also lives in train
        "test": np.array([101, 102], dtype=np.int64),
    }
    rep = run_checks(build_staged(tmp_path, ids=ids))
    assert "leakage train/val" in failed_names(rep)


def test_duplicate_targetids_within_split_are_caught(tmp_path: Path) -> None:
    ids = {
        "train": np.array([1, 1] + list(range(2, 16)), dtype=np.int64),
        "val": np.array([100, 101], dtype=np.int64),
        "test": np.array([102, 103], dtype=np.int64),
    }
    rep = run_checks(build_staged(tmp_path, ids=ids))
    assert "[train] targetids unique" in failed_names(rep)


def test_nonpositive_sigma_is_caught(tmp_path: Path) -> None:
    """A zero-width kernel breaks the convolution likelihood, so it must fail."""
    rep = run_checks(build_staged(tmp_path, sig_lo=0.0))
    assert any("sigma > 0" in name for name in failed_names(rep))


def test_missing_required_input_is_caught(tmp_path: Path) -> None:
    """An INPUT is required under both staging contracts.

    This used to delete `log_lx` and assert on "required datasets". log_lx is a
    LABEL, and the current contract says the staged file must not carry one, so
    its absence is now the expected state rather than a fault. The label rule is
    asserted separately, in both directions, further down.
    """
    staged = build_staged(tmp_path)
    with h5py.File(staged / "desi_train.hdf5", "a") as handle:
        del handle["flux_w2"]
    rep = run_checks(staged)
    assert "[train] required inputs" in failed_names(rep)


def test_misaligned_row_counts_are_caught(tmp_path: Path) -> None:
    staged = build_staged(tmp_path)
    with h5py.File(staged / "desi_train.hdf5", "a") as handle:
        del handle["flux_w1"]
        handle.create_dataset("flux_w1", data=np.zeros(3, dtype=np.float32))
    rep = run_checks(staged)
    assert "[train] row counts aligned" in failed_names(rep)


def test_clean_filter_catches_surviving_bad_match(tmp_path: Path) -> None:
    staged = build_staged(tmp_path)
    all_ids = np.arange(0, 20, dtype=np.int64)
    keep = np.ones(len(all_ids), dtype=bool)
    keep[3] = False  # targetid 3 is an NWAY-rejected match but sits in train
    csv_path = tmp_path / "match_quality.csv"
    pd.DataFrame({"targetid": all_ids, "keep": keep}).to_csv(csv_path, index=False)

    rep = run_checks(staged, match_quality_csv=csv_path, expect_clean=True)
    assert "clean filter applied" in failed_names(rep)

    # Same data, filter not asserted -> a warning, not a failure.
    rep_lenient = run_checks(staged, match_quality_csv=csv_path, expect_clean=False)
    assert "clean filter applied" not in failed_names(rep_lenient)


def test_clean_filter_passes_when_all_matches_are_kept(tmp_path: Path) -> None:
    staged = build_staged(tmp_path)
    all_ids = np.arange(0, 20, dtype=np.int64)
    csv_path = tmp_path / "match_quality.csv"
    pd.DataFrame({"targetid": all_ids, "keep": np.ones(len(all_ids), dtype=bool)}).to_csv(csv_path, index=False)
    rep = run_checks(staged, match_quality_csv=csv_path, expect_clean=True)
    assert rep.failed == 0, f"unexpected failures: {failed_names(rep)}"


def test_bad_image_shape_is_caught(tmp_path: Path) -> None:
    staged = build_staged(tmp_path)
    with h5py.File(staged / "desi_val.hdf5", "a") as handle:
        n = handle["desi_targetid"].shape[0]
        del handle["image_flux"]
        handle.create_dataset("image_flux", data=np.zeros((n, 3, 64, 64), dtype=np.float32))
    rep = run_checks(staged)
    assert "[val] image shape" in failed_names(rep)


def test_nonfinite_redshift_is_caught(tmp_path: Path) -> None:
    staged = build_staged(tmp_path)
    with h5py.File(staged / "desi_test.hdf5", "a") as handle:
        handle["redshift"][0] = np.nan
    rep = run_checks(staged)
    assert "[test] redshift valid" in failed_names(rep)


@pytest.mark.parametrize("target", ["log_ml_flux_1", "log_lx"])
def test_nonfinite_target_is_caught(tmp_path: Path, target: str) -> None:
    staged = build_staged(tmp_path)
    with h5py.File(staged / "desi_train.hdf5", "a") as handle:
        handle[target][0] = np.nan
    rep = run_checks(staged)
    assert f"[train] {target} all finite" in failed_names(rep)


def test_row_misalignment_is_caught_by_consistency(tmp_path: Path) -> None:
    """Roll redshift by one row: every schema/range check still passes, but
    log_lx no longer matches z+flux — the misalignment only consistency sees."""
    staged = build_staged(tmp_path)
    with h5py.File(staged / "desi_train.hdf5", "a") as handle:
        z = handle["redshift"][:]
        del handle["redshift"]
        handle.create_dataset("redshift", data=np.roll(z, 1))
    rep = run_checks(staged)
    assert "[train] log_lx consistent w/ z+flux" in failed_names(rep)
    assert "[train] log_flux consistent" not in failed_names(rep)  # flux itself untouched


def test_unit_mistake_is_caught_by_physical_range(tmp_path: Path) -> None:
    """Linear flux written where log flux belongs (a classic unit slip)."""
    staged = build_staged(tmp_path)
    with h5py.File(staged / "desi_val.hdf5", "a") as handle:
        n = handle["desi_targetid"].shape[0]
        del handle["log_ml_flux_1"]
        handle.create_dataset("log_ml_flux_1", data=np.full(n, 1e-13, dtype=np.float64))
    rep = run_checks(staged)
    assert "[val] log_ml_flux_1 in physical range" in failed_names(rep)


def test_negative_ivar_is_caught(tmp_path: Path) -> None:
    staged = build_staged(tmp_path)
    with h5py.File(staged / "desi_test.hdf5", "a") as handle:
        handle["spectra_ivar"][0, 0] = -1.0
    rep = run_checks(staged)
    assert "[test] ivar non-negative" in failed_names(rep)


def test_descending_wavelength_grid_is_caught(tmp_path: Path) -> None:
    staged = build_staged(tmp_path)
    with h5py.File(staged / "desi_train.hdf5", "a") as handle:
        grid = handle["spectra_lambda"][:]
        del handle["spectra_lambda"]
        handle.create_dataset("spectra_lambda", data=grid[::-1].copy())
    rep = run_checks(staged)
    assert "[train] wavelength grid" in failed_names(rep)


def test_rare_outliers_warn_but_mass_violations_fail(tmp_path: Path, monkeypatch) -> None:
    """The range check is a unit-slip detector: one extreme row warns (below the
    fraction threshold), while most-rows-out still fails."""
    monkeypatch.setattr(vs, "RANGE_FAIL_FRAC", 0.5)  # tiny test sets: 1/16 row ~ 6%
    staged = build_staged(tmp_path)
    with h5py.File(staged / "desi_train.hdf5", "a") as handle:
        handle["log_lx"][0] = 48.5  # a single wrong-match-like extreme
    rep = run_checks(staged)
    assert "[train] log_lx in physical range" not in failed_names(rep)
    warned = [name for status, name, _ in rep.rows if status == "WARN"]
    assert "[train] log_lx in physical range" in warned


def test_duplicates_allowed_in_superset_mode(tmp_path: Path) -> None:
    ids = {
        "train": np.array([1, 1] + list(range(2, 16)), dtype=np.int64),
        "val": np.array([100, 101], dtype=np.int64),
        "test": np.array([102, 103], dtype=np.int64),
    }
    staged = build_staged(tmp_path, ids=ids)
    rep = vs.Report()
    handles = {split: h5py.File(staged / f"desi_{split}.hdf5", "r") for split in vs.SPLITS}
    try:
        vs.check_splits(rep, handles, expect_rows=None, limited=False, allow_duplicates=True)
    finally:
        for handle in handles.values():
            handle.close()
    assert "[train] targetids unique" not in failed_names(rep)
    warned = [name for status, name, _ in rep.rows if status == "WARN"]
    assert "[train] targetids unique" in warned
    # Leakage stays fatal even in superset mode.
    assert not [n for n in failed_names(rep) if n.startswith("leakage")]


# --- staging contract -------------------------------------------------------
#
# There are two contracts and the validator has to be able to assert either.
# It used to demand the labels unconditionally, which made it reject every
# dr2v2 staged dir -- the exact directories sbatch/_dataset.sh gates on their
# ABSENCE. train.sbatch runs this script as its pre-GPU gate, so that
# disagreement meant a fresh clone failed validation before training started.


def _schema_report(staged: Path, labels: str) -> vs.Report:
    rep = vs.Report()
    handles = {split: h5py.File(staged / f"desi_{split}.hdf5", "r") for split in vs.SPLITS}
    try:
        vs.check_schema(rep, handles, labels)
    finally:
        for handle in handles.values():
            handle.close()
    return rep


def test_inputs_only_staging_passes_the_forbidden_contract(tmp_path: Path) -> None:
    staged = build_staged(tmp_path, labels=False, extras=False)
    rep = _schema_report(staged, "forbidden")
    assert failed_names(rep) == []


def test_inputs_only_staging_is_rejected_when_labels_are_required(tmp_path: Path) -> None:
    staged = build_staged(tmp_path, labels=False, extras=False)
    rep = _schema_report(staged, "required")
    assert "[train] staged labels present" in failed_names(rep)


def test_labelled_staging_is_rejected_by_the_inputs_only_contract(tmp_path: Path) -> None:
    """The check that matters: a leftover eRASS1 label must not pass silently."""
    staged = build_staged(tmp_path)
    rep = _schema_report(staged, "forbidden")
    assert "[train] staged labels absent (inputs-only)" in failed_names(rep)


def test_auto_reports_the_contract_without_failing_either_way(tmp_path: Path) -> None:
    for labels in (True, False):
        root = tmp_path / f"s{int(labels)}"
        root.mkdir()
        staged = build_staged(root, labels=labels, extras=labels)
        rep = _schema_report(staged, "auto")
        assert failed_names(rep) == []
        contract = [d for s, n, d in rep.rows if n == "[train] staging contract"]
        assert contract and contract[0].startswith("labelled" if labels else "inputs-only")


def test_a_missing_input_still_fails_under_every_mode(tmp_path: Path) -> None:
    """Relaxing the label rule must not relax the input rule."""
    staged = build_staged(tmp_path, labels=False, extras=False)
    with h5py.File(staged / "desi_train.hdf5", "a") as handle:
        del handle["flux_w2"]
    for mode in ("auto", "required", "forbidden"):
        assert "[train] required inputs" in failed_names(_schema_report(staged, mode))
