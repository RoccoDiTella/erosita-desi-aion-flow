"""Tests for the clean-split runtime view over the staged superset.

The view must reproduce build_clean_manifest semantics from the staged files:
CSV membership = the clean filter, first-occurrence = dedup, CSV split = the
re-split — without a second staged copy of the data.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from shareable_aion_flow.data_to_aion_embeddings import (
    AIONHDF5Dataset,
    build_dataloaders,
    clean_view_row_maps,
    read_view_target_values,
)

N_PIX = 16


def write_split(path: Path, targetids: list[int], base_value: float) -> None:
    n = len(targetids)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("desi_targetid", data=np.asarray(targetids, dtype=np.int64))
        handle.create_dataset("spectra", data=np.zeros((n, N_PIX), dtype=np.float32))
        handle.create_dataset("spectra_ivar", data=np.ones((n, N_PIX), dtype=np.float32))
        handle.create_dataset("spectra_lambda", data=np.linspace(3600, 9800, N_PIX).astype(np.float32))
        handle.create_dataset("redshift", data=np.full(n, 0.5, dtype=np.float32))
        for band in ("flux_w1", "flux_w2", "flux_w3"):
            handle.create_dataset(band, data=np.zeros(n, dtype=np.float32))
        handle.create_dataset("image_flux", data=np.zeros((n, 4, 8, 8), dtype=np.float32))
        # target value encodes identity: base + targetid, so we can verify exactly
        # which physical rows the view served.
        handle.create_dataset(
            "log_ml_flux_1",
            data=(base_value + np.asarray(targetids, dtype=np.float64)).astype(np.float32),
        )
        handle.create_dataset("flux_sig_lo", data=np.full(n, 0.1, dtype=np.float32))
        handle.create_dataset("flux_sig_hi", data=np.full(n, 0.2, dtype=np.float32))


def build_superset(tmp_path: Path) -> Path:
    """Native splits: train=[1,2,3,4], val=[5,6], test=[7, 2(dup)]."""
    staged = tmp_path / "staged"
    staged.mkdir()
    write_split(staged / "desi_train.hdf5", [1, 2, 3, 4], base_value=0.0)
    write_split(staged / "desi_val.hdf5", [5, 6], base_value=0.0)
    write_split(staged / "desi_test.hdf5", [7, 2], base_value=0.0)
    return staged


def write_view_csv(tmp_path: Path) -> Path:
    # Cleaned view: drop 4 (NWAY reject); move 2 native-train -> view-test and
    # 5 native-val -> view-train; keep 7 in test, 1/6 in train, 3 in val.
    csv_path = tmp_path / "clean_split.csv"
    pd.DataFrame(
        {
            "targetid": [1, 2, 3, 5, 6, 7],
            "split": ["train", "test", "val", "train", "train", "test"],
        }
    ).to_csv(csv_path, index=False)
    return csv_path


def collect_targetids(loader) -> set[int]:
    ids: set[int] = set()
    for batch in loader:
        ids.update(int(t) for t in batch[7])
    return ids


def test_view_membership_split_reassignment_and_dedup(tmp_path: Path) -> None:
    staged = build_superset(tmp_path)
    csv_path = write_view_csv(tmp_path)
    train_loader, val_loader, test_loader = build_dataloaders(
        staged_dir=staged, target_name="log_ml_flux_1", batch_size=2, num_workers=0, clean_split_csv=csv_path
    )
    assert collect_targetids(train_loader) == {1, 5, 6}
    assert collect_targetids(val_loader) == {3}
    assert collect_targetids(test_loader) == {2, 7}
    # 4 (rejected) appears nowhere; 2 appears once (dup in native test skipped).
    total = sum(len(loader.dataset) for loader in (train_loader, val_loader, test_loader))
    assert total == 6


def test_view_serves_the_right_physical_rows(tmp_path: Path) -> None:
    staged = build_superset(tmp_path)
    csv_path = write_view_csv(tmp_path)
    _, _, test_loader = build_dataloaders(
        staged_dir=staged, target_name="log_ml_flux_1", batch_size=4, num_workers=0, clean_split_csv=csv_path
    )
    for batch in test_loader:
        # target encodes targetid -> row content matches row identity.
        np.testing.assert_allclose(batch[6].numpy(), batch[7].to(torch.float32).numpy(), rtol=1e-6)


def test_view_target_values_match_loader(tmp_path: Path) -> None:
    staged = build_superset(tmp_path)
    csv_path = write_view_csv(tmp_path)
    values = read_view_target_values(staged, "log_ml_flux_1", csv_path, split="train")
    assert sorted(values.astype(int).tolist()) == [1, 5, 6]


def test_view_row_maps_dedup_prefers_first_occurrence(tmp_path: Path) -> None:
    staged = build_superset(tmp_path)
    csv_path = write_view_csv(tmp_path)
    maps = clean_view_row_maps(staged, csv_path)
    test_files = {path.name: rows.tolist() for path, rows in maps["test"]}
    # targetid 2 must come from desi_train (its first occurrence), not desi_test.
    assert test_files["desi_train.hdf5"] == [1]
    assert test_files["desi_test.hdf5"] == [0]  # targetid 7 only


def test_rows_subset_dataset_bounds_check(tmp_path: Path) -> None:
    staged = build_superset(tmp_path)
    try:
        AIONHDF5Dataset(staged / "desi_val.hdf5", "log_ml_flux_1", rows=np.array([5]))
    except IndexError:
        pass
    else:
        raise AssertionError("out-of-range rows must raise")


def test_nonfinite_targets_excluded_native(tmp_path: Path) -> None:
    """Rows with unmeasured (NaN) targets never reach the loaders — batch_nll
    refuses them, so a secondary-target run would otherwise crash mid-epoch."""
    staged = build_superset(tmp_path)
    with h5py.File(staged / "desi_train.hdf5", "a") as handle:
        handle["log_ml_flux_1"][1] = np.nan  # targetid 2 unmeasured
    train_loader, _, _ = build_dataloaders(
        staged_dir=staged, target_name="log_ml_flux_1", batch_size=4, num_workers=0
    )
    ids = collect_targetids(train_loader)
    assert 2 not in ids and ids == {1, 3, 4}
    for batch in train_loader:
        assert bool(np.isfinite(batch[6].numpy()).all())


def test_nonfinite_targets_excluded_in_view(tmp_path: Path) -> None:
    staged = build_superset(tmp_path)
    csv_path = write_view_csv(tmp_path)
    with h5py.File(staged / "desi_val.hdf5", "a") as handle:
        handle["log_ml_flux_1"][0] = np.nan  # targetid 5: view-train, unmeasured
    train_loader, _, _ = build_dataloaders(
        staged_dir=staged, target_name="log_ml_flux_1", batch_size=4, num_workers=0, clean_split_csv=csv_path
    )
    assert collect_targetids(train_loader) == {1, 6}
