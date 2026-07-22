"""Data staging, AION tokenization, and dataloaders.

Builds the image-backed train/val/test HDF5 splits from the canonical
eROSITA/DESI raw data, wraps the frozen AION encoder so a batch becomes
``(tokens, group_ids)``, and exposes PyTorch dataloaders over the staged splits.
See ``docs/DATA.md`` for how to obtain the raw inputs.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from astropy.cosmology import Planck18
from astropy.io import fits
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

try:
    from .attention_pooling_head import MODALITY_TO_ID
except ImportError:  # Allows `python data_to_aion_embeddings.py` during local debugging.
    from attention_pooling_head import MODALITY_TO_ID


PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
MANIFEST_DIR = DATA_DIR / "manifests"
STAGED_DIR = DATA_DIR / "staged"
CACHE_DIR = DATA_DIR / "cache"

SOURCE_HDF5 = RAW_DIR / "erosita_desi" / "erosita_spectra_merged_32k.hdf5"
SPLIT_MANIFEST = MANIFEST_DIR / "aion_tvsplit_manifest.csv"
FITS_POOL_DIR = RAW_DIR / "legacysurvey" / "fits_pool"

LEGACY_SURVEY_IMAGE_DATASET = "image_flux"
LEGACY_SURVEY_IMAGE_BANDS = ("DES-G", "DES-R", "DES-I", "DES-Z")
LEGACY_SURVEY_IMAGE_BANDS_QUERY = "griz"
LEGACY_SURVEY_IMAGE_SIZE = 160

TOKEN_KEYS_BY_MODALITY = {
    "spectra": ("tok_spectrum_desi",),
    "z": ("tok_z",),
    "wise": ("tok_flux_w1", "tok_flux_w2", "tok_flux_w3"),
    "image": ("tok_image",),
}

DATASET_ALIASES = {
    "desi_targetid": ("desi_targetid",),
    "spectra": ("spectra", "spectra_flux"),
    "spectra_ivar": ("spectra_ivar",),
    "spectra_lambda": ("spectra_lambda",),
    "redshift": ("redshift", "desi_z"),
    "flux_w1": ("flux_w1",),
    "flux_w2": ("flux_w2",),
    "flux_w3": ("flux_w3",),
    "target_ra": ("target_ra",),
    "target_dec": ("target_dec",),
    "ml_flux_1": ("ml_flux_1",),
    "log_ml_flux_1": ("log_ml_flux_1",),
    "log_lx": ("log_lx",),
    "logmstar": ("logmstar",),
    "logmstar_sig": ("logmstar_sig",),
    "hr32_u": ("hr32_u",),
    "hr32_u_sig": ("hr32_u_sig",),
    "flux_sig_lo": ("flux_sig_lo",),
    "flux_sig_hi": ("flux_sig_hi",),
    LEGACY_SURVEY_IMAGE_DATASET: (LEGACY_SURVEY_IMAGE_DATASET,),
}

# Extra target columns joined from targets_extra.csv (by desi_targetid) into each
# staged split, and the (lower, upper) 1-sigma error column for each target.
EXTRA_TARGET_COLUMNS = ("logmstar", "logmstar_sig", "hr32_u", "hr32_u_sig", "flux_sig_lo", "flux_sig_hi")
TARGET_ERROR_COLS = {
    "log_ml_flux_1": ("flux_sig_lo", "flux_sig_hi"),
    "log_lx": ("flux_sig_lo", "flux_sig_hi"),
    "logmstar": ("logmstar_sig", "logmstar_sig"),
    "hr32_u": ("hr32_u_sig", "hr32_u_sig"),
}


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def resolve_dataset_name(handle: h5py.File, canonical_name: str) -> str:
    for candidate in DATASET_ALIASES.get(canonical_name, (canonical_name,)):
        if candidate in handle:
            return candidate
    raise KeyError(f"Dataset {canonical_name!r} not found in {handle.filename}.")


def read_dataset(
    handle: h5py.File, canonical_name: str, rows: np.ndarray | slice | None = None
) -> np.ndarray:
    dataset = handle[resolve_dataset_name(handle, canonical_name)]
    if rows is None:
        return dataset[:]
    return dataset[rows]


def read_ml_flux_1(handle: h5py.File) -> np.ndarray:
    if "ml_flux_1" in handle:
        return handle["ml_flux_1"][:].astype(np.float32)
    target_names = [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in handle.attrs.get("target_names", [])
    ]
    if "targets" in handle and "ml_flux_1" in target_names:
        return handle["targets"][:, target_names.index("ml_flux_1")].astype(np.float32)
    if "targets" in handle:
        return handle["targets"][:, 0].astype(np.float32)
    raise KeyError("Unable to read ML_FLUX_1 from source HDF5.")


def compute_log_flux(flux: np.ndarray) -> np.ndarray:
    flux = np.asarray(flux, dtype=np.float64)
    out = np.full(flux.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(flux) & (flux > 0.0)
    out[valid] = np.log10(flux[valid])
    return out


def compute_log_luminosity(redshift: np.ndarray, flux: np.ndarray) -> np.ndarray:
    redshift = np.asarray(redshift, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    redshift, flux = np.broadcast_arrays(redshift, flux)
    out = np.full(redshift.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(redshift) & np.isfinite(flux) & (flux > 0.0)
    if valid.any():
        d_l_cm = Planck18.luminosity_distance(redshift[valid]).to("cm").value
        out[valid] = np.log10(4.0 * np.pi * d_l_cm * d_l_cm * flux[valid])
    return out


def _copy_dataset(
    source: h5py.File,
    dest: h5py.File,
    canonical_name: str,
    rows: np.ndarray | None,
    *,
    dtype: np.dtype | type | None = None,
) -> None:
    data = read_dataset(source, canonical_name, rows)
    if dtype is not None:
        data = data.astype(dtype)
    dest.create_dataset(canonical_name, data=data, compression="gzip", compression_opts=4)


def _parse_fits_image(path: Path) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        image = np.asarray(hdul[0].data, dtype=np.float32)
        bands = str(hdul[0].header.get("BANDS", "")).lower()
    if image.shape != (len(LEGACY_SURVEY_IMAGE_BANDS), LEGACY_SURVEY_IMAGE_SIZE, LEGACY_SURVEY_IMAGE_SIZE):
        raise ValueError(f"{path} has image shape {image.shape}, expected (4, 160, 160).")
    if bands and bands != LEGACY_SURVEY_IMAGE_BANDS_QUERY:
        raise ValueError(f"{path} has bands {bands!r}, expected {LEGACY_SURVEY_IMAGE_BANDS_QUERY!r}.")
    if not np.isfinite(image).all():
        raise ValueError(f"{path} contains non-finite image values.")
    return image


def _compression_kwargs(compression: str) -> dict[str, object]:
    if compression == "none":
        return {}
    if compression == "gzip":
        return {"compression": "gzip", "compression_opts": 4}
    if compression == "lzf":
        return {"compression": "lzf"}
    raise ValueError(f"Unsupported compression: {compression!r}")


def build_clean_manifest(
    manifest: pd.DataFrame,
    match_quality_csv: Path,
    *,
    seed: int = 42,
    split_fracs: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> pd.DataFrame:
    """Apply the NWAY clean filter, dedup to one row per targetid, and RE-SPLIT.

    The split is re-derived on the cleaned+deduped sample so train/val/test reflect
    the objects we actually train on. It is targetid-grouped by construction (one
    row per targetid after dedup), so no targetid crosses splits.
    ``match_quality_csv`` provides per-targetid ``keep`` (NWAY-correct match).
    """
    mq = pd.read_csv(match_quality_csv, usecols=["targetid", "keep"])
    keep_ids = set(mq.loc[mq["keep"].astype(bool), "targetid"].astype(np.int64).tolist())
    cleaned = (
        manifest.assign(targetid=manifest["targetid"].astype(np.int64))
        .loc[lambda d: d["targetid"].isin(keep_ids)]
        .drop_duplicates("targetid")
        .reset_index(drop=True)
        .copy()
    )
    ids = cleaned["targetid"].to_numpy(np.int64)
    perm = np.random.default_rng(seed).permutation(len(ids))
    n = len(ids)
    n_train = int(split_fracs[0] * n)
    n_val = int(split_fracs[1] * n)
    split_labels = np.array(["test"] * n, dtype=object)
    split_labels[perm[:n_train]] = "train"
    split_labels[perm[n_train : n_train + n_val]] = "val"
    cleaned["split"] = split_labels
    return cleaned


def prepare_staged_data(
    *,
    source_hdf5: Path = SOURCE_HDF5,
    split_manifest_csv: Path = SPLIT_MANIFEST,
    fits_pool_dir: Path = FITS_POOL_DIR,
    output_dir: Path = STAGED_DIR,
    limit: int | None = None,
    overwrite: bool = False,
    image_compression: str = "none",
    clean: bool = False,
    match_quality_csv: Path | None = None,
    targets_extra_csv: Path | None = None,
) -> dict[str, object]:
    """Build image-backed train/val/test HDF5 files from canonical raw data.

    The split manifest is targetid-safe and carries the source HDF5 row for
    each sample. Rows whose targetid does not have a FITS file are excluded,
    because this package trains the four-modality model.

    ``clean=True`` applies the NWAY clean filter + dedup and RE-SPLITS on the
    cleaned sample (needs ``match_quality_csv``). ``targets_extra_csv`` adds the
    extra target + per-source error columns (logmstar, hr32_u, flux_sig_lo/hi).
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    outputs = [output_dir / f"desi_{split}.hdf5" for split in ("train", "val", "test")]
    if not overwrite and any(path.exists() for path in outputs):
        raise FileExistsError(f"Staged outputs already exist under {output_dir}. Use --overwrite to rebuild.")

    manifest = pd.read_csv(split_manifest_csv)
    original_manifest_rows = int(len(manifest))
    if clean:
        if match_quality_csv is None:
            raise ValueError("clean=True requires match_quality_csv (the NWAY keep filter).")
        manifest = build_clean_manifest(manifest, Path(match_quality_csv))

    targets_extra = None
    if targets_extra_csv is not None:
        targets_extra = pd.read_csv(targets_extra_csv)
        targets_extra["targetid"] = targets_extra["targetid"].astype(np.int64)
        targets_extra = targets_extra.drop_duplicates("targetid").set_index("targetid")
    manifest["fits_path"] = manifest["targetid"].map(lambda targetid: fits_pool_dir / f"{int(targetid)}.fits")
    has_fits = manifest["fits_path"].map(Path.exists)
    missing_fits_count = int((~has_fits).sum())
    manifest = manifest.loc[has_fits].copy()
    with h5py.File(source_hdf5, "r") as source:
        manifest_rows = manifest["source_row"].to_numpy(dtype=np.int64)
        read_order = np.argsort(manifest_rows)
        sorted_rows = manifest_rows[read_order]
        source_targetids_sorted = read_dataset(source, "desi_targetid", sorted_rows).astype(np.int64)
        source_targetids = np.empty_like(source_targetids_sorted)
        source_targetids[read_order] = source_targetids_sorted
        if not np.array_equal(source_targetids, manifest["targetid"].to_numpy(dtype=np.int64)):
            raise ValueError("Split manifest targetids do not match source_hdf5 targetids at source_row.")

        ml_flux_sorted = read_ml_flux_1(source)[sorted_rows].astype(np.float64)
        redshift_sorted = read_dataset(source, "redshift", sorted_rows).astype(np.float64)
        ml_flux = np.empty_like(ml_flux_sorted)
        redshift = np.empty_like(redshift_sorted)
        ml_flux[read_order] = ml_flux_sorted
        redshift[read_order] = redshift_sorted
    log_flux = compute_log_flux(ml_flux)
    log_lx = compute_log_luminosity(redshift, ml_flux)
    finite_targets = np.isfinite(log_flux) & np.isfinite(log_lx)
    dropped_nonfinite_targets = int((~finite_targets).sum())
    manifest = manifest.loc[finite_targets].copy()
    if limit is not None:
        if int(limit) < 3:
            raise ValueError(
                "--limit must be at least 3 so train, val, and test each receive at least one row."
            )
        split_order = ("train", "val", "test")
        base = int(limit) // len(split_order)
        remainder = int(limit) % len(split_order)
        pieces = []
        for split_index, split in enumerate(split_order):
            split_count = base + (1 if split_index < remainder else 0)
            split_rows = manifest.loc[manifest["split"] == split].head(split_count).copy()
            if len(split_rows) != split_count:
                raise ValueError(f"Not enough rows for split {split!r} to satisfy --limit {limit}.")
            pieces.append(split_rows)
        manifest = pd.concat(pieces, axis=0).sort_values("source_row").copy()

    summary: dict[str, object] = {
        "source_hdf5": str(source_hdf5),
        "split_manifest_csv": str(split_manifest_csv),
        "fits_pool_dir": str(fits_pool_dir),
        "output_dir": str(output_dir),
        "image_compression": image_compression,
        "limit": limit,
        "original_manifest_rows": original_manifest_rows,
        "missing_fits_count": missing_fits_count,
        "dropped_nonfinite_targets": dropped_nonfinite_targets,
        "split_counts": {},
    }

    compression_kwargs = _compression_kwargs(image_compression)
    for split in ("train", "val", "test"):
        split_frame = manifest.loc[manifest["split"] == split].sort_values("source_row").copy()
        rows = split_frame["source_row"].to_numpy(dtype=np.int64)
        targetids = split_frame["targetid"].to_numpy(dtype=np.int64)
        dest_path = output_dir / f"desi_{split}.hdf5"
        if dest_path.exists():
            dest_path.unlink()

        with h5py.File(source_hdf5, "r") as source, h5py.File(dest_path, "w") as dest:
            dest.create_dataset("source_row", data=rows, compression="gzip", compression_opts=4)
            for name in (
                "desi_targetid",
                "spectra",
                "spectra_ivar",
                "redshift",
                "flux_w1",
                "flux_w2",
                "flux_w3",
            ):
                _copy_dataset(source, dest, name, rows)
            _copy_dataset(source, dest, "spectra_lambda", None)
            for optional_name in ("target_ra", "target_dec"):
                if any(candidate in source for candidate in DATASET_ALIASES[optional_name]):
                    _copy_dataset(source, dest, optional_name, rows, dtype=np.float32)

            ml_flux = read_ml_flux_1(source)[rows].astype(np.float32)
            dest.create_dataset("ml_flux_1", data=ml_flux, compression="gzip", compression_opts=4)
            redshift = dest["redshift"][:].astype(np.float64)
            dest.create_dataset(
                "log_ml_flux_1",
                data=compute_log_flux(ml_flux).astype(np.float32),
                compression="gzip",
                compression_opts=4,
            )
            dest.create_dataset(
                "log_lx",
                data=compute_log_luminosity(redshift, ml_flux).astype(np.float32),
                compression="gzip",
                compression_opts=4,
            )

            if targets_extra is not None:
                aligned = targets_extra.reindex(targetids)
                for col in EXTRA_TARGET_COLUMNS:
                    values = (
                        aligned[col].to_numpy(dtype=np.float64)
                        if col in aligned.columns
                        else np.full(len(targetids), np.nan)
                    )
                    dest.create_dataset(
                        col, data=values.astype(np.float32), compression="gzip", compression_opts=4
                    )
                if "spectype" in aligned.columns:
                    spectype = aligned["spectype"].fillna("").astype(str).to_numpy()
                    dest.create_dataset("spectype", data=np.asarray(spectype, dtype="S"))

            image_ds = dest.create_dataset(
                LEGACY_SURVEY_IMAGE_DATASET,
                shape=(
                    len(rows),
                    len(LEGACY_SURVEY_IMAGE_BANDS),
                    LEGACY_SURVEY_IMAGE_SIZE,
                    LEGACY_SURVEY_IMAGE_SIZE,
                ),
                dtype=np.float32,
                chunks=(
                    min(16, max(1, len(rows))),
                    len(LEGACY_SURVEY_IMAGE_BANDS),
                    LEGACY_SURVEY_IMAGE_SIZE,
                    LEGACY_SURVEY_IMAGE_SIZE,
                ),
                **compression_kwargs,
            )
            for index, targetid in enumerate(tqdm(targetids, desc=f"stage-{split}-images", unit="img")):
                image_ds[index] = _parse_fits_image(fits_pool_dir / f"{int(targetid)}.fits")

            dest.attrs["image_bands"] = np.asarray(LEGACY_SURVEY_IMAGE_BANDS, dtype="S")
            dest.attrs["image_size"] = LEGACY_SURVEY_IMAGE_SIZE
            dest.attrs["aion_targets"] = np.asarray(["log_ml_flux_1", "log_lx"], dtype="S")

        summary["split_counts"][split] = int(len(rows))  # type: ignore[index]

    write_json(summary_path, summary)
    return summary


class AIONHDF5Dataset(Dataset):
    """PyTorch dataset over one staged split (optionally a row-subset view).

    Each item is the 10-tuple ``(spectra, spectra_ivar, wavelength, redshift, wise,
    image, target, targetid, sig_lo, sig_hi)`` consumed by the AION tokenizer and
    the training loop. ``rows`` restricts the dataset to those row indices — used
    by the clean-split runtime view so the cleaned configuration needs no second
    staged copy of the data.
    """

    def __init__(self, hdf5_path: Path, target_name: str, rows: np.ndarray | None = None) -> None:
        self.hdf5_path = Path(hdf5_path)
        self.target_name = target_name
        self._handle: h5py.File | None = None
        self.err_cols = TARGET_ERROR_COLS.get(target_name)
        self.rows = None if rows is None else np.asarray(rows, dtype=np.int64)
        with h5py.File(self.hdf5_path, "r") as handle:
            n = int(handle["desi_targetid"].shape[0])
            if self.rows is not None and len(self.rows) and (self.rows.min() < 0 or self.rows.max() >= n):
                raise IndexError(f"rows out of range for {self.hdf5_path} (n={n}).")
            self.length = n if self.rows is None else int(len(self.rows))
            self._wavelength = torch.from_numpy(read_dataset(handle, "spectra_lambda").astype(np.float32))
            self.has_err = bool(self.err_cols) and all(name in handle for name in self.err_cols)

    def _ensure_open(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.hdf5_path, "r")
        return self._handle

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        handle = self._ensure_open()
        if self.rows is not None:
            index = int(self.rows[index])
        wise = np.asarray(
            [handle[name][index] for name in ("flux_w1", "flux_w2", "flux_w3")], dtype=np.float32
        )
        if self.has_err:
            sig_lo = float(read_dataset(handle, self.err_cols[0], slice(index, index + 1))[0])
            sig_hi = float(read_dataset(handle, self.err_cols[1], slice(index, index + 1))[0])
        else:
            sig_lo = sig_hi = 0.0
        return (
            torch.from_numpy(read_dataset(handle, "spectra", slice(index, index + 1))[0].astype(np.float32)),
            torch.from_numpy(
                read_dataset(handle, "spectra_ivar", slice(index, index + 1))[0].astype(np.float32)
            ),
            self._wavelength,
            torch.tensor(read_dataset(handle, "redshift", slice(index, index + 1))[0], dtype=torch.float32),
            torch.from_numpy(wise),
            torch.from_numpy(
                read_dataset(handle, LEGACY_SURVEY_IMAGE_DATASET, slice(index, index + 1))[0].astype(
                    np.float32
                )
            ),
            torch.tensor(
                read_dataset(handle, self.target_name, slice(index, index + 1))[0], dtype=torch.float32
            ),
            torch.tensor(
                int(read_dataset(handle, "desi_targetid", slice(index, index + 1))[0]), dtype=torch.int64
            ),
            torch.tensor(sig_lo, dtype=torch.float32),
            torch.tensor(sig_hi, dtype=torch.float32),
        )


def clean_view_row_maps(
    staged_dir: Path, clean_split_csv: Path
) -> dict[str, list[tuple[Path, np.ndarray]]]:
    """Map the clean-split sidecar onto the staged superset files.

    Returns, per view split, the ``(staged_file, row_indices)`` pairs selecting
    exactly the cleaned sample: targetids absent from the CSV are excluded (NWAY
    reject) and only the first occurrence of a duplicated targetid is used
    (dedup), scanning train/val/test in order — mirroring ``build_clean_manifest``.
    """
    assign = pd.read_csv(clean_split_csv)
    split_of = dict(
        zip(assign["targetid"].astype(np.int64).tolist(), assign["split"].astype(str).tolist())
    )
    maps: dict[str, list[tuple[Path, np.ndarray]]] = {"train": [], "val": [], "test": []}
    seen: set[int] = set()
    for split_file in ("train", "val", "test"):
        path = Path(staged_dir) / f"desi_{split_file}.hdf5"
        with h5py.File(path, "r") as handle:
            targetids = handle["desi_targetid"][:].astype(np.int64)
        rows_by_view: dict[str, list[int]] = {"train": [], "val": [], "test": []}
        for row, targetid in enumerate(targetids.tolist()):
            view = split_of.get(targetid)
            if view is None or targetid in seen:
                continue
            seen.add(targetid)
            rows_by_view[view].append(row)
        for view, rows in rows_by_view.items():
            if rows:
                maps[view].append((path, np.asarray(rows, dtype=np.int64)))
    return maps


def _finite_target_rows(
    path: Path, target_name: str, rows: np.ndarray | None, max_target_sigma: float | None = None
) -> np.ndarray | None:
    """Restrict ``rows`` (or the whole file) to usable rows for this target.

    Always requires a finite target (``batch_nll`` refuses NaN). With
    ``max_target_sigma``, additionally requires the target's mean measurement
    sigma to be finite and <= the threshold: the S/N gate for targets with
    pathological error tails (hr32_u: median sigma 0.51 but max ~8e3 from
    band non-detections; those rows are upper limits, not measurements).
    Returns ``None`` when the whole file qualifies.
    """
    with h5py.File(path, "r") as handle:
        values = read_dataset(handle, target_name)
        keep = np.isfinite(values)
        err_cols = TARGET_ERROR_COLS.get(target_name)
        if max_target_sigma is not None and err_cols and all(c in handle for c in err_cols):
            sig = 0.5 * (
                read_dataset(handle, err_cols[0]).astype(np.float64)
                + read_dataset(handle, err_cols[1]).astype(np.float64)
            )
            keep &= np.isfinite(sig) & (sig <= float(max_target_sigma))
    if rows is None:
        if bool(keep.all()):
            return None
        kept = np.flatnonzero(keep).astype(np.int64)
    else:
        kept = rows[keep[rows]]
    dropped = (len(values) if rows is None else len(rows)) - len(kept)
    if dropped:
        print(f"[loader] {path.name}: dropped {dropped} rows (non-finite {target_name}"
              + (f" or sigma > {max_target_sigma}" if max_target_sigma is not None else "") + ")", flush=True)
    return kept


def build_dataloaders(
    *,
    staged_dir: Path = STAGED_DIR,
    target_name: str = "log_ml_flux_1",
    batch_size: int = 256,
    eval_batch_size: int | None = None,
    num_workers: int = 0,
    seed: int | None = None,
    clean_split_csv: Path | None = None,
    max_target_sigma: float | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test dataloaders over the staged HDF5 splits.

    With ``clean_split_csv``, the loaders serve the cleaned+re-split VIEW of the
    staged superset (row selection at load time) instead of the files' native
    splits — one staged copy serves both the paper and cleaned configurations.
    Rows whose target is non-finite are always excluded (see _finite_target_rows).
    """
    eval_batch_size = eval_batch_size or batch_size
    view_maps = clean_view_row_maps(Path(staged_dir), clean_split_csv) if clean_split_csv else None
    loaders: list[DataLoader] = []
    train_generator = None
    if seed is not None:
        train_generator = torch.Generator()
        train_generator.manual_seed(int(seed))
    for split in ("train", "val", "test"):
        if view_maps is None:
            path = Path(staged_dir) / f"desi_{split}.hdf5"
            rows = _finite_target_rows(path, target_name, None, max_target_sigma)
            dataset: Dataset = AIONHDF5Dataset(path, target_name, rows=rows)
        else:
            parts = []
            for path, rows in view_maps[split]:
                kept = _finite_target_rows(path, target_name, rows, max_target_sigma)
                if kept is None or len(kept):
                    parts.append(AIONHDF5Dataset(path, target_name, rows=kept if kept is not None else rows))
            if not parts:
                raise ValueError(f"Clean-split view has no rows for split {split!r}.")
            dataset = ConcatDataset(parts)
        loaders.append(
            DataLoader(
                dataset,
                batch_size=batch_size if split == "train" else eval_batch_size,
                shuffle=(split == "train"),
                num_workers=num_workers,
                pin_memory=True,
                generator=train_generator if split == "train" else None,
            )
        )
    return loaders[0], loaders[1], loaders[2]


def read_target_values(hdf5_path: Path, target_name: str) -> np.ndarray:
    """Read the one-dimensional target column from a staged split."""
    with h5py.File(hdf5_path, "r") as handle:
        return read_dataset(handle, target_name).astype(np.float32)


def read_view_target_values(
    staged_dir: Path, target_name: str, clean_split_csv: Path, split: str = "train"
) -> np.ndarray:
    """Target values for one split of the clean view (standardizer/prior fitting)."""
    maps = clean_view_row_maps(Path(staged_dir), clean_split_csv)
    parts: list[np.ndarray] = []
    for path, rows in maps[split]:
        with h5py.File(path, "r") as handle:
            parts.append(read_dataset(handle, target_name)[rows].astype(np.float32))
    if not parts:
        raise ValueError(f"Clean-split view has no rows for split {split!r}.")
    return np.concatenate(parts)


def move_batch_to_device(batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(item.to(device, non_blocking=True) for item in batch)


def resolve_column_offset(changed_by_probe: dict[int, list[int]], n_cols: int) -> int:
    """Column offset of codec wavelength-token 0 inside the code tensor.

    The 272 wavelength tokens occupy a contiguous block of the ``n_cols`` code
    columns, so the offset is one of 0..(n_cols - 272) (typically {0, 1}: the
    normalization token sits before or after the block). Each candidate is
    scored by the summed distance of the probe-changed columns to their spiked
    token position; the minimum wins. Robust to receptive-field leakage
    flipping asymmetric neighbours (which biases mean/median estimates).
    """
    n_candidates = n_cols - N_SPEC_TOKENS_EXPECTED
    if n_candidates < 0:
        raise RuntimeError(f"Code tensor has {n_cols} columns < {N_SPEC_TOKENS_EXPECTED} wavelength tokens.")
    if not any(changed_by_probe.values()):
        raise RuntimeError("Spectrum-code column probe found no changed columns.")
    best_offset, best_score = 0, float("inf")
    for offset in range(n_candidates + 1):
        score = sum(
            abs(c - (offset + q_probe))
            for q_probe, cols in changed_by_probe.items()
            for c in cols
        )
        if score < best_score:
            best_offset, best_score = offset, score
    return best_offset


N_SPEC_TOKENS_EXPECTED = 272  # codec grid: 8704 px / 32 px per token


class AIONTokenEncoder(nn.Module):
    """Frozen AION encoder wrapped to return tokens and per-token modality ids.

    ``encode_tokens`` turns a batch into ``(tokens [B, T, 768], group_ids [B, T])``,
    where each group id is a direct modality (0=spectra, 1=z, 2=wise, 3=image) for the
    attention head. ``aion`` is imported lazily so staging and tests run without it.
    """

    def __init__(self, model_name: str = "polymathic-ai/aion-base", freeze: bool = True) -> None:
        super().__init__()
        try:
            from aion import AION
            from aion.codecs import CodecManager
            from aion.modalities import (
                DESISpectrum,
                LegacySurveyFluxW1,
                LegacySurveyFluxW2,
                LegacySurveyFluxW3,
                LegacySurveyImage,
                Z,
            )
        except ImportError as exc:
            raise ImportError("Install polymathic-aion to use the AION encoder.") from exc

        self.backbone = AION.from_pretrained(model_name)
        self.codec_manager = None
        self.codec_cls = CodecManager
        self.spectrum_cls = DESISpectrum
        self.z_cls = Z
        self.image_cls = LegacySurveyImage
        self.wise_classes = (LegacySurveyFluxW1, LegacySurveyFluxW2, LegacySurveyFluxW3)
        self.freeze = freeze
        if freeze:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def _codec(self, device: torch.device):
        if self.codec_manager is None or str(device) != getattr(self.codec_manager, "device", None):
            self.codec_manager = self.codec_cls(device=str(device))
        return self.codec_manager

    def _modalities(
        self,
        flux: torch.Tensor,
        ivar: torch.Tensor,
        wavelength: torch.Tensor,
        redshift: torch.Tensor,
        wise: torch.Tensor,
        image: torch.Tensor,
        combo: Iterable[str],
    ) -> list[object]:
        requested = set(combo)
        modalities: list[object] = []
        if "spectra" in requested:
            modalities.append(
                self.spectrum_cls(flux=flux, ivar=ivar, mask=ivar <= 0.0, wavelength=wavelength)
            )
        if "z" in requested:
            modalities.append(self.z_cls(value=redshift))
        if "wise" in requested:
            for index, wise_cls in enumerate(self.wise_classes):
                modalities.append(wise_cls(value=wise[:, index]))
        if "image" in requested:
            modalities.append(self.image_cls(flux=image, bands=list(LEGACY_SURVEY_IMAGE_BANDS)))
        return modalities

    @staticmethod
    def _num_tokens(token_dict: dict[str, torch.Tensor]) -> int:
        return sum(tensor.shape[1] if tensor.dim() > 1 else 1 for tensor in token_dict.values())

    def _group_ids_from_modality_mask(
        self, modality_mask: torch.Tensor, invalid: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Map AION modality ids to head group ids; ``invalid`` positions become -1 (padding)."""
        group_ids = torch.full_like(modality_mask, -1, dtype=torch.long)
        modality_info = getattr(self.backbone, "modality_info", {})
        for group_name, token_keys in TOKEN_KEYS_BY_MODALITY.items():
            for token_key in token_keys:
                if token_key in modality_info:
                    group_ids[modality_mask.eq(int(modality_info[token_key]["id"]))] = MODALITY_TO_ID[
                        group_name
                    ]
        unmapped = group_ids.lt(0)
        if invalid is not None:
            unmapped = unmapped & ~invalid
        if bool(unmapped.any()):
            raise RuntimeError("AION returned token ids that are not mapped to spectra, z, WISE, or image.")
        if invalid is not None:
            group_ids = group_ids.masked_fill(invalid, -1)
        return group_ids

    def _spec_key(self, token_dict: dict) -> str:
        for key in token_dict:
            if "spectrum" in key:
                return key
        raise KeyError(f"No spectrum token key in {list(token_dict)}")

    @staticmethod
    def _median_continuum(flux: torch.Tensor, k: int = 101) -> torch.Tensor:
        """Per-source 101-px (~80 A) running-median continuum estimate."""
        pad = k // 2
        padded = torch.nn.functional.pad(flux.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
        return padded.unfold(-1, k, 1).median(dim=-1).values

    def _locate_spec_columns(self, device: torch.device, wavelength: torch.Tensor) -> int:
        """Code-column offset of CODEC wavelength-token 0 in the code tensor.

        Probed once with two spikes placed at raw pixels of two KNOWN codec
        tokens (the codec grid starts at 3500 A, the raw DESI grid at 3600 A,
        so codec token q covers raw pixels [32q - 125, 32q - 93)). The offset
        is geometrically constrained to 0..(n_cols - 272) (the norm token sits
        before or after the contiguous wavelength block), so it is resolved by
        scoring those few candidates against the changed columns -- robust to
        asymmetric receptive-field leakage that biased a median estimate.
        """
        if getattr(self, "_spec_col_offset", None) is not None:
            return self._spec_col_offset
        try:
            from .line_shapley import token_to_raw_pixels
        except ImportError:
            from line_shapley import token_to_raw_pixels

        codec = self._codec(device)
        n_raw = wavelength.shape[-1]
        wl = wavelength[:1] if wavelength.dim() > 1 else wavelength.unsqueeze(0)
        base = torch.ones(1, n_raw, device=device)
        ivar1 = torch.ones_like(base)

        def code_of(fl: torch.Tensor) -> torch.Tensor:
            td = codec.encode(self.spectrum_cls(flux=fl, ivar=ivar1, mask=ivar1 <= 0, wavelength=wl))
            return td[self._spec_key(td)]

        base_code = code_of(base)
        changed_by_probe: dict[int, list[int]] = {}
        for q_probe in (20, 60):
            p_lo, p_hi = token_to_raw_pixels(q_probe)
            spike = base.clone()
            spike[0, p_lo:p_hi] = 25.0
            changed = (base_code != code_of(spike)).nonzero()
            changed_by_probe[q_probe] = sorted(set(int(c) for c in changed[:, -1].tolist()))
        self._spec_col_offset = resolve_column_offset(changed_by_probe, int(base_code.shape[-1]))
        return self._spec_col_offset

    def encode_tokens(
        self,
        batch: tuple[torch.Tensor, ...],
        combo: tuple[str, ...],
        spectrum_token_mask: torch.Tensor | None = None,
        mask_mode: str = "drop",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch; ``spectrum_token_mask`` (bool [B, 272]) removes wavelength tokens.

        ``mask_mode="drop"`` (default) removes masked tokens the AION-native way:
        they are marked invalid in the encoder input mask, so backbone attention
        never sees them (partial input is the pretraining task) and they come
        back as group id -1 (padding) for the head to skip. Nothing is replaced
        or imputed; codec receptive-field leakage into neighbouring KEPT tokens
        is handled upstream by the guard band in ``player_token_map``.
        ``"replace"`` is the legacy behaviour: swap the masked token codes for
        the source's own 101-px median-filtered continuum codes. The
        normalization token is never masked in either mode, so both variants
        condition on overall brightness.
        """
        flux, ivar, wavelength, redshift, wise, image = batch[:6]
        codec = self._codec(flux.device)
        if spectrum_token_mask is not None:
            spectrum_token_mask = spectrum_token_mask.to(flux.device)
        modalities = self._modalities(flux, ivar, wavelength, redshift, wise, image, combo)
        token_dict = codec.encode(*modalities)
        input_mask_dict = None
        if spectrum_token_mask is not None and "spectra" in combo:
            if mask_mode not in ("drop", "replace"):
                raise ValueError(f"Unknown mask_mode {mask_mode!r}.")
            key = self._spec_key(token_dict)
            offset = self._locate_spec_columns(flux.device, wavelength)
            n_tok = spectrum_token_mask.shape[1]
            n_cols = int(token_dict[key].shape[1])
            if offset + n_tok > n_cols:
                raise RuntimeError(
                    f"Spectrum-code offset {offset} + {n_tok} tokens exceeds the "
                    f"{n_cols}-column code tensor -- column probe misresolved."
                )
            if mask_mode == "drop":
                col_mask = torch.zeros(
                    token_dict[key].shape[:2], dtype=torch.bool, device=flux.device
                )
                col_mask[:, offset : offset + n_tok] = spectrum_token_mask
                input_mask_dict = {key: col_mask}
            else:
                cont = self._median_continuum(flux)
                cont_dict = codec.encode(
                    self.spectrum_cls(flux=cont, ivar=ivar, mask=ivar <= 0.0, wavelength=wavelength)
                )
                codes = token_dict[key].clone()
                cont_codes = cont_dict[self._spec_key(cont_dict)]
                sub = codes[:, offset : offset + n_tok]
                sub[spectrum_token_mask] = cont_codes[:, offset : offset + n_tok][spectrum_token_mask]
                codes[:, offset : offset + n_tok] = sub
                token_dict[key] = codes
        num_tokens = self._num_tokens(token_dict)

        context_manager = torch.no_grad() if self.freeze else torch.enable_grad()
        with context_manager:
            encoder_tokens, encoder_emb, encoder_mask, encoder_mod_mask = self.backbone.embed_inputs(
                token_dict,
                mask=input_mask_dict,
                num_encoder_tokens=num_tokens,
            )
            tokens = self.backbone._encode(encoder_tokens, encoder_emb, encoder_mask)
        # encoder_mask marks invalid (dropped) positions with 1/True; those become
        # group id -1 so the attention head excludes them from pooling.
        invalid = None
        if input_mask_dict is not None:
            invalid = encoder_mask.reshape(encoder_mod_mask.shape).bool()
        return tokens, self._group_ids_from_modality_mask(encoder_mod_mask, invalid=invalid)
