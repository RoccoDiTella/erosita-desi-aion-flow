from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
import torch
from astropy.cosmology import Planck18
from astropy.io import fits
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from .attention_pooling_head import MODALITIES, MODALITY_TO_ID
except ImportError:  # Allows `python data_to_aion_embeddings.py` during local debugging.
    from attention_pooling_head import MODALITIES, MODALITY_TO_ID


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
    LEGACY_SURVEY_IMAGE_DATASET: (LEGACY_SURVEY_IMAGE_DATASET,),
}


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def resolve_dataset_name(handle: h5py.File, canonical_name: str) -> str:
    for candidate in DATASET_ALIASES.get(canonical_name, (canonical_name,)):
        if candidate in handle:
            return candidate
    raise KeyError(f"Dataset {canonical_name!r} not found in {handle.filename}.")


def read_dataset(handle: h5py.File, canonical_name: str, rows: np.ndarray | slice | None = None) -> np.ndarray:
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


def prepare_staged_data(
    *,
    source_hdf5: Path = SOURCE_HDF5,
    split_manifest_csv: Path = SPLIT_MANIFEST,
    fits_pool_dir: Path = FITS_POOL_DIR,
    output_dir: Path = STAGED_DIR,
    limit: int | None = None,
    overwrite: bool = False,
    image_compression: str = "none",
) -> dict[str, object]:
    """Build image-backed train/val/test HDF5 files from canonical raw data.

    The split manifest is targetid-safe and carries the source HDF5 row for
    each sample. Rows whose targetid does not have a FITS file are excluded,
    because this package trains the four-modality model.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    outputs = [output_dir / f"desi_{split}.hdf5" for split in ("train", "val", "test")]
    if not overwrite and any(path.exists() for path in outputs):
        raise FileExistsError(f"Staged outputs already exist under {output_dir}. Use --overwrite to rebuild.")

    manifest = pd.read_csv(split_manifest_csv)
    original_manifest_rows = int(len(manifest))
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
            raise ValueError("--limit must be at least 3 so train, val, and test each receive at least one row.")
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

            image_ds = dest.create_dataset(
                LEGACY_SURVEY_IMAGE_DATASET,
                shape=(len(rows), len(LEGACY_SURVEY_IMAGE_BANDS), LEGACY_SURVEY_IMAGE_SIZE, LEGACY_SURVEY_IMAGE_SIZE),
                dtype=np.float32,
                chunks=(min(16, max(1, len(rows))), len(LEGACY_SURVEY_IMAGE_BANDS), LEGACY_SURVEY_IMAGE_SIZE, LEGACY_SURVEY_IMAGE_SIZE),
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
    def __init__(self, hdf5_path: Path, target_name: str) -> None:
        self.hdf5_path = Path(hdf5_path)
        self.target_name = target_name
        self._handle: h5py.File | None = None
        with h5py.File(self.hdf5_path, "r") as handle:
            self.length = int(handle["desi_targetid"].shape[0])
            self._wavelength = torch.from_numpy(read_dataset(handle, "spectra_lambda").astype(np.float32))

    def _ensure_open(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.hdf5_path, "r")
        return self._handle

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        handle = self._ensure_open()
        wise = np.asarray([handle[name][index] for name in ("flux_w1", "flux_w2", "flux_w3")], dtype=np.float32)
        return (
            torch.from_numpy(read_dataset(handle, "spectra", slice(index, index + 1))[0].astype(np.float32)),
            torch.from_numpy(read_dataset(handle, "spectra_ivar", slice(index, index + 1))[0].astype(np.float32)),
            self._wavelength,
            torch.tensor(read_dataset(handle, "redshift", slice(index, index + 1))[0], dtype=torch.float32),
            torch.from_numpy(wise),
            torch.from_numpy(read_dataset(handle, LEGACY_SURVEY_IMAGE_DATASET, slice(index, index + 1))[0].astype(np.float32)),
            torch.tensor(read_dataset(handle, self.target_name, slice(index, index + 1))[0], dtype=torch.float32),
            torch.tensor(int(read_dataset(handle, "desi_targetid", slice(index, index + 1))[0]), dtype=torch.int64),
        )


def build_dataloaders(
    *,
    staged_dir: Path = STAGED_DIR,
    target_name: str = "log_ml_flux_1",
    batch_size: int = 256,
    eval_batch_size: int | None = None,
    num_workers: int = 0,
    seed: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    eval_batch_size = eval_batch_size or batch_size
    loaders: list[DataLoader] = []
    train_generator = None
    if seed is not None:
        train_generator = torch.Generator()
        train_generator.manual_seed(int(seed))
    for split in ("train", "val", "test"):
        dataset = AIONHDF5Dataset(Path(staged_dir) / f"desi_{split}.hdf5", target_name)
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
    with h5py.File(hdf5_path, "r") as handle:
        return read_dataset(handle, target_name).astype(np.float32)


def move_batch_to_device(batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(item.to(device, non_blocking=True) for item in batch)


class AIONTokenEncoder(nn.Module):
    """Native AION encoder wrapper returning full tokens and direct modality ids."""

    def __init__(self, model_name: str = "polymathic-ai/aion-base", freeze: bool = True) -> None:
        super().__init__()
        try:
            from aion import AION
            from aion.codecs import CodecManager
            from aion.modalities import DESISpectrum, LegacySurveyFluxW1, LegacySurveyFluxW2, LegacySurveyFluxW3, LegacySurveyImage, Z
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
            modalities.append(self.spectrum_cls(flux=flux, ivar=ivar, mask=ivar <= 0.0, wavelength=wavelength))
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

    def _group_ids_from_modality_mask(self, modality_mask: torch.Tensor) -> torch.Tensor:
        group_ids = torch.full_like(modality_mask, -1, dtype=torch.long)
        modality_info = getattr(self.backbone, "modality_info", {})
        for group_name, token_keys in TOKEN_KEYS_BY_MODALITY.items():
            for token_key in token_keys:
                if token_key in modality_info:
                    group_ids[modality_mask.eq(int(modality_info[token_key]["id"]))] = MODALITY_TO_ID[group_name]
        if bool(group_ids.lt(0).any()):
            raise RuntimeError("AION returned token ids that are not mapped to spectra, z, WISE, or image.")
        return group_ids

    def encode_tokens(
        self,
        batch: tuple[torch.Tensor, ...],
        combo: tuple[str, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flux, ivar, wavelength, redshift, wise, image, _target, _targetid = batch
        codec = self._codec(flux.device)
        modalities = self._modalities(flux, ivar, wavelength, redshift, wise, image, combo)
        token_dict = codec.encode(*modalities)
        num_tokens = self._num_tokens(token_dict)

        context_manager = torch.no_grad() if self.freeze else torch.enable_grad()
        with context_manager:
            encoder_tokens, encoder_emb, encoder_mask, encoder_mod_mask = self.backbone.embed_inputs(
                token_dict,
                mask=None,
                num_encoder_tokens=num_tokens,
            )
            tokens = self.backbone._encode(encoder_tokens, encoder_emb, encoder_mask)
        return tokens, self._group_ids_from_modality_mask(encoder_mod_mask)
