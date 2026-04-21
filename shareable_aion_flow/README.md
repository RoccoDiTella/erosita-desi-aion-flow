# Shareable AION Attention-Flow Package

This directory contains the clean implementation of the paper-facing AION
attention-flow model for predicting `log_ml_flux_1` from eROSITA/DESI matched
sources.

The implementation intentionally exports one architecture: the clean q4/l2
paper model. It removes legacy padding/register-token code and expresses token
modality conditioning as a direct affine calibration, but it is functionally the
same architecture used for the paper result.

## Data Layout

Canonical raw data lives under:

```text
data/raw/erosita_desi/
  erosita_spectra_merged_32k.hdf5
  erosita_desi_matches_Xray_properties.csv
  erosita_desi_dr1_matches_all_properties.csv

data/raw/legacysurvey/
  fits_pool/
  survey-bricks-dr10.fits.gz

data/manifests/
  aion_tvsplit_manifest.csv
  aion_targetids_ra_dec_split_dr10_south.csv
  aion_targetids_ra_dec_split_legacy_decoded_dr10_south_healpix.csv
```

The old `aion_project/data/...` paths are kept as symlinks for compatibility.

The FITS pool is one file per `targetid`. A root-level `fits_pool.zip` archive
also exists in this workspace, but the model code uses the directory of native
FITS files.

## Model

For each batch and modality combination, the code constructs native AION input
objects:

- `DESISpectrum(flux, ivar, mask=ivar <= 0, wavelength)`
- `Z(value=redshift)`
- `LegacySurveyFluxW1/W2/W3`
- `LegacySurveyImage(flux=[DES-G, DES-R, DES-I, DES-Z])`

Missing modalities are omitted from the AION input list. No fake masking is
used.

AION returns full unpooled token sequences:

```text
tokens:    [B, T, 768]
group_ids: [B, T] with 0=spectra, 1=redshift, 2=WISE, 3=images
```

The attention-flow head is:

```text
AION tokens
-> modality affine calibration
-> four learned global queries
-> 16-way presence embedding added to the queries
-> two attention-pooling blocks with 8 heads
-> flatten 4 x 768 = 3072
-> MLP adapter 3072 -> 512 -> 512 -> 256
-> conditional Zuko NSF
```

The default target is `log_ml_flux_1`. `log_lx` is supported with `--target log_lx`.

## Commands

Build staged image-backed HDF5 files:

```bash
python3 -m aion_project.shareable_aion_flow.main prepare-data --overwrite
```

Run a small staging smoke:

```bash
python3 -m aion_project.shareable_aion_flow.main prepare-data \
  --output-dir /tmp/aion_shareable_smoke \
  --limit 12 \
  --overwrite
```

Train:

```bash
python3 -m aion_project.shareable_aion_flow.main train \
  --staged-dir aion_project/shareable_aion_flow/data/staged \
  --target log_ml_flux_1 \
  --epochs 50 \
  --batch-size 448 \
  --eval-after-train
```

Evaluate all 15 modality combinations:

```bash
python3 -m aion_project.shareable_aion_flow.main eval \
  --checkpoint aion_project/shareable_aion_flow/outputs/<run-id>/best.pt \
  --staged-dir aion_project/shareable_aion_flow/data/staged \
  --target log_ml_flux_1 \
  --output-dir aion_project/shareable_aion_flow/outputs/<run-id>
```

The readable table reports `R^2`, information gain,
`exp(mean information gain)`, and RMSE. NLL is kept out of the table.

