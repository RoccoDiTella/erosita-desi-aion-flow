# Shareable AION Attention-Flow

Clean standalone package for the paper-facing AION attention-flow model on the eROSITA/DESI matched sample.

This repo exports one architecture only: the clean q4/l2 paper model.

## What Is Included

- `shareable_aion_flow/`: the package code
- `shareable_aion_flow/data/manifests/`: small split and image-coverage manifests
- `shareable_aion_flow/tests/`: lightweight tests for the clean attention/flow components

## What Is Not Included

Large raw data is intentionally not tracked in GitHub:

- `shareable_aion_flow/data/raw/erosita_desi/`
- `shareable_aion_flow/data/raw/legacysurvey/fits_pool/`
- staged HDF5 files under `shareable_aion_flow/data/staged/`
- caches and outputs

The code expects the canonical raw layout to look like:

```text
shareable_aion_flow/data/raw/erosita_desi/
  erosita_spectra_merged_32k.hdf5
  erosita_desi_matches_Xray_properties.csv
  erosita_desi_dr1_matches_all_properties.csv

shareable_aion_flow/data/raw/legacysurvey/
  fits_pool/
  survey-bricks-dr10.fits.gz
```

The included manifests are:

```text
shareable_aion_flow/data/manifests/
  aion_tvsplit_manifest.csv
  aion_targetids_ra_dec_split_dr10_south.csv
  aion_targetids_ra_dec_split_legacy_decoded_dr10_south_healpix.csv
```

## Model

For each batch and modality combination, the code constructs native AION inputs:

- `DESISpectrum(flux, ivar, mask=ivar <= 0, wavelength)`
- `Z(value=redshift)`
- `LegacySurveyFluxW1/W2/W3`
- `LegacySurveyImage(flux=[DES-G, DES-R, DES-I, DES-Z])`

Missing modalities are omitted from the AION input list. No fake masking is used.

AION returns unpooled token sequences:

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

Default target: `log_ml_flux_1`

Also supported: `log_lx`

## Installation

```bash
pip install -e .
pip install -e .[dev]
pip install -e .[flow]
pip install -e .[aion]
```

For full training/evaluation you need both `zuko` and `polymathic-aion` available.

## Commands

Build staged image-backed HDF5 files:

```bash
python3 -m shareable_aion_flow.main prepare-data --overwrite
```

Run a small staging smoke:

```bash
python3 -m shareable_aion_flow.main prepare-data \
  --output-dir /tmp/aion_shareable_smoke \
  --limit 12 \
  --overwrite
```

Train:

```bash
python3 -m shareable_aion_flow.main train \
  --staged-dir shareable_aion_flow/data/staged \
  --target log_ml_flux_1 \
  --epochs 50 \
  --batch-size 448 \
  --eval-after-train
```

Evaluate all 15 modality combinations:

```bash
python3 -m shareable_aion_flow.main eval \
  --checkpoint shareable_aion_flow/outputs/<run-id>/best.pt \
  --staged-dir shareable_aion_flow/data/staged \
  --output-dir shareable_aion_flow/outputs/<run-id>
```

The readable table reports `R^2`, information gain, `exp(mean information gain)`, and RMSE. NLL is intentionally not shown in the table.
