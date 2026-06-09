"""Unit tests for target derivation, modality-combo helpers, and regression metrics."""

from __future__ import annotations

import numpy as np
from astropy.cosmology import Planck18

from shareable_aion_flow.attention_pooling_head import (
    MODALITIES,
    all_nonempty_modality_combos,
    combo_name,
)
from shareable_aion_flow.data_to_aion_embeddings import compute_log_flux, compute_log_luminosity
from shareable_aion_flow.evals import regression_metrics


def test_compute_log_flux_handles_invalid_values() -> None:
    flux = np.array([100.0, 1.0, 0.0, -3.0, np.nan, np.inf])
    out = compute_log_flux(flux)
    np.testing.assert_allclose(out[:2], [2.0, 0.0])
    assert np.isnan(out[2:]).all()


def test_compute_log_luminosity_matches_planck18() -> None:
    redshift = np.array([0.5, 1.0])
    flux = np.array([1e-13, 2e-13])
    d_l_cm = Planck18.luminosity_distance(redshift).to("cm").value
    expected = np.log10(4.0 * np.pi * d_l_cm**2 * flux)
    np.testing.assert_allclose(compute_log_luminosity(redshift, flux), expected)


def test_compute_log_luminosity_masks_invalid_inputs() -> None:
    redshift = np.array([0.5, np.nan, 0.5, 0.5])
    flux = np.array([1e-13, 1e-13, 0.0, np.nan])
    out = compute_log_luminosity(redshift, flux)
    assert np.isfinite(out[0])
    assert np.isnan(out[1:]).all()


def test_all_nonempty_modality_combos_enumerates_fifteen() -> None:
    combos = all_nonempty_modality_combos()
    assert len(combos) == 15
    assert len(set(combos)) == 15
    assert all(1 <= len(combo) <= len(MODALITIES) for combo in combos)


def test_combo_name_uses_canonical_modality_order() -> None:
    assert combo_name(("image", "spectra")) == "spectra+image"
    assert combo_name({"wise", "z", "spectra", "image"}) == "spectra+z+wise+image"


def test_regression_metrics_perfect_prediction() -> None:
    y = np.linspace(-1.0, 1.0, 50)
    metrics = regression_metrics(y, y)
    assert metrics["r2"] == 1.0
    assert metrics["r2_trimmed_0.05"] == 1.0
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0


def test_regression_metrics_trimming_drops_worst_residuals() -> None:
    y_true = np.linspace(0.0, 1.0, 20)
    y_pred = y_true.copy()
    y_pred[7] += 10.0  # One catastrophic outlier; 5% of n=20 is exactly one point.
    metrics = regression_metrics(y_true, y_pred)
    assert metrics["r2"] < 0.0
    assert metrics["r2_trimmed_0.05"] == 1.0
