"""Unit tests for the target standardizer and KDE prior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shareable_aion_flow.normalizing_flow import KDEPrior, TargetStandardizer


def test_target_standardizer_rejects_empty_or_constant_values() -> None:
    with pytest.raises(ValueError):
        TargetStandardizer.fit(np.array([np.nan, np.inf]))
    with pytest.raises(ValueError):
        TargetStandardizer.fit(np.array([1.0, 1.0, 1.0]))


def test_kde_prior_round_trips_metadata(tmp_path: Path) -> None:
    prior = KDEPrior(np.array([-1.0, 0.0, 1.0]), metadata={"target": "log_ml_flux_1"})
    path = tmp_path / "prior.npz"
    prior.save(path)
    loaded = KDEPrior.load(path)
    assert loaded.metadata["target"] == "log_ml_flux_1"
    assert np.isfinite(loaded.log_prob_numpy(np.array([0.0]))).all()


def test_kde_prior_round_trips_float_bandwidth(tmp_path: Path) -> None:
    values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    prior = KDEPrior(values, bw_method=0.5)
    path = tmp_path / "prior.npz"
    prior.save(path)
    loaded = KDEPrior.load(path)
    assert loaded.bw_method == 0.5
    np.testing.assert_allclose(
        loaded.log_prob_numpy(np.array([0.0, 0.3])),
        prior.log_prob_numpy(np.array([0.0, 0.3])),
    )
