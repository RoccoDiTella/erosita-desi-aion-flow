from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from normalizing_flow import KDEPrior, TargetStandardizer  # noqa: E402


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

