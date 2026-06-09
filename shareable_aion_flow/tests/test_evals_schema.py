"""Schema guards: the shipped results CSV matches the canonical eval schema and is
consumable by the table builder.

This locks the contract that broke once before -- the on-disk results and the code
that writes/reads them must agree, so ``make-table`` never fails on the repo's own data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals import METRIC_COLUMNS, build_results_table  # noqa: E402

RESULTS_CSV = Path(__file__).resolve().parents[2] / "results" / "test_flow_metrics.csv"


def test_shipped_results_match_canonical_schema() -> None:
    columns = list(pd.read_csv(RESULTS_CSV, nrows=0).columns)
    assert columns == METRIC_COLUMNS


def test_build_results_table_consumes_shipped_results() -> None:
    table = build_results_table(pd.read_csv(RESULTS_CSV))
    assert {"r2", "info_gain_nats", "exp_info_gain", "rmse"}.issubset(table.columns)
    all_inputs = table.loc[table["input_group"] == "spectra+z+wise+image"].iloc[0]
    assert (all_inputs["spectra"], all_inputs["z"], all_inputs["wise"], all_inputs["image"]) == (
        "S",
        "Z",
        "W",
        "I",
    )
