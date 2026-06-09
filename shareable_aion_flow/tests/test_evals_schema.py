"""Schema guards: the shipped results CSV matches the canonical eval schema and is
consumable by the table builder."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from shareable_aion_flow.evals import METRIC_COLUMNS, build_results_table

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


def test_build_results_table_orders_rows_by_r2() -> None:
    table = build_results_table(pd.read_csv(RESULTS_CSV))
    r2 = table["r2"].to_numpy()
    assert (r2[:-1] <= r2[1:]).all()
