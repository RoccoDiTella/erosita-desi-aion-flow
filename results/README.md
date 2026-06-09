# Results

Reference numbers for the paper's q4/l2 attention-flow model (target `log_ml_flux_1`, held-out test
set, n = 3,054).

| File | Contents |
|---|---|
| `results_table.png` | Paper Table 1 — all 15 modality combinations and the emission-line baseline, with 1σ bootstrap intervals. |
| `performance_by_redshift.png` | Predictive performance vs. redshift for several input combinations. |
| `test_flow_metrics.csv` | Per-combination metrics: `r2`, `r2_trimmed_0.05`, `rmse`, `mae`, mean posterior/prior log-prob (nats), `info_gain_nats`, `exp_info_gain`, and the fraction of sources whose per-object info gain is negative / near-zero / positive. |
| `baseline_lines_only_metrics.csv` | The classical emission-line baseline ([O III] + [Ne V] + Hα + Hβ), same schema. |

`test_flow_metrics.csv` is written in exactly the schema `python -m shareable_aion_flow.main eval`
produces and `make-table` renders; the values here are the paper's. These are point estimates — the
1σ bootstrap intervals are drawn in `results_table.png`. Headline: all-inputs **R² = 0.549**,
**exp(IG) = 1.405**; spectra+WISE+images is nominally highest (R² = 0.554) and ties all-inputs within
1σ; the emission-line baseline reaches **R² = 0.392**.

Re-training from this repo's simplified recipe targets the same neighborhood rather than these exact
values — see "Reproducing the paper" in the top-level README.
