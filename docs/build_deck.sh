#!/bin/bash
# Rebuild the deck end-to-end. Every figure it uses is REGENERATED here -- there
# are no committed one-off PNGs left in the narrative except the architecture
# poster, the Buchner screenshot, and the emission-line coverage panel.
#
#   bash docs/build_deck.sh <eval-dir> <epoch-history.json> [<lines-baseline.csv>]
#
#   <eval-dir>          directory holding multi_test_metrics.csv (+ hr_implied_target.csv)
#   <epoch-history>     per-epoch JSON with probe/nll_* and val/nll_* keys
#   <lines-baseline>    results/baseline_lines_only_clean.csv (default if present)
#
# Data-side figures come from the raw local products, so they rebuild without
# the cluster. Override the paths with the AIONFLOW_* env vars below.
#
# Figure provenance (all regenerated unless marked):
#   fig_corner.png, fig_examples_spectra.png,   docs/make_data_figures.py
#     fig_examples_images.png, data_counts.csv
#   fig_loss_total.png, fig_loss_by_target.png  docs/make_loss_curves.py
#   fig_results_v3.png                          docs/make_results_figure.py
#   fig_two_stage_heads.png,                    docs/make_two_stage_figures.py
#     fig_two_stage_select.png, two_stage_heads.csv
#   fig_modality_upset.png                      docs/make_modality_upset.py
#   fig_architecture_poster.png,                NOT GENERATED - kept in git
#     johannes_buchner_comment.png (gitignored),
#     fig_line_coverage.png
#
set -e
PY=${PYTHON:-python3}   # `python` is often only a shell alias
EVAL=${1:?eval dir with multi_test_metrics.csv}
HIST=${2:-}
cd "$(dirname "$0")/.."
LINES=${3:-results/baseline_lines_only_clean.csv}

# Raw local products for the data slides.
SD=${AIONFLOW_SD:-$HOME/astroai/stanford_deadline/data}
ERO=${AIONFLOW_ERO:-$HOME/astroai/stanford_deadline/aion_project/shareable_aion_flow/data/raw/erosita_desi}
POOL=${AIONFLOW_POOL:-$HOME/astroai/stanford_deadline/aion_project/shareable_aion_flow/data/raw/legacysurvey/fits_pool}

if [ -f "$ERO/erosita_spectra_merged_32k.hdf5" ]; then
  $PY docs/make_data_figures.py \
    --all-properties "$ERO/erosita_desi_dr1_matches_all_properties.csv" \
    --spectra-hdf5 "$ERO/erosita_spectra_merged_32k.hdf5" \
    --sidecar "$SD/targets_sidecar.csv" \
    --clean-split-csv "$SD/clean_split.csv" \
    --fits-pool "$POOL"
else
  echo "[build_deck] raw data not found under $ERO - keeping existing data figures"
fi

if [ -n "$HIST" ] && [ -f "$HIST" ]; then
  $PY docs/make_loss_curves.py --epoch-history "$HIST"
else
  echo "[build_deck] no epoch history given - keeping existing loss figures"
fi

$PY docs/make_results_figure.py --metrics "$EVAL/multi_test_metrics.csv" \
  --hr-csv "$EVAL/hr_implied_target.csv" ${LINES:+--baseline "$LINES"}
$PY docs/make_modality_upset.py --metrics "$EVAL/multi_test_metrics.csv" \
  ${LINES:+--baseline "$LINES"}

# Two-stage record: per-head loss, and how the body was chosen. Needs a run dir
# holding history.jsonl plus refit_epoch*.json (one per swept snapshot).
RUNDIR=${AIONFLOW_RUNDIR:-results/dr2_37257713}
if [ -f "$RUNDIR/history.jsonl" ]; then
  $PY docs/make_two_stage_figures.py --run-dir "$RUNDIR"
else
  echo "skip two-stage figures: no $RUNDIR/history.jsonl" >&2
fi

$PY docs/make_slides.py --mt-metrics "$EVAL/multi_test_metrics.csv" \
  --counts-csv docs/figures/data_counts.csv
$PY docs/make_html_deck.py --mt-metrics "$EVAL/multi_test_metrics.csv" \
  --hr-csv "$EVAL/hr_implied_target.csv" --output docs/results.html
