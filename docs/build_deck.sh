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
# DECK_SAMPLE picks the row-set declaration the WHOLE deck is rendered on, and
# every figure prints it. `sbatch/eval_multi.sbatch` runs `--sample both`, so
# multi_test_metrics.csv carries two rows per (head, input_group) and the deck
# has to choose; the consumers refuse to choose for you.
#   common (default)  every source scored carries all four inputs, so the row set
#                     is identical across combinations. This is the ONLY setting
#                     in which the information-gain bars may be read against each
#                     other, which is what the upset figure does on every slide.
#   native            each combination on the rows that support it: larger row
#                     sets, but differences between combinations then mix an
#                     input effect with a row-set effect.
#
# Data-side figures come from the raw local products, so they rebuild without
# the cluster. Override the paths with the AIONFLOW_* env vars below.
#
# Figure provenance (all regenerated unless marked):
#   fig_corner.png, fig_examples_spectra.png,   docs/make_data_figures.py
#     fig_examples_images.png, data_counts.csv
#   fig_loss_total.png, fig_loss_by_target.png  docs/make_loss_curves.py
#   fig_results_v3.png                          docs/make_results_figure.py
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

METRICS="$EVAL/multi_test_metrics.csv"
DECK_SAMPLE=${DECK_SAMPLE:-common}

# PREFLIGHT, before a single figure is regenerated. Resolving DECK_SAMPLE is the
# one step that can refuse (a table written by `--sample native` cannot render a
# `common` deck), and the four consumers that do it run LAST. Without this check
# the data and loss figures rebuild, the first metrics consumer exits, and the
# deck is left half new and half old with no slides.pdf to show which is which.
# It calls docs/deck_sample.select_sample rather than restating the rule.
if [ ! -f "$METRICS" ]; then
  echo "[build_deck] no $METRICS - run sbatch/eval_multi.sbatch for this checkpoint first" >&2
  exit 1
fi
$PY - "$METRICS" "$DECK_SAMPLE" <<'PREFLIGHT'
import sys
sys.path.insert(0, "docs")
import pandas as pd
from deck_sample import sample_caption, sample_meaning, select_sample
path, sample = sys.argv[1], sys.argv[2]
rows, chosen = select_sample(pd.read_csv(path), sample, source=path)
print(f"[build_deck] rendering the whole deck on {sample_caption(chosen, rows)}")
print(f"[build_deck]   {sample_meaning(chosen)}")
PREFLIGHT

# Raw local products for the data slides.
SD=${AIONFLOW_SD:-$HOME/astroai/stanford_deadline/data}
# The dr2v2 pair, in one place. Override for a dr2 reproduction.
SIDECAR=${AIONFLOW_SIDECAR:-$SD/dr2/targets_sidecar_dr2_v2.csv}
SPLIT=${AIONFLOW_SPLIT:-$SD/dr2/clean_split_dr2_v2.csv}
ERO=${AIONFLOW_ERO:-$HOME/astroai/stanford_deadline/aion_project/shareable_aion_flow/data/raw/erosita_desi}
POOL=${AIONFLOW_POOL:-$HOME/astroai/stanford_deadline/aion_project/shareable_aion_flow/data/raw/legacysurvey/fits_pool}

# Data slides + docs/figures/data_counts.csv. Repointed to DR2 on 2026-08-06:
# this block fed make_data_figures.py the DR1 sidecar and the DR1 clean split, so
# every count on the data slide was the eRASS1 sample (25,200) while the deck's
# results came from the eRASS:3 one (25,582).
#
# TWO RESIDUALS, both inside make_data_figures.py and both still DR1:
#  1. It recomputes log_ml_flux_1 and log_lx from the DR1 merged HDF5's
#     ml_flux_1, OVERWRITING the DR2 columns the sidecar supplies. Those two
#     rows of data_counts.csv stay eRASS1-derived.
#  2. Its `props.merge(hd)` is an inner join against that same DR1 HDF5, which
#     holds 24,181 of the 25,582 DR2 clean-split targetids (measured
#     2026-08-06), so the figures are drawn on a subset until the re-stage.
# Neither is fixable from this file. See plan step 17.
if [ -f "$ERO/erosita_spectra_merged_32k.hdf5" ]; then
  $PY docs/make_data_figures.py \
    --all-properties "$ERO/erosita_desi_dr1_matches_all_properties.csv" \
    --spectra-hdf5 "$ERO/erosita_spectra_merged_32k.hdf5" \
    --sidecar "$SIDECAR" \
    --clean-split-csv "$SPLIT" \
    --fits-pool "$POOL"
else
  echo "[build_deck] raw data not found under $ERO - keeping existing data figures"
fi

if [ -n "$HIST" ] && [ -f "$HIST" ]; then
  $PY docs/make_loss_curves.py --epoch-history "$HIST"
else
  echo "[build_deck] no epoch history given - keeping existing loss figures"
fi

# HR32 is OPTIONAL: sbatch/eval_multi.sbatch leaves HR_REF_CSV empty by default,
# and scripts/hr_from_joint.py refuses any joint that is not exactly 2-D over
# (log_flux_p2, log_flux_p3). So this file is often absent, and passing --hr-csv
# for a file that is not there used to make the panel vanish silently. Pass it
# only when it exists; the consumers state the absence and its reason (read from
# the run's own joint_dims) either way.
HR_CSV="$EVAL/hr_implied_target.csv"
HR_SUMMARY="$EVAL/hr_joint_summary.csv"
HR_ARGS=()
if [ -f "$HR_CSV" ]; then
  HR_ARGS+=(--hr-csv "$HR_CSV")
else
  echo "[build_deck] no $HR_CSV: the HR panel will be replaced by a stated reason" >&2
fi
HR_SUM_ARGS=()
if [ -f "$HR_SUMMARY" ]; then
  HR_SUM_ARGS+=(--hr-summary-csv "$HR_SUMMARY")
else
  echo "[build_deck] no $HR_SUMMARY: no joint-vs-independent hardness slide" >&2
fi

$PY docs/make_results_figure.py --metrics "$METRICS" --sample "$DECK_SAMPLE" \
  "${HR_ARGS[@]}" ${LINES:+--baseline "$LINES"}
$PY docs/make_modality_upset.py --metrics "$METRICS" --sample "$DECK_SAMPLE" \
  ${LINES:+--baseline "$LINES"}

# Per-run figures: performance across redshift, and the Lx-sSFR posterior
# correlation distribution. This block used to be labelled "two-stage record"
# and described refit_epoch*.json; two-phase training is gone and so are those
# files, but neither figure below was ever part of it.
RUNDIR=${AIONFLOW_RUNDIR:-results/dr2_37257713}
if [ -f "$RUNDIR/history.jsonl" ]; then
  if [ -f "$RUNDIR/poststruct_allmod.npz" ]; then
    $PY docs/make_redshift_performance.py --npz "$RUNDIR/poststruct_allmod.npz" \
      --sidecar "$SIDECAR" --out docs/figures/fig_perf_vs_z.png
  fi
  if [ -f "$RUNDIR/poststruct_noWISE.npz" ]; then
    $PY scripts/posterior_corr_hist.py --npz "$RUNDIR/poststruct_noWISE.npz" \
      --pair log_lx log_ssfr --group GALAXY --out docs/figures/fig_hist_lx_ssfr.png
  fi
else
  echo "skip per-run figures: no $RUNDIR/history.jsonl" >&2
fi

$PY docs/make_slides.py --mt-metrics "$METRICS" --sample "$DECK_SAMPLE" \
  --counts-csv docs/figures/data_counts.csv "${HR_ARGS[@]}" "${HR_SUM_ARGS[@]}"
$PY docs/make_html_deck.py --mt-metrics "$METRICS" --sample "$DECK_SAMPLE" \
  "${HR_ARGS[@]}" "${HR_SUM_ARGS[@]}" --output docs/results.html
