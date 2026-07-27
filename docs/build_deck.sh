#!/bin/bash
# Rebuild both decks from one V3 eval directory.
#   bash docs/build_deck.sh <eval-dir> <vpai-flux-metrics.csv>
set -e
EVAL=${1:?eval dir with multi_test_metrics.csv}
VPAI=${2:?V_PAI single-target flux metrics csv}
cd "$(dirname "$0")/.."
python docs/make_slides.py --mt-metrics "$EVAL/multi_test_metrics.csv" \
  --compare-metrics "$VPAI" --hr-csv "$EVAL/hr_implied_target.csv"
python docs/make_html_deck.py --mt-metrics "$EVAL/multi_test_metrics.csv" \
  --hr-csv "$EVAL/hr_implied_target.csv" --compare-metrics "$VPAI" --output docs/results.html
