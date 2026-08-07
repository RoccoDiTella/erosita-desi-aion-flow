#!/bin/bash
#SBATCH -J aionflow-eval
#SBATCH --account=finkbeiner_lab
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -t 0-02:00
#SBATCH -o logs/eval_%j.out
#SBATCH -e logs/eval_%j.err
#SBATCH --open-mode=append
# Test-set eval of a train-multi checkpoint: the 15-combo per-head tables and
# the SFR-vs-mass baseline check. Everything the deck needs from a run comes out
# of this one job.
#
#   DATASET=dr2 CKPT=$AIONFLOW_ROOT/outputs/<run>/best.pt \
#     scripts/submit.sh --export=ALL sbatch/eval_multi.sbatch
#
# Eval never injects: the scripts below have no error-mode flag at all and score
# the flow directly. Whether the CHECKPOINT was trained with injection is read
# from its own config, so an old injected run stays scored the way it was
# trained. Nothing here needs --no-inject because nothing here can inject.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/fasrc_env.sh
source sbatch/_dataset.sh

CKPT="${CKPT:?set CKPT to the checkpoint to evaluate}"

# HR32 is OPTIONAL and off by default, for two independent reasons.
# (1) There is no DR2 hardness reference. `targets_extra.csv` carries hr32 and
#     hr32_ok, but they are eRASS1 ML_RATE_P2/P3 -- comparing a DR2-trained
#     posterior against a DR1 measurement is a depth change dressed as a model
#     result. Nothing in data/dr2/ has an hr32 column.
# (2) The implied-HR quadrature is an exactly-2-D shear and the current joint is
#     4-D over (M*, SFR, Lx, P3) with no P2, so hr_from_joint has no P2/P3 pair
#     to marginalise. It applies to a P2xP3 joint only.
# Set HR_REF_CSV explicitly when both hold, i.e. for a P2xP3-joint run scored
# against a hardness reference measured at the same depth as its labels.
HR_REF_CSV="${HR_REF_CSV:-}"

# The row set every number in the table was measured on. NOT left to the script
# default: an unstated sample is the defect this flag exists to fix, and the
# launcher is where it becomes a property of the run rather than of whoever
# typed the command.
#
# `both` is the default because one job then serves both readers:
#   common  the same sources for all 15 combinations, so an information-gain,
#           R2 or RMSE difference between two of them is an input effect. Run B's
#           modality table is only legal here (eval_core.metric_delta refuses
#           anything else), and this is what docs/build_deck.sh renders.
#   native  each combination on the rows that support it: the largest honest row
#           set per combination, and what the headline all-inputs numbers are
#           best quoted on.
# Cost is bounded: the AION encoder pass is shared between the two, only the
# flow evaluations are repeated. Set SAMPLE=common or SAMPLE=native to halve it.
SAMPLE="${SAMPLE:-both}"

COMMON=(--checkpoint "$CKPT" --staged-dir "$STAGED_DIR"
        --clean-split-csv "$CLEAN_SPLIT_CSV"
        --extra-targets-csv "$EXTRA_TARGETS_CSV")

echo "=== per-head tables + SFR-vs-mass baseline (sample=$SAMPLE) ==="
python scripts/eval_multitarget.py "${COMMON[@]}" --sample "$SAMPLE" \
  ${HR_REF_CSV:+--hr-ref-csv "$HR_REF_CSV"}

if [ -n "$HR_REF_CSV" ]; then
  # Two different HR products, both wired into the deck:
  #   hr_joint_summary.csv    written above by eval_multitarget: SAMPLED HR out of
  #                           the joint, against the independent-band baseline
  #                           recomputed on the same rows. Works for any joint
  #                           that CONTAINS both bands.
  #   hr_implied_target.csv   written here: the exact quadrature, 2-D joints only.
  #
  # Only the SECOND is restricted, so "the joint is 4-D" must not fail the job:
  # every table above is already written and correct. hr_from_joint exits 3 for
  # exactly that case and non-zero-non-3 for a real failure, which still stops
  # the job. Without this split, setting HR_REF_CSV on a Run-A-shaped run would
  # turn a complete eval into a FAILED slurm job.
  echo "=== HR32 as an implied target (exact quadrature) ==="
  set +e
  python scripts/hr_from_joint.py "${COMMON[@]}" --hr-ref-csv "$HR_REF_CSV"
  hr_rc=$?
  set -e
  if [ "$hr_rc" -eq 3 ]; then
    echo "=== implied-HR quadrature skipped: not applicable to this checkpoint's joint ==="
    echo "    (reason printed above). hr_joint_summary.csv from the sampled path still"
    echo "    carries the joint-vs-independent comparison; the deck states the absence"
    echo "    of hr_implied_target.csv and reads the reason off this run's own joint_dims."
  elif [ "$hr_rc" -ne 0 ]; then
    echo "hr_from_joint.py failed with exit $hr_rc" >&2
    exit "$hr_rc"
  fi
else
  echo "=== HR32 skipped: no HR_REF_CSV (see the comment in this file) ==="
  echo "    the deck states the absence and its reason; it does not drop the panel"
fi

echo "=== outputs ==="
ls -la "$(dirname "$CKPT")/eval"
