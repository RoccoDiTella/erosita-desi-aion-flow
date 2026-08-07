#!/bin/bash
#SBATCH -J aionflow-sweep
#SBATCH --account=finkbeiner_lab
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -t 0-03:00
#SBATCH -o logs/sweep_%j.out
#SBATCH -e logs/sweep_%j.err
#SBATCH --open-mode=append
#
# ONE allocation, many measurements. Sizes batch, memory, LR and early stopping
# for Run A on the hardware the real run uses.
#
#   scripts/submit.sh sbatch/hparam_sweep.sbatch
#
# WHY ONE JOB RATHER THAN A JOB PER SETTING. Every separate job re-pays env
# activation, the AION weight load and the HDF5/dataloader spin-up before it
# computes anything. Across ~7 settings that is most of the wall-clock, and the
# GPU sits idle for it while still being charged. A single allocation that loops
# keeps the card busy from start to finish. Fairshare bills GPU wall-time, so an
# idle fraction is a direct cost to the lab.
#
# WHY -p gpu AND NOT gpu_test. gpu_test is an A100 MIG 3g.20gb slice. The real
# run goes to a full A100-80GB, and the smoke used 3.2 GB at batch 64, so the
# interesting batch sizes do not fit on the slice. Measuring there would mean
# extrapolating 4x beyond the largest point measured -- which is how the
# unjustified 80G memory request got into this repo in the first place.
#
# MEMORY. 64G is deliberately generous and is the one number here NOT yet
# measured at scale: the smoke used MaxRSS 2.7 GB at batch 64 with 0 workers,
# and this job runs batch up to 2048 with 8 workers, each of which holds
# prefetched batches. Host RAM per GPU on these nodes is far above this, so 64G
# is not hoarding a scarce resource the way an inflated GPU count would be.
# `sacct -j <id> --format=MaxRSS` at the end of this job is what sizes the real
# run; do not copy 64G forward without reading it.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/fasrc_env.sh
# _dataset.sh refuses to guess, by design. This sweep sizes RUN A, so it names
# dr2v2 rather than inheriting whatever happens to be in the caller's env --
# but an explicit DATASET= still wins, so the same script can size a dr2
# reproduction without editing.
export DATASET="${DATASET:-dr2v2}"
source sbatch/_dataset.sh

OUT="${AIONFLOW_ROOT}/outputs/sweep-${SLURM_JOB_ID}"
mkdir -p "$OUT"
COMMON=(
  --staged-dir "$STAGED_DIR"
  --clean-split-csv "$CLEAN_SPLIT_CSV"
  --extra-targets-csv "$EXTRA_TARGETS_CSV"
  --drop-heads logmstar log_mbh_pan25 log_mbh_vo09 log_flux_p3
  --fixed-combo spectra+z --joint-marginal log_sfr --select-metric joint_complete
  --no-inject --bucketed --accumulate-buckets --grad-checkpoint
  --num-workers 8 --output-dir "$OUT"
  # Each setting becomes its own wandb run, named by --run-id, so the sweep is
  # readable without the SSH socket -- which has died three times in two hours,
  # once mid-command. A long run that can only be monitored over a socket that
  # keeps dropping is a long run nobody is watching.
  --wandb
)

echo "############ PHASE 1: batch size ############"
# 1 epoch each. What we want out of this is samples/s and vram/peak_gb, both of
# which train-multi already logs per epoch, plus whether it OOMs at all.
for B in 256 512 1024 2048; do
  echo "=== batch $B ==="
  if ! python -m shareable_aion_flow.main train-multi "${COMMON[@]}" \
        --epochs 1 --batch-size "$B" --lr 3e-4 \
        --run-id "bs${B}" 2>&1 | grep -E "^epoch=|Error|OutOfMemory|CUDA"; then
    echo "batch $B FAILED (likely OOM) -- stopping the batch phase here"
    break
  fi
done

echo "############ PHASE 2: learning rate ############"
# Held at batch 512 rather than at whatever phase 1 liked, so the LR comparison
# is not confounded by a batch change. Read phase 1 for throughput; read this
# for where train and val diverge.
#
# 6 epochs is enough to SEE the divergence, not to converge. Past runs reported
# overfitting as the binding constraint, and the GALAXY arm carrying Run A's
# claim is only ~1,400 sources, so the epoch at which gap/nll_joint starts
# climbing is the number this phase exists to produce.
for LR in 1e-4 3e-4 1e-3; do
  echo "=== lr $LR ==="
  python -m shareable_aion_flow.main train-multi "${COMMON[@]}" \
    --epochs 6 --batch-size 512 --lr "$LR" \
    --early-stop-patience 99 \
    --run-id "lr${LR}" 2>&1 | grep -E "^epoch=|Error|CUDA" || true
done

echo "############ SUMMARY ############"
python - "$OUT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
print(f"{'run':>10} {'epoch':>5} {'samp/s':>8} {'VRAM GB':>8} {'sel':>8} {'joint':>8} {'gap_joint':>10}")
for d in sorted(root.iterdir()):
    h = d / "history.jsonl"
    if not h.is_file():
        continue
    for line in h.read_text().splitlines():
        r = json.loads(line)
        print(f"{d.name:>10} {r.get('epoch',0):>5} "
              f"{r.get('throughput/samples_per_second',0):>8.1f} "
              f"{r.get('vram/peak_gb',0):>8.2f} "
              f"{r.get('val/selection',float('nan')):>8.4f} "
              f"{r.get('val/nll_joint',float('nan')):>8.4f} "
              f"{r.get('gap/nll_joint',float('nan')):>10.4f}")
print()
print("PICK: largest batch whose samples/s still improves and whose VRAM leaves")
print("headroom; the LR whose val/selection minimum is lowest; early-stop")
print("patience from the epoch where gap/nll_joint starts climbing.")
print("Then read `sacct -j $SLURM_JOB_ID --format=MaxRSS` and size --mem from it.")
PY
echo "sweep outputs in $OUT"
