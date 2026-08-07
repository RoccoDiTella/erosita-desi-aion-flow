#!/bin/bash
# Resolve the target-table trio (split CSV, sidecar, staged inputs) and refuse to
# continue if the sidecar is not the one the caller thinks it is. SOURCE this,
# never execute it; it exits the caller on failure, which is the point.
#
#   DATASET=dr2 scripts/submit.sh --export=ALL sbatch/train_multi.sbatch
#
# WHY THERE IS NO DEFAULT. Every launcher used to default to the DR1 trio while
# posterior_structure.sbatch defaulted to the DR2 one, so a checkpoint trained on
# one relabel could be analysed against the other and nothing anywhere said so.
# Both directions are silent: DR1 labels are a strict subset of the DR2 columns,
# so a mixed run does not crash, it just reports the wrong depth. Naming the
# dataset at submit time puts it in the job's environment, in the SLURM record,
# and in this file's failure message.
set -eo pipefail

: "${DATASET:?set DATASET=dr2 explicitly -- there is no default (see sbatch/_dataset.sh)}"

case "$DATASET" in
  dr2)
    # SUPERSEDED 2026-08-06 by dr2v2. Kept only so an already-published number
    # can be reproduced against the artifacts it was actually measured on:
    # 25,582 rows, targetid-grouped (leaky), no reliability cut, staged WITH a
    # DR1 log_ml_flux_1 column. Do not start new work on this branch.
    CLEAN_SPLIT_CSV="${CLEAN_SPLIT_CSV:-$AIONFLOW_DATA/clean_split_dr2.csv}"
    EXTRA_TARGETS_CSV="${EXTRA_TARGETS_CSV:-$AIONFLOW_DATA/targets_sidecar_dr2.csv}"
    STAGED_DIR="${STAGED_DIR:-$AIONFLOW_STAGED}"
    ;;
  dr2v2)
    # The rebuilt chain: make_dr2_targets -> make_targets_sidecar ->
    # build_manifest -> make_split --require-spectrum -> build_manifest --split
    # -> prepare-data. Four differences from dr2, all of which matter:
    #   * detuid-grouped split, so the 756 detections carrying more than one
    #     DESI fibre stop putting the same X-ray photons in train and test
    #   * NWAY counterpart-reliability cut applied
    #   * DR2 flux and Lx read from the sidecar; the staged HDF5 no longer
    #     carries a label at all, so it cannot select rows by a DR1 flux
    #   * staged file carries has_spectrum/has_z/has_wise/has_image, so modality
    #     presence is the manifest's verdict rather than a guess from tensors
    # 22,800 split rows (18,275/2,274/2,251) from a 25,454-row target table.
    CLEAN_SPLIT_CSV="${CLEAN_SPLIT_CSV:-$AIONFLOW_DATA/clean_split_dr2_v2.csv}"
    EXTRA_TARGETS_CSV="${EXTRA_TARGETS_CSV:-$AIONFLOW_DATA/targets_sidecar_dr2_v2.csv}"
    STAGED_DIR="${STAGED_DIR:-$AIONFLOW_DATA/staged_v2}"
    ;;
  dr1)
    # DELETED 2026-08-06, not merely defaulted away from. `targets_sidecar.csv`
    # has 39 columns and none of det_like_0, det_like_p1..p4, log_ml_flux_1,
    # log_lx, flux_sig_lo/hi (measured). Every band target now gates on a
    # det_like column, so load_multi_target_matrix raises before the first
    # forward -- a DR1 branch could only ever resolve paths to a crash. Recover
    # it from git history if you need the DR1 numbers reproduced.
    echo "FAIL: DATASET=dr1 is retired. The DR1 sidecar lacks all nine columns" >&2
    echo "      the detection-gated targets need; it cannot load. See git log." >&2
    exit 1
    ;;
  *)
    echo "FAIL: unknown DATASET='$DATASET' (dr2v2 for new work, dr2 to reproduce a published number)" >&2
    exit 1
    ;;
esac

echo "[dataset] DATASET=$DATASET"
echo "[dataset]   split   $CLEAN_SPLIT_CSV"
echo "[dataset]   sidecar $EXTRA_TARGETS_CSV"
echo "[dataset]   staged  $STAGED_DIR"

# Fail before burning GPU-hours. Two checks, and the second is the one that
# matters: the required-column list is DERIVED from the target spec, so a head
# added to MULTI_TARGETS extends this gate automatically, but a file can satisfy
# every column name and still be the wrong release. The sigma distribution
# cannot. eRASS:3 is 2.7x the eRASS1 exposure over this footprint, which shows
# up as a median split-normal lower error of 0.1177 dex against DR1's 0.1826
# (both measured 2026-08-06 on the 23,283 sources the two clean splits share).
# That is a property of the photons, so a renamed file, a half-merged sidecar,
# or a DR1 file copied over the DR2 name is caught here rather than in a plot.
# `python` exists inside the cluster conda env but not on a bare workstation,
# where only `python3` does. The gate has to run in both places or it stops
# being a gate and starts being a thing you skip locally.
PY_BIN="${PY_BIN:-$(command -v python || command -v python3)}"
[ -x "$PY_BIN" ] || { echo "FAIL: no python interpreter on PATH" >&2; exit 1; }

"$PY_BIN" - "$EXTRA_TARGETS_CSV" "$CLEAN_SPLIT_CSV" "$STAGED_DIR" "$DATASET" <<'PY'
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from shareable_aion_flow.multitarget import _ALL_TARGETS, DET_LIKE_MIN

sidecar, split_csv, staged = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])

for path in (sidecar, split_csv):
    if not path.is_file():
        sys.exit(f"FAIL: {path} does not exist")
for name in ("train", "val", "test"):
    if not (staged / f"desi_{name}.hdf5").is_file():
        sys.exit(f"FAIL: {staged}/desi_{name}.hdf5 does not exist")

need = {"targetid"}
for spec in _ALL_TARGETS:
    if not spec["sidecar"]:
        continue                      # read from the staged HDF5, not from here
    need.add(spec["name"])
    need.update(spec["sig"] or ())
    if spec["det"]:
        need.add(spec["det"][0])

frame = pd.read_csv(sidecar)
missing = sorted(need - set(frame.columns))
if missing:
    sys.exit(f"FAIL: {sidecar} is missing {len(missing)} required columns: {missing}")

# The photon-depth check. Bounds are deliberately wide enough to survive a
# re-derivation of the sidecar and narrow enough to exclude DR1 by 5x the gap.
med = float(np.nanmedian(frame["flux_sig_lo"]))
if not (0.10 < med < 0.14):
    sys.exit(
        f"FAIL: median flux_sig_lo = {med:.4f}, outside the eRASS:3 range "
        f"(0.10, 0.14). DR1 measures 0.1826. This is not a DR2 sidecar, or it "
        f"is a partial merge -- check {sidecar}")

split = pd.read_csv(split_csv)
missing_split = sorted({"targetid", "split"} - set(split.columns))
if missing_split:
    sys.exit(f"FAIL: {split_csv} is missing {missing_split}")

# The staged file must be INPUTS ONLY, and it must carry the presence flags.
# Both directions are load-bearing. A staged label is a second, older copy of
# the truth that something will eventually read by accident (a DR1 flux
# selecting a DR2 sample is exactly how that went last time). Absent presence
# flags silently demote modality_presence() to its weaker content heuristic,
# which cannot see zwarn or the WISE inverse variances, so an information-gain
# table would be computed on a row set nobody declared.
import h5py

from shareable_aion_flow.data_to_aion_embeddings import PRESENCE_FLAGS

dataset = sys.argv[4]
with h5py.File(staged / "desi_train.hdf5", "r") as h:
    keys = set(h.keys())

if dataset == "dr2v2":
    banned = sorted(keys & {"log_ml_flux_1", "log_lx", "ml_flux_1", "logmstar", "hr32_u"})
    if banned:
        sys.exit(f"FAIL: {staged} carries staged labels {banned}; the rebuilt "
                 f"staging is inputs only -- re-run prepare-data")
    absent = [f for f in PRESENCE_FLAGS if f not in keys]
    if absent:
        sys.exit(f"FAIL: {staged} is missing presence flags {absent}; rebuild the "
                 f"manifest with scripts/build_manifest.py and re-run prepare-data")
    print(f"[dataset]   staged is inputs-only and carries {len(PRESENCE_FLAGS)} presence flags")

det0 = frame["det_like_0"].to_numpy(float)
print(f"[gate] sidecar OK: {sidecar}")
print(f"[gate]   rows {len(frame):,}  median flux_sig_lo {med:.4f}  "
      f"det_like_0 > {DET_LIKE_MIN:g} for {int((det0 > DET_LIKE_MIN).sum()):,}")
print(f"[gate] split OK: {split_csv}  rows {len(split):,}  "
      f"{split['split'].value_counts().to_dict()}")
PY
