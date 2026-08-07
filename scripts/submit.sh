#!/bin/bash
# Convenience wrapper: inject the SLURM account from .fasrc.env, ensure logs/
# exists, then sbatch. Extra args pass through (options before the script), e.g.
#   DATASET=dr2 RUN_TAG=mt-dr2 scripts/submit.sh --export=ALL sbatch/train_multi.sbatch
#   scripts/submit.sh -p gpu_test -t 00:40:00 sbatch/train_smoke.sbatch   # override partition
#
# ACCOUNT. -A comes from FASRC_ACCOUNT in .fasrc.env and is `finkbeiner_lab`.
# There is no hardcoded account in this file and there must not be one: the
# value lives in exactly one place so a scheduler-verified change is a one-line
# edit. It OVERRIDES the #SBATCH --account header in each sbatch file; the
# headers repeat the same value so a bare `sbatch` is not silently uncharged.
#
# Do NOT pass `-p siag_gpu` here. siag_lab has NOT landed -- verified against
# the scheduler 2026-08-06: `sbatch --test-only -A siag_lab -p siag_gpu` returns
# "Invalid account or account/partition combination", while
# `-A finkbeiner_lab -p gpu` schedules. siag_gpu IS visible in `sinfo`, so
# seeing the partition proves nothing about membership. Only --test-only does.
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/.fasrc.env"
mkdir -p logs
exec sbatch -A "${FASRC_ACCOUNT:?set FASRC_ACCOUNT in .fasrc.env}" "$@"
