#!/bin/bash
# Convenience wrapper: inject the SLURM account from .fasrc.env, ensure logs/
# exists, then sbatch. Extra args pass through (options before the script), e.g.
#   scripts/submit.sh sbatch/train_smoke.sbatch
#   scripts/submit.sh -p siag_gpu sbatch/train.sbatch     # override partition
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/.fasrc.env"
mkdir -p logs
exec sbatch -A "${FASRC_ACCOUNT:?set FASRC_ACCOUNT in .fasrc.env}" "$@"
