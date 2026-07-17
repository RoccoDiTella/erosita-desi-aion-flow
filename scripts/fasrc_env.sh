#!/bin/bash
# Shared FASRC environment setup. Sourced by scripts/fasrc_validate.sh and the
# sbatch jobs. Reads .fasrc.env (repo root) for paths, loads the conda module,
# activates the project env, and exports HF_HOME.
#
# Note: no `set -u` here -- conda's activate script references unset vars.
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$REPO_ROOT/.fasrc.env" ]]; then
  set -a; source "$REPO_ROOT/.fasrc.env"; set +a
else
  echo "ERROR: $REPO_ROOT/.fasrc.env not found. Copy .fasrc.env.example and fill it." >&2
  return 1 2>/dev/null || exit 1
fi

module load Miniforge3/26.1.0-fasrc01
source activate "${AIONFLOW_ENV:?set AIONFLOW_ENV in .fasrc.env}"
export HF_HOME="${HF_HOME:-$AIONFLOW_ROOT/hf_cache}"
