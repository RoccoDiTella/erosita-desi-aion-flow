#!/bin/bash
# One-time FASRC conda env build. Run on the cluster (over the socket) AFTER the
# code is staged, from the staged repo root:
#   ssh -S ~/.ssh/cm-fasrc <host> 'bash <staged-repo>/scripts/fasrc_build_env.sh'
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/.fasrc.env"

module load Miniforge3/26.1.0-fasrc01
# idempotent: skip create if the env already exists (so a re-run just resumes at pip).
[ -x "$AIONFLOW_ENV/bin/python" ] || mamba create -y -p "$AIONFLOW_ENV" python=3.11
source activate "$AIONFLOW_ENV"

# CUDA-matched torch FIRST so nothing pulls a CPU wheel afterward.
pip install --quiet torch --index-url https://download.pytorch.org/whl/cu124
# repo base deps (numpy, scipy, h5py, astropy, pandas, matplotlib, tqdm); torch already satisfied.
pip install --quiet -e "$REPO_ROOT"
# flow + AION: install light deps explicitly, then aion --no-deps (avoids re-pulling torch).
pip install --quiet zuko safetensors einops jaxtyping huggingface_hub tokenizers
pip install --quiet --no-deps polymathic-aion
# experiment tracking (optional at runtime; --wandb is opt-in)
pip install --quiet wandb

python - <<'PY'
import torch, zuko, safetensors, aion  # noqa: F401
print("env OK | torch", torch.__version__, "| cuda", torch.version.cuda, "| cuda_available", torch.cuda.is_available())
PY
echo "ENV BUILD DONE -> $AIONFLOW_ENV"
