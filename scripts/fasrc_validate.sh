#!/bin/bash
# Infra gate -- must pass on a GPU node before any training (ported from the
# RunPod "imports + codec smoke + CUDA tensor" discipline). Data-independent:
# the AION codec smoke uses a synthetic spectrum, so this can run before staging.
#
# Interactive use:
#   salloc -p gpu_test -t 0-00:30 --gpus 1 --mem 16G -A <account>
#   bash scripts/fasrc_validate.sh
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/fasrc_env.sh"

echo "== imports =="
python - <<'PY'
import torch, zuko, safetensors
import aion, aion.codecs, aion.modalities
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
print("aion / zuko / safetensors import OK")
PY

echo "== CUDA tensor smoke =="
python - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA device visible on this node"
x = torch.randn(2048, 2048, device="cuda")
print("cuda matmul OK, checksum:", float((x @ x).sum()))
PY

echo "== AION spectrum codec smoke (synthetic spectrum, CPU) =="
python - <<'PY'
import torch
from aion.codecs import CodecManager
from aion.modalities import DESISpectrum
n = 7781
flux = torch.rand(1, n).abs()
ivar = torch.ones(1, n)
lam = torch.linspace(3600.0, 9824.0, n)[None, :]
tok = CodecManager(device="cpu").encode(
    DESISpectrum(flux=flux, ivar=ivar, mask=ivar <= 0, wavelength=lam)
)
key = max(tok, key=lambda k: tok[k].shape[-1])
print("codec OK; spectrum token tensor", key, tuple(tok[key].shape))
PY

echo "ALL INFRA CHECKS PASSED"
