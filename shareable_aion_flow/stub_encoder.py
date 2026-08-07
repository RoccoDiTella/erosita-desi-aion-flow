"""A deterministic stand-in for AIONTokenEncoder, so the train loop runs off-GPU.

WHY THIS EXISTS. `aion` is a heavyweight optional dependency and is not
installed on a workstation, so `run_train_multi` could not be executed at all
outside the cluster. Every wiring mistake -- a flag that does not reach the
optimiser, a head set that does not match the sidecar, a scheduler that divides
by zero on a short run, a checkpoint that omits a state dict -- was therefore
only discoverable by burning an A100 allocation and reading a traceback in a
SLURM log twenty minutes later.

This stub matches the parts of the encoder interface the trainer actually
touches (`cls_token`, `cls_read_adapters`, `encode_tokens`) and nothing else. It
produces contexts that are a smooth, deterministic function of redshift, the
head index and the combo, so the flows have something learnable and a loss that
goes down means the plumbing works.

IT DOES NOT TEST THE SCIENCE. The contexts carry no astrophysics; a smoke run's
NLL is meaningless as a result. What it tests is that the run STARTS, steps,
validates, checkpoints and exits. Treat a green smoke as "the pipeline is
wired", never as "the model works".

    AIONFLOW_STUB_ENCODER=1 python -m shareable_aion_flow.main train-multi ...
    AIONFLOW_STUB_ENCODER=1 python scripts/eval_multitarget.py --device cpu ...

`build_encoder` is the ONE place that reads the switch. It used to live inside
`run_train_multi` only, which meant the train loop ran off-GPU while everything
downstream of it -- eval, posterior structure, the HR quadrature -- built the
real encoder unconditionally and could not run at all without `aion`. The
train->eval chain was therefore only smoke-testable up to the checkpoint, and
the eval half was first exercised on the cluster after the expensive part had
already succeeded. Every consumer now goes through this function, so a stub
run covers the whole chain.
"""

from __future__ import annotations

import os

import torch
from torch import nn

EMBED_DIM = 768


def stub_requested() -> bool:
    return os.environ.get("AIONFLOW_STUB_ENCODER", "").strip() not in ("", "0", "false", "False")


def build_encoder(num_cls: int, device, grad_checkpoint: bool = False, tag: str = "aionflow"):
    """The read-only-CLS encoder: StubTokenEncoder if the switch is set, else AION.

    THE SINGLE READER OF `AIONFLOW_STUB_ENCODER`. Call this instead of
    constructing `AIONTokenEncoder` directly, so train, eval, posterior
    structure and the HR quadrature all honour the same switch and a
    workstation can run the whole chain.

    `tag` only prefixes the log line ("multi", "eval", "post", "hr"), so a log
    says which stage is stubbed. The warning is printed on EVERY stub build and
    is deliberately loud: the contexts are synthetic, so no number produced
    downstream of a stub encoder means anything.
    """
    if stub_requested():
        print(f"[{tag}] AIONFLOW_STUB_ENCODER set: using StubTokenEncoder. "
              "PIPELINE SMOKE ONLY -- the contexts are synthetic and every metric "
              "from this run is meaningless.", flush=True)
        return StubTokenEncoder(num_cls=num_cls).to(device)
    # Deferred, and via the package/flat fallback the rest of this repo uses:
    # importing the real encoder module at import time would drag the heavy
    # dependency chain into the very environment the stub exists to serve.
    try:
        from .data_to_aion_embeddings import AIONTokenEncoder
    except ImportError:
        from data_to_aion_embeddings import AIONTokenEncoder
    try:
        return AIONTokenEncoder(
            freeze=False, cls_mode=True, cls_variant="readonly",
            num_cls=num_cls, grad_checkpoint=grad_checkpoint,
        ).to(device)
    except ImportError as exc:
        # The bare "No module named 'aion'" gives no clue that this repo has a
        # supported way to run without it.
        raise ImportError(
            f"{exc}  (no `aion` here? set AIONFLOW_STUB_ENCODER=1 for a "
            f"plumbing-only smoke run -- its numbers are meaningless)") from exc


class StubTokenEncoder(nn.Module):
    """Deterministic [B, num_cls, 768] contexts. No AION, no GPU, no weights.

    `cls_read_adapters` is a real ModuleList of small Linears rather than an
    empty placeholder: the trainer puts them in their own parameter group with
    their own learning rate and reports per-depth movement, so a stub without
    them would skip exactly the code path that LR decisions are read from.
    """

    def __init__(self, num_cls: int, depth: int = 12, **_ignored) -> None:
        super().__init__()
        self.num_cls = int(num_cls)
        self.cls_token = nn.Parameter(torch.randn(self.num_cls, EMBED_DIM) * 0.02)
        self.cls_read_adapters = nn.ModuleList(
            nn.Linear(EMBED_DIM, EMBED_DIM, bias=False) for _ in range(depth))
        for a in self.cls_read_adapters:                 # near-identity at init
            nn.init.eye_(a.weight)
            with torch.no_grad():
                a.weight.add_(torch.randn_like(a.weight) * 1e-3)

    def encoder_trainable_state(self) -> dict:
        """Only the trainable parts, mirroring the real encoder's contract.

        The real encoder holds a frozen AION backbone it deliberately does NOT
        checkpoint; only the CLS tokens and read adapters are ours. A stub that
        omitted this method let a smoke run pass every training step and then
        die at the first `save()`, which is the least useful place to fail.
        """
        return {"cls_token": self.cls_token.detach().cpu(),
                **{f"cls_read_adapters.{i}.weight": a.weight.detach().cpu()
                   for i, a in enumerate(self.cls_read_adapters)}}

    def load_encoder_trainable_state(self, state: dict) -> None:
        with torch.no_grad():
            if "cls_token" in state:
                self.cls_token.copy_(state["cls_token"].to(self.cls_token.device))
            for i, a in enumerate(self.cls_read_adapters):
                k = f"cls_read_adapters.{i}.weight"
                if k in state:
                    a.weight.copy_(state[k].to(a.weight.device))

    def encode_tokens(self, batch, combo, modality_dropout=None, **_ignored):
        """(cls_seq [B, num_cls, 768], None). Signature mirrors the real encoder."""
        flux, _ivar, _lam, redshift, wise, image = batch[:6]
        device = self.cls_token.device
        z = redshift.reshape(-1, 1, 1).to(device=device, dtype=torch.float32)
        b = z.shape[0]

        # One scalar per modality, so a combo change is visible in the context
        # and a modality ablation is not a no-op in a smoke run.
        parts = {
            "spectra": torch.nan_to_num(flux.to(device).float()).mean(dim=1).reshape(-1, 1, 1),
            "z": z,
            "wise": torch.nan_to_num(wise.to(device).float()).mean(dim=1).reshape(-1, 1, 1),
            "image": torch.nan_to_num(image.to(device).float()).flatten(1).mean(dim=1).reshape(-1, 1, 1),
        }
        acc = torch.zeros(b, 1, 1, device=device)
        for k, name in enumerate(("spectra", "z", "wise", "image")):
            if name not in combo:
                continue
            v = parts[name]
            if modality_dropout is not None and name in modality_dropout:
                keep = (~modality_dropout[name].to(device).bool()).float().reshape(-1, 1, 1)
                v = v * keep
            acc = acc + (k + 1.0) * torch.tanh(v)

        heads = torch.arange(self.num_cls, device=device, dtype=torch.float32).view(1, -1, 1)
        feats = torch.arange(EMBED_DIM, device=device, dtype=torch.float32).view(1, 1, -1)
        base = torch.sin(0.05 * (1.0 + feats) * (acc + z) + 0.37 * heads)
        cls = self.cls_token.unsqueeze(0) + base
        for a in self.cls_read_adapters:
            cls = cls + 0.01 * a(cls)
        return cls, None
