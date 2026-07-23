"""Multi-target read-only-CLS training: 6 targets, one frozen forward.

Targets: p1/p2/p3 band rates (runtime sidecar), broad-band flux, Lx, logmstar
(P4 and HR excluded per their null results). Architecture: per-target CLS
vectors + SHARED per-block Q/V read adapters (data stream frozen, no_grad),
one SHARED 768->512->256 MLP over the stacked CLS states, one small NSF flow
per target. Joint loss: per-target NLL on standardized values, split-normal
noise injection where sigma exists, per-source availability masks, and
detached-EMA loss normalization so harder targets do not dominate gradients.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

try:
    from .data_to_aion_embeddings import read_dataset
    from .normalizing_flow import ConditionalNSFFlow, TargetStandardizer, sample_split_normal
except ImportError:
    from data_to_aion_embeddings import read_dataset
    from normalizing_flow import ConditionalNSFFlow, TargetStandardizer, sample_split_normal

# name, sigma columns (None = no error model), availability sigma gate
MULTI_TARGETS: list[dict] = [
    {"name": "log_flux_p1", "sig": ("log_flux_p1_sig_lo", "log_flux_p1_sig_hi"), "max_sigma": 1.0, "sidecar": True},
    {"name": "log_flux_p2", "sig": ("log_flux_p2_sig_lo", "log_flux_p2_sig_hi"), "max_sigma": 1.0, "sidecar": True},
    {"name": "log_flux_p3", "sig": ("log_flux_p3_sig_lo", "log_flux_p3_sig_hi"), "max_sigma": 1.0, "sidecar": True},
    {"name": "log_ml_flux_1", "sig": ("flux_sig_lo", "flux_sig_hi"), "max_sigma": None, "sidecar": False},
    {"name": "log_lx", "sig": ("flux_sig_lo", "flux_sig_hi"), "max_sigma": None, "sidecar": False},
    {"name": "logmstar", "sig": None, "max_sigma": None, "sidecar": False},
]
N_TARGETS = len(MULTI_TARGETS)


def load_multi_target_matrix(
    staged_path: Path, extra_targets_csv: Path | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(targets [n,6], sig_lo [n,6], sig_hi [n,6]) aligned to a staged split file.

    Unavailable entries are NaN in ``targets`` (masked in the loss); missing
    sigmas are 0 (no injection). Sidecar targets join by targetid.
    """
    import pandas as pd

    with h5py.File(staged_path, "r") as handle:
        n = int(handle["desi_targetid"].shape[0])
        tids = handle["desi_targetid"][:].astype(np.int64)
        store_cols = {}
        for spec in MULTI_TARGETS:
            cols = [spec["name"]] + (list(spec["sig"]) if spec["sig"] else [])
            for c in cols:
                if not spec["sidecar"] and c in handle:
                    store_cols[c] = read_dataset(handle, c).astype(np.float64)
    side = None
    if extra_targets_csv is not None:
        side = pd.read_csv(extra_targets_csv).drop_duplicates("targetid").set_index("targetid")
        side = side.reindex(tids)

    y = np.full((n, N_TARGETS), np.nan)
    slo = np.zeros((n, N_TARGETS))
    shi = np.zeros((n, N_TARGETS))
    for j, spec in enumerate(MULTI_TARGETS):
        if spec["sidecar"]:
            if side is None:
                continue
            def col(c, _side=side):
                return _side[c].to_numpy(np.float64) if c in _side.columns else None
        else:
            def col(c, _store=store_cols):
                return _store.get(c)
        vals = col(spec["name"])
        if vals is None:
            continue
        y[:, j] = vals
        if spec["sig"]:
            lo, hi = col(spec["sig"][0]), col(spec["sig"][1])
            if lo is not None and hi is not None:
                slo[:, j] = np.nan_to_num(np.abs(lo))
                shi[:, j] = np.nan_to_num(np.abs(hi))
        if spec["max_sigma"] is not None:
            mean_sig = 0.5 * (slo[:, j] + shi[:, j])
            bad = ~np.isfinite(mean_sig) | (mean_sig > spec["max_sigma"])
            y[bad, j] = np.nan
    return y, slo, shi


class SharedCLSHead(nn.Module):
    """One shared 768->512->256 MLP applied across all stacked CLS streams."""

    def __init__(self, embed_dim: int = 768, hidden: int = 512, context_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, context_dim),
            nn.LayerNorm(context_dim),
        )
        self.context_dim = context_dim

    def forward(self, cls_states: torch.Tensor) -> torch.Tensor:  # [B, K, 768] -> [B, K, 256]
        return self.net(cls_states)


class MultiTargetFlows(nn.Module):
    def __init__(self, context_dim: int = 256, n_targets: int = N_TARGETS) -> None:
        super().__init__()
        self.flows = nn.ModuleList(ConditionalNSFFlow(context_dim=context_dim) for _ in range(n_targets))


class EMALossWeights:
    """Detached EMA of each target's mean loss; weight = 1/EMA (unit-scale grads)."""

    def __init__(self, n_targets: int = N_TARGETS, beta: float = 0.98) -> None:
        self.ema = np.ones(n_targets)
        self.beta = float(beta)
        self.seen = np.zeros(n_targets, dtype=bool)

    def update_and_weights(self, losses: list[float | None]) -> np.ndarray:
        for j, val in enumerate(losses):
            if val is None or not np.isfinite(val):
                continue
            if not self.seen[j]:
                self.ema[j] = val
                self.seen[j] = True
            else:
                self.ema[j] = self.beta * self.ema[j] + (1 - self.beta) * val
        return 1.0 / np.clip(np.abs(self.ema), 0.2, None)

    def state_dict(self) -> dict:
        return {"ema": self.ema.tolist(), "seen": self.seen.tolist(), "beta": self.beta}


def multi_target_nll(
    *,
    contexts: torch.Tensor,            # [B, K, 256]
    flows: MultiTargetFlows,
    targets: torch.Tensor,             # [B, K] (NaN = unavailable)
    sig_lo: torch.Tensor,              # [B, K]
    sig_hi: torch.Tensor,
    standardizers: list[TargetStandardizer],
    weights: np.ndarray,               # [K] detached loss weights
    inject: bool = True,
) -> tuple[torch.Tensor, list[float | None]]:
    """Weighted joint NLL over available (source, target) pairs.

    Returns (total_loss, per-target UNweighted mean losses for the EMA update).
    """
    total = contexts.new_zeros(())
    raw: list[float | None] = []
    for j, flow in enumerate(flows.flows):
        mask = torch.isfinite(targets[:, j])
        if not bool(mask.any()):
            raw.append(None)
            continue
        std = standardizers[j]
        y = std.transform_tensor(targets[mask, j])
        if inject:
            lo = (sig_lo[mask, j].abs() / std.std).clamp_min(0.0)
            hi = (sig_hi[mask, j].abs() / std.std).clamp_min(0.0)
            has = (lo + hi) > 1e-8
            if bool(has.any()):
                eps = sample_split_normal(lo.clamp_min(1e-6), hi.clamp_min(1e-6))
                y = torch.where(has, y + eps, y)
        nll = -flow.log_prob(y, contexts[mask, j]).mean()
        raw.append(float(nll.item()))
        total = total + float(weights[j]) * nll
    return total, raw
