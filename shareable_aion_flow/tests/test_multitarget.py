"""Unit tests for the multi-target loss, EMA weights, and shared CLS head."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multitarget import (  # noqa: E402
    N_HEADS,
    N_TARGETS,
    EMALossWeights,
    MultiTargetFlows,
    SharedCLSHead,
    multi_target_nll,
)
from normalizing_flow import TargetStandardizer  # noqa: E402


class _StubFlow(nn.Module):
    """log_prob = -(y - w·context_mean)^2, differentiable."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(1))

    def log_prob(self, y, context):
        return -((y - self.w * context.mean(dim=-1)) ** 2)


class _StubJointFlow(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(1))

    def log_prob(self, pair, context):
        return -((pair.mean(dim=-1) - self.w * context.mean(dim=-1)) ** 2)


def _stub_flows() -> MultiTargetFlows:
    flows = MultiTargetFlows.__new__(MultiTargetFlows)
    nn.Module.__init__(flows)
    flows.flows = nn.ModuleList(_StubFlow() for _ in range(N_TARGETS))
    flows.joint = _StubJointFlow()
    return flows


def test_shared_head_shape() -> None:
    head = SharedCLSHead()
    out = head(torch.randn(3, N_HEADS, 768))
    assert out.shape == (3, N_HEADS, 256)


def test_masked_loss_skips_nan_targets_and_flows_gradients() -> None:
    B = 8
    flows = _stub_flows()
    targets = torch.randn(B, N_TARGETS)
    targets[:, 2] = float("nan")           # target 2 fully unavailable
    targets[0, 0] = float("nan")           # one source missing target 0
    contexts = torch.randn(B, N_HEADS, 256, requires_grad=True)
    stds = [TargetStandardizer(0.0, 1.0) for _ in range(N_TARGETS)]
    total, raw = multi_target_nll(
        contexts=contexts, flows=flows, targets=targets,
        sig_lo=torch.zeros(B, N_TARGETS), sig_hi=torch.zeros(B, N_TARGETS),
        standardizers=stds, weights=np.ones(N_HEADS), inject=False,
    )
    assert len(raw) == N_HEADS
    assert raw[2] is None and all(r is not None for j, r in enumerate(raw) if j != 2)
    total.backward()
    grad = contexts.grad
    assert grad is not None
    assert grad[:, 2].abs().sum() == 0          # no gradient from the missing target
    assert grad[0, 0].abs().sum() == 0          # masked source contributes nothing
    assert grad[1, 0].abs().sum() > 0
    assert grad[:, N_TARGETS].abs().sum() > 0   # joint head receives gradient


def test_injection_only_where_sigma_present() -> None:
    torch.manual_seed(0)
    B = 512
    flows = _stub_flows()
    targets = torch.zeros(B, N_TARGETS)
    contexts = torch.zeros(B, N_HEADS, 256)
    stds = [TargetStandardizer(0.0, 1.0) for _ in range(N_TARGETS)]
    sig = torch.zeros(B, N_TARGETS)
    sig[:, 0] = 0.5                              # only target 0 has errors
    _, raw = multi_target_nll(
        contexts=contexts, flows=flows, targets=targets,
        sig_lo=sig, sig_hi=sig, standardizers=stds,
        weights=np.ones(N_HEADS), inject=True,
    )
    assert raw[0] > 0.01                         # noise was injected -> nonzero sq loss
    for j in range(1, N_HEADS):
        assert abs(raw[j]) < 1e-8                # sigma-free targets/joint: no injection


def test_ema_weights_normalize_scales() -> None:
    ema = EMALossWeights(beta=0.5)
    for _ in range(60):
        w = ema.update_and_weights([4.0, 1.0, None, 0.5, 2.0, 1.0, 1.0, 1.0])
    assert w[0] < w[1] < w[3]                    # harder target -> smaller weight
    assert np.isclose(w[0] * 4.0, 1.0, rtol=0.05)
    assert np.isclose(w[3] * 0.5, 1.0, rtol=0.05)
    assert w[2] == 1.0                           # never-seen target keeps unit weight
