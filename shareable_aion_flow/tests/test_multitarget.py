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

    def log_prob_draws(self, y_draws, context):
        return -((y_draws - self.w * context.mean(dim=-1)) ** 2)


class _StubJointFlow(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(1))

    def log_prob(self, pair, context):
        return -((pair.mean(dim=-1) - self.w * context.mean(dim=-1)) ** 2)

    def log_prob_draws(self, pair, context):
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


def test_length_buckets_partition_all_15_combos() -> None:
    from multitarget import LENGTH_BUCKETS, bucket_assignments, bucket_modality_dropout
    from attention_pooling_head import all_nonempty_modality_combos

    all_combos = all_nonempty_modality_combos()
    covered = set()
    for b in LENGTH_BUCKETS:
        assert not (covered & b["combos"])       # disjoint
        covered |= b["combos"]
    assert covered == set(all_combos)            # exhaustive

    combos = [("z",), ("spectra", "z"), ("spectra", "z", "wise", "image"), ("wise", "image"), ("z",)]
    parts = bucket_assignments(combos)
    assert sum(len(idx) for _, idx in parts) == len(combos)
    for bucket, idx in parts:
        for i in idx:
            assert tuple(combos[i]) in bucket["combos"]
        drop = bucket_modality_dropout(bucket, combos, idx)
        for k, group in enumerate(bucket["union"]):
            for row, i in enumerate(idx):
                assert drop[group][row].item() == (group not in combos[i])


def test_broadcast_draws_match_per_draw_calls() -> None:
    torch.manual_seed(0)
    from normalizing_flow import ConditionalNSFFlow

    for features in (1, 2):
        flow = ConditionalNSFFlow(context_dim=8, transforms=2, hidden_features=(16,), features=features)
        ctx = torch.randn(5, 8)
        shape = (3, 5) if features == 1 else (3, 5, 2)
        draws = torch.randn(*shape)
        broadcast = flow.log_prob_draws(draws, ctx)
        assert broadcast.shape == (3, 5)
        for k in range(3):
            if features == 1:
                single = flow.log_prob(draws[k], ctx)
            else:
                single = flow.distribution(ctx).log_prob(draws[k])
            assert torch.allclose(broadcast[k], single, atol=1e-5)


def test_configure_heads_drops_and_restores() -> None:
    import multitarget as mt

    # Relative to the full target list, not a literal: heads get appended over
    # time (log_sfr was) and a hard-coded count turns that into a false failure.
    n_all = len(mt._ALL_TARGETS)
    try:
        mt.configure_heads(("log_flux_p4",))
        assert mt.N_TARGETS == n_all - 1 and mt.N_HEADS == n_all
        assert "log_flux_p4" not in mt.HEAD_NAMES
        # the joint head must still point at the right two bands after reindexing
        j2, j3 = mt.JOINT_IDX
        assert mt.MULTI_TARGETS[j2]["name"] == "log_flux_p2"
        assert mt.MULTI_TARGETS[j3]["name"] == "log_flux_p3"
        # dropping a band the joint head needs is refused
        try:
            mt.configure_heads(("log_flux_p2",))
        except ValueError:
            pass
        else:
            raise AssertionError("dropping a joint-pair band must raise")
    finally:
        mt.configure_heads(())
        assert mt.N_TARGETS == n_all and mt.N_HEADS == n_all + 1


def test_head_influence_attributes_gradient_to_the_right_head() -> None:
    from multitarget import head_influence

    flows = _stub_flows()
    B = 6
    targets = torch.randn(B, N_TARGETS)
    contexts = torch.randn(B, N_HEADS, 256, requires_grad=True)
    stds = [TargetStandardizer(0.0, 1.0) for _ in range(N_TARGETS)]
    _, _, terms = multi_target_nll(
        contexts=contexts, flows=flows, targets=targets,
        sig_lo=torch.zeros(B, N_TARGETS), sig_hi=torch.zeros(B, N_TARGETS),
        standardizers=stds, weights=np.ones(N_HEADS), inject=False, return_terms=True,
    )
    inf = head_influence(terms, contexts)
    assert all(f"influence/{h}" in inf for h in ("log_ml_flux_1", "log_lx"))
    shares = [v for k, v in inf.items() if k.startswith("influence_share/")]
    assert abs(sum(shares) - 1.0) < 1e-5          # shares are a distribution


def test_configure_heads_from_config_reloads_older_checkpoints() -> None:
    """A checkpoint trained before a head was appended must still build."""
    import multitarget as mt

    n_all = len(mt._ALL_TARGETS)
    try:
        # a pre-log_sfr checkpoint: it stored the head names it actually had
        legacy = [t["name"] for t in mt._ALL_TARGETS if t["name"] != "log_sfr"]
        mt.configure_heads_from_config({"heads": legacy + ["p2xp3_joint"]})
        assert "log_sfr" not in mt.HEAD_NAMES
        assert mt.N_TARGETS == n_all - 1
        flows = mt.MultiTargetFlows(context_dim=8)
        assert len(flows.flows) == n_all - 1        # matches the old state_dict
        # the joint pair still resolves after the reindex
        j2, j3 = mt.JOINT_IDX
        assert mt.MULTI_TARGETS[j2]["name"] == "log_flux_p2"
        assert mt.MULTI_TARGETS[j3]["name"] == "log_flux_p3"

        # even older: no `heads` key at all, only drop_heads
        mt.configure_heads_from_config({"drop_heads": ["log_flux_p4"]})
        assert "log_flux_p4" not in mt.HEAD_NAMES and "log_sfr" in mt.HEAD_NAMES

        # empty / missing config falls back to the full current set
        mt.configure_heads_from_config({})
        assert mt.N_TARGETS == n_all
        mt.configure_heads_from_config(None)
        assert mt.N_TARGETS == n_all
    finally:
        mt.configure_heads(())


def test_model_pieces_follow_dropped_head_count() -> None:
    """Regression: default args must not capture the head count at import time."""
    import multitarget as mt

    n_all = len(mt._ALL_TARGETS)
    try:
        mt.configure_heads(("log_flux_p4",))
        flows = mt.MultiTargetFlows(context_dim=8)
        assert len(flows.flows) == mt.N_TARGETS == n_all - 1
        assert len(mt.EMALossWeights().ema) == mt.N_HEADS == n_all
    finally:
        mt.configure_heads(())
    flows = mt.MultiTargetFlows(context_dim=8)
    assert len(flows.flows) == mt.N_TARGETS == n_all
    assert len(mt.EMALossWeights().ema) == mt.N_HEADS == n_all + 1
