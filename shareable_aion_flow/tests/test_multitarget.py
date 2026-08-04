"""Unit tests for the multi-target loss, EMA weights, and shared CLS head."""

from __future__ import annotations

import json
import math
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
        # the joint head still points at its own dimensions after reindexing
        names = [mt.MULTI_TARGETS[j]["name"] for j in mt.JOINT_IDX]
        assert names == [n for n in mt.JOINT_PAIR if n in
                         {t["name"] for t in mt.MULTI_TARGETS}]
        # dropping a REQUIRED joint dimension is refused; a marginalisable one is not
        try:
            mt.configure_heads(("log_sfr",))
        except ValueError:
            pass
        else:
            raise AssertionError("dropping a required joint dimension must raise")
        mt.configure_heads(("log_flux_p3",))     # marginalisable -> allowed
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
        # a pre-joint checkpoint restores the 2-D P2xP3 joint it was trained with
        assert mt.JOINT_PAIR == ("log_flux_p2", "log_flux_p3")
        assert mt.HEAD_NAMES[-1] == "p2xp3_joint"
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


def test_joint_marginalises_a_missing_dimension_instead_of_imputing() -> None:
    """A row missing a marginalisable joint dimension still trains, via quadrature.

    The alternative -- imputing that dimension from the prior -- would pull the
    conditional toward the unconditional. This checks the quadrature path runs,
    contributes gradient, and gives a numerically stable marginal.
    """
    import multitarget as mt

    torch.manual_seed(0)
    try:
        mt.configure_heads(("log_flux_p4", "logmstar", "log_mbh_pan25", "log_mbh_vo09",
                            "log_flux_p1", "log_flux_p2", "log_ml_flux_1"))
        n_t, n_h = mt.N_TARGETS, mt.N_HEADS
        marg = [j for j in mt.JOINT_IDX
                if mt.MULTI_TARGETS[j]["name"] in mt.JOINT_MARGINAL]
        assert len(marg) == 1, "test assumes exactly one marginalisable dimension"
        jm = marg[0]

        B = 12
        flows = mt.MultiTargetFlows(context_dim=256)
        targets = torch.randn(B, n_t)
        targets[B // 2:, jm] = float("nan")        # half the rows lack it
        contexts = torch.randn(B, n_h, 256, requires_grad=True)
        stds = [TargetStandardizer(0.0, 1.0) for _ in range(n_t)]
        total, raw = mt.multi_target_nll(
            contexts=contexts, flows=flows, targets=targets,
            sig_lo=torch.zeros(B, n_t), sig_hi=torch.zeros(B, n_t),
            standardizers=stds, weights=np.ones(n_h), inject=False,
        )
        assert raw[-1] is not None and np.isfinite(raw[-1])
        total.backward()
        assert contexts.grad is not None
        assert contexts.grad[:, n_t].abs().sum() > 0      # joint context got gradient

        # the quadrature grid is fine enough: refining it barely moves the answer
        coarse = mt.JOINT_QUAD_NODES
        try:
            mt.JOINT_QUAD_NODES = 4 * coarse
            _, raw_fine = mt.multi_target_nll(
                contexts=contexts.detach(), flows=flows, targets=targets,
                sig_lo=torch.zeros(B, n_t), sig_hi=torch.zeros(B, n_t),
                standardizers=stds, weights=np.ones(n_h), inject=False,
            )
        finally:
            mt.JOINT_QUAD_NODES = coarse
        assert abs(raw_fine[-1] - raw[-1]) < 0.05
    finally:
        mt.configure_heads(())


def test_joint_only_trains_the_joint_and_leaves_marginal_flows_untouched() -> None:
    """Phase 1: only the joint contributes loss; marginal flows get NO gradient.

    Skipping the term matters more than zero-weighting it. A zero-weighted term
    still produces grads of 0 rather than None, and AdamW's decoupled weight
    decay would then quietly shrink flows that phase 2 is about to refit.
    """
    import multitarget as mt

    torch.manual_seed(0)
    try:
        mt.configure_heads(("log_flux_p4", "logmstar", "log_mbh_pan25", "log_mbh_vo09",
                            "log_flux_p1", "log_flux_p2", "log_ml_flux_1"))
        n_t, n_h = mt.N_TARGETS, mt.N_HEADS
        B = 12
        flows = mt.MultiTargetFlows(context_dim=256)
        targets = torch.randn(B, n_t)
        contexts = torch.randn(B, n_h, 256, requires_grad=True)
        stds = [TargetStandardizer(0.0, 1.0) for _ in range(n_t)]
        total, raw = mt.multi_target_nll(
            contexts=contexts, flows=flows, targets=targets,
            sig_lo=torch.zeros(B, n_t), sig_hi=torch.zeros(B, n_t),
            standardizers=stds, weights=np.ones(n_h), inject=False,
            joint_only=True,
        )
        assert all(r is None for r in raw[:n_t]), "marginal heads must not report a loss"
        assert raw[-1] is not None and np.isfinite(raw[-1]), "joint must still train"

        total.backward()
        joint_grad = sum(float(p.grad.abs().sum()) for p in flows.joint.parameters()
                         if p.grad is not None)
        assert joint_grad > 0, "joint flow got no gradient"
        for f in flows.flows:
            assert all(p.grad is None for p in f.parameters()), \
                "marginal flow grads must be None, not zero, so AdamW skips them"
        # the shared trunk is still driven, via the joint's context slot
        assert float(contexts.grad[:, n_t].abs().sum()) > 0
    finally:
        mt.configure_heads(())


def approx(x, tol=1e-9):
    class _A:
        def __eq__(self, other): return abs(other - x) <= tol
    return _A()


def test_warmup_then_cosine_is_one_factor_preserving_group_ratios() -> None:
    """Warmup and cosine compose in a single LambdaLR factor.

    LambdaLR scales each group's OWN initial LR, so the adapter/flow/trunk ratios
    survive the schedule. Chaining two schedulers would not guarantee that.
    """
    total_steps, warmup = 100, 10

    def factor(step: int) -> float:
        if warmup and step < warmup:
            return float(step + 1) / float(warmup)
        done = (step - warmup) / max(1, total_steps - warmup)
        return float(0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, done)))))

    assert factor(0) == approx(0.1)          # ramps from a tenth, not zero
    assert factor(warmup - 1) == approx(1.0)  # reaches full LR at the end of warmup
    assert factor(warmup) == approx(1.0)      # cosine starts at its maximum
    assert factor(total_steps - 1) < 0.01            # and anneals to ~0
    mid = [factor(s) for s in range(warmup, total_steps)]
    assert all(a >= b - 1e-12 for a, b in zip(mid, mid[1:])), "cosine must be monotone"


def test_both_tracker_paths_mirror_to_history_jsonl(tmp_path) -> None:
    """The wandb path must mirror too, not just the disabled path.

    Caught by a smoke: the mirror was wired into NullRun only, because the
    WandbRun construction is an ASSIGNMENT rather than a return, and a test that
    exercised only NullRun passed happily while the real runs wrote nothing.
    """
    import tracking

    class FakeWandb:
        def __init__(self): self.logged = []
        def log(self, metrics, step=None): self.logged.append((metrics, step))

    fake = FakeWandb()
    for run, label in ((tracking.NullRun(tracking.JsonlMirror(tmp_path / "a")), "null"),
                       (tracking.WandbRun(fake, object(),
                                          tracking.JsonlMirror(tmp_path / "b")), "wandb")):
        run.log({"move/adapters_low": 1.5e-3}, step=42)
    for sub in ("a", "b"):
        path = tmp_path / sub / "history.jsonl"
        assert path.exists(), f"{sub} path wrote no history.jsonl"
        row = json.loads(path.read_text().splitlines()[0])
        assert row["_step"] == 42 and row["move/adapters_low"] == 1.5e-3
    assert fake.logged, "wandb must still receive the payload"


def test_blank_cutout_drops_the_image_modality_for_that_source_only() -> None:
    """A source staged without a cutout must still train the image-free combos.

    Staging used to discard every source lacking a Legacy Survey cutout, which
    silently threw away ~60% of the sample after one partial unzip and would
    discard the entire DR2 expansion, since cutouts take ~6 days to fetch. Such
    sources now carry an all-zero image and have the image modality dropped
    per-source -- not per-batch, so a neighbour WITH a cutout is unaffected.
    """
    import multitarget as mt

    bucket = {"union": ("spectra", "z", "wise", "image")}
    combos = [("spectra", "z", "wise", "image")] * 4      # all four ask for images
    idx = np.arange(4)
    images = torch.rand(4, 4, 8, 8)
    images[1] = 0.0                                        # source 1 has no cutout
    images[3] = 0.0                                        # source 3 has no cutout

    out = mt.bucket_modality_dropout(bucket, combos, idx, images=images)
    assert out["image"].tolist() == [False, True, False, True], \
        "only the blank-image sources may have the image modality dropped"
    for other in ("spectra", "z", "wise"):
        assert not out[other].any(), f"{other} must be untouched by a blank image"

    # and when the combo already excludes images, nothing changes
    combos_noimg = [("spectra", "z")] * 4
    out2 = mt.bucket_modality_dropout(bucket, combos_noimg, idx, images=images)
    assert out2["image"].all(), "image is dropped for every source when not in the combo"

    # without the images argument the behaviour is exactly as before
    out3 = mt.bucket_modality_dropout(bucket, combos, idx)
    assert not out3["image"].any()
