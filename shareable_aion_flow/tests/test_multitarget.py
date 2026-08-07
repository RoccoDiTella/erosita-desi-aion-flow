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


def test_band_availability_is_gated_on_detection_not_error_bar() -> None:
    """An undetected band must read as MISSING, however tight its error bar.

    Gating on sigma alone admitted upper limits as though they were
    measurements. That is the likeliest source of the faint-band sigma
    overestimation: P1 and P3 sat far above their sigma-derived R^2 ceilings,
    and the hardness-ratio test implied a negative true variance, which is
    impossible unless E[sigma^2] is inflated.

    Asserts the RULE against mt.DET_LIKE_MIN rather than a literal, so the
    threshold lives in exactly one place. It moved 5 -> 6 on 2026-08-06 to match
    the eRASS Main catalogue's own inclusion rule.
    """
    import multitarget as mt

    spec = next(t for t in mt.MULTI_TARGETS if t["name"] == "log_flux_p3")
    assert spec["det"] == ("det_like_p3", mt.DET_LIKE_MIN)
    # The sigma gate is deliberately retired. On the bands the detection cut
    # removes a strict superset of the rows (measured: zero additional rows in
    # P1-P4 at either threshold), and carrying both made the selection function
    # impossible to state as a single cut.
    assert spec["max_sigma"] is None

    # a source with a beautiful error bar but no detection is NOT available
    thr = mt.DET_LIKE_MIN
    det = np.array([thr + 4.0, thr - 0.1, thr + 0.1, np.nan])
    tight_sigma = np.full(4, 0.01)
    y = np.array([-13.0, -13.0, -13.0, -13.0])
    ok = np.isfinite(y)
    if spec["max_sigma"] is not None:
        ok &= 0.5 * (tight_sigma + tight_sigma) <= spec["max_sigma"]
    ok &= np.isfinite(det) & (det > thr)
    assert ok.tolist() == [True, False, True, False], \
        "detection must decide availability even when the sigma gate passes"


def test_joint_is_addressable_by_name_not_position() -> None:
    """`j2, j3 = JOINT_IDX` survived a whole run cycle. Name resolution or nothing.

    Two scripts unpacked the joint as a 2-tuple long after it became 4-D, and a
    third assumed flow columns 0 and 1 were (P2, P3). Positional indexing of the
    joint is always silently wrong rather than loudly wrong, so these are the
    invariants that make name indexing safe.
    """
    import multitarget as mt

    dims = mt.joint_dims()
    assert len(dims) == len(mt.JOINT_IDX)
    names = [t["name"] for t in mt.MULTI_TARGETS]
    # Flow column order is JOINT_PAIR DECLARATION order, not MULTI_TARGETS order.
    assert list(dims) == [n for n in mt.JOINT_PAIR if n in names]
    # ...and the two orders really are different, which is the whole hazard:
    # if they ever coincide this test still passes but stops proving anything,
    # so assert the divergence explicitly for the default head set.
    if len(dims) > 1 and sorted(dims, key=names.index) != list(dims):
        assert [mt.target_col(n) for n in dims] != list(range(len(dims)))
    for k, name in enumerate(dims):
        assert mt.joint_col(name) == k
        assert names[mt.target_col(name)] == name
    for bogus in ("definitely_not_a_dimension", "log_flux_p2"):
        if bogus in dims:
            continue
        try:
            mt.joint_col(bogus)
        except KeyError:
            pass
        else:
            raise AssertionError(f"joint_col({bogus!r}) must raise, not return a position")


def test_joint_availability_agrees_across_backends() -> None:
    """One availability rule, or the trainer and eval report different samples.

    have_req and have_all are genuinely different populations (fully observed
    versus quadrature-marginalised), and collapsing them into one "joint n" is a
    count-weighted mixture of two dimensionalities.
    """
    import multitarget as mt
    import torch

    if not mt.JOINT_MARGINAL:
        return
    n = len(mt.MULTI_TARGETS)
    marg = mt.JOINT_MARGINAL[0]
    req = next(d for d in mt.joint_dims() if d not in mt.JOINT_MARGINAL)

    t = np.ones((3, n))
    t[1, mt.target_col(marg)] = np.nan       # marginalisable missing -> still usable
    t[2, mt.target_col(req)] = np.nan        # required missing       -> unusable
    want_req, want_all = [True, True, False], [True, False, False]

    for arr in (t, torch.tensor(t)):
        have_req, have_all = mt.joint_availability(arr)
        assert [bool(x) for x in have_req] == want_req
        assert [bool(x) for x in have_all] == want_all


def test_multi_target_mode_does_not_let_a_staged_column_select_rows(tmp_path) -> None:
    """`target_name=None` must not filter. The DR1 flux column must not gate DR2.

    This is the test whose absence made the bug invisible. Every multi-target
    caller passed the literal "log_ml_flux_1", which `_finite_target_rows` reads
    FROM THE STAGED HDF5 (DR1/eRASS1). Once log_ml_flux_1 and log_lx moved to the
    DR2 sidecar, the LABELS were DR2 while the SAMPLE was still whichever rows
    eRASS1 happened to detect: a DR1 detection limit silently selecting a DR2
    experiment, at 2.7x less exposure.

    Nothing crashed and no number looked wrong, which is exactly why it needed an
    assertion rather than a reviewer.
    """
    import h5py
    from data_to_aion_embeddings import _finite_target_rows

    path = tmp_path / "desi_train.hdf5"
    with h5py.File(path, "w") as h:
        h.create_dataset("desi_targetid", data=np.arange(6, dtype=np.int64))
        # rows 1 and 4 are undetected in the DR1 column, i.e. exactly the rows a
        # DR1 gate would silently remove from a DR2 experiment
        h.create_dataset("log_ml_flux_1",
                         data=np.array([-12.0, np.nan, -12.5, -13.0, np.nan, -11.5]))

    rows = np.arange(6, dtype=np.int64)

    # single-target mode: the column DOES gate, which is correct for that mode
    kept = _finite_target_rows(path, "log_ml_flux_1", rows)
    assert kept is not None
    assert kept.tolist() == [0, 2, 3, 5]

    # multi-target mode: membership is the caller's business, so nothing is cut
    assert _finite_target_rows(path, None, rows) is rows
    assert _finite_target_rows(path, None, None) is None


def test_a_head_with_no_column_anywhere_is_refused_not_silently_skipped(tmp_path) -> None:
    """An absent column must raise, not produce a head that never trains.

    `vals is None -> continue` left the target all-NaN, and multi_target_nll
    skips a head whose targets are all NaN, so the head was built, optimised
    over, and silently never trained. In a loss curve that is indistinguishable
    from converging immediately. `logmstar` is the live example: the FastSpecFit
    mass is no longer staged, so any run that forgets --drop-heads logmstar
    would have trained a ghost head.
    """
    import h5py
    import pandas as pd
    import multitarget as mt

    staged = tmp_path / "desi_train.hdf5"
    tids = np.arange(4, dtype=np.int64)
    with h5py.File(staged, "w") as h:
        h.create_dataset("desi_targetid", data=tids)

    side = pd.DataFrame({"targetid": tids})
    for spec in mt._ALL_TARGETS:                 # everything EXCEPT logmstar
        if not spec["sidecar"]:
            continue
        side[spec["name"]] = 1.0
        for c in spec["sig"] or ():
            side[c] = 0.1
        if spec["det"]:
            side[spec["det"][0]] = mt.DET_LIKE_MIN + 1.0
    csv = tmp_path / "sidecar.csv"
    side.to_csv(csv, index=False)

    try:
        mt.load_multi_target_matrix(staged, csv)
    except ValueError as exc:
        assert "logmstar" in str(exc)
        assert "--drop-heads" in str(exc)
    else:
        raise AssertionError("a head with no column anywhere must raise")

    # ...and dropping it is the documented escape hatch, which must then work.
    try:
        mt.configure_heads(("logmstar",))
        y, slo, shi = mt.load_multi_target_matrix(staged, csv)
        assert y.shape == (4, mt.N_TARGETS)
        assert np.isfinite(y).all()
    finally:
        mt.configure_heads(())
