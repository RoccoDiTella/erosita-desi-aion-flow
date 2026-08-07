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
        y, slo, shi, pois = mt.load_multi_target_matrix(staged, csv)
        assert y.shape == (4, mt.N_TARGETS)
        assert np.isfinite(y).all()
        # no Poisson head is active, so the counts channel is all-NaN padding
        assert pois.shape == (4, mt.N_TARGETS, 3) and not np.isfinite(pois).any()
    finally:
        mt.configure_heads(())


def test_configure_joint_marginal_recovers_rows_missing_the_optional_dimension() -> None:
    """A row with M* and Lx but no SFR must count as usable once SFR is declared
    marginalisable.

    Catches configure_joint_marginal failing to actually update the JOINT_MARGINAL
    global that joint_availability reads -- have_req would then equal have_all and
    the recovered rows (the whole point of Run A's --joint-marginal) would vanish.
    """
    import multitarget as mt

    try:
        mt.configure_heads(("logmstar", "log_mbh_pan25", "log_mbh_vo09", "log_flux_p3"))
        mt.configure_joint_marginal(("log_sfr",))

        n = mt.N_TARGETS
        mstar, sfr = mt.target_col("logmstar_cigale"), mt.target_col("log_sfr")
        y = np.ones((6, n))
        y[1:3, sfr] = np.nan          # SFR missing, M* and Lx present -> still usable
        y[4, mstar] = np.nan          # a REQUIRED dimension missing -> unusable
        have_req, have_all = mt.joint_availability(y)
        assert have_req.sum() > have_all.sum()
        assert bool(have_req[1]) and not bool(have_all[1])    # marginal branch
        assert not bool(have_req[4])                          # required branch excludes it
    finally:
        mt.JOINT_MARGINAL = mt._DEFAULT_JOINT_MARGINAL
        mt.configure_heads(())


def test_configure_joint_marginal_rejects_a_name_that_is_not_a_joint_dimension() -> None:
    """A typo'd --joint-marginal name must be refused, not silently ignored.

    A silently-ignored name would leave every joint dimension required and the
    run would keep dropping exactly the rows the flag claimed to rescue.
    """
    import multitarget as mt

    try:
        mt.configure_joint_marginal(("not_a_real_dimension",))
    except ValueError as exc:
        assert "not_a_real_dimension" in str(exc)
    else:
        raise AssertionError("an unknown --joint-marginal name must raise")
    finally:
        mt.JOINT_MARGINAL = mt._DEFAULT_JOINT_MARGINAL


def test_configure_joint_marginal_rejects_covering_every_dimension() -> None:
    """Marginalising every joint dimension must raise.

    A row with none of them present would contribute an empty likelihood, i.e.
    the quadrature would integrate over the whole joint rather than condition on
    anything -- a silently meaningless head rather than a loud refusal.
    """
    import multitarget as mt

    try:
        mt.configure_joint_marginal(mt.JOINT_PAIR)
    except ValueError:
        pass
    else:
        raise AssertionError("covering every joint dimension must raise")
    finally:
        mt.JOINT_MARGINAL = mt._DEFAULT_JOINT_MARGINAL


def test_configure_joint_marginal_none_is_a_noop() -> None:
    """`--joint-marginal` omitted (None) must leave JOINT_MARGINAL untouched.

    Catches a None-clobbers-to-something-else bug: since the default already
    equals ("log_flux_p3",), the probe value is deliberately a DIFFERENT
    dimension so a reset-to-default would still be caught.
    """
    import multitarget as mt

    try:
        mt.configure_joint_marginal(("log_sfr",))
        before = mt.JOINT_MARGINAL
        mt.configure_joint_marginal(None)
        assert mt.JOINT_MARGINAL == before
    finally:
        mt.JOINT_MARGINAL = mt._DEFAULT_JOINT_MARGINAL


def test_multi_target_nll_reports_joint_branch_stats() -> None:
    """The `stats` out-param must expose both joint branches separately.

    A pooled joint number mixes densities over different dimensionalities (see
    the comment above the call site in multi_target_nll), so `stats` is the only
    way --select-metric joint_complete can select on the complete branch alone.
    Catches the out-param being silently dropped, only half-filled, or the two
    branch counts not partitioning the usable rows.
    """
    import multitarget as mt

    torch.manual_seed(0)
    B = 12
    flows = _stub_flows()
    targets = torch.randn(B, N_TARGETS)
    jm = mt.target_col(mt.JOINT_MARGINAL[0])
    targets[B // 2:, jm] = float("nan")          # half the rows lack the marginal dim
    contexts = torch.randn(B, N_HEADS, 256)
    stds = [TargetStandardizer(0.0, 1.0) for _ in range(N_TARGETS)]
    stats: dict = {}
    mt.multi_target_nll(
        contexts=contexts, flows=flows, targets=targets,
        sig_lo=torch.zeros(B, N_TARGETS), sig_hi=torch.zeros(B, N_TARGETS),
        standardizers=stds, weights=np.ones(N_HEADS), inject=False, stats=stats,
    )
    assert set(stats) == {"joint_complete_nll", "joint_complete_n",
                          "joint_quadrature_nll", "joint_quadrature_n"}
    have_req, _ = mt.joint_availability(targets)
    assert stats["joint_complete_n"] + stats["joint_quadrature_n"] == int(have_req.sum())
    assert math.isfinite(stats["joint_complete_nll"])
    assert math.isfinite(stats["joint_quadrature_nll"])


def test_train_multi_argparse_wires_the_new_flags() -> None:
    """--select-metric, --joint-marginal, --fixed-combo must reach the parsed
    namespace under the values passed.

    run_train_multi trusts args.select_metric / args.joint_marginal /
    args.fixed_combo directly, so a wiring slip (wrong dest, missing
    add_argument) would silently no-op the flag while the CLI still accepted it.
    """
    import sys

    import main

    argv = ["prog", "train-multi",
            "--staged-dir", "/tmp/staged", "--clean-split-csv", "/tmp/split.csv",
            "--extra-targets-csv", "/tmp/extra.csv", "--run-id", "r1",
            "--select-metric", "joint_complete",
            "--joint-marginal", "log_sfr",
            "--fixed-combo", "spectra+z"]
    old_argv = sys.argv
    sys.argv = argv
    try:
        args = main.parse_args()
    finally:
        sys.argv = old_argv
    assert args.command == "train-multi"
    assert args.select_metric == "joint_complete"
    assert args.joint_marginal == ["log_sfr"]
    assert args.fixed_combo == "spectra+z"


def test_train_multi_argparse_rejects_an_invalid_select_metric() -> None:
    """An unlisted --select-metric value must be refused at parse time.

    Otherwise it would reach run_train_multi and fail -- or worse, silently
    mis-select a checkpoint -- deep into a training run instead of at startup.
    """
    import sys

    import main

    argv = ["prog", "train-multi",
            "--staged-dir", "/tmp/staged", "--clean-split-csv", "/tmp/split.csv",
            "--extra-targets-csv", "/tmp/extra.csv", "--run-id", "r1",
            "--select-metric", "bogus"]
    old_argv = sys.argv
    sys.argv = argv
    try:
        try:
            main.parse_args()
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("an invalid --select-metric must exit, not parse")
    finally:
        sys.argv = old_argv


# ===========================================================================
# A1: declarable joints
# ===========================================================================

def test_configure_joint_declares_the_joint_in_flow_column_order() -> None:
    """--joint must rebind the joint, its column order, and reset the marginal.

    The joint was a module constant, so Run A (M*, SFR, Lx) and Run B
    (a band joint) could not both be expressed by the same code.
    """
    import multitarget as mt

    try:
        mt.configure_joint(("log_flux_p3", "log_flux_p2"))     # deliberately not P2,P3
        mt.configure_heads(("logmstar",))
        assert mt.JOINT_PAIR == ("log_flux_p3", "log_flux_p2")
        assert mt.joint_dims() == ("log_flux_p3", "log_flux_p2")
        # column order is the DECLARATION order, not MULTI_TARGETS order
        assert mt.joint_col("log_flux_p3") == 0 and mt.joint_col("log_flux_p2") == 1
        assert mt.target_col("log_flux_p2") < mt.target_col("log_flux_p3")
        # declaring a joint resets the marginal: the old names belong to the old joint
        assert mt.JOINT_MARGINAL == ()
    finally:
        mt.JOINT_PAIR, mt.JOINT_MARGINAL = mt._DEFAULT_JOINT_PAIR, mt._DEFAULT_JOINT_MARGINAL
        mt.configure_heads(())


def test_configure_joint_refuses_declarations_that_cannot_mean_anything() -> None:
    """Every bad --joint must raise at startup, not produce a strange model."""
    import multitarget as mt

    bad = [
        ("log_lx",),                                   # 1-D "joint" carries no correlation
        ("log_lx", "log_lx"),                          # repeated dimension
        ("log_lx", "definitely_not_a_head"),           # typo
        ("log_lx", "log_rate_p2"),                     # mixes likelihood kinds
    ]
    try:
        for names in bad:
            try:
                mt.configure_joint(names)
            except ValueError:
                pass
            else:
                raise AssertionError(f"--joint {names} must raise")
        assert mt.JOINT_PAIR == mt._DEFAULT_JOINT_PAIR   # and nothing was half-applied
        mt.configure_joint(None)                          # omitted is a no-op
        assert mt.JOINT_PAIR == mt._DEFAULT_JOINT_PAIR
    finally:
        mt.JOINT_PAIR, mt.JOINT_MARGINAL = mt._DEFAULT_JOINT_PAIR, mt._DEFAULT_JOINT_MARGINAL
        mt.configure_heads(())


def test_checkpoint_joint_survives_a_same_arity_edit_to_the_module_constant() -> None:
    """The live hazard: JOINT_PAIR is re-derived at load time.

    A same-arity edit to the module constant silently relabels every joint
    column of every stored checkpoint AND still loads clean under strict=True,
    because the flow shape is unchanged. The checkpoint's own `joint_dims` has
    to win. Simulated here by loading a config whose joint is a PERMUTATION of
    the current default -- same arity, same head set, different meaning.
    """
    import multitarget as mt

    permuted = tuple(reversed(mt._DEFAULT_JOINT_PAIR))
    assert len(permuted) == len(mt._DEFAULT_JOINT_PAIR) and permuted != mt._DEFAULT_JOINT_PAIR
    heads = [t["name"] for t in mt._ALL_TARGETS if t["name"] != "logmstar"] + ["joint"]
    try:
        mt.configure_heads_from_config({
            "heads": heads, "joint_dims": list(permuted),
            "joint_marginal": [mt._DEFAULT_JOINT_MARGINAL[0]],
        })
        assert mt.joint_dims() == permuted, "the stored joint must win over the constant"
        assert mt.JOINT_MARGINAL == (mt._DEFAULT_JOINT_MARGINAL[0],)
        assert "logmstar" not in mt.HEAD_NAMES and mt.HEAD_NAMES[-1] == "joint"
        # ...and a Poisson (opt-in) head listed by a Run B checkpoint comes BACK
        pois_heads = [t["name"] for t in mt.POISSON_TARGETS[2:4]]
        mt.configure_heads_from_config({
            "heads": ["log_lx"] + pois_heads + ["joint"],
            "joint_dims": pois_heads, "joint_marginal": [],
        })
        assert mt.HEAD_NAMES == ["log_lx"] + pois_heads + ["joint"]
        assert mt.joint_dims() == tuple(pois_heads)
    finally:
        mt.JOINT_PAIR, mt.JOINT_MARGINAL = mt._DEFAULT_JOINT_PAIR, mt._DEFAULT_JOINT_MARGINAL
        mt.configure_heads(())


def test_configure_heads_from_config_still_loads_every_pre_joint_dims_checkpoint() -> None:
    """BACKWARD COMPATIBILITY. Checkpoints written before `joint_dims` existed.

    Three eras, all of which must still build the right flows:
      * `p2xp3_joint` in `heads`  -- the 2-D band joint, differently NAMED
      * `heads` but no joint info -- the joint hardcoded at the time, i.e. today's
                                     module default
      * no `heads` at all         -- only `drop_heads`
    """
    import multitarget as mt

    n_all = len(mt._ALL_TARGETS)
    try:
        # era 1: pre-2026-08, 2-D P2xP3 joint, and no log_sfr head yet
        legacy = [t["name"] for t in mt._ALL_TARGETS if t["name"] != "log_sfr"]
        mt.configure_heads_from_config({"heads": legacy + ["p2xp3_joint"]})
        assert mt.JOINT_PAIR == ("log_flux_p2", "log_flux_p3")
        assert mt.joint_dims() == ("log_flux_p2", "log_flux_p3")
        assert mt.HEAD_NAMES[-1] == "p2xp3_joint"
        assert mt.N_TARGETS == n_all - 1
        flows = mt.MultiTargetFlows(context_dim=8)
        assert len(flows.flows) == n_all - 1        # matches the old state_dict
        assert flows.joint.features == 2

        # era 2: `heads` present, no joint_dims -> the module default joint
        mt.configure_heads_from_config({"heads": [t["name"] for t in mt._ALL_TARGETS] + ["joint"]})
        assert mt.JOINT_PAIR == mt._DEFAULT_JOINT_PAIR
        assert mt.JOINT_MARGINAL == mt._DEFAULT_JOINT_MARGINAL
        assert mt.N_TARGETS == n_all

        # era 3: no `heads` at all
        mt.configure_heads_from_config({"drop_heads": ["log_flux_p4"]})
        assert "log_flux_p4" not in mt.HEAD_NAMES and "log_sfr" in mt.HEAD_NAMES
        assert mt.JOINT_PAIR == mt._DEFAULT_JOINT_PAIR

        # and an empty / missing config is still the full current default
        mt.configure_heads_from_config({})
        assert mt.N_TARGETS == n_all
        mt.configure_heads_from_config(None)
        assert mt.N_TARGETS == n_all
    finally:
        mt.JOINT_PAIR, mt.JOINT_MARGINAL = mt._DEFAULT_JOINT_PAIR, mt._DEFAULT_JOINT_MARGINAL
        mt.configure_heads(())


def test_train_multi_argparse_wires_joint_and_add_heads() -> None:
    """--joint and --add-heads must reach the namespace under the values passed."""
    import sys

    import main

    argv = ["prog", "train-multi",
            "--staged-dir", "/tmp/staged", "--clean-split-csv", "/tmp/split.csv",
            "--extra-targets-csv", "/tmp/extra.csv", "--run-id", "r1",
            "--add-heads", "log_rate_p2", "log_rate_p3",
            "--joint", "log_rate_p2,log_rate_p3"]
    old_argv = sys.argv
    sys.argv = argv
    try:
        args = main.parse_args()
    finally:
        sys.argv = old_argv
    assert args.add_heads == ["log_rate_p2", "log_rate_p3"]
    assert args.joint == "log_rate_p2,log_rate_p3"


# ===========================================================================
# B2: the Poisson head
# ===========================================================================

class _GaussianFlow(nn.Module):
    """A learnable N(mu, sigma) with the ConditionalNSFFlow call surface.

    Deliberately not an NSF: these tests are about the LIKELIHOOD and the
    quadrature, and a closed-form density makes "did the posterior land on the
    truth" a statement about the estimator rather than about zuko's optimiser.
    Independent across dimensions, which is what makes the joint-vs-product
    check below a real check.
    """

    def __init__(self, features: int = 1, mu: float = 0.0, log_sigma: float = 0.0,
                 ctx_weight: float = 0.0) -> None:
        super().__init__()
        self.features = features
        self.mu = nn.Parameter(torch.full((features,), float(mu)))
        self.log_sigma = nn.Parameter(torch.full((features,), float(log_sigma)))
        # 0 keeps the density closed-form and context-free, which is what the
        # numerical checks want; the plumbing checks set it nonzero so gradient
        # actually reaches the shared context.
        self.ctx_weight = float(ctx_weight)

    def _lp(self, x, context):              # x: [..., m, D]; context: [m, ctx]
        sigma = self.log_sigma.exp()
        centre = self.mu
        if self.ctx_weight:
            centre = centre + self.ctx_weight * context.mean(dim=-1).unsqueeze(-1)
        z = (x - centre) / sigma
        return (-0.5 * z**2 - self.log_sigma - 0.5 * float(np.log(2 * np.pi))).sum(dim=-1)

    def log_prob_draws(self, y_draws, context):
        x = y_draws.unsqueeze(-1) if y_draws.dim() == 2 else y_draws
        return self._lp(x, context)

    def log_prob(self, y, context):
        return self._lp(y.unsqueeze(-1) if y.dim() == 1 else y, context)

    def sample(self, context, num_samples):
        eps = torch.randn(num_samples, context.shape[0], self.features)
        return (self.mu + self.log_sigma.exp() * eps).squeeze(-1)


def _pois_1d(counts, bkg, exposure):
    """(counts, bkg, exposure) as [m, 1] float tensors."""
    return tuple(torch.tensor(np.asarray(a, dtype=np.float64), dtype=torch.float32).view(-1, 1)
                 for a in (counts, bkg, exposure))


def test_poisson_band_logprob_matches_scipy_and_handles_zero_counts() -> None:
    """The pmf itself, against an independent implementation, N = 0 included.

    N = 0 is the regime the whole move to counts was made for, so it is checked
    as ordinary data -- same code path, same call, no branch.
    """
    import multitarget as mt
    from scipy.stats import poisson

    std = TargetStandardizer(-2.0, 0.5)
    u = torch.tensor([[-3.0], [-1.0], [0.0], [1.0], [3.0]])
    counts = torch.tensor([[0.0, 1.0, 7.0, 250.0]])
    bkg = torch.tensor([[0.0, 0.3, 2.5, 40.0]])
    exposure = torch.tensor([[300.0, 300.0, 1200.0, 300.0]])
    got = mt.poisson_band_logprob(u=u, counts=counts, bkg=bkg, exposure=exposure,
                                  standardizer=std).numpy()
    lam = 10.0 ** (std.mean + std.std * u.numpy())
    want = poisson.logpmf(counts.numpy(), lam * exposure.numpy() + bkg.numpy())
    assert np.isfinite(got).all()
    assert np.abs(got - want).max() < 2e-3, np.abs(got - want).max()


def test_poisson_zero_counts_gives_an_upper_limit_not_a_nan() -> None:
    """N = 0 must produce a finite, upper-limit-SHAPED posterior.

    Upper-limit-shaped means three things, all asserted: the marginal likelihood
    is finite (nothing NaNs), the posterior is pushed DOWN relative to the prior
    rather than merely widened, and a deeper non-detection is a TIGHTER limit --
    the 84th percentile of log10 lambda must fall as exposure rises. A posterior
    that ignored exposure would pass the first two and fail the third.
    """
    import multitarget as mt

    std = TargetStandardizer(-2.0, 0.6)
    flow = _GaussianFlow()
    exposures = np.array([100.0, 1000.0, 10000.0])
    counts, bkg, exp = _pois_1d(np.zeros(3), np.zeros(3), exposures)
    ctx = torch.zeros(3, 4)
    nll = mt.poisson_marginal_nll(flow=flow, context=ctx, standardizers=[std],
                                  counts=counts, bkg=bkg, exposure=exp)
    assert torch.isfinite(nll).all(), nll
    assert (nll > 0).all()          # -log P(N=0) is a probability, so positive

    with torch.no_grad():
        u_grid, logw, _ = mt.poisson_log_integrand(
            flow=flow, context=ctx, standardizers=[std],
            counts=counts, bkg=bkg, exposure=exp)
    w = torch.softmax(logw, dim=0).numpy()          # [G, 3] normalized posterior
    u = u_grid[:, :, 0].numpy()                     # per-source nodes
    assert np.isfinite(w).all()
    q84 = [u[np.searchsorted(np.cumsum(w[:, i]), 0.84), i] for i in range(3)]
    prior84 = 0.9944                                 # N(0,1) 84th percentile
    assert all(q < prior84 for q in q84), q84        # pushed down, not just widened
    assert q84[0] > q84[1] > q84[2], q84             # deeper non-detection, tighter limit
    # and the posterior has no upper shoulder: mass piles up at the faint end
    assert w[:, 2].argmax() < len(u) // 2


def _brute_force_marginal_nll(counts, bkg, exposure, std, mu, sigma,
                              lo=-25.0, hi=25.0, n=400001):
    """-log INT q(u) p(N|u) du by a huge uniform numpy grid. Independent of
    multitarget: numpy + scipy only, no adaptive placement, no torch."""
    from scipy.special import logsumexp
    from scipy.stats import norm, poisson

    u = np.linspace(lo, hi, n)
    du = u[1] - u[0]
    out = []
    for N, B, t_ in zip(np.atleast_1d(counts), np.atleast_1d(bkg), np.atleast_1d(exposure)):
        lam = 10.0 ** (std.mean + std.std * u)
        lw = norm.logpdf(u, mu, sigma) + poisson.logpmf(N, lam * t_ + B)
        out.append(-(logsumexp(lw) + np.log(du)))
    return np.array(out)


def test_poisson_quadrature_agrees_with_a_brute_force_integral() -> None:
    """The 48-node quadrature must reproduce a 400,001-node brute force.

    The reference is written from scratch in numpy/scipy in this file -- not the
    same code with a bigger K -- because the failure this catches is in the NODE
    PLACEMENT, and a self-comparison cannot see a systematically misplaced grid.

    Checked across regimes whose integrands look nothing alike: zero counts (a
    flat likelihood, prior-dominated), one count, a moderate source, a bright one
    whose likelihood peak is ~8x narrower than a FIXED grid's node spacing, and a
    background-dominated one. The bright source is the whole point: on a fixed
    grid it was wrong by nats.
    """
    import multitarget as mt

    std = TargetStandardizer(-2.0, 0.6)
    mu, sigma = 0.15, 0.9
    flow = _GaussianFlow(mu=mu, log_sigma=float(np.log(sigma)))
    N = np.array([0.0, 1.0, 12.0, 800.0, 3.0, 9500.0])
    B = np.array([0.0, 0.4, 3.0, 20.0, 25.0, 60.0])
    T = np.array([150.0, 400.0, 1200.0, 300.0, 900.0, 2000.0])
    counts, bkg, exp = _pois_1d(N, B, T)
    with torch.no_grad():
        got = mt.poisson_marginal_nll(
            flow=flow, context=torch.zeros(len(N), 4), standardizers=[std],
            counts=counts, bkg=bkg, exposure=exp).numpy()
    want = _brute_force_marginal_nll(N, B, T, std, mu, sigma)
    err = np.abs(got - want)
    assert err.max() < 2e-2, dict(zip(N.tolist(), err.tolist()))


def test_a_fixed_grid_would_miss_a_bright_source_which_is_why_it_is_adaptive() -> None:
    """The regression this exists to prevent, asserted as a MEASUREMENT.

    Places the same number of nodes on the old FIXED +/-5 sigma grid and shows it
    is wrong by more than a nat on a bright source, while the adaptive placement
    is right. If someone reverts poisson_quad_proposal to a constant, this fails
    with the number instead of the loss curve quietly moving.
    """
    import multitarget as mt

    std = TargetStandardizer(-2.0, 0.6)
    mu, sigma = 0.15, 0.9
    flow = _GaussianFlow(mu=mu, log_sigma=float(np.log(sigma)))
    N, B, T = np.array([800.0]), np.array([20.0]), np.array([300.0])
    counts, bkg, exp = _pois_1d(N, B, T)
    ctx = torch.zeros(1, 4)
    want = _brute_force_marginal_nll(N, B, T, std, mu, sigma)[0]

    with torch.no_grad():
        adaptive = float(mt.poisson_marginal_nll(
            flow=flow, context=ctx, standardizers=[std],
            counts=counts, bkg=bkg, exposure=exp)[0])
        nodes, dv = mt._quad_nodes(torch.device("cpu"), torch.float32)
        lp = mt.poisson_band_logprob(u=nodes.view(-1, 1), counts=counts.view(1, -1),
                                     bkg=bkg.view(1, -1), exposure=exp.view(1, -1),
                                     standardizer=std)
        lq = flow.log_prob_draws(nodes.view(-1, 1).expand(-1, 1), ctx)
        fixed = float(-(torch.logsumexp(lq + lp, dim=0) + float(np.log(dv)))[0])

    assert abs(adaptive - want) < 2e-2, (adaptive, want)
    assert abs(fixed - want) > 1.0, (fixed, want)


def test_poisson_joint_factorises_when_the_flow_does() -> None:
    """A 2-band joint under an INDEPENDENT prior must equal the sum of two 1-D
    marginals, exactly.

    This is the only closed-form check available on the 2-D quadrature: the
    Poisson factorises across bands, so the bands are coupled only through
    p(lambda|x). Make that independent and the marginal likelihood must
    separate. If the grid, the Jacobian bookkeeping (D * log du) or the column
    order were wrong, this is where it shows.
    """
    import multitarget as mt

    stds = [TargetStandardizer(-2.0, 0.5), TargetStandardizer(-2.3, 0.7)]
    joint = _GaussianFlow(features=2, mu=0.1, log_sigma=0.05)
    single = _GaussianFlow(features=1, mu=0.1, log_sigma=0.05)
    counts = torch.tensor([[0.0, 3.0], [11.0, 0.0], [420.0, 260.0]])
    bkg = torch.tensor([[0.0, 1.2], [2.0, 0.0], [15.0, 9.0]])
    exp = torch.tensor([[300.0, 300.0], [900.0, 900.0], [250.0, 250.0]])
    ctx = torch.zeros(3, 4)
    both = mt.poisson_marginal_nll(flow=joint, context=ctx, standardizers=stds,
                                   counts=counts, bkg=bkg, exposure=exp)
    parts = [mt.poisson_marginal_nll(flow=single, context=ctx, standardizers=[stds[d]],
                                     counts=counts[:, d:d+1], bkg=bkg[:, d:d+1],
                                     exposure=exp[:, d:d+1]) for d in (0, 1)]
    assert torch.allclose(both, parts[0] + parts[1], atol=2e-3), (both, parts)

    # and an absent band drops its factor: the joint then equals the OTHER band
    # alone, because the missing dimension is simply integrated out of the prior.
    present = torch.tensor([[True, False], [True, False], [True, False]])
    dropped = mt.poisson_marginal_nll(flow=joint, context=ctx, standardizers=stds,
                                      counts=counts, bkg=bkg, exposure=exp, present=present)
    assert torch.allclose(dropped, parts[0], atol=2e-3), (dropped, parts[0])


def test_poisson_recovers_a_known_rate() -> None:
    """Simulate counts from a known lambda, fit, and the posterior lands on it.

    The generative truth is a Gaussian in log10 lambda with a known mean and
    width; the fit sees only (N, B, t) through the Poisson marginal likelihood
    and must recover BOTH -- recovering the mean alone would be consistent with
    a posterior that had collapsed or blown up.
    """
    import multitarget as mt

    torch.manual_seed(0)
    rng = np.random.default_rng(11)
    n = 4000
    true_mean, true_sd = -1.6, 0.35                 # log10 ct/s
    log_lam = rng.normal(true_mean, true_sd, n)
    exposure = rng.uniform(200.0, 1500.0, n)
    bkg = rng.uniform(0.0, 3.0, n)
    counts = rng.poisson(10.0**log_lam * exposure + bkg).astype(float)
    assert (counts == 0).sum() > 0, "the test sample must contain real zeros"

    std = TargetStandardizer(-1.0, 1.0)             # deliberately WRONG units
    flow = _GaussianFlow(mu=0.0, log_sigma=0.0)
    c, b, t = _pois_1d(counts, bkg, exposure)
    ctx = torch.zeros(n, 4)
    opt = torch.optim.Adam(flow.parameters(), lr=0.05)
    for _ in range(400):
        opt.zero_grad()
        loss = mt.poisson_marginal_nll(flow=flow, context=ctx, standardizers=[std],
                                       counts=c, bkg=b, exposure=t).mean()
        loss.backward()
        opt.step()
    got_mean = std.mean + std.std * float(flow.mu.item())
    got_sd = std.std * float(flow.log_sigma.exp().item())
    assert abs(got_mean - true_mean) < 0.03, (got_mean, true_mean)
    assert abs(got_sd - true_sd) < 0.05, (got_sd, true_sd)


def test_poisson_loss_ignores_the_standardization_anchor_entirely() -> None:
    """Every nat must come from (N, B, t). The anchor is units, not a label.

    `targets[:, j]` for a Poisson head is a plug-in log10 rate used to fit the
    standardizer and to mark availability. If it ever leaked into the
    likelihood, the head would be quietly fitting a floored, background-
    subtracted point estimate instead of the counts -- which is exactly the
    thing counts were adopted to stop doing. Perturbing it must change nothing.
    """
    import multitarget as mt

    torch.manual_seed(0)
    try:
        bands = ("log_rate_p2", "log_rate_p3")
        mt.configure_joint(bands)
        mt.configure_heads(tuple(t["name"] for t in mt._ALL_TARGETS), bands)
        assert mt.HEAD_NAMES == list(bands) + ["joint"]
        n_t, n_h = mt.N_TARGETS, mt.N_HEADS

        B = 16
        flows = mt.MultiTargetFlows.__new__(mt.MultiTargetFlows)
        nn.Module.__init__(flows)
        flows.flows = nn.ModuleList(_GaussianFlow() for _ in range(n_t))
        flows.joint = _GaussianFlow(features=2)
        stds = [TargetStandardizer(-2.0, 0.5) for _ in range(n_t)]
        rng = np.random.default_rng(3)
        pois = torch.zeros(B, n_t, 3)
        pois[:, :, 0] = torch.tensor(rng.poisson(4.0, (B, n_t)).astype(np.float32))
        pois[:, :, 1] = torch.tensor(rng.uniform(0, 2, (B, n_t)).astype(np.float32))
        pois[:, :, 2] = torch.tensor(rng.uniform(200, 900, (B, n_t)).astype(np.float32))
        anchor = torch.zeros(B, n_t)
        ctx = torch.randn(B, n_h, 256)

        args = dict(contexts=ctx, flows=flows, sig_lo=torch.zeros(B, n_t),
                    sig_hi=torch.zeros(B, n_t), standardizers=stds,
                    weights=np.ones(n_h), inject=False, pois=pois)
        _, raw_a = mt.multi_target_nll(targets=anchor, **args)
        _, raw_b = mt.multi_target_nll(targets=anchor + 7.5, **args)
        assert all(r is not None for r in raw_a)
        for a, b_ in zip(raw_a, raw_b):
            assert abs(a - b_) < 1e-6, (a, b_)
    finally:
        mt.JOINT_PAIR, mt.JOINT_MARGINAL = mt._DEFAULT_JOINT_PAIR, mt._DEFAULT_JOINT_MARGINAL
        mt.configure_heads(())


def test_poisson_head_needs_its_counts_and_says_so() -> None:
    """multi_target_nll without `pois` must raise, not score the anchor.

    A silently-scored anchor is indistinguishable from a working head in a loss
    curve, which is the same failure mode as the all-NaN ghost head.
    """
    import multitarget as mt

    try:
        bands = ("log_rate_p2", "log_rate_p3")
        mt.configure_joint(bands)
        mt.configure_heads(tuple(t["name"] for t in mt._ALL_TARGETS), bands)
        n_t, n_h = mt.N_TARGETS, mt.N_HEADS
        flows = mt.MultiTargetFlows.__new__(mt.MultiTargetFlows)
        nn.Module.__init__(flows)
        flows.flows = nn.ModuleList(_GaussianFlow() for _ in range(n_t))
        flows.joint = _GaussianFlow(features=2)
        try:
            mt.multi_target_nll(
                contexts=torch.randn(4, n_h, 256), flows=flows,
                targets=torch.zeros(4, n_t), sig_lo=torch.zeros(4, n_t),
                sig_hi=torch.zeros(4, n_t),
                standardizers=[TargetStandardizer(0.0, 1.0)] * n_t,
                weights=np.ones(n_h), inject=False)
        except ValueError as exc:
            assert "log_rate_p2" in str(exc) and "pois" in str(exc)
        else:
            raise AssertionError("a Poisson head with no counts must raise")
    finally:
        mt.JOINT_PAIR, mt.JOINT_MARGINAL = mt._DEFAULT_JOINT_PAIR, mt._DEFAULT_JOINT_MARGINAL
        mt.configure_heads(())


def test_poisson_joint_trains_through_multi_target_nll() -> None:
    """The band-joint path end to end: both branches, finite, and gradient flows."""
    import multitarget as mt

    torch.manual_seed(0)
    try:
        bands = ("log_rate_p2", "log_rate_p3")
        mt.configure_joint(bands)
        mt.configure_heads(tuple(t["name"] for t in mt._ALL_TARGETS), bands)
        mt.configure_joint_marginal(("log_rate_p3",))
        n_t, n_h = mt.N_TARGETS, mt.N_HEADS

        B = 12
        flows = mt.MultiTargetFlows.__new__(mt.MultiTargetFlows)
        nn.Module.__init__(flows)
        flows.flows = nn.ModuleList(_GaussianFlow(ctx_weight=0.3) for _ in range(n_t))
        flows.joint = _GaussianFlow(features=2, ctx_weight=0.3)
        stds = [TargetStandardizer(-2.0, 0.5) for _ in range(n_t)]
        rng = np.random.default_rng(5)
        pois = torch.zeros(B, n_t, 3)
        pois[:, :, 0] = torch.tensor(rng.poisson(3.0, (B, n_t)).astype(np.float32))
        pois[:, :, 2] = torch.tensor(rng.uniform(200, 900, (B, n_t)).astype(np.float32))
        anchor = torch.zeros(B, n_t)
        p3 = mt.target_col("log_rate_p3")
        anchor[B // 2:, p3] = float("nan")             # half the rows lack the marginal band
        pois[B // 2:, p3, :] = float("nan")
        ctx = torch.randn(B, n_h, 256, requires_grad=True)
        stats: dict = {}
        total, raw = mt.multi_target_nll(
            contexts=ctx, flows=flows, targets=anchor, sig_lo=torch.zeros(B, n_t),
            sig_hi=torch.zeros(B, n_t), standardizers=stds, weights=np.ones(n_h),
            inject=False, stats=stats, pois=pois)
        assert raw[-1] is not None and np.isfinite(raw[-1])
        assert stats["joint_complete_n"] + stats["joint_quadrature_n"] == B
        assert stats["joint_complete_n"] == B // 2
        total.backward()
        assert ctx.grad[:, n_t].abs().sum() > 0        # the joint context got gradient
        assert torch.isfinite(ctx.grad).all()
    finally:
        mt.JOINT_PAIR, mt.JOINT_MARGINAL = mt._DEFAULT_JOINT_PAIR, mt._DEFAULT_JOINT_MARGINAL
        mt.configure_heads(())


def test_negative_counts_are_dropped_and_negative_background_is_clipped(tmp_path) -> None:
    """The two measured catalogue hazards, decided at load time.

    APE_CTS is int16 and WRAPS in the parent catalogue (20 band-1 rows already
    negative); APE_BKG goes slightly negative on 2. A wrapped count is not a
    count and must not be clipped to zero -- that would turn the very brightest
    sources into non-detections. A negative background IS clipped: it is an
    estimate of a non-negative nuisance, not a datum. Zero exposure is dropped:
    the likelihood then does not contain lambda at all.
    """
    import h5py
    import pandas as pd
    import multitarget as mt

    staged = tmp_path / "desi_train.hdf5"
    tids = np.arange(5, dtype=np.int64)
    with h5py.File(staged, "w") as h:
        h.create_dataset("desi_targetid", data=tids)
    side = pd.DataFrame({
        "targetid": tids,
        "ape_cts_p2": [7, -32000, 3, 0, 12],       # row 1 wrapped
        "ape_bkg_p2": [0.5, 0.5, -0.4, 0.0, 1.0],  # row 2 negative background
        "ape_exp_p2": [300.0, 300.0, 300.0, 300.0, 0.0],   # row 4 zero exposure
        "ape_cts_p3": [4, 4, 4, 4, 4],                     # the other band is clean
        "ape_bkg_p3": [0.2, 0.2, 0.2, 0.2, 0.2],
        "ape_exp_p3": [300.0, 300.0, 300.0, 300.0, 300.0],
    })
    csv = tmp_path / "sidecar.csv"
    side.to_csv(csv, index=False)

    try:
        bands = ("log_rate_p2", "log_rate_p3")
        mt.configure_joint(bands)
        mt.configure_heads(tuple(t["name"] for t in mt._ALL_TARGETS), bands)
        y, _, _, pois = mt.load_multi_target_matrix(staged, csv)
        keep = np.isfinite(y[:, 0])
        assert keep.tolist() == [True, False, True, True, False], keep
        assert np.isfinite(pois[:, 0, 0]).tolist() == keep.tolist()
        assert pois[2, 0, 1] == 0.0, "a negative background must be clipped to zero"
        assert pois[0, 0, 1] == 0.5                       # and a good one untouched
        assert pois[3, 0, 0] == 0.0                       # N = 0 is kept: ordinary data
        assert np.isfinite(y[3, 0])
    finally:
        mt.JOINT_PAIR, mt.JOINT_MARGINAL = mt._DEFAULT_JOINT_PAIR, mt._DEFAULT_JOINT_MARGINAL
        mt.configure_heads(())


def test_exposure_and_background_never_reach_the_model_inputs() -> None:
    """The (N, B, t) columns must appear nowhere in the input pipeline.

    Same principle, and the same failure mode, as the standing ban on
    conditioning on per-source sigma: exposure and background are per-source and
    KNOWN, so feeding them lets the model infer how well measured a source is
    instead of what it is. They sit in the same sidecar table as the counts, so
    the temptation is structural and an assertion is cheaper than vigilance.
    """
    import multitarget as mt

    cols = {c for spec in mt.POISSON_TARGETS for c in spec["pois"]}
    assert cols, "POISSON_TARGETS must declare their (counts, bkg, exposure) columns"
    for name in ("data_to_aion_embeddings.py", "attention_pooling_head.py", "stub_encoder.py"):
        text = (ROOT / name).read_text()
        found = sorted(c for c in cols if c in text)
        assert not found, f"{name} mentions likelihood-only columns {found}"


def test_eval_core_refuses_a_poisson_checkpoint_rather_than_scoring_the_anchor() -> None:
    """eval scores a density AT a value. A latent-rate head has no value.

    Left unguarded, pointing eval_multitarget.py at a Run B checkpoint produces a
    complete metrics table computed on the standardization anchor: plausible
    numbers, wrong quantity, no error. Refusal is the only safe default until
    eval learns the count marginal likelihood.
    """
    import multitarget as mt
    import pytest
    import eval_core

    try:
        bands = ("log_rate_p2", "log_rate_p3")
        mt.configure_joint(bands)
        mt.configure_heads(tuple(t["name"] for t in mt._ALL_TARGETS), bands)
        with pytest.raises(SystemExit, match="log_rate_p2"):
            eval_core.assert_no_poisson_heads()
    finally:
        mt.JOINT_PAIR, mt.JOINT_MARGINAL = mt._DEFAULT_JOINT_PAIR, mt._DEFAULT_JOINT_MARGINAL
        mt.configure_heads(())
    eval_core.assert_no_poisson_heads()          # the ordinary head set is fine
