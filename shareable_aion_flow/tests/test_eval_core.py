"""Tests for the evaluation path: sample declaration, the joint block, HR by name.

These exist because the eval path had no tests at all, which is how
``j2, j3 = JOINT_IDX`` -- a two-element unpack of a four-element tuple -- shipped
a full run cycle, and how the SFR-vs-mass guard sat keyed on a head name that is
dropped from every run and therefore never fired once.

Everything here runs on CPU with no ``aion`` installed, no checkpoint and no
GPU: a tiny staged fixture, a deterministic stub ``encode``, and small real zuko
flows so the flow API is exercised rather than mocked away.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import shareable_aion_flow.multitarget as mt  # noqa: E402
from shareable_aion_flow import eval_core  # noqa: E402
from shareable_aion_flow.data_to_aion_embeddings import build_dataloaders  # noqa: E402
from shareable_aion_flow.normalizing_flow import ConditionalNSFFlow, TargetStandardizer  # noqa: E402

CTX = 8
N_PIX = 16
N_TRAIN, N_VAL, N_TEST = 140, 10, 90


# ------------------------------------------------------------------ fixture
def _write_split(path: Path, targetids: np.ndarray, blank_image: np.ndarray, rng) -> None:
    n = len(targetids)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("desi_targetid", data=targetids.astype(np.int64))
        handle.create_dataset("spectra", data=rng.normal(size=(n, N_PIX)).astype(np.float32))
        handle.create_dataset("spectra_ivar", data=np.ones((n, N_PIX), dtype=np.float32))
        handle.create_dataset("spectra_lambda",
                              data=np.linspace(3600, 9800, N_PIX).astype(np.float32))
        handle.create_dataset("redshift", data=rng.uniform(0.1, 2.0, n).astype(np.float32))
        for band in ("flux_w1", "flux_w2", "flux_w3"):
            handle.create_dataset(band, data=rng.uniform(1.0, 5.0, n).astype(np.float32))
        images = rng.uniform(0.1, 1.0, (n, 4, 8, 8)).astype(np.float32)
        images[blank_image] = 0.0            # staged without a Legacy Survey cutout
        handle.create_dataset("image_flux", data=images)
        handle.create_dataset("log_ml_flux_1",
                              data=rng.normal(-13.0, 0.4, n).astype(np.float32))
        handle.create_dataset("flux_sig_lo", data=np.full(n, 0.12, dtype=np.float32))
        handle.create_dataset("flux_sig_hi", data=np.full(n, 0.13, dtype=np.float32))


def _sidecar_frame(targetids: np.ndarray, rng) -> pd.DataFrame:
    """One row per targetid carrying every column MULTI_TARGETS reads.

    Built from the spec list rather than a hand-written column list, so a new
    sidecar target does not silently get all-NaN labels here and pass.
    """
    n = len(targetids)
    frame = pd.DataFrame({"targetid": targetids.astype(np.int64)})
    centres = {"log_ml_flux_1": -13.0, "log_lx": 44.0, "log_flux_p1": -13.4,
               "log_flux_p2": -13.2, "log_flux_p3": -13.1, "log_flux_p4": -13.6,
               "log_sfr": 1.0, "logmstar_cigale": 10.8,
               "log_mbh_pan25": 8.2, "log_mbh_vo09": 8.1}
    for spec in mt._ALL_TARGETS:
        if not spec["sidecar"]:
            continue                       # logmstar reads from the staged file, which has none
        name = spec["name"]
        frame[name] = rng.normal(centres[name], 0.5, n)
        if spec["sig"]:
            for col in spec["sig"]:
                if col not in frame:
                    frame[col] = np.full(n, 0.12)
    # log_sfr is correlated with the CIGALE mass through the main sequence, which
    # is exactly what the SFR-vs-mass baseline has to be able to beat.
    frame["log_sfr"] = 0.8 * (frame["logmstar_cigale"] - 10.8) + rng.normal(1.0, 0.3, n)
    frame["det_like_0"] = 20.0
    for band in ("p1", "p2", "p3", "p4"):
        frame[f"det_like_{band}"] = 20.0
    # P1 undetected for half the sample: heads legitimately have different n_test,
    # and the table has to say so per row rather than imply one shared denominator.
    frame.loc[frame.index % 2 == 0, "det_like_p1"] = mt.DET_LIKE_MIN - 1.0
    return frame


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    """(staged_dir, clean_split_csv, sidecar_csv) for a 240-source toy sample."""
    rng = np.random.default_rng(7)
    root = tmp_path_factory.mktemp("staged")
    ids = {"train": np.arange(1, N_TRAIN + 1),
           "val": np.arange(1001, 1001 + N_VAL),
           "test": np.arange(2001, 2001 + N_TEST)}
    for split, tids in ids.items():
        blank = np.zeros(len(tids), dtype=bool)
        blank[::5] = True                  # 20% of every split has no cutout
        _write_split(root / f"desi_{split}.hdf5", tids, blank, rng)
    all_ids = np.concatenate([ids[s] for s in ("train", "val", "test")])
    sidecar = root / "sidecar.csv"
    _sidecar_frame(all_ids, rng).to_csv(sidecar, index=False)
    split_csv = root / "clean_split.csv"
    pd.DataFrame({"targetid": all_ids,
                  "split": sum(([s] * len(ids[s]) for s in ("train", "val", "test")), [])}
                 ).to_csv(split_csv, index=False)
    return root, split_csv, sidecar


#: Heads a real run must drop, and therefore so must a fixture. `logmstar` is
#: the FastSpecFit mass: it is `sidecar: False`, so it reads from the staged
#: HDF5, and the rebuilt staging is inputs-only and no longer carries it.
#: load_multi_target_matrix now REFUSES a head with no column in either source
#: rather than handing it an all-NaN target that posts no loss and reads as
#: instant convergence, so a fixture that keeps it tests a configuration no run
#: can actually use.
FIXTURE_DROP_HEADS = ("logmstar",)


@pytest.fixture(autouse=True)
def _drop_unstaged_heads():
    mt.configure_heads(FIXTURE_DROP_HEADS)
    yield
    mt.configure_heads(FIXTURE_DROP_HEADS)


@pytest.fixture
def restore_heads():
    yield
    mt.JOINT_PAIR, mt.JOINT_MARGINAL = mt._DEFAULT_JOINT_PAIR, mt._DEFAULT_JOINT_MARGINAL
    mt.configure_heads(FIXTURE_DROP_HEADS)


# ------------------------------------------------------------------ stubs
def stub_encode(batch, combo):
    """Deterministic contexts [B, N_HEADS, CTX]. No AION, no weights, no GPU.

    Reads N_HEADS from the module at CALL time for the same reason eval_core
    does: the head set is rebound after the checkpoint is read.
    """
    z = batch[3].reshape(-1, 1, 1).to(torch.float32)
    heads = torch.arange(mt.N_HEADS, dtype=torch.float32).view(1, -1, 1)
    feats = torch.arange(CTX, dtype=torch.float32).view(1, 1, -1)
    return torch.sin(z * (1.0 + feats) + 0.37 * heads + float(len(combo)))


def small_flows(n_targets: int | None = None, n_joint: int | None = None):
    """Real zuko flows, small enough to run 15 combos in a unit test."""
    n_targets = mt.N_TARGETS if n_targets is None else n_targets
    n_joint = len(mt.joint_dims()) if n_joint is None else n_joint
    flows = mt.MultiTargetFlows.__new__(mt.MultiTargetFlows)
    nn.Module.__init__(flows)
    flows.flows = nn.ModuleList(
        ConditionalNSFFlow(context_dim=CTX, transforms=2, hidden_features=(16,))
        for _ in range(n_targets))
    flows.joint = ConditionalNSFFlow(context_dim=CTX, transforms=2, hidden_features=(16,),
                                     features=max(2, n_joint))
    return flows


class _FixedDist:
    """A joint distribution whose draws sit at a known value in each column."""

    def __init__(self, fills: list[float], n: int) -> None:
        self.fills, self.n = fills, n

    def log_prob(self, x):
        return torch.zeros(x.shape[:-1])

    def sample(self, shape):
        base = torch.tensor(self.fills, dtype=torch.float32)
        jitter = torch.linspace(-0.05, 0.05, int(shape[0])).view(-1, 1, 1)
        return base.view(1, 1, -1).expand(int(shape[0]), self.n, len(self.fills)) + jitter


class _FixedJointFlow(nn.Module):
    def __init__(self, fills: list[float]) -> None:
        super().__init__()
        self.fills = fills
        self.features = len(fills)

    def distribution(self, context):
        return _FixedDist(self.fills, context.shape[0])

    def log_prob_draws(self, x, context):
        return torch.zeros(x.shape[:-1])


class _FixedFlow(nn.Module):
    """A 1-D marginal whose draws sit at a known value, with a per-band spread.

    The two bands need DIFFERENT spreads or their difference is deterministic and
    the independent-band hardness posterior has zero width, which would let a
    baseline that is not being computed at all pass for one that is.
    """

    def __init__(self, value: float, spread: float = 0.05) -> None:
        super().__init__()
        self.value, self.spread, self.features = value, spread, 1

    def log_prob(self, y, context):
        return torch.zeros(y.shape[0])

    def sample(self, context, num_samples: int):
        jitter = (torch.linspace(-1.0, 1.0, int(num_samples)) * self.spread).view(-1, 1)
        return torch.full((int(num_samples), context.shape[0]), self.value) + jitter


def fixed_flows(fills: dict[str, tuple[float, float]], joint_fills: list[float]):
    flows = mt.MultiTargetFlows.__new__(mt.MultiTargetFlows)
    nn.Module.__init__(flows)
    flows.flows = nn.ModuleList(
        _FixedFlow(*fills.get(spec["name"], (0.0, 0.05))) for spec in mt.MULTI_TARGETS)
    flows.joint = _FixedJointFlow(joint_fills)
    return flows


def make_pieces(staged, *, flows=None, batch_size=N_TEST):
    root, split_csv, sidecar = staged
    _, _, test_loader = build_dataloaders(
        staged_dir=root, target_name="log_ml_flux_1", batch_size=batch_size,
        eval_batch_size=batch_size, num_workers=0, clean_split_csv=split_csv)
    lookup = mt.MultiTargetLookup(root, sidecar)
    train_tids = pd.read_csv(split_csv).query("split == 'train'").targetid.to_numpy(np.int64)
    train_y = lookup.values_for(train_tids)
    stds = []
    for j in range(mt.N_TARGETS):
        vals = train_y[:, j][np.isfinite(train_y[:, j])]
        stds.append(TargetStandardizer.fit(vals) if vals.size > 1 else TargetStandardizer(0.0, 1.0))
    return test_loader, lookup, train_y, stds, (flows if flows is not None else small_flows())


# ------------------------------------------------------------------ tests
def test_the_positional_joint_unpack_this_module_exists_to_delete() -> None:
    """`j2, j3 = JOINT_IDX` on the default joint. Pinned so it cannot come back."""
    if len(mt.JOINT_IDX) == 2:
        pytest.skip("joint is 2-D; the unpack is accidentally valid and proves nothing")
    with pytest.raises(ValueError):
        _j2, _j3 = mt.JOINT_IDX
    # name resolution answers the same question without the assumption
    assert mt.joint_col(mt.joint_dims()[0]) == 0


def test_every_head_name_literal_in_eval_exists_in_all_targets() -> None:
    """A head-name literal that no longer names a head must break a test.

    "logmstar" in the SFR-vs-mass guard did not: it just made the branch
    unreachable, for four months, without a single warning.
    """
    # Non-head strings in the eval path that happen to be spelled like one.
    # Adding to this set is deliberate: it forces a human to look at the literal.
    non_heads = {"log_post", "log_prior"}
    all_names = {t["name"] for t in mt._ALL_TARGETS}
    looks_like_a_head = re.compile(r"log[a-z0-9_]*")
    files = [REPO_ROOT / "shareable_aion_flow" / "eval_core.py",
             REPO_ROOT / "scripts" / "eval_multitarget.py",
             REPO_ROOT / "scripts" / "hr_from_joint.py"]
    for path in files:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            text = node.value
            if not looks_like_a_head.fullmatch(text) or text in non_heads:
                continue
            assert text in all_names, (
                f"{path.name}:{node.lineno} refers to head {text!r}, which is not in "
                f"_ALL_TARGETS ({sorted(all_names)})")
    # and the constants eval resolves by name are real heads
    for name in eval_core.HR_BANDS + eval_core.MASS_HEADS + (eval_core.SFR_HEAD,):
        assert name in all_names


def test_joint_dims_must_match_the_flow_width() -> None:
    flows = small_flows()
    eval_core.assert_joint_matches_flow(flows)          # matched: silent
    flows.joint.features = len(mt.joint_dims()) + 1
    with pytest.raises(SystemExit):
        eval_core.assert_joint_matches_flow(flows)


def test_joint_block_runs_on_a_4d_joint(staged) -> None:
    """The default joint is 4-D and is scored by name, in flow column order."""
    loader, lookup, train_y, stds, flows = make_pieces(staged)
    result = eval_core.run_eval(
        loader=loader, lookup=lookup, encode=stub_encode, flows=flows, standardizers=stds,
        train_y=train_y, combos=[("spectra", "z")], num_samples=4, joint_samples=4, log=lambda m: None)
    joint = [r for r in result.table if r["head"] == mt.HEAD_NAMES[-1]]
    assert len(joint) == 1
    row = joint[0]
    assert row["head"] == "joint" and row["joint_dims"] == "+".join(mt.joint_dims())
    assert np.isfinite(row["nll"])
    # The two joint populations are reported separately, never summed.
    assert row["n_joint_full"] > 0
    assert row["n_joint_full"] + row["n_joint_marginalised"] == row["n_test"]
    # HR is impossible from this joint and says so instead of indexing 0 and 1.
    assert not result.hr_records
    assert any("hardness needs" in n for n in result.notes)


def test_joint_block_runs_on_a_2d_p2xp3_joint(staged, restore_heads) -> None:
    """The legacy 2-D joint a pre-2026-08 checkpoint restores must still score.

    The stored head list omits `logmstar` deliberately: a real checkpoint from
    that era recorded the heads it was TRAINED with, and every one of those runs
    passed --drop-heads logmstar because the FastSpecFit mass was already
    superseded. Putting it back here would test a checkpoint nobody produced.
    """
    mt.configure_heads_from_config({"heads": [t["name"] for t in mt._ALL_TARGETS
                                              if t["name"] not in FIXTURE_DROP_HEADS]
                                             + ["p2xp3_joint"]})
    assert mt.joint_dims() == eval_core.HR_BANDS
    loader, lookup, train_y, stds, flows = make_pieces(staged)
    result = eval_core.run_eval(
        loader=loader, lookup=lookup, encode=stub_encode, flows=flows, standardizers=stds,
        train_y=train_y, combos=[("spectra", "z", "wise", "image")],
        num_samples=4, joint_samples=8, log=lambda m: None)
    joint = [r for r in result.table if r["head"] == mt.HEAD_NAMES[-1]]
    assert len(joint) == 1 and np.isfinite(joint[0]["nll"])
    # the retro-name is carried through instead of a "p2xp3_joint" literal in eval
    assert joint[0]["head"] == "p2xp3_joint"
    assert joint[0]["n_joint_marginalised"] == 0      # nothing marginalisable in a 2-D joint
    assert result.hr_records and set(result.hr_records[0]) >= {"targetid", "hr_p16", "hr_p50", "hr_p84"}


def test_hr_is_resolved_by_name_against_a_3d_joint(staged, restore_heads) -> None:
    """Joint ordered (log_sfr, P2, P3): the bands are at columns 1 and 2, not 0 and 1.

    With a fixed-draw joint the answer is analytic, so reading the wrong columns
    is not a slightly different number, it is a different one by ~1.4 in HR.
    """
    mt.JOINT_PAIR = (eval_core.SFR_HEAD,) + eval_core.HR_BANDS
    mt.JOINT_MARGINAL = ()
    mt.configure_heads(FIXTURE_DROP_HEADS)
    assert mt.joint_col(eval_core.HR_BANDS[0]) == 1 and mt.joint_col(eval_core.HR_BANDS[1]) == 2

    lf_sfr, lf2, lf3 = 10.0, -13.0, -12.5
    ind2, ind3 = -13.2, -12.4
    flows = fixed_flows({eval_core.HR_BANDS[0]: (ind2, 0.05), eval_core.HR_BANDS[1]: (ind3, 0.25)},
                        [lf_sfr, lf2, lf3])
    loader, lookup, train_y, _stds, flows = make_pieces(staged, flows=flows)
    stds = [TargetStandardizer(0.0, 1.0) for _ in range(mt.N_TARGETS)]   # physical == standardized
    result = eval_core.run_eval(
        loader=loader, lookup=lookup, encode=stub_encode, flows=flows, standardizers=stds,
        train_y=train_y, combos=[("spectra", "z", "wise", "image")],
        num_samples=4, joint_samples=9, log=lambda m: None)

    assert result.hr_records
    got = np.median([r["hr_p50"] for r in result.hr_records])
    want = float(eval_core.hr_from_log_fluxes(np.array([lf2]), np.array([lf3]))[0])
    wrong = float(eval_core.hr_from_log_fluxes(np.array([lf_sfr]), np.array([lf2]))[0])
    assert abs(got - want) < 0.02, f"HR {got:+.3f} is not the (P2,P3) answer {want:+.3f}"
    assert abs(got - wrong) > 1.0, "reading joint columns 0 and 1 would be indistinguishable"

    # step 28: the independent-marginal baseline is computed here, on these rows
    want_ind = float(eval_core.hr_from_log_fluxes(np.array([ind2]), np.array([ind3]))[0])
    got_ind = np.median([r["hr_ind_p50"] for r in result.hr_records])
    assert abs(got_ind - want_ind) < 0.02
    assert result.hr_baseline_width is not None and result.hr_baseline_width > 0.0
    assert abs(result.hr_baseline_width - 0.551) > 1e-6, "0.551 was the hardcoded number"


def test_sfr_vs_mass_baseline_fires_for_the_default_head_set(staged) -> None:
    """The guard has to run without --drop-heads gymnastics, on the CIGALE mass."""
    loader, lookup, train_y, stds, flows = make_pieces(staged)
    result = eval_core.run_eval(
        loader=loader, lookup=lookup, encode=stub_encode, flows=flows, standardizers=stds,
        train_y=train_y, combos=[("spectra", "z", "wise", "image")],
        num_samples=4, joint_samples=4, log=lambda m: None)
    # the head the OLD code keyed on produced nothing, which is why it never fired
    assert "logmstar" not in result.allin_pred
    assert result.sfr_baseline is not None, result.notes
    assert result.sfr_baseline["mass_head"] == "logmstar_cigale"
    for key in ("ms_slope", "r2_sfr_head", "r2_true_mstar", "r2_pred_mstar", "verdict"):
        assert key in result.sfr_baseline


def test_sfr_vs_mass_baseline_explains_itself_when_it_cannot_run() -> None:
    row, reason = eval_core.sfr_vs_mass_baseline(
        allin_pred={}, train_y=np.zeros((10, mt.N_TARGETS)), head_names=list(mt.HEAD_NAMES))
    assert row is None and eval_core.SFR_HEAD in reason
    row, reason = eval_core.sfr_vs_mass_baseline(
        allin_pred={eval_core.SFR_HEAD: pd.DataFrame({"targetid": [1], "pred": [0.0], "true": [0.0]})},
        train_y=np.zeros((10, mt.N_TARGETS)), head_names=list(mt.HEAD_NAMES))
    assert row is None and "stellar-mass" in reason


def test_sample_declaration_is_stamped_on_every_row(staged) -> None:
    loader, lookup, train_y, stds, flows = make_pieces(staged)
    result = eval_core.run_eval(
        loader=loader, lookup=lookup, encode=stub_encode, flows=flows, standardizers=stds,
        train_y=train_y, combos=[("z",), ("spectra", "z", "wise", "image")],
        sample="both", num_samples=4, joint_samples=4, log=lambda m: None)
    required = {"sample", "n_test", "n_common", "n_sample", "frac_of_test", "cross_combo_comparable"}
    for row in result.table:
        assert required <= set(row)
        assert row["sample"] in ("common", "native")
        assert 0 < row["n_sample"] <= row["n_test"]
        assert row["cross_combo_comparable"] == (row["sample"] == "common")
    # 20% of the fixture is staged without a cutout, so `common` is strictly smaller
    z_native = next(r for r in result.table
                    if r["head"] == "log_lx" and r["input_group"] == "z" and r["sample"] == "native")
    z_common = next(r for r in result.table
                    if r["head"] == "log_lx" and r["input_group"] == "z" and r["sample"] == "common")
    assert z_common["n_sample"] < z_native["n_sample"]
    assert z_common["n_sample"] == z_common["n_common"]


def test_ig_delta_is_refused_outside_the_common_sample(staged) -> None:
    loader, lookup, train_y, stds, flows = make_pieces(staged)
    result = eval_core.run_eval(
        loader=loader, lookup=lookup, encode=stub_encode, flows=flows, standardizers=stds,
        train_y=train_y, combos=[("z",), ("spectra", "z", "wise", "image")],
        sample="both", num_samples=4, joint_samples=4, log=lambda m: None)

    def pick(head, group, sample):
        return next(r for r in result.table if r["head"] == head
                    and r["input_group"] == group and r["sample"] == sample)

    common_a = pick("log_lx", "z", "common")
    common_b = pick("log_lx", "spectra+z+wise+image", "common")
    assert np.isfinite(eval_core.information_gain_delta(common_b, common_a))
    # R2, RMSE and mean NLL carry exactly the same hazard: their denominator is
    # recomputed per combo, so they are flagged and gated by the same rule.
    for metric in ("r2", "rmse_dex", "nll"):
        assert np.isfinite(eval_core.metric_delta(common_b, common_a, metric))
    with pytest.raises(ValueError):
        eval_core.information_gain_delta(pick("log_lx", "z", "native"), common_a)
    with pytest.raises(ValueError):
        eval_core.metric_delta(common_a, common_b, "n_test")
    with pytest.raises(ValueError):
        eval_core.information_gain_delta(common_a, pick("log_sfr", "z", "common"))
    mismatched = dict(common_b)
    mismatched["n_sample"] = common_a["n_sample"] - 1
    with pytest.raises(ValueError):
        eval_core.information_gain_delta(mismatched, common_a)


def test_run_eval_end_to_end_on_the_tiny_staged_fixture(staged) -> None:
    """All 15 combos, CPU, stub encoder, no checkpoint, no aion."""
    loader, lookup, train_y, stds, flows = make_pieces(staged)
    result = eval_core.run_eval(
        loader=loader, lookup=lookup, encode=stub_encode, flows=flows, standardizers=stds,
        train_y=train_y, num_samples=4, joint_samples=4, log=lambda m: None)
    table = pd.DataFrame(result.table)
    assert set(table.input_group) == {
        "+".join(m for m in eval_core.MODALITIES if m in c)
        for c in eval_core.all_nonempty_modality_combos()}
    assert table["nll"].notna().all()
    # every scalar head with labels appears for every combo
    heads = set(table["head"]) - {mt.HEAD_NAMES[-1]}
    assert {"log_lx", "log_sfr", "logmstar_cigale"} <= heads
    assert "logmstar" not in heads               # no labels, and no crash either
    # P1 is detected for half the sample; the table says so per head
    p1 = table[table["head"] == "log_flux_p1"].iloc[0]
    lx = table[table["head"] == "log_lx"].iloc[0]
    assert p1.n_test < lx.n_test
    # an image combo is scored only on rows that have an image
    img = table[(table["head"] == "log_lx") & (table["input_group"] == "image")].iloc[0]
    assert img.n_sample < lx.n_test


def test_priors_are_none_where_the_train_view_has_no_labels(staged) -> None:
    """A head with a column but no usable labels must not take eval down.

    This used to be tested with `logmstar`, whose column is absent entirely.
    That case is now REFUSED at load time -- an all-NaN target trains silently
    forever, so load_multi_target_matrix raises rather than allowing it. The
    case that survives, and still has to degrade gracefully, is a head whose
    column exists and whose train-view values happen to be all NaN: a band
    detected in no training source, say.
    """
    _loader, _lookup, train_y, stds, _flows = make_pieces(staged)
    blank = mt.target_col("log_flux_p4")
    train_y = train_y.copy()
    train_y[:, blank] = np.nan
    priors = eval_core.build_priors(train_y, stds)
    assert priors[blank] is None
    assert priors[mt.target_col("log_lx")] is not None


# ------------------------------------------------------------------ exact HR quadrature
# scripts/hr_from_joint.py refuses today's 4-D joint, correctly. It is also the
# ONLY code that can answer Run B's headline question -- does a 2-D P2xP3 joint
# beat two independent marginals on hardness -- and until these tests it had no
# coverage at all, so "it refuses the joint we have" was indistinguishable from
# "it is broken for the joint we are about to train".
def _hr_from_joint():
    """Load the script as a module. It imports cleanly without aion or a GPU."""
    path = REPO_ROOT / "scripts" / "hr_from_joint.py"
    spec = importlib.util.spec_from_file_location("hr_from_joint_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _GaussianJointFlow(nn.Module):
    """A 2-D Gaussian in the flow's OWN (standardized) coordinates.

    Anisotropic on purpose: with equal scales on the two columns, swapping which
    column is P2 and which is P3 is invisible, and a test that cannot see the
    swap cannot check that ``joint_col`` is being honoured.
    """

    def __init__(self, a0: float, a1: float, rho: float) -> None:
        super().__init__()
        self.a = (a0, a1)
        self.rho, self.features = rho, 2

    def log_prob_draws(self, x, context):
        z0, z1 = x[..., 0] / self.a[0], x[..., 1] / self.a[1]
        r = self.rho
        quad = (z0 ** 2 - 2 * r * z0 * z1 + z1 ** 2) / (1.0 - r ** 2)
        norm = (math.log(2 * math.pi) + math.log(self.a[0]) + math.log(self.a[1])
                + 0.5 * math.log(1.0 - r ** 2))
        return -0.5 * quad - norm


def _analytic_line_logpdf(hrj, flow, std2, std3, col2, col3, d_phys):
    """log p(d) for the same model, in closed form.

    The quadrature integrates p(u, u + d - DELTA) du, which is the density of
    W = V - U evaluated at d - DELTA. For a Gaussian joint that is another
    Gaussian, so the whole integral has an answer to compare against.
    """
    sd_u = std2.std * flow.a[col2]
    sd_v = std3.std * flow.a[col3]
    var_w = sd_u ** 2 + sd_v ** 2 - 2.0 * flow.rho * sd_u * sd_v
    mean_w = std3.mean - std2.mean
    w = d_phys - hrj.DELTA
    return -0.5 * ((w - mean_w) ** 2 / var_w) - 0.5 * math.log(2 * math.pi * var_w)


def test_the_hr_quadrature_integrates_to_the_right_density_for_a_2d_joint() -> None:
    """The shear integral, against the closed form, for a P2xP3 joint.

    This is the number Run B's hardness comparison is built on. It is not a
    smoke test: a wrong Jacobian, a missing ``log(du)``, or standardized units
    mistaken for physical ones all still produce a finite, plausible curve.
    """
    hrj = _hr_from_joint()
    std2 = TargetStandardizer(-13.20, 0.40)
    std3 = TargetStandardizer(-13.05, 0.30)
    flow = _GaussianJointFlow(a0=0.80, a1=0.45, rho=0.55)
    u_nodes = torch.linspace(std2.mean - 6.0 * std2.std, std2.mean + 6.0 * std2.std, 384)

    d_phys = torch.tensor([-0.30, 0.0, 0.15, 0.42], dtype=torch.float32)
    context = torch.zeros(len(d_phys), CTX)
    got = hrj.line_log_density(flow, context, d_phys, std2, std3, u_nodes,
                               col2=0, col3=1).numpy()
    want = np.array([_analytic_line_logpdf(hrj, flow, std2, std3, 0, 1, float(d))
                     for d in d_phys])
    assert np.allclose(got, want, atol=2e-3), f"{got} vs {want}"


def test_the_hr_quadrature_honours_which_flow_column_is_which_band() -> None:
    """col2/col3 are passed in from joint_col; assuming 0 and 1 must be visible.

    A 2-D joint declared (P3, P2) is a legal JOINT_PAIR and puts the bands in
    the other order. The old code hardcoded columns 0 and 1.
    """
    hrj = _hr_from_joint()
    std2 = TargetStandardizer(-13.20, 0.40)
    std3 = TargetStandardizer(-13.05, 0.30)
    flow = _GaussianJointFlow(a0=0.80, a1=0.45, rho=0.55)
    u_nodes = torch.linspace(std2.mean - 6.0 * std2.std, std2.mean + 6.0 * std2.std, 384)
    d_phys = torch.tensor([0.15], dtype=torch.float32)
    context = torch.zeros(1, CTX)

    straight = float(hrj.line_log_density(flow, context, d_phys, std2, std3, u_nodes,
                                          col2=0, col3=1)[0])
    swapped = float(hrj.line_log_density(flow, context, d_phys, std2, std3, u_nodes,
                                         col2=1, col3=0)[0])
    assert abs(straight - swapped) > 0.1, "the fixture cannot see a column swap"
    assert abs(straight - _analytic_line_logpdf(hrj, flow, std2, std3, 0, 1, 0.15)) < 2e-3
    assert abs(swapped - _analytic_line_logpdf(hrj, flow, std2, std3, 1, 0, 0.15)) < 2e-3


def test_both_hr_paths_agree_on_the_sign_of_the_ecf_offset() -> None:
    """The quadrature and the sampled block must mean the same thing by HR.

    Run B compares them, so a flipped ``DELTA = C_P2 - C_P3`` would not show up
    as an error anywhere: both halves stay finite and inside (-1, 1), and the
    hardness is simply wrong by ~0.29 in this configuration. Peak of the
    quadrature's posterior against eval_core.hr_from_log_fluxes on the same
    fluxes is the only check that catches it.
    """
    hrj = _hr_from_joint()
    lf2, lf3 = -13.20, -13.05
    std2, std3 = TargetStandardizer(lf2, 0.40), TargetStandardizer(lf3, 0.30)
    flow = _GaussianJointFlow(a0=0.30, a1=0.30, rho=0.0)
    u_nodes = torch.linspace(std2.mean - 6.0 * std2.std, std2.mean + 6.0 * std2.std, 384)

    hr_grid = np.linspace(-0.97, 0.97, 601)
    d_grid = torch.tensor(hrj.hr_to_d(hr_grid), dtype=torch.float32)
    context = torch.zeros(len(d_grid), CTX)
    # density in d; the HR Jacobian is monotone-positive but not flat, so apply
    # it exactly as main() does before locating the mode
    lp = hrj.line_log_density(flow, context, d_grid, std2, std3, u_nodes,
                              col2=0, col3=1).numpy()
    lp = lp + np.log(2.0 / (hrj.LN10 * (1.0 - hr_grid ** 2)))
    peak_hr = float(hr_grid[int(np.argmax(lp))])

    want = float(eval_core.hr_from_log_fluxes(np.array([lf2]), np.array([lf3]))[0])
    flipped = float(np.tanh((lf3 - lf2 - hrj.DELTA) * hrj.LN10 / 2.0))
    assert abs(peak_hr - want) < 0.02, f"quadrature peaks at {peak_hr:+.3f}, sampled says {want:+.3f}"
    assert abs(want - flipped) > 0.1, "the fixture cannot see a sign flip in DELTA"
    # and the two hardness parametrisations are inverses of each other
    probe = np.array([-0.8, -0.2, 0.0, 0.35, 0.9])
    assert np.allclose(hrj.d_to_hr(hrj.hr_to_d(probe)), probe, atol=1e-6)


def test_a_non_2d_joint_is_a_stated_skip_not_a_failed_job() -> None:
    """`hr_from_joint` exits 3, and sbatch/eval_multi.sbatch treats 3 as a skip.

    Under `set -e` an undifferentiated exit 1 marks the whole eval job FAILED
    after every table is already written, so the two must agree on the code.
    """
    hrj = _hr_from_joint()
    assert hrj.NOT_APPLICABLE == 3
    launcher = (REPO_ROOT / "sbatch" / "eval_multi.sbatch").read_text()
    assert f'"$hr_rc" -eq {hrj.NOT_APPLICABLE}' in launcher, (
        "eval_multi.sbatch does not special-case hr_from_joint's not-applicable exit code")
    assert '"$hr_rc" -ne 0' in launcher, "a real hr_from_joint failure must still stop the job"
    # the refusal itself is reachable only through main(); assert the condition
    # it is keyed on rather than re-deriving it
    assert len(mt.joint_dims()) != 2 or set(mt.joint_dims()) == set(eval_core.HR_BANDS)


def test_modality_presence_reads_wise_on_flux_not_finiteness() -> None:
    """A finiteness rule would mark WISE present for 25,582 of 25,582 staged rows."""
    b = 3
    batch = [torch.randn(b, N_PIX), torch.ones(b, N_PIX), torch.zeros(N_PIX),
             torch.tensor([0.5, 1.0, 1.5]), torch.zeros(b, 3), torch.ones(b, 4, 8, 8)]
    batch[4][0] = torch.tensor([2.0, 3.0, 1.0])       # measured
    batch[4][1] = torch.tensor([0.0, 0.0, 0.0])       # present as a column, absent as a flux
    batch[4][2] = torch.tensor([float("nan")] * 3)
    batch[5][1] = 0.0                                  # staged without a cutout
    present = eval_core.modality_presence(tuple(batch))
    assert present["wise"].tolist() == [True, False, False]
    assert present["image"].tolist() == [True, False, True]
    assert present["z"].all() and present["spectra"].all()
