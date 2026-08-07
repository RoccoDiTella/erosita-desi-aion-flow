"""Evaluation internals for the multi-target model: importable and testable.

All of this used to live inside ``scripts/eval_multitarget.py:main()``. That is
why ``j2, j3 = JOINT_IDX`` -- a two-element unpack of a tuple that had become
four elements long -- survived a whole run cycle: a 200-line ``main()`` whose
preconditions are a GPU, a checkpoint, AION and a staged HDF5 cannot be tested,
so it was not, and the joint block raises ValueError the first time it is
pointed at a model trained on the (M*, SFR, Lx, P3) joint. The script now keeps
argparse, ``torch.load``, device plumbing and CSV writing; every decision is
here, behind a function with arguments.

The seam is ``encode(batch, combo) -> contexts [B, N_HEADS, ctx]``: the script
passes a frozen AION encoder followed by the shared MLP, the tests pass a
deterministic stub. Nothing in this module imports ``aion`` or reads a
checkpoint.

Head configuration is read through the ``multitarget`` MODULE on every call and
never captured at import time. ``configure_heads_from_config`` rebinds those
globals *after* the checkpoint is loaded, so ``from multitarget import
HEAD_NAMES`` would freeze the pre-checkpoint default set -- the same class of
mistake as indexing the joint by position.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

try:
    from . import multitarget as _mt
    from .attention_pooling_head import MODALITIES, all_nonempty_modality_combos, combo_name
    from .normalizing_flow import KDEPrior
except ImportError:                                   # bare-module import (tests, scripts)
    import multitarget as _mt
    from attention_pooling_head import MODALITIES, all_nonempty_modality_combos, combo_name
    from normalizing_flow import KDEPrior

# Exact log10(FLUX/RATE) global ECFs per band, from the eRASS catalogue rather
# than fitted here. The band rate is the flux divided by a CONSTANT, so hardness
# is a function of the difference of the two log fluxes and of nothing else --
# which is what makes the shear marginalisation in hr_from_joint.py exact.
C_P2, C_P3 = -12.1332, -12.0060

# ---------------------------------------------------------------------------
# Head names this module refers to BY NAME, collected in one place.
#
# Every head-name literal in the eval path is here and nowhere else, and
# test_eval_core asserts each one exists in multitarget._ALL_TARGETS. The reason
# is not tidiness: the SFR-vs-mass guard was keyed on the literal "logmstar",
# a head that is dropped from every run, so the check silently never fired and
# nothing noticed for four months. A name that no longer exists must break a
# test, not quietly disable a branch.
# ---------------------------------------------------------------------------
HR_BANDS = ("log_flux_p2", "log_flux_p3")     # (soft, hard) for HR32; order is meaning, not layout
SFR_HEAD = "log_sfr"
MASS_HEADS = ("logmstar_cigale", "logmstar")  # preference order: CIGALE fits the AGN, FastSpecFit does not

# Metrics whose value depends on WHICH rows were scored, so comparing them across
# two combos is only meaningful when both used the same rows. IG is the obvious
# one and the only one the plan flagged; r2 and rmse_dex are worse, because the
# variance denominator is recomputed on each combo's own rows, so a combo
# evaluated on an easier subset posts a better R2 without predicting anything
# better. Mean NLL is in the same position (IG is NLL minus a prior term that is
# also a row average), so it is gated too.
ROW_SET_DEPENDENT_METRICS = ("info_gain_nats", "r2", "rmse_dex", "nll")

SAMPLE_CHOICES = ("common", "native", "both")


# --------------------------------------------------------------- sample declaration
# Presence is defined ONCE, in data_to_aion_embeddings, and imported here.
#
# This module briefly carried its own copy. It disagreed with the manifest's
# rule on two counts (no zwarn, no WISE inverse variance), so `--sample common`
# admitted 352 zwarn-flagged rows the manifest declares unusable, and any
# information gain was computed on a row set nobody had declared. A duplicated
# definition of "which rows count" is the same class of defect as a duplicated
# definition of "which column is P3".
try:
    from .data_to_aion_embeddings import modality_presence  # noqa: F401
except ImportError:                                          # flat-layout imports
    from data_to_aion_embeddings import modality_presence    # noqa: F401


def resolve_samples(sample: str) -> tuple[str, ...]:
    if sample not in SAMPLE_CHOICES:
        raise ValueError(f"sample must be one of {SAMPLE_CHOICES}, got {sample!r}")
    return ("common", "native") if sample == "both" else (sample,)


def sample_row_mask(present: dict[str, torch.Tensor], combo, sample: str) -> torch.Tensor:
    """Rows this combo may be scored on under the requested sample declaration.

    ``native``  rows that support THIS combo -- the largest honest row set, and
                the one every per-combo number was implicitly computed on before,
                except that the old code did not check presence at all and scored
                image combos on sources with no cutout (uniform attention over
                tokens that are not there).
    ``common``  rows that support EVERY modality, so the row set is identical for
                all 15 combos and a difference between two of them is a statement
                about inputs rather than about which sources happened to survive.
    """
    if sample == "common":
        wanted = MODALITIES
    elif sample == "native":
        wanted = tuple(m for m in MODALITIES if m in set(combo))
    else:
        raise ValueError(f"unknown sample {sample!r}")
    any_key = next(iter(present))
    mask = torch.ones_like(present[any_key])
    for name in wanted:
        mask = mask & present[name]
    return mask


def metric_delta(row_a: dict, row_b: dict, metric: str = "info_gain_nats") -> float:
    """``metric(a) - metric(b)`` across two combos, refused unless it means something.

    Every metric in ROW_SET_DEPENDENT_METRICS is computed against a denominator
    that the row set fixes -- the KDE prior's mean log-density for IG, the
    sample variance for R2 and RMSE. Subtracting two of them across combos whose
    rows differ measures the difference in row sets, not the difference in
    inputs, and there is nothing in the number that says which one you got.
    """
    if metric not in ROW_SET_DEPENDENT_METRICS:
        raise ValueError(f"{metric!r} is not one of {ROW_SET_DEPENDENT_METRICS}")
    if row_a["head"] != row_b["head"]:
        raise ValueError(f"different heads: {row_a['head']!r} vs {row_b['head']!r}")
    for row in (row_a, row_b):
        if row["sample"] != "common":
            raise ValueError(
                f"{metric} deltas are only defined within sample='common'; "
                f"{row['head']}/{row['input_group']} was scored on sample={row['sample']!r} "
                f"({row['n_sample']} of {row['n_test']} test rows)")
    if row_a["n_sample"] != row_b["n_sample"]:
        raise ValueError(
            f"sample='common' but n_sample differs ({row_a['n_sample']} vs {row_b['n_sample']}): "
            "the two rows did not see the same sources, so the delta is not a modality effect")
    return float(row_a[metric]) - float(row_b[metric])


def information_gain_delta(row_a: dict, row_b: dict) -> float:
    """IG(a) - IG(b). Run B's headline number, and only legal within `common`."""
    return metric_delta(row_a, row_b, "info_gain_nats")


# --------------------------------------------------------------- joint helpers
def assert_joint_matches_flow(flows) -> None:
    """The joint flow's width must equal the number of declared joint dimensions.

    A checkpoint trained with a 2-D P2xP3 joint loaded under the current 4-D
    declaration (or the reverse) fails at ``load_state_dict`` if you are lucky
    and produces a silently wrong density if you are not.
    """
    declared = len(_mt.joint_dims())
    actual = int(getattr(flows.joint, "features", declared))
    if actual != declared:
        raise SystemExit(
            f"joint flow has {actual} features but the head configuration declares "
            f"{declared} dimensions {_mt.joint_dims()}. The checkpoint and multitarget.JOINT_PAIR "
            "disagree; configure_heads_from_config(ckpt['config']) must run before the flows are built.")
    assert_no_poisson_heads()


def assert_no_poisson_heads() -> None:
    """Refuse a Run B checkpoint until this module learns the count likelihood.

    Every metric here -- NLL, information gain, R^2, RMSE -- is computed by
    evaluating the flow's density AT THE TARGET VALUE. A Poisson latent-rate head
    has no target value: ``targets[:, j]`` is a plug-in standardization anchor,
    and scoring it would produce a full, plausible, entirely wrong table rather
    than an error. That is precisely the class of silent failure this module was
    refactored out of a 200-line main() to prevent, so it is a refusal and not a
    warning.

    The right metric for those heads is the marginal likelihood of the COUNTS,
    ``multitarget.poisson_marginal_nll`` -- the same number the trainer reports.
    """
    pois = [t["name"] for t in _mt.MULTI_TARGETS if t.get("pois")]
    if pois:
        raise SystemExit(
            f"this checkpoint has Poisson latent-rate heads {pois}, which eval_core "
            f"cannot score: it evaluates a flow density at a target value, and those "
            f"heads have no target value -- only counts, background and exposure. "
            f"Scoring them would silently rate the standardization anchor. Use the "
            f"count marginal likelihood (multitarget.poisson_marginal_nll) instead.")


def hr_skip_reason(dims: tuple[str, ...]) -> str | None:
    """Why the joint cannot produce a hardness ratio, or None if it can.

    Hardness needs BOTH band fluxes as joint dimensions. The old code assumed
    joint flow columns 0 and 1 were (P2, P3), which is false for any joint that
    is not literally the 2-D P2xP3 one -- today's is (M*, SFR, Lx, P3).
    """
    missing = [b for b in HR_BANDS if b not in dims]
    if missing:
        return (f"joint dimensions are {dims}; hardness needs {HR_BANDS} and is missing "
                f"{missing}. No HR posteriors from this checkpoint.")
    return None


def hr_from_log_fluxes(lf2: np.ndarray, lf3: np.ndarray) -> np.ndarray:
    """HR32 from physical log fluxes, via the exact per-band ECFs.

    rho = R3/R2 = 10^((logF3 - C3) - (logF2 - C2)); HR = (rho-1)/(rho+1). The
    exponent is a log-flux difference and is physically within a couple of dex,
    so it is clipped at +/-30: an untrained or degenerate flow can otherwise
    draw an exponent that overflows to inf, and inf/inf is NaN, which then
    poisons the median of the whole posterior rather than saturating it at the
    correct +/-1 limit.
    """
    expo = np.clip((lf3 - C_P3) - (lf2 - C_P2), -30.0, 30.0)
    rho = 10.0 ** expo
    return (rho - 1.0) / (rho + 1.0)


def hr_percentiles(hr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p16, p50, p84 = np.percentile(hr, [16, 50, 84], axis=0)
    return p16, p50, p84


def independent_hr_skip_reason(head_names: list[str]) -> str | None:
    missing = [b for b in HR_BANDS if b not in head_names]
    if missing:
        return (f"independent-band HR baseline needs the {HR_BANDS} marginal heads; "
                f"this run does not have {missing}")
    return None


# --------------------------------------------------------------- priors
def build_priors(train_y: np.ndarray, standardizers) -> list[KDEPrior | None]:
    """One KDE prior per scalar head, None where the train view has no labels.

    A head can be configured and carry no data at all: ``logmstar`` reads from
    the staged HDF5, which the DR2 rebuild stops writing, so its column is all
    NaN while the head still exists for checkpoint compatibility. Returning None
    (and reporting IG as NaN) is the honest answer; ``KDEPrior`` would raise and
    take the whole evaluation with it.
    """
    priors: list[KDEPrior | None] = []
    for j in range(train_y.shape[1]):
        vals = train_y[:, j][np.isfinite(train_y[:, j])]
        if vals.size < 2 or float(np.std(vals)) <= 1e-12:
            priors.append(None)
            continue
        priors.append(KDEPrior(standardizers[j].transform_numpy(vals), bw_method="scott"))
    return priors


# --------------------------------------------------------------- accumulators
@dataclass
class _ScalarAcc:
    nll: float = 0.0
    plp: float = 0.0
    n: int = 0
    pred: list = field(default_factory=list)
    true: list = field(default_factory=list)
    tid: list = field(default_factory=list)


@dataclass
class _JointAcc:
    """The joint head has TWO populations and they must not be added up.

    ``n_full``  every joint dimension observed -> the ordinary likelihood
    ``n_marg``  a marginalisable dimension missing -> the trainer integrates it
                out by quadrature, which is a density over FEWER dimensions

    A single "joint n" is a count-weighted mixture of two dimensionalities and
    cannot be compared with anything, so eval scores the NLL on ``n_full`` only
    and reports ``n_marg`` beside it.
    """
    nll: float = 0.0
    n_full: int = 0
    n_marg: int = 0


@dataclass
class EvalResult:
    table: list[dict]
    hr_records: list[dict]
    allin_pred: dict
    sfr_baseline: dict | None
    hr_baseline_width: float | None
    notes: list[str]
    head_names: list[str]
    joint_dims: tuple[str, ...]
    samples: tuple[str, ...]
    hr_sample: str


def _stamp(row: dict, *, sample: str, n_test: int, n_common: int, n_sample: int) -> dict:
    """Attach the sample declaration every table row has to carry.

    Without these four numbers a row of the combo table is not interpretable:
    "IG 0.31 for spectra+z+wise+image" says nothing until you know it was scored
    on 22% of the test set and the row it is being compared against was not.
    """
    row.update({
        "sample": sample,
        "n_test": int(n_test),
        "n_common": int(n_common),
        "n_sample": int(n_sample),
        "frac_of_test": (float(n_sample) / n_test) if n_test else float("nan"),
        # r2/rmse/IG are only comparable ACROSS combos when the row set is fixed
        "cross_combo_comparable": sample == "common",
    })
    return row


def run_eval(
    *,
    loader,
    lookup,
    encode,
    flows,
    standardizers,
    train_y: np.ndarray | None = None,
    priors: list | None = None,
    device: torch.device | None = None,
    combos=None,
    sample: str = "native",
    num_samples: int = 128,
    joint_samples: int = 256,
    hr_combo: str | None = None,
    allin_combo: str | None = None,
    presence_fn=modality_presence,
    log=print,
) -> EvalResult:
    """Score every modality combo on the test loader. One encoder pass per combo.

    ``encode(batch, combo) -> [B, N_HEADS, ctx]`` is the only model-side
    dependency; ``lookup.batch(targetids, device) -> (y, sig_lo, sig_hi)`` the
    only data-side one.
    """
    device = torch.device("cpu") if device is None else device
    head_names = list(_mt.HEAD_NAMES)
    n_targets = _mt.N_TARGETS
    dims = _mt.joint_dims()
    assert_joint_matches_flow(flows)
    samples = resolve_samples(sample)
    combos = list(all_nonempty_modality_combos() if combos is None else combos)
    hr_combo = combo_name(MODALITIES) if hr_combo is None else hr_combo
    allin_combo = combo_name(MODALITIES) if allin_combo is None else allin_combo
    if priors is None:
        priors = (build_priors(train_y, standardizers) if train_y is not None
                  else [None] * n_targets)

    notes: list[str] = []
    hr_reason = hr_skip_reason(dims)
    hr_cols: tuple[int, int] | None = None
    if hr_reason is not None:
        notes.append(f"[HR] skipped: {hr_reason}")
    else:
        # Index by NAME. Under a 4-D joint, columns 0 and 1 are M* and SFR.
        hr_cols = (_mt.joint_col(HR_BANDS[0]), _mt.joint_col(HR_BANDS[1]))
    ind_reason = independent_hr_skip_reason(head_names)
    if hr_reason is None and ind_reason is not None:
        notes.append(f"[HR] independent-band baseline skipped: {ind_reason}")
    # The per-source HR posteriors are one set of records, so they are built
    # under one sample. `hr_combo` is the all-modality combo by default, whose
    # native and common row sets coincide by construction.
    hr_sample = samples[0]

    table: list[dict] = []
    hr_records: list[dict] = []
    allin_pred: dict = {}
    ind_widths: list[np.ndarray] = []

    with torch.no_grad():
        for combo in combos:
            name = combo_name(combo)
            accs = {s: [_ScalarAcc() for _ in range(n_targets)] for s in samples}
            jaccs = {s: _JointAcc() for s in samples}
            n_test_scalar = np.zeros(n_targets, dtype=np.int64)
            n_test_joint = 0
            n_common_scalar = np.zeros(n_targets, dtype=np.int64)
            n_common_joint = 0

            for batch in loader:
                batch = tuple(t.to(device, non_blocking=True) for t in batch)
                y, _, _ = lookup.batch(batch[7], device)
                present = presence_fn(batch)
                common = sample_row_mask(present, combo, "common")
                have_req, have_all = _mt.joint_availability(y)

                # Denominators: the head's whole test pool, and its common-sample
                # part. Neither depends on the combo -- that is the point of
                # printing them next to n_sample.
                finite = torch.isfinite(y)
                n_test_scalar += finite.sum(dim=0).cpu().numpy().astype(np.int64)
                n_common_scalar += (finite & common.unsqueeze(1)).sum(dim=0).cpu().numpy().astype(np.int64)
                n_test_joint += int(have_req.sum())
                n_common_joint += int((have_req & common).sum())

                masks = {s: sample_row_mask(present, combo, s) for s in samples}
                if not any(bool(m.any()) for m in masks.values()):
                    continue
                contexts = encode(batch, combo)

                for s in samples:
                    base = masks[s]
                    if not bool(base.any()):
                        continue
                    acc, jacc = accs[s], jaccs[s]
                    for j in range(n_targets):
                        mask = finite[:, j] & base
                        if not bool(mask.any()):
                            continue
                        ys = standardizers[j].transform_tensor(y[mask, j])
                        ctx_j = contexts[mask, j]
                        lp = flows.flows[j].log_prob(ys, ctx_j)
                        samp = flows.flows[j].sample(ctx_j, num_samples=num_samples)
                        m = int(mask.sum())
                        acc[j].nll += float(-lp.mean()) * m
                        if priors[j] is not None:
                            acc[j].plp += float(priors[j].log_prob_tensor(ys).mean()) * m
                        else:
                            acc[j].plp = float("nan")
                        acc[j].n += m
                        acc[j].pred.append(samp.mean(dim=0).cpu().numpy())
                        acc[j].true.append(ys.cpu().numpy())
                        acc[j].tid.append(batch[7][mask].cpu().numpy())

                    # ---- joint head, indexed by name in FLOW COLUMN ORDER ----
                    jm_full = have_all & base
                    jm_req = have_req & base
                    jacc.n_full += int(jm_full.sum())
                    jacc.n_marg += int((jm_req & ~jm_full).sum())
                    if not bool(jm_full.any()):
                        continue
                    ctx_joint = contexts[jm_full, n_targets]
                    vec = torch.stack(
                        [standardizers[_mt.target_col(d)].transform_tensor(y[jm_full, _mt.target_col(d)])
                         for d in dims], dim=-1)
                    jlp = flows.joint.distribution(ctx_joint).log_prob(vec)
                    jacc.nll += float(-jlp.mean()) * int(jm_full.sum())

                    if hr_cols is not None and name == hr_combo and s == hr_sample:
                        draws = flows.joint.distribution(ctx_joint).sample((joint_samples,))
                        c2, c3 = hr_cols
                        s2, s3 = standardizers[_mt.target_col(HR_BANDS[0])], \
                            standardizers[_mt.target_col(HR_BANDS[1])]
                        lf2 = draws[..., c2].cpu().numpy() * s2.std + s2.mean
                        lf3 = draws[..., c3].cpu().numpy() * s3.std + s3.mean
                        p16, p50, p84 = hr_percentiles(hr_from_log_fluxes(lf2, lf3))
                        rec = {"targetid": batch[7][jm_full].cpu().numpy(),
                               "hr_p16": p16, "hr_p50": p50, "hr_p84": p84}
                        if ind_reason is None:
                            # Step 28: the independent-marginal baseline, on the
                            # SAME rows with the SAME constants. It used to be a
                            # hardcoded 0.551 carried over from a different run,
                            # which is not a baseline, it is a memory.
                            i2, i3 = _mt.target_col(HR_BANDS[0]), _mt.target_col(HR_BANDS[1])
                            d2 = flows.flows[i2].sample(contexts[jm_full, i2], num_samples=joint_samples)
                            d3 = flows.flows[i3].sample(contexts[jm_full, i3], num_samples=joint_samples)
                            lf2i = d2.cpu().numpy() * standardizers[i2].std + standardizers[i2].mean
                            lf3i = d3.cpu().numpy() * standardizers[i3].std + standardizers[i3].mean
                            q16, q50, q84 = hr_percentiles(hr_from_log_fluxes(lf2i, lf3i))
                            rec.update({"hr_ind_p16": q16, "hr_ind_p50": q50, "hr_ind_p84": q84})
                            ind_widths.append(0.5 * (q84 - q16))
                        for k in range(len(rec["targetid"])):
                            hr_records.append({key: (int(val[k]) if key == "targetid" else float(val[k]))
                                               for key, val in rec.items()})

            for s in samples:
                acc, jacc = accs[s], jaccs[s]
                for j in range(n_targets):
                    if acc[j].n == 0:
                        continue
                    yp = np.concatenate(acc[j].pred)
                    yt = np.concatenate(acc[j].true)
                    ss = float(np.sum((yt - yt.mean()) ** 2))
                    table.append(_stamp({
                        "head": head_names[j], "input_group": name,
                        "nll": acc[j].nll / acc[j].n,
                        "r2": 1.0 - float(np.sum((yt - yp) ** 2)) / ss if ss > 0 else np.nan,
                        "rmse_dex": float(np.sqrt(np.mean((yt - yp) ** 2))) * standardizers[j].std,
                        "info_gain_nats": -(acc[j].nll / acc[j].n) - (acc[j].plp / acc[j].n),
                        "joint_dims": "",
                        "n_joint_full": np.nan, "n_joint_marginalised": np.nan,
                    }, sample=s, n_test=int(n_test_scalar[j]), n_common=int(n_common_scalar[j]),
                        n_sample=acc[j].n))
                if jacc.n_full or jacc.n_marg:
                    table.append(_stamp({
                        # The head's own name, not the "p2xp3_joint" literal: that
                        # string is the legacy retro-name for a 2-D joint and is
                        # simply false for any other one.
                        "head": head_names[-1], "input_group": name,
                        "nll": (jacc.nll / jacc.n_full) if jacc.n_full else np.nan,
                        "r2": np.nan, "rmse_dex": np.nan, "info_gain_nats": np.nan,
                        "joint_dims": "+".join(dims),
                        "n_joint_full": jacc.n_full, "n_joint_marginalised": jacc.n_marg,
                    }, sample=s, n_test=n_test_joint, n_common=n_common_joint,
                        n_sample=jacc.n_full))

            if name == allin_combo:
                acc = accs[samples[0]]
                allin_pred = {
                    head_names[j]: pd.DataFrame({
                        "targetid": np.concatenate(acc[j].tid),
                        "pred": np.concatenate(acc[j].pred) * standardizers[j].std + standardizers[j].mean,
                        "true": np.concatenate(acc[j].true) * standardizers[j].std + standardizers[j].mean,
                    })
                    for j in range(n_targets) if acc[j].n
                }
            log(f"[eval] combo {name} done")

    sfr_row, sfr_reason = sfr_vs_mass_baseline(
        allin_pred=allin_pred, train_y=train_y, head_names=head_names)
    if sfr_row is None:
        notes.append(f"[SFR vs mass-only baselines] skipped: {sfr_reason}")

    baseline_width = float(np.median(np.concatenate(ind_widths))) if ind_widths else None
    return EvalResult(
        table=table, hr_records=hr_records, allin_pred=allin_pred, sfr_baseline=sfr_row,
        hr_baseline_width=baseline_width, notes=notes, head_names=head_names,
        joint_dims=dims, samples=samples, hr_sample=hr_sample)


# --------------------------------------------------------------- SFR vs mass
def sfr_vs_mass_baseline(*, allin_pred: dict, train_y, head_names: list[str]) -> tuple[dict | None, str]:
    """Is the SFR head doing anything a stellar-mass predictor could not?

    SFR correlates with M* through the star-forming main sequence, so a head
    that merely re-predicted the mass would still post a respectable SFR R2.
    Two reference points, both scored on the same test rows as the head:
      true-M*  the ceiling for ANY mass-only predictor (uses the real mass)
      pred-M*  what this model could achieve by routing its own mass head's
               output through the main sequence
    The SFR head has to beat both.

    Returns ``(row, reason)``; ``row`` is None when the check could not run and
    ``reason`` says why. It never silently does nothing, which is what it did
    for four months: it was keyed on the literal "logmstar", a head dropped from
    every run since CIGALE masses landed, so the branch was unreachable.
    """
    if train_y is None:
        return None, "no train view supplied, so the main sequence cannot be fitted"
    if SFR_HEAD not in allin_pred:
        return None, f"the reference combo produced no {SFR_HEAD} predictions"
    mass = next((m for m in MASS_HEADS if m in allin_pred and m in head_names), None)
    if mass is None:
        return None, (f"no stellar-mass predictions: none of {MASS_HEADS} is a trained head with "
                      f"test rows (head set: {head_names})")
    sfr_df, m_df = allin_pred[SFR_HEAD], allin_pred[mass]
    merged = sfr_df.merge(m_df, on="targetid", suffixes=("_sfr", "_m"))
    js, jm = head_names.index(SFR_HEAD), head_names.index(mass)
    ok = np.isfinite(train_y[:, js]) & np.isfinite(train_y[:, jm])
    if int(ok.sum()) <= 100:
        return None, f"only {int(ok.sum())} train rows have both {SFR_HEAD} and {mass}; need > 100"
    if len(merged) <= 50:
        return None, f"only {len(merged)} test rows have both {SFR_HEAD} and {mass}; need > 50"
    slope, icpt = np.polyfit(train_y[ok, jm], train_y[ok, js], 1)   # main sequence, TRAIN only
    yt = merged.true_sfr.to_numpy()
    ss = float(np.sum((yt - yt.mean()) ** 2))

    def r2(pred) -> float:
        return 1.0 - float(np.sum((yt - pred) ** 2)) / ss

    r2_head = r2(merged.pred_sfr.to_numpy())
    r2_true_m = r2(slope * merged.true_m.to_numpy() + icpt)
    r2_pred_m = r2(slope * merged.pred_m.to_numpy() + icpt)
    best = max(r2_true_m, r2_pred_m)
    return {
        "mass_head": mass, "n": int(len(merged)), "n_train_pairs": int(ok.sum()),
        "ms_slope": float(slope), "ms_intercept": float(icpt),
        "r2_sfr_head": r2_head, "r2_true_mstar": r2_true_m, "r2_pred_mstar": r2_pred_m,
        "margin": r2_head - best,
        "verdict": "MEASURING SFR" if r2_head > best + 0.02 else "NOT clearly beating mass alone",
    }, ""


# --------------------------------------------------------------- HR summaries
def hr_summary_rows(hr_records: list[dict], ref: pd.DataFrame) -> list[dict]:
    """Joint-implied HR against the measured HR32, plus the same for the baseline.

    ``ref`` is indexed by targetid and carries ``hr32`` and ``hr32_ok``.
    """
    if not hr_records:
        return []
    hr_df = pd.DataFrame(hr_records)
    aligned = ref.reindex(hr_df.targetid.values)
    meas = aligned.hr32.to_numpy(np.float64)
    ok = aligned.hr32_ok.to_numpy(bool) if "hr32_ok" in aligned.columns else np.zeros(len(meas), bool)
    rows = []
    for label, sel in (("all finite", np.isfinite(meas)), ("hr32_ok", ok & np.isfinite(meas))):
        if int(sel.sum()) < 30:
            continue
        denom = float(np.sum((meas[sel] - meas[sel].mean()) ** 2))
        row = {"subset": label, "n": int(sel.sum())}
        for tag, cols in (("joint", ("hr_p16", "hr_p50", "hr_p84")),
                          ("independent", ("hr_ind_p16", "hr_ind_p50", "hr_ind_p84"))):
            if cols[1] not in hr_df.columns:
                continue
            pred = hr_df[cols[1]].to_numpy(np.float64)
            width = 0.5 * (hr_df[cols[2]].to_numpy(np.float64) - hr_df[cols[0]].to_numpy(np.float64))
            row[f"corr_{tag}"] = float(np.corrcoef(pred[sel], meas[sel])[0, 1])
            row[f"r2_{tag}"] = 1.0 - float(np.sum((meas[sel] - pred[sel]) ** 2)) / denom
            row[f"halfwidth_{tag}"] = float(np.median(width[sel]))
        rows.append(row)
    return rows
