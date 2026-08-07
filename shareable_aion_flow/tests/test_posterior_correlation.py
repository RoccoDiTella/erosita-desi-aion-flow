"""Tests for scripts/posterior_correlation.py -- Run A's actual scientific result.

The test that matters most is `test_recovers_known_conditional_correlation`:
a joint whose TRUE conditional correlation is fixed by construction, checked
against both estimators. Everything else guards a specific way the claim could
be wrong rather than merely broken:

  * the two spellings of the partial correlation (closed form vs precision
    matrix) must agree, because the run plan asserts the equivalence;
  * the kernel estimator must see a rho that FLIPS SIGN with the control while
    the partial correlation averages it to nothing -- that is the entire reason
    two estimators exist;
  * both nulls must be centred on zero with the analytic Gaussian width, since
    every "excess over null" statement is measured against them;
  * the sanity check must hold where it is a theorem (r_xy <= 0) and must be
    allowed to fail where it is not (asymmetric r_xk / r_yk).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "posterior_correlation", REPO_ROOT / "scripts" / "posterior_correlation.py")
assert _spec is not None and _spec.loader is not None
pc = importlib.util.module_from_spec(_spec)
sys.modules["posterior_correlation"] = pc
_spec.loader.exec_module(pc)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def raw_from_conditional(rho_true: float, r_xk: float, r_yk: float) -> float:
    """Invert the partial-correlation formula: r_xy that yields rho_true.

    Parametrising by the CONDITIONAL correlation rather than the raw one is not
    cosmetic. It makes the quantity under test the input, and it guarantees a
    positive-definite matrix for every |rho_true| < 1 -- the raw-correlation
    parametrisation silently admits infeasible triples (r_xy = -0.3 with
    r_xk = 0.65, r_yk = 0.55 has a negative determinant and no joint exists).
    """
    return r_xk * r_yk + rho_true * np.sqrt((1 - r_xk ** 2) * (1 - r_yk ** 2))


def true_partial(r_xy: float, r_xk: float, r_yk: float) -> float:
    return (r_xy - r_xk * r_yk) / np.sqrt((1 - r_xk ** 2) * (1 - r_yk ** 2))


def gaussian_trio(n_sources: int, n_draws: int, rho_true: float, r_xk: float,
                  r_yk: float, rng: np.random.Generator,
                  sd=(1.0, 1.0, 1.0)) -> np.ndarray:
    """[N, S, 3] draws whose TRUE Corr(x, y | control) is exactly `rho_true`."""
    r_xy = raw_from_conditional(rho_true, r_xk, r_yk)
    r = np.array([[1.0, r_xy, r_xk],
                  [r_xy, 1.0, r_yk],
                  [r_xk, r_yk, 1.0]])
    assert np.linalg.eigvalsh(r).min() > -1e-12, "infeasible correlation triple"
    chol = np.linalg.cholesky(r + np.eye(3) * 1e-12)
    z = rng.standard_normal((n_sources, n_draws, 3))
    return ((z @ chol.T) * np.asarray(sd)).astype(np.float32)


# --------------------------------------------------------------------------
# THE test: a known conditional correlation, recovered by both estimators
# --------------------------------------------------------------------------
@pytest.mark.parametrize("want, r_xk, r_yk", [
    (-0.56, 0.70, 0.60),    # conditional NEGATIVE while the raw corr is +0.10
    (0.00, 0.60, 0.70),     # conditional exactly zero (r_xy = r_xk r_yk)
    (0.55, 0.50, 0.40),     # conditional strongly positive
    (-0.72, 0.65, 0.55),    # raw and conditional both negative
])
def test_recovers_known_conditional_correlation(want, r_xk, r_yk) -> None:
    """Gaussian joint: the true Corr(x, y | k) is fixed by construction.

    For a joint Gaussian the conditional correlation is CONSTANT in k and equals
    the partial correlation exactly, so the same target checks both estimators
    and the kernel estimator's agreement is a real test rather than a tautology.
    """
    rng = np.random.default_rng(11)
    r_xy = raw_from_conditional(want, r_xk, r_yk)
    trio = gaussian_trio(40, 20_000, want, r_xk, r_yk, rng)

    out = pc.estimate(trio, (0.1, 0.3, 0.5, 0.7, 0.9), 1.0, 25.0, 512)
    assert np.nanmean(out["rho_partial"]) == pytest.approx(want, abs=0.02)

    # Kernel: local-constant weighting has an O(h^2) bias that pulls rho toward
    # the RAW correlation, so it is checked at a tighter bandwidth and with a
    # tolerance that still excludes the raw value it could have collapsed to.
    tight = pc.estimate(trio, (0.5,), 0.35, 25.0, 512)
    got = float(np.nanmean(tight["rho_kernel"]))
    assert got == pytest.approx(want, abs=0.06)
    if abs(want - r_xy) > 0.15:
        assert abs(got - want) < abs(got - r_xy), "kernel collapsed to the raw corr"


def test_true_conditional_correlation_is_flat_for_a_gaussian() -> None:
    """The Gaussian case has no M*-dependence, so the grid must come out flat."""
    rng = np.random.default_rng(3)
    trio = gaussian_trio(60, 20_000, -0.56, 0.70, 0.60, rng)
    k = pc.kernel_conditional_corr(trio, (0.1, 0.3, 0.5, 0.7, 0.9), 0.5, 25.0)
    prof = np.nanmean(k["rho"], axis=0)
    assert prof.max() - prof.min() < 0.05, f"spurious trend in rho(k): {prof}"
    assert np.nanmedian(np.abs(pc.trend_slope(k["grid"], k["rho"]))) < 0.15


# --------------------------------------------------------------------------
# the two estimators are NOT the same thing -- the non-Gaussian case
# --------------------------------------------------------------------------
def test_kernel_sees_a_sign_flip_that_partial_correlation_averages_away() -> None:
    """A joint where rho(k) flips sign: the Gaussian shortcut must fail here.

    Built so that Corr(x, y | k) = +a for k < 0 and -a for k > 0. The partial
    correlation, being a single linear number, has to land near zero; the kernel
    estimator has to recover both signs. If this test ever passes for the partial
    correlation too, the two estimators are not measuring different things and
    the whole two-estimator design is pointless.
    """
    rng = np.random.default_rng(5)
    n, s = 24, 30_000
    k = rng.standard_normal((n, s))
    e1 = rng.standard_normal((n, s))
    e2 = rng.standard_normal((n, s))
    lam = np.where(k < 0, 0.85, -0.85)          # sign of the shared component
    x = e1
    y = lam * e1 + np.sqrt(1 - 0.85 ** 2) * e2
    trio = np.stack([x, y, k], axis=-1).astype(np.float32)

    out = pc.estimate(trio, (0.05, 0.25, 0.5, 0.75, 0.95), 0.4, 25.0, 512)
    assert abs(np.nanmean(out["rho_partial"])) < 0.15, "partial should average to ~0"

    prof = np.nanmean(out["rho_kernel_grid"], axis=0)
    assert prof[0] > 0.4, f"kernel missed the positive branch: {prof}"
    assert prof[-1] < -0.4, f"kernel missed the negative branch: {prof}"
    assert np.nanmedian(out["rho_kernel_slope"]) < -0.3


# --------------------------------------------------------------------------
# the two spellings of the partial correlation
# --------------------------------------------------------------------------
def test_closed_form_and_precision_matrix_agree() -> None:
    rng = np.random.default_rng(7)
    trio = np.concatenate([
        gaussian_trio(20, 4000, -0.56, 0.7, 0.6, rng),
        gaussian_trio(20, 4000, -0.80, 0.3, 0.8, rng),
        gaussian_trio(20, 4000, +0.50, -0.2, 0.5, rng)])
    c = pc.corr_from_draws(trio)
    closed = pc.partial_corr_closed_form(c[:, 0, 1], c[:, 0, 2], c[:, 1, 2])
    prec = pc.partial_corr_precision(c)
    assert np.allclose(closed, prec, atol=1e-8)


def test_corr_from_draws_matches_numpy_corrcoef() -> None:
    rng = np.random.default_rng(13)
    trio = gaussian_trio(5, 2000, 0.3, 0.5, -0.4, rng)   # rho_true = +0.3
    got = pc.corr_from_draws(trio)
    for i in range(trio.shape[0]):
        assert np.allclose(got[i], np.corrcoef(trio[i].T.astype(np.float64)),
                           atol=1e-6)


# --------------------------------------------------------------------------
# the sanity check: theorem where it is one, allowed to fail where it is not
# --------------------------------------------------------------------------
def test_partial_is_more_negative_than_raw_whenever_that_is_a_theorem() -> None:
    """r_xy <= 0 with r_xk, r_yk > 0 forces rho < r_xy. No exceptions.

    r_xy is walked down to the edge of the feasible (positive-definite) region
    for each control pair, so the theorem is checked right where it is tightest.
    """
    rng = np.random.default_rng(17)
    for r_xk, r_yk in ((0.3, 0.3), (0.8, 0.2), (0.6, 0.7), (0.5, 0.5)):
        c0 = r_xk * r_yk
        d0 = np.sqrt((1 - r_xk ** 2) * (1 - r_yk ** 2))
        lo = c0 - d0                                  # smallest feasible r_xy
        assert lo < 0, "pick a pair whose feasible region reaches r_xy <= 0"
        for frac in (0.95, 0.6, 0.2, 0.0):
            rho_true = true_partial(lo * frac, r_xk, r_yk)
            trio = gaussian_trio(30, 8000, rho_true, r_xk, r_yk, rng)
            c = pc.corr_from_draws(trio)
            rho = pc.partial_corr_closed_form(c[:, 0, 1], c[:, 0, 2], c[:, 1, 2])
            ok = (c[:, 0, 2] > 0) & (c[:, 1, 2] > 0) & (c[:, 0, 1] <= 0)
            assert ok.sum(), (r_xk, r_yk, frac)
            assert (rho[ok] < c[ok, 0, 1]).all(), (r_xk, r_yk, frac)


def test_sanity_check_has_a_legitimate_counterexample() -> None:
    """The run plan states the check as unconditional; it is not.

    rho < r holds iff r(1-d) < r_xk r_yk. With strongly ASYMMETRIC r_xk / r_yk
    and a large positive r_xy the inequality reverses, and the script must not
    treat that as a bug. This pins the exact corner so a future reader does not
    'fix' a violation that is arithmetic.
    """
    r_xy, r_xk, r_yk = 0.30, 0.90, 0.10
    rho = pc.partial_corr_closed_form(
        np.array([r_xy]), np.array([r_xk]), np.array([r_yk]))[0]
    assert r_xk > 0 and r_yk > 0
    assert rho > r_xy, "expected the documented counterexample to violate the check"
    d = np.sqrt((1 - r_xk ** 2) * (1 - r_yk ** 2))
    assert r_xy * (1 - d) > r_xk * r_yk          # the exact algebraic condition


# --------------------------------------------------------------------------
# the nulls
# --------------------------------------------------------------------------
def test_permutation_null_destroys_the_per_source_structure() -> None:
    rng = np.random.default_rng(19)
    want = -0.56                                          # by construction
    trio = gaussian_trio(400, 600, want, 0.70, 0.60, rng)
    assert want < -0.4                                    # the signal is real

    nullblock = pc.permutation_null(trio, rng)
    out = pc.estimate(nullblock, (0.5,), 1.0, 25.0, 512)
    assert abs(np.nanmean(out["rho_partial"])) < 0.05, "null is not centred on 0"
    assert np.nanmean(out["rho_partial"]) > want + 0.3, "null kept the signal"
    # the marginals must survive: only the PAIRING is destroyed
    assert np.allclose(np.sort(nullblock[:, :, 0], axis=1),
                       np.sort(trio[:, :, 0], axis=1))
    assert np.allclose(np.sort(nullblock[:, :, 2], axis=1),
                       np.sort(trio[:, :, 2], axis=1))
    assert np.allclose(np.sort(nullblock[:, :, 1].ravel()),
                       np.sort(trio[:, :, 1].ravel()))


def test_derangement_moves_every_source() -> None:
    rng = np.random.default_rng(23)
    for n in (2, 3, 17, 250):
        p = pc._derangement(n, rng)
        assert sorted(p.tolist()) == list(range(n))
        assert (p != np.arange(n)).all()


def test_synthetic_null_has_zero_true_conditional_correlation() -> None:
    """It must keep r_xk and r_yk and kill only the conditional correlation."""
    rng = np.random.default_rng(29)
    trio = gaussian_trio(300, 4000, -0.56, 0.70, 0.60, rng)
    c = pc.corr_from_draws(trio)
    sd = trio.std(axis=1)
    mean = trio.mean(axis=1)

    nullblock = pc.synthetic_independent_null(c, sd, mean, 4000, rng)
    cn = pc.corr_from_draws(nullblock)
    assert np.nanmean(cn[:, 0, 2]) == pytest.approx(np.nanmean(c[:, 0, 2]), abs=0.02)
    assert np.nanmean(cn[:, 1, 2]) == pytest.approx(np.nanmean(c[:, 1, 2]), abs=0.02)
    out = pc.estimate(nullblock, (0.5,), 0.5, 25.0, 512)
    assert abs(np.nanmean(out["rho_partial"])) < 0.03
    assert abs(np.nanmean(out["rho_kernel"])) < 0.06


def test_synthetic_null_width_matches_the_analytic_gaussian_form() -> None:
    """sd of a partial correlation from S draws with 1 control is ~1/sqrt(S-4).

    This is the internal consistency check the script prints: if the synthetic
    null does not reproduce its own closed form, every 'excess over null' number
    downstream is meaningless.
    """
    rng = np.random.default_rng(31)
    s = 512
    trio = gaussian_trio(4000, s, -0.56, 0.70, 0.60, rng)
    c = pc.corr_from_draws(trio)
    nullblock = pc.synthetic_independent_null(
        c, trio.std(axis=1), trio.mean(axis=1), s, rng)
    got = float(np.nanstd(pc.estimate(nullblock, (0.5,), 1.0, 25.0, 512)["rho_partial"]))
    assert got == pytest.approx(1.0 / np.sqrt(s - 4), rel=0.12)


def test_a_true_zero_still_produces_a_negative_tail() -> None:
    """Why the nulls gate the claim: 'some sources are negative' is free.

    With S=512 draws and a TRUE conditional correlation of exactly zero, close to
    half the sources come out negative and a few percent clear -0.1. Any reading
    of the observed tail has to be net of this.
    """
    rng = np.random.default_rng(37)
    trio = gaussian_trio(3000, 512, 0.0, 0.60, 0.70, rng)    # true partial = 0
    rho = pc.estimate(trio, (0.5,), 1.0, 25.0, 512)["rho_partial"]
    assert 0.45 < np.mean(rho < 0) < 0.55
    assert np.mean(rho < -0.1) > 0.005


# --------------------------------------------------------------------------
# summary plumbing
# --------------------------------------------------------------------------
def test_tail_table_reports_the_explained_fraction() -> None:
    obs = np.concatenate([np.full(80, -0.5), np.full(20, 0.5)])
    perm = np.concatenate([np.full(40, -0.5), np.full(60, 0.5)])
    rows = {r["threshold"]: r for r in pc.tail_table(obs, perm, perm,
                                                     thresholds=(-0.2,))}
    r = rows[-0.2]
    assert r["obs"] == pytest.approx(0.8)
    assert r["perm"] == pytest.approx(0.4)
    assert r["explained_perm"] == pytest.approx(0.5)
    assert r["excess_perm"] == pytest.approx(0.4)


def test_describe_handles_all_nan() -> None:
    d = pc.describe(np.full(5, np.nan))
    assert d["n"] == 0 and np.isnan(d["mean"])


def test_kernel_marks_low_effective_sample_size_as_nan() -> None:
    rng = np.random.default_rng(41)
    trio = gaussian_trio(8, 60, -0.56, 0.7, 0.6, rng)
    k = pc.kernel_conditional_corr(trio, (0.5,), 0.05, min_ess=25.0)
    assert np.isnan(k["rho"]).all(), "tiny bandwidth must not yield a rho"
    assert (k["ess"] < 25.0).all()


# --------------------------------------------------------------------------
# end to end, on an npz shaped exactly like posterior_structure.py's
# --------------------------------------------------------------------------
def test_end_to_end_writes_a_per_source_csv(tmp_path: Path) -> None:
    rng = np.random.default_rng(43)
    n, s = 120, 400
    want = -0.56                                          # by construction
    trio = gaussian_trio(n, s, want, 0.70, 0.60, rng, sd=(0.45, 0.60, 0.42))
    trio += np.array([44.0, 1.0, 10.5], dtype=np.float32)
    # npz column order deliberately differs from (x, y, control) so the script's
    # dim lookup is exercised rather than assumed.
    draws = trio[:, :, [2, 1, 0]]
    dims = np.array(["logmstar_cigale", "log_sfr", "log_lx"])
    npz = tmp_path / "poststruct.npz"
    np.savez_compressed(
        npz, draws=draws.astype(np.float32),
        per_source=pc.corr_from_draws(draws).astype(np.float32),
        targetid=np.arange(n, dtype=np.int64),
        group=np.where(np.arange(n) % 4 == 0, "GALAXY", "QSO"),
        dims=dims, truth=draws.mean(1).astype(np.float32),
        mean=draws.mean(1).astype(np.float32), sd=draws.std(1).astype(np.float32))

    out_csv = tmp_path / "per.csv"
    pc.main(["--npz", str(npz), "--out-csv", str(out_csv), "--no-fig",
             "--null-reps", "2"])

    import pandas as pd
    df = pd.read_csv(out_csv)
    assert len(df) == n
    for col in ("targetid", "group", "rho_partial", "rho_kernel",
                "null_perm_rho_partial", "null_synth_rho_partial",
                "rho_kernel_minus_partial", "sig_neg_synth_partial",
                "truth_control", "post_sd_control"):
        assert col in df.columns, col
    assert set(df.group) == {"GALAXY", "QSO"}
    assert df.rho_partial.mean() == pytest.approx(want, abs=0.05)
    assert abs(df.null_synth_rho_partial.mean()) < 0.05


def test_rejects_an_npz_without_draws(tmp_path: Path) -> None:
    npz = tmp_path / "old.npz"
    np.savez_compressed(npz, per_source=np.zeros((3, 3, 3), np.float32),
                        targetid=np.arange(3), group=np.array(["ALL"] * 3),
                        dims=np.array(["logmstar_cigale", "log_sfr", "log_lx"]),
                        truth=np.zeros((3, 3), np.float32),
                        mean=np.zeros((3, 3), np.float32),
                        sd=np.ones((3, 3), np.float32))
    with pytest.raises(SystemExit, match="save-draws"):
        pc.main(["--npz", str(npz), "--no-fig"])
