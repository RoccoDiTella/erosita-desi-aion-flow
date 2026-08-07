"""Tests for ``docs/deck_sample.py``: the deck's one rule for picking a row set.

``scripts/eval_multitarget.py --sample both`` writes TWO rows per
(head, input_group). Before this module every deck consumer resolved that by
accident -- ``dict(zip(input_group, value))`` keeps whichever row pandas saw
last, ``make_slides`` rendered a 15-combination table as 30 rows, and the two
figure scripts took ``.iloc[0]`` of a two-row frame. The failure was silent and
always fell the same way: it kept ``native`` and discarded ``common``, which is
the only declaration in which an information-gain difference between two
combinations is a statement about inputs.

So the property under test is not "select_sample returns a frame". It is
"nothing in the deck can publish a number whose row set it did not choose on
purpose". Every test here is either that, or the loud refusal that replaces a
silent pick.

Pure pandas, no torch, no h5py: the deck rebuilds on a laptop, and so must its
tests.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs"
if str(DOCS) not in sys.path:
    sys.path.insert(0, str(DOCS))

import deck_sample  # noqa: E402

MODALITIES = ("spectra", "z", "wise", "image")
FOUR_D = "logmstar_cigale+log_sfr+log_lx+log_flux_p3"
P2XP3 = "+".join(deck_sample.HR_BANDS)


def combos() -> list[str]:
    out = []
    for k in (1, 2, 3, 4):
        out += ["+".join(c) for c in itertools.combinations(MODALITIES, k)]
    return out


def metrics_frame(samples=("common", "native"), heads=("log_lx", "log_sfr"),
                  joint_dims: str = FOUR_D) -> pd.DataFrame:
    """A multi_test_metrics.csv shaped exactly like the eval writes it.

    The information gain is a function of the SAMPLE, so a consumer that keeps
    the wrong row silently posts a different number rather than an obviously
    broken one -- which is what happened.
    """
    ig = {"common": 0.25, "native": 0.90}
    n_sample = {"common": 2100, "native": 5200}
    rows = []
    for s in samples:
        for head in heads:
            for group in combos():
                rows.append({
                    "head": head, "input_group": group, "sample": s,
                    "nll": 1.0, "r2": 0.5, "rmse_dex": 0.3,
                    "info_gain_nats": ig[s], "joint_dims": "",
                    "n_test": 5600, "n_common": 2100, "n_sample": n_sample[s],
                    "frac_of_test": n_sample[s] / 5600,
                    "cross_combo_comparable": s == "common",
                })
        rows.append({
            "head": "joint", "input_group": combos()[-1], "sample": s,
            "nll": 2.0, "r2": float("nan"), "rmse_dex": float("nan"),
            "info_gain_nats": float("nan"), "joint_dims": joint_dims,
            "n_test": 4000, "n_common": 1800, "n_sample": n_sample[s],
            "frac_of_test": n_sample[s] / 4000,
            "cross_combo_comparable": s == "common",
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------- the defect itself
def test_last_wins_would_have_kept_native_and_select_sample_does_not() -> None:
    """Pins the exact silent failure, both halves of it.

    Half one: the old ``dict(zip(...))`` really does resolve to the native row on
    a table written by ``--sample both``, so the common-sample numbers were being
    thrown away rather than merely duplicated. Half two: select_sample returns
    the one that was asked for.
    """
    frame = metrics_frame()
    lx = frame[frame["head"] == "log_lx"]
    last_wins = dict(zip(lx.input_group, lx.info_gain_nats))
    assert last_wins["spectra+z+wise+image"] == 0.90, "fixture no longer reproduces the bug"

    chosen, name = deck_sample.select_sample(frame, "common", log=lambda _m: None)
    assert name == "common"
    picked = dict(zip(chosen[chosen["head"] == "log_lx"].input_group,
                      chosen[chosen["head"] == "log_lx"].info_gain_nats))
    assert picked["spectra+z+wise+image"] == 0.25
    assert set(picked) == set(combos())          # nothing dropped by the filter


def test_one_row_per_head_and_input_group_after_selection() -> None:
    frame = metrics_frame()
    chosen, _ = deck_sample.select_sample(frame, "native", log=lambda _m: None)
    assert not chosen.duplicated(subset=["head", "input_group"]).any()
    assert len(chosen) == len(frame) // 2


# ------------------------------------------------------- loud refusals
def test_two_declarations_and_no_request_is_refused() -> None:
    with pytest.raises(SystemExit) as err:
        deck_sample.select_sample(metrics_frame(), None, log=lambda _m: None)
    text = str(err.value)
    assert "common" in text and "native" in text and "--sample" in text


def test_one_declaration_and_no_request_is_accepted() -> None:
    """A table from `--sample common` needs no flag: there is nothing to choose."""
    chosen, name = deck_sample.select_sample(
        metrics_frame(samples=("common",)), None, log=lambda _m: None)
    assert name == "common" and len(chosen) == 2 * len(combos()) + 1


def test_a_missing_sample_column_is_refused_not_defaulted() -> None:
    """A pre-declaration table must not be published as though it were native.

    It probably IS native, and guessing that is exactly the reasoning this
    module exists to forbid; the message tells the reader how to say so on
    purpose instead.
    """
    frame = metrics_frame(samples=("native",)).drop(columns=["sample"])
    with pytest.raises(SystemExit) as err:
        deck_sample.select_sample(frame, None, log=lambda _m: None)
    assert "no 'sample' column" in str(err.value)


def test_empty_and_unknown_declarations_are_refused() -> None:
    blank = metrics_frame(samples=("common",))
    blank.loc[blank.index[:3], "sample"] = None
    with pytest.raises(SystemExit) as err:
        deck_sample.select_sample(blank, None, log=lambda _m: None)
    assert "Refusing to guess" in str(err.value)

    odd = metrics_frame(samples=("common",))
    odd.loc[odd.index[:3], "sample"] = "everything"
    with pytest.raises(SystemExit) as err:
        deck_sample.select_sample(odd, None, log=lambda _m: None)
    assert "everything" in str(err.value)


def test_asking_for_a_declaration_the_table_does_not_carry_is_refused() -> None:
    with pytest.raises(SystemExit) as err:
        deck_sample.select_sample(metrics_frame(samples=("native",)), "common",
                                  log=lambda _m: None)
    assert "only has ['native']" in str(err.value)


def test_duplicate_rows_within_one_declaration_are_refused() -> None:
    """Two rows for one (head, input_group) inside ONE sample is still ambiguous.

    Filtering on the column is not by itself a guarantee of uniqueness, and a
    consumer that then calls ``.iloc[0]`` is back where it started.
    """
    frame = metrics_frame(samples=("common",))
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(SystemExit) as err:
        deck_sample.select_sample(frame, "common", log=lambda _m: None)
    assert "repeated rows" in str(err.value)


# ------------------------------------------------------- what gets published
def test_caption_and_warning_say_which_row_set_the_bars_are_on() -> None:
    frame, _ = deck_sample.select_sample(metrics_frame(), "common", log=lambda _m: None)
    caption = deck_sample.sample_caption("common", frame)
    assert "sample=common" in caption and "2,100" in caption and "5,600" in caption
    # a figure whose bars are read against each other must carry the warning when
    # the row sets behind them differ
    assert deck_sample.cross_combo_warning("common") is None
    assert "not a modality effect" in deck_sample.cross_combo_warning("native")
    assert deck_sample.sample_meaning("common") != deck_sample.sample_meaning("native")


# ------------------------------------------------------- hardness absence (C5)
def test_hr_absence_reason_reads_the_joint_off_this_runs_own_table() -> None:
    """A 4-D joint cannot produce the panel, and the deck says which joint it was."""
    note = deck_sample.hr_absence_reason(metrics_frame(joint_dims=FOUR_D))
    assert "4-D" in note and "log_flux_p2" in note and "hr_implied_target.csv" in note


def test_hr_absence_reason_distinguishes_a_p2xp3_run_that_simply_did_not_run_it() -> None:
    """For Run B's 2-D joint the quadrature APPLIES, so the reason is different.

    Reporting "the joint is the wrong shape" for a run whose joint is exactly
    right would send the reader to fix the model instead of setting HR_REF_CSV.
    """
    note = deck_sample.hr_absence_reason(metrics_frame(joint_dims=P2XP3))
    assert "quadrature applies" in note and "HR_REF_CSV" in note
    assert "-D" not in note.split("quadrature applies")[0].replace("2-D", "")


def test_hr_absence_reason_admits_when_the_table_does_not_name_the_joint() -> None:
    frame = metrics_frame().drop(columns=["joint_dims"])
    note = deck_sample.hr_absence_reason(frame)
    assert "does not name the joint" in note


def test_hr_note_is_none_when_the_file_is_there(tmp_path) -> None:
    path = tmp_path / "hr_implied_target.csv"
    path.write_text("targetid,hr_meas,hr_p50\n1,0.1,0.1\n")
    assert deck_sample.hr_note_for(metrics_frame(), path, log=lambda _m: None) is None
    assert deck_sample.hr_note_for(metrics_frame(), tmp_path / "gone.csv",
                                   log=lambda _m: None) is not None
    # and with no path at all, which is what build_deck.sh passes when the file
    # was never produced
    assert deck_sample.hr_note_for(metrics_frame(), None, log=lambda _m: None) is not None


# ------------------------------------------------------- no consumer opts out
def test_every_deck_consumer_of_the_metrics_table_goes_through_select_sample() -> None:
    """A new consumer must not be able to reintroduce the last-wins read.

    Keyed on ``input_group`` because that column only appears in a per-combination
    metrics table; anything reading it is reading rows that ``--sample both``
    duplicates.
    """
    consumers = {p.name for p in DOCS.glob("*.py")
                 if "input_group" in p.read_text() and p.name != "deck_sample.py"}
    assert consumers, "no deck consumer found; the glob or the column name changed"
    missing = {name for name in consumers
               if "select_sample" not in (DOCS / name).read_text()}
    assert not missing, (
        f"{sorted(missing)} read input_group without deck_sample.select_sample, so a "
        "--sample both table would resolve itself by whichever row came last")
