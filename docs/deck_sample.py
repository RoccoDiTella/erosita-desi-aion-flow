#!/usr/bin/env python
"""One rule for "which rows of multi_test_metrics.csv is this figure about".

``scripts/eval_multitarget.py --sample both`` writes TWO rows per
(head, input_group): one scored on the rows that support that combination
(``native``) and one scored on the rows that support EVERY modality
(``common``), which is the only row set in which a difference between two
combinations is a statement about inputs rather than about which sources
happened to survive.

Before this module no deck consumer looked at the column:

* ``make_modality_upset.py`` and ``make_html_deck.py`` built
  ``dict(zip(input_group, value))``, which keeps the LAST row and therefore
  silently discarded the common-sample number that is the entire reason the
  flag exists.
* ``make_slides.py`` rendered one visual row per CSV row, so a 15-combination
  table came out with 30 rows.
* ``make_results_figure.py`` and ``make_v3b_figures.py`` filtered
  ``input_group == "spectra+z+wise+image"`` and took ``.iloc[0]`` of what had
  become two rows.

Every consumer now calls :func:`select_sample`, which returns exactly one row
per (head, input_group) or exits with a message saying what to pass. Silently
taking the last row is the one behaviour this module exists to make impossible.

Pure pandas on purpose: the deck must rebuild on a laptop with no torch, no
h5py and no AION, so this cannot live in ``shareable_aion_flow/eval_core.py``.
"""

from __future__ import annotations

import pandas as pd

SAMPLE_COLUMN = "sample"
SAMPLE_CHOICES = ("common", "native")
ROW_KEY = ("head", "input_group")

# Mirrors eval_core.HR_BANDS. Used ONLY to explain in prose why a hardness panel
# is absent -- nothing here indexes a flow, a joint column or a target matrix by
# these names, so this copy cannot drift into a wrong number the way a second
# copy of a column index would.
HR_BANDS = ("log_flux_p2", "log_flux_p3")

_MEANING = {
    "common": ("every source counted here carries all four inputs, so the row set is "
               "identical for every combination and a difference between two of them "
               "is an input effect"),
    "native": ("each combination is scored on the rows that support it, so the row sets "
               "differ between combinations and differences between them are NOT "
               "input effects"),
}
_MEANING_SHORT = {
    "common": "identical row set for every combination",
    "native": "row sets differ between combinations",
}


def add_sample_argument(parser, *, default: str | None = None) -> None:
    """``--sample``: which declaration this figure renders. No default guessing.

    Left as None the consumer accepts a table that carries exactly one
    declaration and refuses one that carries two, which is what ``--sample both``
    produces.
    """
    parser.add_argument(
        "--sample", choices=SAMPLE_CHOICES, default=default,
        help="row-set declaration to render: common = sources with all four inputs "
             "(required for any cross-combination comparison), native = each "
             "combination on the rows that support it. Required when the metrics "
             "table carries both.")


def select_sample(frame: pd.DataFrame, requested: str | None = None, *,
                  source: str = "the metrics table",
                  log=print) -> tuple[pd.DataFrame, str]:
    """Return ``(rows of exactly one sample, its name)``.

    Exits loudly when the column is missing, when the table carries more than
    one declaration and none was requested, or when the filtered table still has
    two rows for one (head, input_group) -- all three of which used to resolve
    themselves by keeping whichever row pandas happened to see last.
    """
    if SAMPLE_COLUMN not in frame.columns:
        raise SystemExit(
            f"[deck] {source} has no '{SAMPLE_COLUMN}' column.\n"
            "        Every row scripts/eval_multitarget.py writes carries one, because a\n"
            "        metric without its row set is not interpretable: 'IG 0.31' says nothing\n"
            "        until you know whether it was measured on 22% of the test set or all of\n"
            "        it. This table predates that, so the deck refuses to publish from it.\n"
            "        Re-run the eval. If you know the table is pre-declaration and every row\n"
            "        is the native row set, say so explicitly:\n"
            f"          python3 -c \"import pandas,sys; d=pandas.read_csv(sys.argv[1]); "
            f"d['{SAMPLE_COLUMN}']='native'; d.to_csv(sys.argv[1],index=False)\" <csv>")

    col = frame[SAMPLE_COLUMN]
    if col.isna().any():
        raise SystemExit(
            f"[deck] {source}: {int(col.isna().sum())} rows have an empty "
            f"'{SAMPLE_COLUMN}'. Refusing to guess which row set they came from.")
    present = tuple(sorted(set(col.astype(str))))
    unknown = [s for s in present if s not in SAMPLE_CHOICES]
    if unknown:
        raise SystemExit(
            f"[deck] {source}: unknown sample declaration(s) {unknown}; "
            f"expected some of {list(SAMPLE_CHOICES)}.")

    if requested is None:
        if len(present) != 1:
            raise SystemExit(
                f"[deck] {source} carries {len(present)} sample declarations {list(present)} "
                "(that is what `--sample both` writes), so every (head, input_group) appears "
                "more than once.\n"
                "        Pass --sample common or --sample native. Taking whichever row came "
                "last is how the common-sample numbers were being discarded.")
        chosen = present[0]
    else:
        if requested not in present:
            raise SystemExit(
                f"[deck] --sample {requested} was asked for but {source} only has "
                f"{list(present)}. Re-run the eval with --sample {requested} (or --sample both).")
        chosen = requested

    sub = frame[col.astype(str) == chosen].copy()
    key = [c for c in ROW_KEY if c in sub.columns]
    if key:
        dup = sub.duplicated(subset=key, keep=False)
        if bool(dup.any()):
            offenders = sub.loc[dup, key].astype(str).agg(" / ".join, axis=1).unique()[:5]
            raise SystemExit(
                f"[deck] {source}: sample='{chosen}' still has repeated rows for "
                f"{list(offenders)}. One row per {tuple(key)} is required; the deck will not "
                "pick one for you.")
    log(f"[deck] sample={chosen}: {len(sub)} of {len(frame)} rows from {source}")
    return sub, chosen


def sample_meaning(sample: str, *, short: bool = False) -> str:
    """One clause a reader can act on, for a caption or a subtitle."""
    table = _MEANING_SHORT if short else _MEANING
    return table.get(sample, f"sample={sample}")


def sample_caption(sample: str, frame: pd.DataFrame | None = None) -> str:
    """`sample=common · 5,431 of 22,800 rows scored` for a figure or slide."""
    text = f"sample={sample}"
    if frame is None or not len(frame) or "n_sample" not in frame.columns:
        return text
    lo, hi = int(frame.n_sample.min()), int(frame.n_sample.max())
    scored = f"{lo:,}" if lo == hi else f"{lo:,} to {hi:,}"
    if "n_test" in frame.columns and frame.n_test.notna().any():
        pool = int(frame.n_test.max())
        return f"{text} · {scored} of {pool:,} labelled test rows scored"
    return f"{text} · {scored} rows scored"


def cross_combo_warning(sample: str) -> str | None:
    """Text to print ON a figure whose bars are compared against each other."""
    if sample == "common":
        return None
    return ("row sets differ between bars (sample=" + sample + "), so a difference "
            "between two bars is not a modality effect")


# --------------------------------------------------------------- hardness ratio
def joint_dims_from_metrics(frame: pd.DataFrame) -> tuple[str, ...] | None:
    """The joint this run actually trained, read off the metrics table."""
    if "joint_dims" not in frame.columns:
        return None
    vals = [str(v) for v in frame.joint_dims.dropna().unique() if str(v).strip()]
    if len(vals) != 1:
        return None
    return tuple(vals[0].split("+"))


def hr_absence_reason(frame: pd.DataFrame, path=None) -> str:
    """Why there is no hardness-ratio panel, stated from THIS run's own table.

    The panel used to disappear without a word: ``build_deck.sh`` passed
    ``--hr-csv`` unconditionally and both consumers guarded on ``.exists()``, so
    an unproducible file and a file nobody asked for looked identical on the
    page. The reason is derived rather than hardcoded because it is
    run-dependent: a 2-D P2xP3 joint can produce the panel and a 4-D one cannot.
    """
    where = f" ({path})" if path else ""
    dims = joint_dims_from_metrics(frame)
    head = f"No hardness-ratio panel: hr_implied_target.csv was not produced by this run{where}."
    if dims is None:
        return (f"{head} The metrics table does not name the joint, so the reason cannot be read "
                f"off this run. scripts/hr_from_joint.py runs only for an exactly 2-D "
                f"{HR_BANDS} joint, and only when HR_REF_CSV is set in sbatch/eval_multi.sbatch.")
    if len(dims) == 2 and set(dims) == set(HR_BANDS):
        return (f"{head} The joint is 2-D over {dims}, so the quadrature applies; the stage is "
                "opt-in and did not run. Set HR_REF_CSV in sbatch/eval_multi.sbatch to a "
                "hardness reference measured at the same depth as the labels.")
    return (f"{head} This run's joint is {dims} ({len(dims)}-D). The implied-HR quadrature is an "
            f"exactly 2-D shear over {HR_BANDS}, so scripts/hr_from_joint.py refuses this "
            "checkpoint rather than reinterpreting two of its axes as the bands.")


def hr_note_for(frame: pd.DataFrame, hr_csv=None, log=print) -> str | None:
    """None when the HR file is there, otherwise the sentence to publish."""
    if hr_csv is not None and getattr(hr_csv, "exists", lambda: False)():
        return None
    note = hr_absence_reason(frame, hr_csv)
    log(f"[deck] {note}")
    return note
