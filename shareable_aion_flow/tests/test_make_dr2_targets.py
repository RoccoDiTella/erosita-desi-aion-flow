"""Tests for the counts columns in scripts/make_dr2_targets.py.

The point of carrying counts is that a NON-DETECTION is an exact zero and must
survive the build untouched, so every test here is about a value the previous
log-flux path would have destroyed or silently repaired: N = 0, a negative
background, and an int16 APE_CTS that has already wrapped in the parent
catalogue. Each builds a tiny synthetic pair of FITS tables, runs the script,
and reads the CSV back.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

REPO_ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "make_dr2_targets", REPO_ROOT / "scripts" / "make_dr2_targets.py")
assert _spec is not None and _spec.loader is not None
mdt = importlib.util.module_from_spec(_spec)
sys.modules["make_dr2_targets"] = mdt
_spec.loader.exec_module(mdt)

BANDS = mdt.BANDS                       # (1, 2, 3, 4)
SUFFIX = [s for _, s in mdt.BAND_SUFFIX]  # ("1", "p1", ... "p4")
N = 4
RA = np.array([10.0, 10.1, 10.2, 10.3])
DEC = np.array([-5.0, -5.1, -5.2, -5.3])
DETUID = np.array([f"eDET{i:03d}" for i in range(N)])


def write_counterparts(path: Path) -> None:
    cols = [fits.Column(name="DETUID", format="10A", array=DETUID),
            fits.Column(name="LS10_RA", format="D", array=RA),
            fits.Column(name="LS10_DEC", format="D", array=DEC)]
    for name in mdt.COUNTERPART_COLUMNS:
        if name == "NWAY_match_flag":
            cols.append(fits.Column(name=name, format="J", array=np.ones(N, int)))
        elif name in ("LS10_RELEASE", "LS10_BRICKID", "LS10_OBJID"):
            cols.append(fits.Column(name=name, format="J", array=np.arange(N)))
        elif name in ("class_gal_exgal", "simbad_known_galactic"):
            cols.append(fits.Column(name=name, format="J", array=np.zeros(N, int)))
        else:
            cols.append(fits.Column(name=name, format="E", array=np.full(N, 0.9, np.float32)))
    fits.BinTableHDU.from_columns(cols).writeto(path, overwrite=True)


def write_main(path: Path, *, ape_cts: dict[str, np.ndarray],
               ape_bkg: dict[str, np.ndarray] | None = None) -> None:
    """Main catalogue. `ape_cts` maps our band suffix -> int16 counts, verbatim."""
    rng = np.random.default_rng(0)
    cols = [fits.Column(name="DETUID", format="10A", array=DETUID),
            fits.Column(name="DET_LIKE_0", format="E", array=np.full(N, 20.0, np.float32))]
    for fits_b, ours in mdt.BAND_SUFFIX:
        flux = np.full(N, 1e-13, np.float32)
        cols += [
            fits.Column(name=f"ML_FLUX_{fits_b}", format="E", array=flux),
            fits.Column(name=f"ML_FLUX_LOWERR_{fits_b}", format="E", array=flux * 0.1),
            fits.Column(name=f"ML_FLUX_UPERR_{fits_b}", format="E", array=flux * 0.1),
            fits.Column(name=f"ML_EXP_{fits_b}", format="E",
                        array=np.full(N, 300.0, np.float32)),
            fits.Column(name=f"APE_CTS_{fits_b}", format="I",
                        array=ape_cts[ours].astype(np.int16)),
            fits.Column(name=f"APE_BKG_{fits_b}", format="E",
                        array=(np.full(N, 0.5, np.float32) if ape_bkg is None
                               else ape_bkg[ours].astype(np.float32))),
            fits.Column(name=f"APE_EXP_{fits_b}", format="E",
                        array=np.full(N, 301.0, np.float32)),
            fits.Column(name=f"APE_RADIUS_{fits_b}", format="E",
                        array=np.full(N, 7.5, np.float32)),
            fits.Column(name=f"APE_POIS_{fits_b}", format="E",
                        array=rng.random(N).astype(np.float32)),
            fits.Column(name=f"ML_CTS_{fits_b}", format="E",
                        array=np.full(N, 12.0, np.float32)),
            fits.Column(name=f"ML_RATE_{fits_b}", format="E",
                        array=np.full(N, 0.04, np.float32)),
            fits.Column(name=f"ML_EEF_{fits_b}", format="E",
                        array=np.full(N, 0.8836025, np.float32)),
        ]
        if fits_b != "1":
            cols.append(fits.Column(name=f"DET_LIKE_{fits_b}", format="E",
                                    array=np.full(N, 10.0, np.float32)))
    fits.BinTableHDU.from_columns(cols).writeto(path, overwrite=True)


def write_desi(path: Path) -> None:
    pd.DataFrame({"targetid": np.arange(1000, 1000 + N),
                  "target_ra": RA, "target_dec": DEC,
                  "z": np.full(N, 0.5), "spectype": ["QSO"] * N,
                  "zwarn": np.zeros(N, int), "survey": ["main"] * N,
                  "program": ["dark"] * N, "healpix": np.arange(N)}).to_csv(path, index=False)


def build(tmp_path: Path, **kwargs) -> pd.DataFrame:
    write_counterparts(tmp_path / "cp.fits")
    write_main(tmp_path / "main.fits", **kwargs)
    write_desi(tmp_path / "desi.csv")
    out = tmp_path / "targets.csv"
    argv = ["make_dr2_targets.py", "--counterparts", str(tmp_path / "cp.fits"),
            "--main", str(tmp_path / "main.fits"), "--desi", str(tmp_path / "desi.csv"),
            "--out", str(out)]
    old, sys.argv = sys.argv, argv
    try:
        mdt.main()
    finally:
        sys.argv = old
    return pd.read_csv(out)


def zeros_and_ones(cts: int = 7) -> dict[str, np.ndarray]:
    """One non-detection (row 0) and three ordinary sources, in every band."""
    return {s: np.array([0, cts, cts + 1, cts + 2]) for s in SUFFIX}


def test_every_counts_column_is_carried(tmp_path):
    frame = build(tmp_path, ape_cts=zeros_and_ones())
    for suffix in SUFFIX:
        for name in mdt.APE_COLUMNS.values():
            assert f"{name}_{suffix}" in frame.columns
        for name in mdt.ML_COUNT_COLUMNS.values():
            if name == "ml_exp" and suffix == "1":
                continue          # the broad band's exposure is `ero_exp`
            assert f"{name}_{suffix}" in frame.columns
        assert frame[f"ape_cts_{suffix}"].notna().all()
        assert frame[f"ape_bkg_{suffix}"].notna().all()
        assert frame[f"ape_exp_{suffix}"].notna().all()
    # ML_EXP_1 is not duplicated: it is already carried, under its own name.
    assert "ml_exp_1" not in frame.columns
    assert np.allclose(frame["ero_exp"], 300.0)


def test_zero_counts_survive_as_zero(tmp_path):
    """The whole reason for the column set: log10(0) drops a row, N = 0 does not."""
    frame = build(tmp_path, ape_cts=zeros_and_ones())
    assert len(frame) == N
    for suffix in SUFFIX:
        col = frame[f"ape_cts_{suffix}"]
        assert col.iloc[0] == 0
        assert pd.api.types.is_integer_dtype(col), f"ape_cts_{suffix} is {col.dtype}, not integer"
        assert (col.to_numpy() == np.array([0, 7, 8, 9])).all()


def test_counts_are_not_transformed_or_thresholded(tmp_path):
    """No scaling, no background subtraction, no detection cut -- catalogue values."""
    cts = {s: np.array([0, 3, 41, 900]) for s in SUFFIX}
    frame = build(tmp_path, ape_cts=cts)
    for suffix in SUFFIX:
        assert (frame[f"ape_cts_{suffix}"].to_numpy() == cts[suffix]).all()
        assert np.allclose(frame[f"ape_bkg_{suffix}"], 0.5)
        assert np.allclose(frame[f"ape_exp_{suffix}"], 301.0)
        assert np.allclose(frame[f"ml_cts_{suffix}"], 12.0)
        assert np.allclose(frame[f"ml_rate_{suffix}"], 0.04)
        assert np.allclose(frame[f"ml_eef_{suffix}"], 0.8836025, atol=1e-7)


def test_int16_overflow_is_carried_negative_and_reported(tmp_path, capsys):
    """A wrapped APE_CTS stays wrapped: +65536 is not invertible, so it is not applied."""
    cts = zeros_and_ones()
    cts["p2"] = np.array([0, 7, 8, -23089])          # a real eRASS:3 wrapped value
    frame = build(tmp_path, ape_cts=cts)
    assert frame["ape_cts_p2"].iloc[3] == -23089
    printed = capsys.readouterr().out
    assert "ape_cts_p2" in printed and "NEGATIVE" in printed
    for suffix in SUFFIX:
        if suffix != "p2":
            assert (frame[f"ape_cts_{suffix}"] >= 0).all()


def test_negative_background_is_carried_and_reported(tmp_path, capsys):
    bkg = {s: np.full(N, 0.5) for s in SUFFIX}
    bkg["1"] = np.array([-0.05, 0.5, 0.5, 0.5])
    frame = build(tmp_path, ape_cts=zeros_and_ones(), ape_bkg=bkg)
    assert frame["ape_bkg_1"].iloc[0] == pytest.approx(-0.05, abs=1e-6)
    assert "ape_bkg_1" in capsys.readouterr().out


def test_clean_sample_prints_no_counts_warning(tmp_path, capsys):
    """Silence is the signal: a warning that always fires reports nothing."""
    build(tmp_path, ape_cts=zeros_and_ones())
    assert "[counts]" not in capsys.readouterr().out
