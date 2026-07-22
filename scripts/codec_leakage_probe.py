#!/usr/bin/env python
"""Measure the spectrum codec's EFFECTIVE receptive field on REAL spectra.

The codec encoder is a ConvNeXt (kernel-7 depthwise convs over 4 scales,
theoretical receptive field ~1,700 px = ~26 tokens per side), so token CODES
near a line can carry line information even when the line's own tokens are
dropped. Remedy (user-chosen design): drop the line window PLUS a guard band of
the empirically measured effective radius -- nothing replaced or imputed.

Method (user-specified): take real staged spectra, add a line-sized Gaussian
perturbation (amplitude scaled to the LOCAL continuum) at known codec tokens,
re-encode, and record which code columns flip vs the unperturbed spectrum.
Two channels are separated:
- LOCAL (receptive field): flip fraction vs distance from the line; the guard
  radius is where this profile decays to the far-field floor.
- GLOBAL (normalization): adding a line shifts the spectrum's global
  normalization, which can flip codes anywhere. This is real line information
  in the norm channel, but it is non-local by nature -- a guard band cannot
  remove it; it is reported as the far-field floor + norm-token flip rate.

Output ends with a machine-readable line: RECOMMENDED_GUARD=<n>.

    python scripts/codec_leakage_probe.py --staged-dir <staged_paper> [--n-spectra 16]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shareable_aion_flow.data_to_aion_embeddings import AIONTokenEncoder  # noqa: E402
from shareable_aion_flow.line_shapley import token_to_raw_pixels  # noqa: E402

PROBE_TOKENS = (30, 120, 200)
LINE_SHAPES = ((500.0, 3.0), (5000.0, 1.0))  # (FWHM km/s, peak amp x local continuum)
FLOOR_RANGE = (20, 100)  # |distance| range treated as the norm-mediated far field


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--staged-dir", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-spectra", type=int, default=16)
    args = ap.parse_args()

    device = torch.device("cpu")
    encoder = AIONTokenEncoder(freeze=True).to(device)
    codec = encoder._codec(device)

    with h5py.File(args.staged_dir / f"desi_{args.split}.hdf5", "r") as h:
        flux_np = h["spectra"][: args.n_spectra].astype(np.float32)
        ivar_np = h["spectra_ivar"][: args.n_spectra].astype(np.float32)
        wave_np = h["spectra_lambda"][:].astype(np.float32)
    n_raw = flux_np.shape[1]
    wave = torch.tensor(wave_np).unsqueeze(0)

    def code_of(flux_row: np.ndarray, ivar_row: np.ndarray) -> torch.Tensor:
        fl = torch.tensor(flux_row).unsqueeze(0)
        iv = torch.tensor(ivar_row).unsqueeze(0)
        td = codec.encode(
            encoder.spectrum_cls(flux=fl, ivar=iv, mask=iv <= 0, wavelength=wave)
        )
        return td[encoder._spec_key(td)]

    offset = encoder._locate_spec_columns(device, wave)
    print(f"wavelength-token column offset: {offset}; probing {len(flux_np)} real spectra")

    flip_counts: dict[int, int] = {}
    trials = 0
    norm_flips = 0
    for s in range(len(flux_np)):
        base_code = code_of(flux_np[s], ivar_np[s])
        for q_line in PROBE_TOKENS:
            p_lo, p_hi = token_to_raw_pixels(q_line)
            if p_lo < 0 or p_hi > n_raw:
                continue
            center = float(wave_np[(p_lo + p_hi) // 2])
            local = float(np.median(flux_np[s, max(p_lo - 200, 0): min(p_hi + 200, n_raw)]))
            if not np.isfinite(local) or local <= 0:
                continue
            for fwhm, amp_x in LINE_SHAPES:
                sigma_a = center * (fwhm / 299792.458) / 2.355
                line = amp_x * local * np.exp(-0.5 * ((wave_np - center) / sigma_a) ** 2)
                pert_code = code_of(flux_np[s] + line.astype(np.float32), ivar_np[s])
                changed = (base_code != pert_code)[0].nonzero().ravel().tolist()
                trials += 1
                for c in changed:
                    d = int(c) - offset - q_line
                    if int(c) - offset < 0:
                        norm_flips += 1
                    else:
                        flip_counts[d] = flip_counts.get(d, 0) + 1

    if trials == 0:
        raise RuntimeError("No usable probe trials (bad spectra or token windows).")
    frac = {d: n / trials for d, n in flip_counts.items()}
    far = [f for d, f in frac.items() if FLOOR_RANGE[0] <= abs(d) <= FLOOR_RANGE[1]]
    floor = float(np.mean(far)) if far else 0.0
    print(f"trials: {trials}; norm-token flip rate: {norm_flips / trials:.2f}; "
          f"far-field flip floor (|d| {FLOOR_RANGE[0]}-{FLOOR_RANGE[1]}): {floor:.3f}")
    print("flip fraction vs token distance from the line (0 = the line's own token):")
    for d in range(-6, 7):
        bar = "#" * int(40 * frac.get(d, 0.0))
        print(f"  d={d:+3d}: {frac.get(d, 0.0):5.2f} {bar}")
    guard = 0
    for d in range(1, 11):
        f_here = max(frac.get(d, 0.0), frac.get(-d, 0.0))
        if f_here > floor + 0.10:
            guard = d
    print(f"RECOMMENDED_GUARD={max(1, guard)}")


if __name__ == "__main__":
    main()
