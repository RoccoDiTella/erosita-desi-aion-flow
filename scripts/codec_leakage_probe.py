#!/usr/bin/env python
"""Measure the spectrum codec's EFFECTIVE receptive field at the token level.

The codec encoder is a ConvNeXt (kernel-7 depthwise convs over 4 scales,
theoretical receptive field ~1,700 px = ~26 tokens per side), so token CODES
near a line can carry line information even when the line's own tokens are
dropped. Remedy (user-chosen design): drop the line window PLUS a guard band of
the empirically measured effective radius -- nothing is replaced or imputed.

This probe injects a Gaussian emission line at known codec tokens on a smooth
continuum and reports which code columns change relative to the line-free
spectrum, as a function of distance from the line. The last line of output is
machine-readable: RECOMMENDED_GUARD=<n> (max observed leak distance beyond the
line's own 3-sigma token window, over all probed positions and widths).

Runs on CPU in ~a minute. Usage: python scripts/codec_leakage_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shareable_aion_flow.data_to_aion_embeddings import AIONTokenEncoder  # noqa: E402
from shareable_aion_flow.line_shapley import (  # noqa: E402
    GRID_STEP,
    PIX_PER_TOKEN,
    RAW_LO,
    token_to_raw_pixels,
)

N_RAW = 7781  # DESI grid 3600-9824 A at 0.8 A


def gaussian_line(wave: np.ndarray, center: float, fwhm_kms: float, amp: float) -> np.ndarray:
    sigma = center * (fwhm_kms / 299792.458) / 2.355
    return amp * np.exp(-0.5 * ((wave - center) / sigma) ** 2)


def main() -> None:
    device = torch.device("cpu")
    encoder = AIONTokenEncoder(freeze=True).to(device)
    codec = encoder._codec(device)

    wave_np = RAW_LO + GRID_STEP * np.arange(N_RAW)
    wave = torch.tensor(wave_np, dtype=torch.float32).unsqueeze(0)
    ivar = torch.ones(1, N_RAW)
    # smooth power-law-ish continuum, mean ~1 (typical normalized DESI scale)
    cont_np = (wave_np / 6000.0) ** -0.5
    cont = torch.tensor(cont_np, dtype=torch.float32).unsqueeze(0)

    def code_of(flux: torch.Tensor) -> torch.Tensor:
        td = codec.encode(
            encoder.spectrum_cls(flux=flux, ivar=ivar, mask=ivar <= 0, wavelength=wave)
        )
        return td[encoder._spec_key(td)]

    base_code = code_of(cont)
    offset = encoder._locate_spec_columns(device, wave)
    print(f"code columns: {base_code.shape[-1]}, wavelength-token column offset: {offset}")

    max_leak = 0
    for q_line in (30, 120, 200):
        p_lo, p_hi = token_to_raw_pixels(q_line)
        center = float(wave_np[(p_lo + p_hi) // 2])
        for fwhm, amp in ((500.0, 2.0), (5000.0, 1.0)):
            line = torch.tensor(
                gaussian_line(wave_np, center, fwhm, amp), dtype=torch.float32
            ).unsqueeze(0)
            changed = (base_code != code_of(cont + line))[0].nonzero().ravel().tolist()
            wl_cols = [c - offset for c in changed if 0 <= c - offset]
            norm_changed = len(wl_cols) != len(changed)
            dists = sorted(c - q_line for c in wl_cols)
            sig_tok = center * (fwhm / 299792.458) / 2.355 / (GRID_STEP * PIX_PER_TOKEN)
            window = max(1, int(np.ceil(3 * sig_tok)))
            leaked = [d for d in dists if abs(d) > window]
            leak_radius = max((abs(d) - window for d in leaked), default=0)
            max_leak = max(max_leak, leak_radius)
            print(
                f"token {q_line} fwhm={fwhm:6.0f} km/s amp={amp}: "
                f"{len(dists)} cols changed, 3-sigma window +-{window}: "
                f"leaked outside at {sorted(set(leaked))} -> radius {leak_radius} "
                f"(norm token changed: {norm_changed})"
            )
    print(f"RECOMMENDED_GUARD={max(1, max_leak)}")


if __name__ == "__main__":
    main()
