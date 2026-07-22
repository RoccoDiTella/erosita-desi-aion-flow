#!/usr/bin/env python
"""Model-level leakage: how much injected-line information survives token drops?

The codec-level probe (codec_leakage_probe.py) counts CODE flips, which
saturates: a flipped code is not extractable information. This calibration
measures the quantity the Shapley game actually depends on -- the change in the
trained surrogate's log-likelihood caused by a line the model should not be
able to see:

    leak(g) = mean_s | LL(spectrum_s + injected line; window+-g dropped)
                      - LL(spectrum_s;                 window+-g dropped) |

For guard g large enough, leak(g) -> 0. The reference scale is the same
contrast with NO tokens dropped (the line's full effect on the model). Compare
leak(g) against the per-line Shapley values (~1-3 mnats): if leak(guard) is
well below them, the guard suffices.

    python scripts/model_leakage_calibration.py --checkpoint <best.pt> \
        --staged-dir <staged_paper> --clean-split-csv <clean_split.csv> \
        [--n-batches 8] [--batch-size 256]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shareable_aion_flow.data_to_aion_embeddings import build_dataloaders  # noqa: E402
from shareable_aion_flow.line_shapley import (  # noqa: E402
    GRID_STEP,
    N_SPEC_TOKENS,
    masked_log_prob,
    token_to_raw_pixels,
)
from shareable_aion_flow.main import load_checkpoint  # noqa: E402

INJECT_TOKENS = (60, 140, 220)  # spread across the grid; site randomized per source
LINE_SHAPES = ((500.0, 3.0), (5000.0, 1.0))  # (FWHM km/s, peak amp x local continuum)
GUARDS = (0, 1, 2, 3, 5, 8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--staged-dir", type=Path, required=True)
    ap.add_argument("--clean-split-csv", type=Path, default=None)
    ap.add_argument("--n-batches", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, context_encoder, flow, standardizer, config = load_checkpoint(
        args.checkpoint, device=device, dropout=0.0
    )
    target = config.get("target", "log_ml_flux_1")
    _, _, test_loader = build_dataloaders(
        staged_dir=args.staged_dir, target_name=target, batch_size=args.batch_size,
        num_workers=args.num_workers, clean_split_csv=args.clean_split_csv,
    )
    rng = np.random.default_rng(args.seed)

    def lp(batch, mask_np):
        mask = None if mask_np is None else torch.from_numpy(mask_np).to(device)
        if mask is None:
            mask = torch.zeros((batch[0].shape[0], N_SPEC_TOKENS), dtype=torch.bool, device=device)
        return masked_log_prob(
            encoder=encoder, context_encoder=context_encoder, flow=flow,
            batch=batch, standardizer=standardizer,
            spectrum_token_mask=mask, mask_mode="drop",
        )

    abs_leak = {(g, f): [] for g in GUARDS for f, _ in LINE_SHAPES}
    full_effect = {f: [] for f, _ in LINE_SHAPES}
    for bi, batch in enumerate(test_loader):
        if bi >= args.n_batches:
            break
        batch = tuple(t.to(device, non_blocking=True) for t in batch)
        flux = batch[0]
        n_raw = flux.shape[1]
        B = flux.shape[0]
        wave_np = batch[2][0].detach().cpu().numpy()
        sites = rng.choice(INJECT_TOKENS, size=B)
        for fwhm, amp_x in LINE_SHAPES:
            injected = flux.clone()
            window_tokens: list[list[int]] = []
            for b in range(B):
                q = int(sites[b])
                p_lo, p_hi = token_to_raw_pixels(q)
                center = float(wave_np[(p_lo + p_hi) // 2])
                local = float(
                    torch.median(flux[b, max(p_lo - 200, 0): min(p_hi + 200, n_raw)]).item()
                )
                if not np.isfinite(local) or local <= 0:
                    local = float(torch.median(flux[b]).clamp_min(1e-3).item())
                sigma_a = center * (fwhm / 299792.458) / 2.355
                line = amp_x * local * np.exp(-0.5 * ((wave_np - center) / sigma_a) ** 2)
                injected[b] += torch.tensor(line, dtype=flux.dtype, device=device)
                half = max(1, int(np.ceil(3 * sigma_a / (GRID_STEP * 32))))
                window_tokens.append([t for t in range(q - half, q + half + 1) if 0 <= t < N_SPEC_TOKENS])

            batch_inj = (injected,) + batch[1:]
            lp_orig_full = lp(batch, None)
            lp_inj_full = lp(batch_inj, None)
            full_effect[fwhm].extend(np.abs(lp_inj_full - lp_orig_full).tolist())
            for g in GUARDS:
                mask = np.zeros((B, N_SPEC_TOKENS), dtype=bool)
                for b in range(B):
                    q = int(sites[b])
                    lo = min(window_tokens[b]) - g
                    hi = max(window_tokens[b]) + g
                    mask[b, max(lo, 0): min(hi + 1, N_SPEC_TOKENS)] = True
                d = np.abs(lp(batch_inj, mask) - lp(batch, mask))
                abs_leak[(g, fwhm)].extend(d.tolist())

    print(f"{'guard':>6} " + "".join(f"{f'leak fwhm={int(f)}':>18}" for f, _ in LINE_SHAPES)
          + f"{'(mnats, mean |dLL|)':>22}")
    for g in GUARDS:
        row = "".join(f"{1e3 * float(np.mean(abs_leak[(g, f)])):18.3f}" for f, _ in LINE_SHAPES)
        print(f"{g:6d} {row}")
    ref = "".join(f"{1e3 * float(np.mean(full_effect[f])):18.3f}" for f, _ in LINE_SHAPES)
    print(f"{'full':>6} {ref}   <- line NOT dropped (reference: total model effect)")


if __name__ == "__main__":
    main()
