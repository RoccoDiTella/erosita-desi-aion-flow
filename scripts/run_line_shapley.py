#!/usr/bin/env python
"""Run the line/continuum Shapley sweeps on a trained spectra-only checkpoint.

Outputs into <run-dir>/shapley/: shapley_table.csv (per player: phi, SE, n),
heatmap.png (z-bins x rest wavelength), and line_table.png.

    python scripts/run_line_shapley.py --checkpoint <best.pt> \
        --staged-dir <staged_paper> --clean-split-csv <clean_split.csv> \
        [--full-sweeps 2] [--line-sweeps 4] [--batch-size 256]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shareable_aion_flow.data_to_aion_embeddings import build_dataloaders  # noqa: E402
from shareable_aion_flow.line_shapley import N_LINES, player_catalog, run_sweeps  # noqa: E402
from shareable_aion_flow.main import load_checkpoint  # noqa: E402

ACCENT, LINEC = "#0072B2", "#D55E00"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--staged-dir", type=Path, required=True)
    ap.add_argument("--clean-split-csv", type=Path, default=None)
    ap.add_argument("--full-sweeps", type=int, default=2)
    ap.add_argument("--line-sweeps", type=int, default=4)
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
    players = player_catalog()
    acc = run_sweeps(
        encoder=encoder, context_encoder=context_encoder, flow=flow,
        loader=test_loader, standardizer=standardizer, players=players,
        device=device, n_full_sweeps=args.full_sweeps, n_line_sweeps=args.line_sweeps,
        seed=args.seed,
    )

    out_dir = args.checkpoint.parent / "shapley"
    out_dir.mkdir(exist_ok=True)
    phi, se, count = acc.table()
    table = pd.DataFrame({
        "player": [p["name"] for p in players],
        "kind": [p["kind"] for p in players],
        "rest_center_A": [p["rest_center"] for p in players],
        "phi_nats": phi, "se_nats": se, "n_samples": count,
    })
    table.to_csv(out_dir / "shapley_table.csv", index=False)

    # ---- heatmap: z-bins x players on the rest-wavelength axis
    zt = acc.z_table()
    z_labels = ["z<0.5", "0.5-1.0", "1.0-1.7", "z>1.7"]
    order = np.argsort([p["rest_center"] for p in players])
    fig, ax = plt.subplots(figsize=(11, 3.6))
    vmax = np.nanmax(np.abs(zt)) or 1e-3
    im = ax.imshow(zt[:, order], aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(z_labels)), z_labels, fontsize=9)
    names = [players[i]["name"] for i in order]
    ax.set_xticks(range(len(order)),
                  [n if not n.startswith("cont_") else "" for n in names],
                  rotation=45, ha="right", fontsize=8)
    for k, i in enumerate(order):
        if players[i]["kind"] == "line":
            ax.axvline(k, color=LINEC, lw=0.6, alpha=0.5)
    ax.set_xlabel("players, ordered by rest wavelength (unlabeled = continuum bins)")
    fig.colorbar(im, ax=ax, label="Shapley value (nats)")
    ax.set_title("Flux information by spectral region")
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap.png", dpi=180)
    plt.close(fig)

    # ---- line table figure
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ln = table.iloc[:N_LINES].sort_values("phi_nats")
    ax.barh(ln.player, ln.phi_nats, xerr=ln.se_nats, color=ACCENT, capsize=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Shapley value (nats of log-likelihood)")
    ax.set_title("Emission-line contributions to flux prediction")
    fig.tight_layout()
    fig.savefig(out_dir / "line_table.png", dpi=180)
    plt.close(fig)

    print(table.head(N_LINES).to_string(index=False))
    print(f"written to {out_dir}")


if __name__ == "__main__":
    main()
