#!/usr/bin/env python
"""Run the line/continuum Shapley sweeps on a trained spectra-only checkpoint.

Outputs into <run-dir>/shapley/: shapley_table.csv (per player: phi, SE, n,
mean tokens, phi per token), pair_interactions.csv, coalition_summary.csv
(full / lines-only / continuum-only / norm-only log-likelihoods), heatmap.png,
line_table.png, pair_heatmap.png.

    python scripts/run_line_shapley.py --checkpoint <best.pt> \
        --staged-dir <staged_paper> --clean-split-csv <clean_split.csv> \
        [--full-sweeps 2] [--line-sweeps 4] [--pair-sweeps 3] \
        [--mask-mode drop] [--batch-size 256]
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
from shareable_aion_flow.line_shapley import (  # noqa: E402
    N_LINES,
    N_SPEC_TOKENS,
    masked_log_prob,
    player_catalog,
    player_token_map,
    run_sweeps,
)
from shareable_aion_flow.main import load_checkpoint  # noqa: E402

ACCENT, LINEC = "#0072B2", "#D55E00"


@torch.no_grad()
def coalition_summary(*, encoder, context_encoder, flow, loader, standardizer,
                      players, device, mask_mode, line_guard_tokens=1) -> pd.DataFrame:
    """Mean test log-likelihood of four fixed coalitions.

    full = no mask; lines_only = drop all continuum tokens; continuum_only =
    drop all line tokens; norm_only = drop every wavelength token (the
    normalization token always stays: all four condition on overall brightness).
    """
    sums = {"full": 0.0, "continuum_only": 0.0, "lines_only": 0.0, "norm_only": 0.0}
    total = 0
    for batch in loader:
        batch = tuple(t.to(device, non_blocking=True) for t in batch)
        z_np = batch[3].detach().cpu().numpy().ravel()
        B = len(z_np)
        total += B
        masks = {
            "full": np.zeros((B, N_SPEC_TOKENS), dtype=bool),
            "continuum_only": np.zeros((B, N_SPEC_TOKENS), dtype=bool),
            "lines_only": np.zeros((B, N_SPEC_TOKENS), dtype=bool),
            "norm_only": np.ones((B, N_SPEC_TOKENS), dtype=bool),
        }
        for b in range(B):
            tok = player_token_map(float(z_np[b]), players, line_guard_tokens)
            for j, t in enumerate(tok):
                if not len(t):
                    continue
                if j < N_LINES:
                    masks["continuum_only"][b, t] = True
                else:
                    masks["lines_only"][b, t] = True
        for name, mask in masks.items():
            lp = masked_log_prob(
                encoder=encoder, context_encoder=context_encoder, flow=flow,
                batch=batch, standardizer=standardizer,
                spectrum_token_mask=torch.from_numpy(mask).to(device),
                mask_mode=mask_mode,
            )
            sums[name] += float(lp.sum())
    return pd.DataFrame(
        [{"coalition": k, "mean_log_prob_nats": v / total, "n_test": total} for k, v in sums.items()]
    )


def mean_tokens_per_player(loader, players, line_guard_tokens=1) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros(len(players))
    avail = np.zeros(len(players))
    total = 0
    for batch in loader:
        for z in batch[3].detach().cpu().numpy().ravel():
            total += 1
            for j, t in enumerate(player_token_map(float(z), players, line_guard_tokens)):
                if len(t):
                    counts[j] += len(t)
                    avail[j] += 1
    return counts / np.maximum(avail, 1), avail / max(total, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--staged-dir", type=Path, required=True)
    ap.add_argument("--clean-split-csv", type=Path, default=None)
    ap.add_argument("--full-sweeps", type=int, default=2)
    ap.add_argument("--line-sweeps", type=int, default=4)
    ap.add_argument("--pair-sweeps", type=int, default=3)
    ap.add_argument("--mask-mode", choices=["drop", "replace"], default=None,
                    help="Defaults to the checkpoint config's mask_mode, else drop.")
    ap.add_argument("--line-guard-tokens", type=int, default=1,
                    help="Extra tokens dropped on each side of a line window "
                    "(codec effective receptive field; see codec_leakage_probe.py).")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--z-bins", type=int, default=4,
                    help="Redshift bins for the z-resolved table, quantile-spaced so each holds a "
                    "similar number of sources (4 reproduces the original coarse binning).")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, context_encoder, flow, standardizer, config = load_checkpoint(
        args.checkpoint, device=device, dropout=0.0
    )
    mask_mode = args.mask_mode or str(config.get("mask_mode", "drop"))
    target = config.get("target", "log_ml_flux_1")
    _, _, test_loader = build_dataloaders(
        staged_dir=args.staged_dir, target_name=target, batch_size=args.batch_size,
        num_workers=args.num_workers, clean_split_csv=args.clean_split_csv,
    )
    players = player_catalog()
    # quantile z edges from the sources actually being swept
    zs_all = []
    for b in test_loader:
        zs_all.append(b[3].detach().cpu().numpy().ravel())
    zs_all = np.concatenate(zs_all)
    if args.z_bins <= 4:
        z_edges = (0.0, 0.5, 1.0, 1.7, 99.0)
    else:
        qs = np.linspace(0, 100, args.z_bins + 1)[1:-1]
        z_edges = tuple([0.0] + list(np.percentile(zs_all, qs)) + [99.0])
    print(f"[sweep] {args.z_bins} z bins, edges "
          + " ".join(f"{v:.2f}" for v in z_edges[:-1]) + " ...", flush=True)
    print(f"mask_mode={mask_mode}", flush=True)

    out_dir = args.checkpoint.parent / "shapley"
    out_dir.mkdir(exist_ok=True)

    summary = coalition_summary(
        encoder=encoder, context_encoder=context_encoder, flow=flow,
        loader=test_loader, standardizer=standardizer, players=players,
        device=device, mask_mode=mask_mode, line_guard_tokens=args.line_guard_tokens,
    )
    summary.to_csv(out_dir / "coalition_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)

    acc, pair_acc = run_sweeps(
        encoder=encoder, context_encoder=context_encoder, flow=flow,
        loader=test_loader, standardizer=standardizer, players=players,
        device=device, n_full_sweeps=args.full_sweeps, n_line_sweeps=args.line_sweeps,
        n_pair_sweeps=args.pair_sweeps, mask_mode=mask_mode,
        line_guard_tokens=args.line_guard_tokens, seed=args.seed, z_edges=z_edges,
    )

    phi, se, count = acc.table()
    mean_tok, avail_frac = mean_tokens_per_player(test_loader, players, args.line_guard_tokens)
    table = pd.DataFrame({
        "player": [p["name"] for p in players],
        "kind": [p["kind"] for p in players],
        "rest_center_A": [p["rest_center"] for p in players],
        "phi_nats": phi, "se_nats": se, "n_samples": count,
        "mean_tokens": mean_tok,
        "phi_per_token": phi / np.maximum(mean_tok, 1e-9),
        # phi is conditional on availability; the population view weights by it
        "availability_frac": avail_frac,
        "phi_population": phi * avail_frac,
    })
    table.to_csv(out_dir / "shapley_table.csv", index=False)

    imean, ise, icount = pair_acc.table()
    line_names = [p["name"] for p in players[:N_LINES]]
    pair_rows = []
    for i in range(N_LINES):
        for j in range(i + 1, N_LINES):
            pair_rows.append({
                "line_a": line_names[i], "line_b": line_names[j],
                "interaction_nats": imean[i, j], "se_nats": ise[i, j],
                "n_samples": icount[i, j],
            })
    pd.DataFrame(pair_rows).to_csv(out_dir / "pair_interactions.csv", index=False)

    # ---- heatmap: z-bins x players on the rest-wavelength axis
    zt = acc.z_table()
    # persist the z-resolved values too: the redshift-vs-rest-wavelength figure
    # needs them, and re-running a sweep just to recover them is wasteful
    zdf = pd.DataFrame(
        zt, index=[f"zbin_{i}" for i in range(zt.shape[0])],
        columns=[p["name"] for p in players],
    )
    assert zt.shape[0] == len(z_edges) - 1, (
        f"z-table has {zt.shape[0]} bins but {len(z_edges)-1} edges were requested")
    zdf.insert(0, "z_hi", [z_edges[i + 1] for i in range(zt.shape[0])])
    zdf.insert(0, "z_lo", [z_edges[i] for i in range(zt.shape[0])])
    zdf.to_csv(out_dir / "shapley_by_zbin.csv")
    pd.DataFrame(
        acc.z_count, index=[f"zbin_{i}" for i in range(zt.shape[0])],
        columns=[p["name"] for p in players],
    ).to_csv(out_dir / "shapley_by_zbin_counts.csv")
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

    # ---- pair interaction heatmap
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    vmax = np.nanmax(np.abs(imean)) or 1e-4
    masked = np.where(np.eye(N_LINES, dtype=bool), np.nan, imean)
    im = ax.imshow(masked, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(N_LINES), line_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(N_LINES), line_names, fontsize=9)
    fig.colorbar(im, ax=ax, label="Shapley interaction (nats)")
    ax.set_title("Line-pair interactions (blue = redundant, red = synergistic)")
    fig.tight_layout()
    fig.savefig(out_dir / "pair_heatmap.png", dpi=180)
    plt.close(fig)

    print(table.head(N_LINES).to_string(index=False))
    print(f"written to {out_dir}")


if __name__ == "__main__":
    main()
