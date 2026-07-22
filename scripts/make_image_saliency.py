#!/usr/bin/env python
"""Occlusion saliency maps over the Legacy Survey cutouts, compared across targets.

For each selected test source and each per-target checkpoint (e.g. flux, Lx,
logMstar), slide a grid of occlusion cells over the 160x160 griz image (cell
pixels set to 0 = sky), re-encode, and record the drop in image-only
log-likelihood: saliency = LL(base) - LL(occluded). Occlusion happens in PIXEL
space before the codec, so no token machinery is involved. The gallery puts the
same source's maps for different targets side by side -- the question is
whether the model reads different image regions for different physics
(unresolved nucleus vs host extent).

    python scripts/make_image_saliency.py --staged-dir <staged> --clean-split-csv <csv> \
        --checkpoint flux=<run>/best.pt --checkpoint lx=<run>/best.pt [--n-sources 12]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shareable_aion_flow.data_to_aion_embeddings import build_dataloaders  # noqa: E402
from shareable_aion_flow.main import load_checkpoint  # noqa: E402


@torch.no_grad()
def image_ll(encoder, context_encoder, flow, standardizer, batch):
    target_std = standardizer.transform_tensor(batch[6])
    tokens, gids = encoder.encode_tokens(batch, ("image",))
    return flow.log_prob(target_std, context_encoder(tokens, gids)).cpu().numpy().ravel()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--staged-dir", type=Path, required=True)
    ap.add_argument("--clean-split-csv", type=Path, required=True)
    ap.add_argument("--checkpoint", action="append", required=True,
                    help="label=path/to/best.pt (repeatable)")
    ap.add_argument("--n-sources", type=int, default=12)
    ap.add_argument("--grid", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("image_saliency.png"))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {}
    for spec in args.checkpoint:
        label, path = spec.split("=", 1)
        models[label] = load_checkpoint(Path(path), device=device, dropout=0.0)

    # one loader per model target: each flow must be scored at ITS OWN
    # observed target value, or its LL sits in a flat tail and maps go blank
    batches = {}
    for lab, (enc, ctx, flow, std, cfg) in models.items():
        _, _, ldr = build_dataloaders(
            staged_dir=args.staged_dir, target_name=cfg.get("target", "log_ml_flux_1"),
            batch_size=512, num_workers=4, clean_split_csv=args.clean_split_csv,
        )
        batches[lab] = next(iter(ldr))
    ref = batches[next(iter(models))]
    y = ref[6].numpy()
    order = np.argsort(np.where(np.isfinite(y), y, np.inf))
    pick = order[np.linspace(0, args.n_sources * 20, args.n_sources).astype(int)]
    ref_tids = ref[7].numpy()[pick]
    base_batches = {}
    for lab, b in batches.items():
        tid_index = {int(t): i for i, t in enumerate(b[7].numpy())}
        rows = [tid_index.get(int(t), -1) for t in ref_tids]
        base_batches[lab] = tuple(x[rows].to(device) for x in b)
    base_batch = base_batches[next(iter(models))]
    B, G, size = len(pick), args.grid, base_batch[5].shape[-1]
    cell = size // G

    sal = {lab: np.zeros((B, G, G)) for lab in models}
    for lab, (enc, ctx, flow, std, cfg) in models.items():
        base_batch = base_batches[lab]
        base_ll = image_ll(enc, ctx, flow, std, base_batch)
        for gy in range(G):
            for gx in range(G):
                img = base_batch[5].clone()
                img[:, :, gy * cell:(gy + 1) * cell, gx * cell:(gx + 1) * cell] = 0.0
                occ = base_batch[:5] + (img,) + base_batch[6:]
                # occluded batch keeps this model's own target column semantics:
                # LL differences within a model are all that is used
                sal[lab][:, gy, gx] = base_ll - image_ll(enc, ctx, flow, std, occ)
        print(f"{lab}: saliency done", flush=True)

    labels = list(models)
    z = base_batch[3].cpu().numpy()
    fig, axes = plt.subplots(B, 1 + len(labels), figsize=(2.1 * (1 + len(labels)), 2.05 * B))
    for i in range(B):
        img = base_batch[5][i].cpu().numpy()
        rgb = np.arcsinh(np.clip(np.stack([img[2], img[1], img[0]], -1), 0, None) * 8.0)
        axes[i, 0].imshow(rgb / max(rgb.max(), 1e-9))
        axes[i, 0].set_ylabel(f"z={z[i]:.2f}", fontsize=8)
        for j, lab in enumerate(labels):
            m = sal[lab][i]
            v = np.abs(m).max() or 1e-9
            axes[i, j + 1].imshow(m, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
            if i == 0:
                axes[i, j + 1].set_title(lab, fontsize=11)
        for ax in axes[i]:
            ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_title("grz image", fontsize=11)
    fig.suptitle("Occlusion saliency (image-only LL drop), same sources across targets", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=170)
    np.savez(args.out.with_suffix(".npz"), **{lab: sal[lab] for lab in labels},
             targetid=base_batch[7].cpu().numpy(), z=z)
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
