#!/usr/bin/env python
"""Emission-line-only baseline: the same flow, conditioned on 4 line fluxes.

This is the comparison that matters for the results slide. V_PAI is not a
baseline for the multi-target model -- it is the same frozen-AION architecture
with a different head, so beating it says nothing about whether the AION
representation earns its keep. The classical reference is the one the paper
used: a flow given ONLY measured emission-line fluxes, no spectrum, no image,
no WISE, no redshift.

Lines are the paper's set ([O III] 5007, [Ne V] 3426, Halpha, Hbeta;
results/README.md). Missing/out-of-window lines are left at their catalog value
(FastSpecFit writes 0 when a line falls outside DESI coverage) rather than
masked, which is what the legacy baseline did -- the model is free to read
"Halpha = 0" as a redshift cue, and that is part of what a line-only predictor
can honestly do.

The flow is IDENTICAL to the multi-target one (ConditionalNSFFlow, 8 transforms,
hidden 256x256). Only the context encoder differs: a small MLP over 4 standardized
features instead of AION + read-only CLS. Trains on CPU in minutes.

    python scripts/train_lines_baseline.py --all-properties ... --spectra-hdf5 ... \
        --sidecar ... --clean-split-csv ... --out results/baseline_lines_only_clean.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shareable_aion_flow.normalizing_flow import (  # noqa: E402
    ConditionalNSFFlow, KDEPrior, TargetStandardizer,
)

LINES = ["oiii_5007_flux", "nev_3426_flux", "halpha_flux", "hbeta_flux"]
TARGETS = ["log_ml_flux_1", "log_lx", "log_sfr"]


class LineContext(nn.Module):
    """4 line fluxes -> 256-d context. The flow downstream is unchanged."""

    def __init__(self, n_features: int, context_dim: int = 256, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, context_dim), nn.LayerNorm(context_dim),
        )

    def forward(self, x):
        return self.net(x)


def build(args) -> pd.DataFrame:
    props = pd.read_csv(args.all_properties, low_memory=False,
                        usecols=["targetid", "z"] + LINES)
    props = props.drop_duplicates("targetid")

    with h5py.File(args.spectra_hdf5, "r") as h:
        hd = pd.DataFrame({"targetid": h["desi_targetid"][:].astype(np.int64),
                           "ml_flux_1": h["ml_flux_1"][:].astype(np.float64)})
    hd = hd.drop_duplicates("targetid")

    side = pd.read_csv(args.sidecar, usecols=["targetid", "log_sfr"]).drop_duplicates("targetid")
    df = props.merge(hd, on="targetid").merge(side, on="targetid", how="left")

    with np.errstate(divide="ignore", invalid="ignore"):
        df["log_ml_flux_1"] = np.log10(np.where(df.ml_flux_1 > 0, df.ml_flux_1, np.nan))
        # luminosity distance -> log Lx, exactly as the staged pipeline defines it
        from astropy.cosmology import Planck18
        import astropy.units as u
        dl = Planck18.luminosity_distance(np.clip(df.z.to_numpy(float), 1e-4, None)).to(u.cm).value
        df["log_lx"] = df.log_ml_flux_1 + np.log10(4 * np.pi * dl ** 2)

    return df


def line_features(df: pd.DataFrame) -> np.ndarray:
    """asinh keeps the sign of noise-negative line fluxes and compresses the
    4-decade dynamic range without discarding the zeros that encode
    out-of-window lines."""
    f = np.stack([np.arcsinh(df[c].to_numpy(np.float64)) for c in LINES], axis=1)
    return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)


@torch.no_grad()
def evaluate(ctxenc, flow, std, prior, X, y, num_samples=128) -> dict:
    ctx = ctxenc(X)
    ys = std.transform_tensor(y)
    lp = flow.log_prob(ys, ctx)
    samp = flow.sample(ctx, num_samples=num_samples).mean(dim=0)
    yt, yp = ys.numpy(), samp.numpy()
    ss = float(np.sum((yt - yt.mean()) ** 2))
    nll = float(-lp.mean())
    return {
        "n_test": len(yt),
        "nll": nll,
        "r2": 1.0 - float(np.sum((yt - yp) ** 2)) / ss if ss > 0 else np.nan,
        "rmse_dex": float(np.sqrt(np.mean((yt - yp) ** 2))) * std.std,
        "info_gain_nats": -nll - float(prior.log_prob_tensor(ys).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all-properties", type=Path, required=True)
    ap.add_argument("--spectra-hdf5", type=Path, required=True)
    ap.add_argument("--sidecar", type=Path, required=True)
    ap.add_argument("--clean-split-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=40,
                    help="stop after this many epochs with no validation improvement")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    df = build(args)
    split = pd.read_csv(args.clean_split_csv).drop_duplicates("targetid")
    # join the split FIRST, then derive features, so rows can never misalign
    df = df.merge(split, on="targetid", how="inner").reset_index(drop=True)
    feats = line_features(df)
    print(f"[lines] {len(df):,} rows on the clean split; features {LINES}")

    is_tr = (df.split == "train").to_numpy()
    mu, sd = feats[is_tr].mean(0), feats[is_tr].std(0) + 1e-8
    Xall = torch.tensor((feats - mu) / sd, dtype=torch.float32)

    rows = []
    for tname in TARGETS:
        y = df[tname].to_numpy(np.float64)
        ok = np.isfinite(y)
        tr, te = is_tr & ok, (df.split == "test").to_numpy() & ok
        if te.sum() < 50:
            print(f"[lines] {tname}: only {te.sum()} test rows, skipping"); continue
        va = (df.split == "val").to_numpy() & ok
        std = TargetStandardizer.fit(y[tr])
        prior = KDEPrior(std.transform_numpy(y[tr]), bw_method="scott")
        Xtr = Xall[torch.tensor(tr)]
        ytr = torch.tensor(y[tr], dtype=torch.float32)
        Xva = Xall[torch.tensor(va)]
        yva = torch.tensor(y[va], dtype=torch.float32)
        ctxenc, flow = LineContext(feats.shape[1]), ConditionalNSFFlow(context_dim=256)
        opt = torch.optim.AdamW(list(ctxenc.parameters()) + list(flow.parameters()), lr=args.lr)
        n = len(ytr)
        # Four features and a full NSF overfit hard: at 400 blind epochs the log-flux
        # baseline reached R2 0.394 but info gain -0.373 nats, i.e. a sharper-than-
        # justified posterior that scores WORSE than the prior. Select on validation
        # NLL instead, and stop once it stops improving.
        best_val, best_state, best_ep, since = np.inf, None, -1, 0
        for ep in range(args.epochs):
            ctxenc.train(); flow.train()
            perm = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, args.batch_size):
                idx = perm[i:i + args.batch_size]
                loss = -flow.log_prob(std.transform_tensor(ytr[idx]), ctxenc(Xtr[idx])).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss) * len(idx)
            ctxenc.eval(); flow.eval()
            with torch.no_grad():
                vnll = float(-flow.log_prob(std.transform_tensor(yva), ctxenc(Xva)).mean())
            if vnll < best_val - 1e-4:
                best_val, best_ep, since = vnll, ep, 0
                best_state = ({k: v.clone() for k, v in ctxenc.state_dict().items()},
                              {k: v.clone() for k, v in flow.state_dict().items()})
            else:
                since += 1
            if (ep + 1) % 50 == 0:
                print(f"  [{tname}] epoch {ep+1}/{args.epochs} train {tot/n:.4f} "
                      f"val {vnll:.4f} (best {best_val:.4f} @ {best_ep+1})", flush=True)
            if since >= args.patience:
                print(f"  [{tname}] early stop at epoch {ep+1}; "
                      f"best val {best_val:.4f} @ epoch {best_ep+1}", flush=True)
                break
        if best_state is not None:
            ctxenc.load_state_dict(best_state[0]); flow.load_state_dict(best_state[1])
        ctxenc.eval(); flow.eval()
        m = evaluate(ctxenc, flow, std, prior,
                     Xall[torch.tensor(te)], torch.tensor(y[te], dtype=torch.float32))
        m.update({"head": tname, "input_group": "emission_lines_only",
                  "best_epoch": best_ep + 1, "val_nll": best_val})
        rows.append(m)
        print(f"[lines] {tname}: n={m['n_test']} R2={m['r2']:+.3f} "
              f"IG={m['info_gain_nats']:+.3f} nats RMSE={m['rmse_dex']:.3f} dex", flush=True)

    out = pd.DataFrame(rows)[["head", "input_group", "n_test", "nll", "r2", "rmse_dex",
                              "info_gain_nats", "best_epoch", "val_nll"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwritten to {args.out}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
