#!/usr/bin/env python
"""Phase 2: freeze the body, refit each marginal head with its OWN early stopping.

Phase 1 trains the joint alone, so exactly one CLS token and one context slot
were ever driven by a gradient. This script freezes that body, caches its context
vector once per source, and then fits a small flow per target on the cached
vectors. Because the body does not move, each head can stop at its own optimum
without dragging any other head, and the expensive frozen-AION forward happens
ONCE instead of once per epoch per head.

Why this rather than per-head checkpoints: keeping a body per head would mean N
encoder forwards at inference, which is N models wearing a trenchcoat and gives
up the one-run-all-targets claim the project rests on.

Which context slot the marginals read matters. Under --joint-only only the
JOINT's CLS token received gradient; the other slots are still at their
initialisation and carry no trained signal. So the default is to read the joint
slot. `--context-slot own` is available for bodies trained with every head live.

    python scripts/refit_heads.py --checkpoint <run>/best.pt \
        --staged-dir $AIONFLOW_STAGED --clean-split-csv .../clean_split.csv \
        --extra-targets-csv .../targets_sidecar.csv --out <run>/refit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shareable_aion_flow"))

import multitarget as mt                                          # noqa: E402
from data_to_aion_embeddings import AIONTokenEncoder, build_dataloaders  # noqa: E402
from normalizing_flow import ConditionalNSFFlow, TargetStandardizer      # noqa: E402

ALL_COMBO = ("spectra", "z", "wise", "image")


@torch.no_grad()
def cache_contexts(encoder, head, loader, lookup, device):
    """One frozen forward per source, keeping EVERY context slot.

    All slots are kept (25k sources x heads x 256 floats is ~100 MB) so the
    choice of which slot a marginal reads is made at fit time rather than baked
    into an expensive pass we would then have to repeat.
    """
    ctxs, ys = [], []
    for batch in loader:
        batch = tuple(t.to(device, non_blocking=True) for t in batch)
        y, _, _ = lookup.batch(batch[7], device)
        cls_seq, _ = encoder.encode_tokens(batch, ALL_COMBO)
        ctxs.append(head(cls_seq).float().cpu())          # [B, N_HEADS, 256]
        ys.append(y.float().cpu())
    return torch.cat(ctxs), torch.cat(ys)


def fit_one(name, ctx_tr, y_tr, ctx_va, y_va, std, args, device):
    """Fit a 1-D flow on cached contexts, early stopping on ITS own val NLL."""
    ok_tr = torch.isfinite(y_tr)
    ok_va = torch.isfinite(y_va)
    if int(ok_tr.sum()) < 50 or int(ok_va.sum()) < 20:
        return None
    xt, yt = ctx_tr[ok_tr].to(device), std.transform_tensor(y_tr[ok_tr].to(device))
    xv, yv = ctx_va[ok_va].to(device), std.transform_tensor(y_va[ok_va].to(device))

    flow = ConditionalNSFFlow(context_dim=xt.shape[1]).to(device)
    opt = torch.optim.AdamW(flow.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n = xt.shape[0]
    best, best_ep, best_state, curve = float("inf"), 0, None, []
    gen = torch.Generator().manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        flow.train()
        perm = torch.randperm(n, generator=gen)
        for lo in range(0, n, args.batch_size):
            idx = perm[lo:lo + args.batch_size]
            loss = -flow.log_prob_draws(yt[idx].unsqueeze(0), xt[idx]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
            opt.step()
        flow.eval()
        with torch.no_grad():
            v = float(-flow.log_prob_draws(yv.unsqueeze(0), xv).mean())
        curve.append(v)
        if v < best - 1e-5:
            best, best_ep = v, epoch
            best_state = {k: t.detach().clone() for k, t in flow.state_dict().items()}
        elif epoch - best_ep >= args.patience:
            break
    if best_state is not None:
        flow.load_state_dict(best_state)
    return {"name": name, "val_nll": best, "best_epoch": best_ep, "epochs_run": len(curve),
            "n_train": int(ok_tr.sum()), "n_val": int(ok_va.sum()), "curve": curve,
            "state": flow.state_dict()}


def fit_joint(ctx_tr, y_tr, ctx_va, y_va, stds, args, device):
    """Refit the JOINT on the frozen body, so the comparison is like for like.

    Without this the joint is judged on the flow phase 1 happened to leave it
    with, while every marginal it is compared against got a full dedicated fit
    with its own early stopping. That comparison is rigged against the joint and
    would understate the dependence it captures.

    Rows missing the marginalisable dimension are integrated out by the same
    quadrature the trainer uses; rows missing a REQUIRED dimension are skipped.
    """
    names = [t["name"] for t in mt.MULTI_TARGETS]
    idx = [names.index(n) for n in mt.JOINT_PAIR if n in names]
    req = [j for j in idx if names[j] not in mt.JOINT_MARGINAL]
    marg = [j for j in idx if names[j] in mt.JOINT_MARGINAL]
    if len(idx) < 2 or len(marg) > 1:
        return None

    def prep(ctx, y):
        keep = torch.ones(y.shape[0], dtype=torch.bool)
        for j in req:
            keep &= torch.isfinite(y[:, j])
        z = torch.stack([stds[j].transform_tensor(y[keep, j]) for j in idx], dim=-1)
        have = torch.ones(int(keep.sum()), dtype=torch.bool)
        for j in marg:
            have &= torch.isfinite(y[keep, j])
        return ctx[keep].to(device), z.to(device), have.to(device)

    xt, zt, ht = prep(ctx_tr, y_tr)
    xv, zv, hv = prep(ctx_va, y_va)
    K, span = mt.JOINT_QUAD_NODES, mt.JOINT_QUAD_SPAN
    jm = idx.index(marg[0]) if marg else None
    nodes = torch.linspace(-span, span, K, device=device)
    du = float(nodes[1] - nodes[0])

    def nll(flow, x, z, have):
        out = x.new_zeros(())
        n = 0
        if bool(have.any()):
            lp = flow.log_prob_draws(z[have].unsqueeze(0), x[have])
            out = out + -lp.sum(); n += int(have.sum())
        part = ~have
        if jm is not None and bool(part.any()):
            # log_prob_draws takes [draws, batch, features] against [batch, ctx],
            # so the K quadrature nodes ARE the draw axis: no reshaping needed.
            zz = z[part]; m = zz.shape[0]
            grid = zz.unsqueeze(0).repeat(K, 1, 1)      # [K, m, F]
            grid[:, :, jm] = nodes.view(K, 1)
            lp = flow.log_prob_draws(grid, x[part])     # [K, m]
            lp = torch.logsumexp(lp, dim=0) + float(np.log(du))
            out = out + -lp.sum(); n += m
        return out / max(n, 1)

    flow = ConditionalNSFFlow(context_dim=xt.shape[1], features=max(2, len(idx))).to(device)
    opt = torch.optim.AdamW(flow.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, best_ep, curve = float("inf"), 0, []
    gen = torch.Generator().manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        flow.train()
        perm = torch.randperm(xt.shape[0], generator=gen)
        for lo in range(0, xt.shape[0], args.batch_size):
            i = perm[lo:lo + args.batch_size]
            loss = nll(flow, xt[i], zt[i], ht[i])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0); opt.step()
        flow.eval()
        with torch.no_grad():
            v = float(nll(flow, xv, zv, hv))
        curve.append(v)
        if v < best - 1e-5:
            best, best_ep = v, epoch
        elif epoch - best_ep >= args.patience:
            break
    return {"name": "joint_refit", "val_nll": best, "best_epoch": best_ep,
            "epochs_run": len(curve), "n_train": int(xt.shape[0]),
            "n_val": int(xv.shape[0]), "curve": curve}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--staged-dir", type=Path, required=True)
    ap.add_argument("--clean-split-csv", type=Path, required=True)
    ap.add_argument("--extra-targets-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--context-slot", choices=["joint", "own"], default="joint")
    ap.add_argument("--refit-heads", nargs="*", default=None,
                    help="Targets to fit. Defaults to the checkpoint's own heads. A frozen body "
                         "can carry ANY target, so heads phase 1 dropped (flux, P2, ...) can be "
                         "fit here without retraining anything: pass their names, or 'all'.")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--eval-batch-size", type=int, default=224)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})

    # The MODEL must be built with the checkpoint's head count: num_cls and the
    # context width are baked into the trained weights.
    mt.configure_heads_from_config(config)
    ckpt_heads, ckpt_n_heads, joint_slot = list(mt.HEAD_NAMES), mt.N_HEADS, mt.N_TARGETS
    ckpt_stds = {n: s for n, s in zip(mt.HEAD_NAMES, ckpt.get("standardizers", []))}
    print(f"[refit] heads from checkpoint: {ckpt_heads}")
    print(f"[refit] joint dims: {mt.JOINT_PAIR}  marginalisable: {mt.JOINT_MARGINAL}")

    encoder = AIONTokenEncoder(freeze=False, cls_mode=True, cls_variant="readonly",
                               num_cls=mt.N_HEADS).to(device)
    _, unexpected = encoder.load_state_dict(ckpt["encoder_trainable_state_dict"], strict=False)
    if unexpected:
        raise SystemExit(f"unexpected encoder keys: {unexpected[:5]}")
    head = mt.SharedCLSHead().to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    encoder.eval(); head.eval()
    for p in list(encoder.parameters()) + list(head.parameters()):
        p.requires_grad_(False)

    # The TARGET list is decoupled from the model: the body is frozen, so any
    # target can be fit on its context, including ones phase 1 never trained.
    if args.refit_heads == ["all"]:
        mt.configure_heads(())
    elif args.refit_heads:
        keep = set(args.refit_heads)
        mt.configure_heads(tuple(t["name"] for t in mt._ALL_TARGETS if t["name"] not in keep))
    print(f"[refit] fitting targets: {[t['name'] for t in mt.MULTI_TARGETS]}")

    lookup = mt.MultiTargetLookup(args.staged_dir, args.extra_targets_csv)
    import pandas as pd
    assign = pd.read_csv(args.clean_split_csv)
    train_y = lookup.values_for(assign.loc[assign.split == "train", "targetid"].to_numpy("int64"))
    stds = []
    for j, spec in enumerate(mt.MULTI_TARGETS):
        # Reuse the checkpoint's standardizer where the head existed, so those
        # NLLs stay comparable with phase 1; fit fresh for newly added targets.
        if spec["name"] in ckpt_stds:
            stds.append(TargetStandardizer.from_state_dict(ckpt_stds[spec["name"]]))
        else:
            v = train_y[:, j][np.isfinite(train_y[:, j])]
            stds.append(TargetStandardizer.fit(v))
            print(f"[refit] fitted a new standardizer for {spec['name']} ({len(v):,} train values)")
    train_loader, val_loader, _ = build_dataloaders(
        staged_dir=args.staged_dir, target_name="log_ml_flux_1",
        batch_size=args.eval_batch_size, eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers, seed=args.seed,
        clean_split_csv=args.clean_split_csv,
    )

    print(f"[refit] caching contexts ({args.context_slot} slot; one frozen forward per source)")
    ctx_tr, y_tr = cache_contexts(encoder, head, train_loader, lookup, device)
    ctx_va, y_va = cache_contexts(encoder, head, val_loader, lookup, device)
    print(f"[refit] cached train {tuple(ctx_tr.shape)}  val {tuple(ctx_va.shape)}")

    args.out.mkdir(parents=True, exist_ok=True)
    results, states = [], {}
    for j, spec in enumerate(mt.MULTI_TARGETS):
        # Under --joint-only ONLY the joint's CLS token ever saw a gradient, so
        # a per-head slot still holds its initialisation and carries no trained
        # signal. Reading the joint slot is the default for exactly that reason.
        slot = joint_slot if args.context_slot == "joint" else min(j, ckpt_n_heads - 1)
        r = fit_one(spec["name"], ctx_tr[:, slot], y_tr[:, j], ctx_va[:, slot],
                    y_va[:, j], stds[j], args, device)
        if r is None:
            print(f"[refit] {spec['name']}: too few finite values, skipped")
            continue
        states[r["name"]] = r.pop("state")
        results.append(r)
        print(f"[refit] {r['name']:20s} val NLL {r['val_nll']:.4f} "
              f"at epoch {r['best_epoch']:>3d}/{r['epochs_run']:<3d} "
              f"(n_train {r['n_train']:,})")

    jr = fit_joint(ctx_tr[:, joint_slot], y_tr, ctx_va[:, joint_slot], y_va, stds, args, device)
    if jr is not None:
        results.append(jr)
        print(f"[refit] {'joint (refit)':20s} val NLL {jr['val_nll']:.4f} "
              f"at epoch {jr['best_epoch']:>3d}/{jr['epochs_run']:<3d} "
              f"(n_train {jr['n_train']:,})")

    torch.save(states, args.out / "refit_flows.pt")
    (args.out / "refit_report.json").write_text(json.dumps(results, indent=2))

    # The reason the joint head exists: it should beat independent marginals on
    # the dimensions it models. Anything else means it captured no dependence.
    joint_dims = [r for r in results if r["name"] in mt.JOINT_PAIR]
    if joint_dims:
        s = sum(r["val_nll"] for r in joint_dims)
        print(f"\n[refit] sum of refit marginals over joint dims "
              f"({', '.join(r['name'] for r in joint_dims)}): {s:.4f} nats")
        if jr is not None:
            print(f"[refit] joint refit on the same frozen body:         {jr['val_nll']:.4f} nats")
            print(f"[refit] DEPENDENCE CAPTURED: {s - jr['val_nll']:+.4f} nats "
                  "(positive means the joint beats independent marginals)")
    print(f"[refit] wrote {args.out}/refit_report.json")


if __name__ == "__main__":
    main()
