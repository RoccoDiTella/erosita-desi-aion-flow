"""Multi-target read-only-CLS training: one frozen forward, many heads.

Heads: X-ray flux, Lx and p1..p4 band fluxes; stellar mass, log SFR and
black-hole mass (PAN25 and VO09) from the runtime sidecar -- each a 1-D flow --
plus a JOINT 2-D flow over (P2, P3) whose samples give per-source hardness
posteriors with correlated band errors (HR itself is never a direct target).

sSFR is deliberately NOT a head. It is an exact function of log SFR and log M*,
so it carries no label information they do not already hold, and an independent
head could produce a posterior contradicting their difference. It is left to be
implied from an (M*, SFR) joint by the same shear-marginalisation used for HR,
which also yields the error correlation instead of assuming it. Architecture: per-target CLS
vectors + SHARED per-block Q/V read adapters (data stream frozen, no_grad),
one SHARED 768->512->256 MLP over the stacked CLS states, one small NSF flow
per target. Joint loss: per-target NLL on standardized values, split-normal
noise injection where sigma exists, per-source availability masks, and
detached-EMA loss normalization so harder targets do not dominate gradients.
"""

from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

try:
    from .data_to_aion_embeddings import read_dataset
    from .normalizing_flow import ConditionalNSFFlow, TargetStandardizer, sample_split_normal
except ImportError:
    from data_to_aion_embeddings import read_dataset
    from normalizing_flow import ConditionalNSFFlow, TargetStandardizer, sample_split_normal

# name, sigma columns (None = no error model), availability sigma gate
MULTI_TARGETS: list[dict] = [
    {"name": "log_ml_flux_1", "sig": ("flux_sig_lo", "flux_sig_hi"), "max_sigma": None, "sidecar": False},
    {"name": "log_lx", "sig": ("flux_sig_lo", "flux_sig_hi"), "max_sigma": None, "sidecar": False},
    {"name": "logmstar", "sig": None, "max_sigma": None, "sidecar": False},
    {"name": "log_flux_p1", "sig": ("log_flux_p1_sig_lo", "log_flux_p1_sig_hi"), "max_sigma": 1.0, "sidecar": True},
    {"name": "log_flux_p2", "sig": ("log_flux_p2_sig_lo", "log_flux_p2_sig_hi"), "max_sigma": 1.0, "sidecar": True},
    {"name": "log_flux_p3", "sig": ("log_flux_p3_sig_lo", "log_flux_p3_sig_hi"), "max_sigma": 1.0, "sidecar": True},
    {"name": "log_flux_p4", "sig": ("log_flux_p4_sig_lo", "log_flux_p4_sig_hi"), "max_sigma": 1.0, "sidecar": True},
    # APPENDED, never inserted: the flows are indexed positionally, so adding a
    # head anywhere but the end silently renumbers every existing checkpoint.
    # log SFR comes from the CIGALE VAC, a DIFFERENT SED fit than the one that
    # produced logmstar -- see scripts/make_sfr_sidecar.py for why that matters.
    {"name": "log_sfr", "sig": ("log_sfr_sig_lo", "log_sfr_sig_hi"), "max_sigma": 1.0, "sidecar": True},
    # CIGALE stellar mass. Supersedes the FastSpecFit `logmstar` above for this
    # sample: FastSpecFit has NO AGN component, so on 87% QSO it attributes the
    # accretion-disk continuum to stars. CIGALE fits the AGN explicitly. Keeping
    # both in the same fit as log_sfr also makes sSFR well defined. Drop the
    # older head with --drop-heads logmstar.
    {"name": "logmstar_cigale", "sig": ("logmstar_cigale_sig_lo", "logmstar_cigale_sig_hi"),
     "max_sigma": 1.0, "sidecar": True},
    # Black-hole mass, DR1 qmassiron VAC. PAN25 is iron-corrected and primary;
    # VO09 is the classic estimator, kept as a comparison target. SHEN11/LE20/
    # YU23 ride in the sidecar as columns -- they correlate with VO09 at 0.99+
    # (LE20 is exactly 1.0000, a pure rescaling), so training them would be
    # training the same target several times.
    {"name": "log_mbh_pan25", "sig": ("log_mbh_pan25_sig_lo", "log_mbh_pan25_sig_hi"),
     "max_sigma": 1.0, "sidecar": True},
    {"name": "log_mbh_vo09", "sig": ("log_mbh_vo09_sig_lo", "log_mbh_vo09_sig_hi"),
     "max_sigma": 1.0, "sidecar": True},
]
_ALL_TARGETS = list(MULTI_TARGETS)
N_TARGETS = len(MULTI_TARGETS)          # scalar heads

# The joint head. Dimensions are modelled together; `JOINT_MARGINAL` names the
# ones that may be MISSING and are then integrated out by quadrature rather than
# imputed -- imputing from the prior would pull the conditional toward the
# unconditional and bias the target. Every other joint dimension is required:
# a source lacking one is skipped entirely for this head.
#
# log flux is deliberately NOT here alongside log Lx. Lx = flux + log(4 pi D_L(z)^2)
# EXACTLY (verified to 7e-15) and z is a model input, so conditional on the
# inputs the two are in one-to-one correspondence: their joint density would be
# supported on a line, and the flow could drive the NLL to -infinity by
# collapsing that direction.
JOINT_PAIR = ("logmstar_cigale", "log_sfr", "log_lx", "log_flux_p3")
JOINT_MARGINAL = ("log_flux_p3",)
_DEFAULT_JOINT_PAIR, _DEFAULT_JOINT_MARGINAL = JOINT_PAIR, JOINT_MARGINAL
JOINT_QUAD_NODES = 48       # standardized-space nodes for the marginalisation
JOINT_QUAD_SPAN = 5.0       # +/- sigma covered by the grid


def _joint_idx():
    names = {t["name"]: j for j, t in enumerate(MULTI_TARGETS)}
    return tuple(names[n] for n in JOINT_PAIR if n in names)


JOINT_IDX = _joint_idx()
N_HEADS = N_TARGETS + 1
HEAD_NAMES = [t["name"] for t in MULTI_TARGETS] + ["joint"]


def configure_heads(drop: tuple[str, ...] = ()) -> None:
    """Drop scalar heads (e.g. P4, which never learns) before building the model.

    Rebinds the module-level head configuration; call once at startup, before
    anything reads N_TARGETS / N_HEADS / HEAD_NAMES.
    """
    global MULTI_TARGETS, N_TARGETS, JOINT_IDX, N_HEADS, HEAD_NAMES
    MULTI_TARGETS = [t for t in _ALL_TARGETS if t["name"] not in set(drop)]
    N_TARGETS = len(MULTI_TARGETS)
    kept = {t["name"] for t in MULTI_TARGETS}
    required = [n for n in JOINT_PAIR if n not in JOINT_MARGINAL]
    missing = [n for n in required if n not in kept]
    if missing:
        raise ValueError(f"cannot drop {missing}: the joint head requires them")
    JOINT_IDX = _joint_idx()
    N_HEADS = N_TARGETS + 1
    HEAD_NAMES = [t["name"] for t in MULTI_TARGETS] + ["joint"]


def configure_heads_from_config(config: dict | None) -> None:
    """Rebind the head set to match a CHECKPOINT, not the current default list.

    Checkpoints store the head names they were trained with. Reading `drop_heads`
    alone is not enough: when a new head is appended to MULTI_TARGETS (log_sfr
    was, after the band runs), an older checkpoint has fewer flows than the
    current default and would fail to load with a shape mismatch. Deriving the
    drop set from the stored `heads` list keeps every earlier checkpoint
    loadable without hand-passing --drop-heads.
    """
    global JOINT_PAIR, JOINT_MARGINAL
    config = config or {}
    stored = config.get("heads") or []
    # Restore the joint the checkpoint was TRAINED with. Runs before the
    # (M*, SFR, Lx, P3) joint used a 2-D P2xP3 head; loading one of those under
    # the current definition would demand targets it never had.
    if "p2xp3_joint" in stored:
        JOINT_PAIR = ("log_flux_p2", "log_flux_p3")
        JOINT_MARGINAL = ()
    else:
        JOINT_PAIR = _DEFAULT_JOINT_PAIR
        JOINT_MARGINAL = _DEFAULT_JOINT_MARGINAL
    if stored:
        joint_names = {"p2xp3_joint", "joint"}
        keep = {h for h in stored if h not in joint_names}
        drop = tuple(t["name"] for t in _ALL_TARGETS if t["name"] not in keep)
    else:                                    # pre-`heads` checkpoints
        drop = tuple(config.get("drop_heads", ()) or ())
    configure_heads(drop)
    if "p2xp3_joint" in stored:
        HEAD_NAMES[-1] = "p2xp3_joint"


def load_multi_target_matrix(
    staged_path: Path, extra_targets_csv: Path | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(targets [n,7], sig_lo [n,7], sig_hi [n,7]) aligned to a staged split file.

    Unavailable entries are NaN in ``targets`` (masked in the loss); missing
    sigmas are 0 (no injection). Sidecar targets join by targetid.
    """
    import pandas as pd

    with h5py.File(staged_path, "r") as handle:
        n = int(handle["desi_targetid"].shape[0])
        tids = handle["desi_targetid"][:].astype(np.int64)
        store_cols = {}
        for spec in MULTI_TARGETS:
            cols = [spec["name"]] + (list(spec["sig"]) if spec["sig"] else [])
            for c in cols:
                if not spec["sidecar"] and c in handle:
                    store_cols[c] = read_dataset(handle, c).astype(np.float64)
    side = None
    if extra_targets_csv is not None:
        side = pd.read_csv(extra_targets_csv).drop_duplicates("targetid").set_index("targetid")
        side = side.reindex(tids)

    y = np.full((n, N_TARGETS), np.nan)
    slo = np.zeros((n, N_TARGETS))
    shi = np.zeros((n, N_TARGETS))
    for j, spec in enumerate(MULTI_TARGETS):
        if spec["sidecar"]:
            if side is None:
                continue
            def col(c, _side=side):
                return _side[c].to_numpy(np.float64) if c in _side.columns else None
        else:
            def col(c, _store=store_cols):
                return _store.get(c)
        vals = col(spec["name"])
        if vals is None:
            continue
        y[:, j] = vals
        if spec["sig"]:
            lo, hi = col(spec["sig"][0]), col(spec["sig"][1])
            if lo is not None and hi is not None:
                slo[:, j] = np.nan_to_num(np.abs(lo))
                shi[:, j] = np.nan_to_num(np.abs(hi))
        if spec["max_sigma"] is not None:
            mean_sig = 0.5 * (slo[:, j] + shi[:, j])
            bad = ~np.isfinite(mean_sig) | (mean_sig > spec["max_sigma"])
            y[bad, j] = np.nan
    return y, slo, shi


class SharedCLSHead(nn.Module):
    """One shared 768->512->256 MLP applied across all stacked CLS streams."""

    def __init__(self, embed_dim: int = 768, hidden: int = 512, context_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, context_dim),
            nn.LayerNorm(context_dim),
        )
        self.context_dim = context_dim

    def forward(self, cls_states: torch.Tensor) -> torch.Tensor:  # [B, K, 768] -> [B, K, 256]
        return self.net(cls_states)


class MultiTargetFlows(nn.Module):
    """One 1-D flow per scalar target, plus one N-D joint over JOINT_PAIR."""

    def __init__(self, context_dim: int = 256, n_targets: int | None = None) -> None:
        super().__init__()
        # resolved at CALL time: configure_heads() may have changed the head
        # count after import, and a default argument would capture the old one
        n_targets = N_TARGETS if n_targets is None else n_targets
        self.flows = nn.ModuleList(ConditionalNSFFlow(context_dim=context_dim) for _ in range(n_targets))
        self.joint = ConditionalNSFFlow(context_dim=context_dim,
                                       features=max(2, len(JOINT_IDX)))


class EMALossWeights:
    """Detached EMA of each target's mean loss; weight = 1/EMA (unit-scale grads)."""

    def __init__(self, n_targets: int | None = None, beta: float = 0.98) -> None:
        n_targets = N_HEADS if n_targets is None else n_targets   # call-time, see above
        self.ema = np.ones(n_targets)
        self.beta = float(beta)
        self.seen = np.zeros(n_targets, dtype=bool)

    def update_and_weights(self, losses: list[float | None]) -> np.ndarray:
        for j, val in enumerate(losses):
            if val is None or not np.isfinite(val):
                continue
            if not self.seen[j]:
                self.ema[j] = val
                self.seen[j] = True
            else:
                self.ema[j] = self.beta * self.ema[j] + (1 - self.beta) * val
        return 1.0 / np.clip(np.abs(self.ema), 0.2, None)

    def state_dict(self) -> dict:
        return {"ema": self.ema.tolist(), "seen": self.seen.tolist(), "beta": self.beta}


def multi_target_nll(
    *,
    contexts: torch.Tensor,            # [B, K, 256]
    flows: MultiTargetFlows,
    targets: torch.Tensor,             # [B, K] (NaN = unavailable)
    joint_only: bool = False,
    sig_lo: torch.Tensor,              # [B, K]
    sig_hi: torch.Tensor,
    standardizers: list[TargetStandardizer],
    weights: np.ndarray,               # [K] detached loss weights
    inject: bool = True,
    inject_samples: int = 50,
    return_terms: bool = False,
) -> tuple[torch.Tensor, list[float | None]]:
    """Weighted joint NLL over available (source, target) pairs.

    Returns (total_loss, per-target UNweighted mean losses for the EMA update).
    """
    total = contexts.new_zeros(())
    raw: list[float | None] = []
    terms: list[torch.Tensor | None] = []

    def _std_inject(j: int, mask: torch.Tensor) -> torch.Tensor:
        """Standardized targets as draw stacks: [k, m] with k=1 when not injecting.

        Broadcast injection (user design): k independent split-normal draws per
        source, all evaluated under one context-conditioned distribution.
        Sources without sigma get zero perturbation in every draw.
        """
        std = standardizers[j]
        y = std.transform_tensor(targets[mask, j])
        if not inject:
            return y.unsqueeze(0)
        k = max(1, int(inject_samples))
        lo = (sig_lo[mask, j].abs() / std.std).clamp_min(0.0)
        hi = (sig_hi[mask, j].abs() / std.std).clamp_min(0.0)
        has = ((lo + hi) > 1e-8).unsqueeze(0)
        lo_k = lo.clamp_min(1e-6).unsqueeze(0).expand(k, -1)
        hi_k = hi.clamp_min(1e-6).unsqueeze(0).expand(k, -1)
        eps = sample_split_normal(lo_k, hi_k) * has
        return y.unsqueeze(0) + eps

    B = targets.shape[0]
    for j, flow in enumerate(flows.flows):
        # joint_only (phase 1): the marginal flows are not trained at all. Their
        # targets are still loaded and standardized, because the JOINT indexes
        # into the same target matrix -- dropping the heads would break
        # JOINT_IDX. Skipping the term (rather than zero-weighting it) leaves
        # their grads None, so AdamW skips them entirely and weight decay does
        # not quietly shrink flows that phase 2 is about to refit.
        mask = torch.isfinite(targets[:, j])
        if joint_only or not bool(mask.any()):
            raw.append(None)
            terms.append(None)
            continue
        nll = -flow.log_prob_draws(_std_inject(j, mask), contexts[mask, j]).mean()
        raw.append(float(nll.item()))
        terms.append(nll)
        # availability weighting (user): sources without this target do not
        # count, so the head's effective weight in the batch is its coverage.
        total = total + float(weights[j]) * float(mask.float().mean()) * nll

    # ---- joint head -------------------------------------------------------
    # Sources must have every REQUIRED joint dimension. A source missing a
    # MARGINALISABLE one is still used: that dimension is integrated out by
    # quadrature, which is the exact marginal likelihood. Imputing it from the
    # prior instead would pull the conditional toward the unconditional.
    names = [t["name"] for t in MULTI_TARGETS]
    req_idx = [j for j in JOINT_IDX if names[j] not in JOINT_MARGINAL]
    marg_idx = [j for j in JOINT_IDX if names[j] in JOINT_MARGINAL]
    have_req = torch.ones(B, dtype=torch.bool, device=targets.device)
    for j in req_idx:
        have_req = have_req & torch.isfinite(targets[:, j])

    joint_terms, joint_counts = [], []
    if bool(have_req.any()) and len(JOINT_IDX) >= 2:
        ctx_j = contexts[:, N_TARGETS]
        have_all = have_req.clone()
        for j in marg_idx:
            have_all = have_all & torch.isfinite(targets[:, j])
        # (a) fully observed rows: ordinary joint likelihood
        if bool(have_all.any()):
            vec = torch.stack([_std_inject(j, have_all) for j in JOINT_IDX], dim=-1)
            lp = flows.joint.log_prob_draws(vec, ctx_j[have_all])
            joint_terms.append(-lp.mean()); joint_counts.append(int(have_all.sum()))
        # (b) rows missing a marginalisable dimension: integrate it out
        part = have_req & ~have_all
        if bool(part.any()) and len(marg_idx) == 1:
            jm = marg_idx[0]
            nodes = torch.linspace(-JOINT_QUAD_SPAN, JOINT_QUAD_SPAN, JOINT_QUAD_NODES,
                                   device=targets.device, dtype=targets.dtype)
            du = float(nodes[1] - nodes[0])
            obs = {j: _std_inject(j, part) for j in req_idx}      # each [k, m]
            k, m = next(iter(obs.values())).shape
            K = nodes.numel()
            cols = []
            for j in JOINT_IDX:
                if j == jm:
                    cols.append(nodes.view(1, K, 1).expand(k, K, m))
                else:
                    cols.append(obs[j].unsqueeze(1).expand(k, K, m))
            vec = torch.stack(cols, dim=-1).reshape(k * K, m, len(JOINT_IDX))
            lp = flows.joint.log_prob_draws(vec, ctx_j[part]).view(k, K, m)
            lp = torch.logsumexp(lp, dim=1) + float(np.log(du))   # [k, m]
            joint_terms.append(-lp.mean()); joint_counts.append(int(part.sum()))

    if joint_terms:
        tot_n = float(sum(joint_counts))
        nll = sum(t * (c / tot_n) for t, c in zip(joint_terms, joint_counts))
        raw.append(float(nll.item()))
        terms.append(nll)
        total = total + float(weights[N_TARGETS]) * (tot_n / B) * nll
    else:
        raw.append(None)
        terms.append(None)
    return (total, raw, terms) if return_terms else (total, raw)


# ---------------------------------------------------------------- diagnostics
def param_diagnostic_groups(encoder, head, flows) -> dict[str, list[torch.Tensor]]:
    """Named parameter groups for LR / weight-decay diagnostics.

    Adapters are split by encoder depth so a per-depth learning-rate decision
    has evidence behind it.
    """
    groups: dict[str, list[torch.Tensor]] = {"cls_tokens": [encoder.cls_token]}
    adapters = list(encoder.cls_read_adapters)
    n = len(adapters)
    for label, lo, hi in (("adapters_low", 0, n // 3),
                          ("adapters_mid", n // 3, 2 * n // 3),
                          ("adapters_high", 2 * n // 3, n)):
        params = [p for a in adapters[lo:hi] for p in a.parameters()]
        if params:
            groups[label] = params
    groups["shared_mlp"] = list(head.parameters())
    for j, f in enumerate(flows.flows):
        groups[f"flow_{HEAD_NAMES[j]}"] = list(f.parameters())
    # Named for the head it actually is. The label was "flow_p2xp3_joint" from
    # when the only joint was the 2-D P2xP3 one; the joint is now whatever
    # JOINT_PAIR says, so a stale name would mislabel the LR evidence.
    groups[f"flow_{HEAD_NAMES[-1]}"] = list(flows.joint.parameters())
    return groups


@torch.no_grad()
def group_snapshot(groups: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.cat([p.detach().reshape(-1) for p in v]).clone() for k, v in groups.items()}


@torch.no_grad()
def group_metrics(groups, prev: dict[str, torch.Tensor] | None) -> dict[str, float]:
    """Weight norm, grad norm, and relative movement since the last snapshot.

    ``move`` (|w_t - w_{t-k}| / |w_t|) is the update-to-weight ratio aggregated
    over k steps: groups moving orders of magnitude faster or slower than the
    rest are the candidates for a different learning rate; ``wnorm`` drifting
    up means weight decay is not binding.
    """
    out: dict[str, float] = {}
    for name, params in groups.items():
        flat = torch.cat([p.detach().reshape(-1) for p in params])
        wn = float(flat.norm())
        out[f"wnorm/{name}"] = wn
        gs = [p.grad.detach().reshape(-1) for p in params if p.grad is not None]
        if gs:
            out[f"gnorm/{name}"] = float(torch.cat(gs).norm())
        if prev is not None and name in prev and wn > 0:
            out[f"move/{name}"] = float((flat - prev[name]).norm() / wn)
    return out


def head_influence(loss_terms, contexts) -> dict[str, float]:
    """How hard each head pulls on the SHARED representation.

    d(head loss)/d(context) at the shared/private boundary: one small backward
    per head through its flow only. This is the honest measure of a head's
    influence on the shared trunk -- loss weight is not it, since a converged
    head has small gradients whatever its weight.
    """
    out: dict[str, float] = {}
    for j, term in enumerate(loss_terms):
        if term is None:
            continue
        g = torch.autograd.grad(term, contexts, retain_graph=True, allow_unused=True)[0]
        if g is None:
            continue
        out[f"influence/{HEAD_NAMES[j]}"] = float(g.detach().norm())
    tot = sum(out.values())
    if tot > 0:
        out.update({k.replace("influence/", "influence_share/"): v / tot for k, v in list(out.items())})
    return out


class MultiTargetLookup:
    """targetid -> (targets [7], sig_lo [7], sig_hi [7]) across all staged splits."""

    def __init__(self, staged_dir: Path, extra_targets_csv: Path | None) -> None:
        ys, slos, shis, tids = [], [], [], []
        for split in ("train", "val", "test"):
            path = Path(staged_dir) / f"desi_{split}.hdf5"
            y, lo, hi = load_multi_target_matrix(path, extra_targets_csv)
            with h5py.File(path, "r") as handle:
                tids.append(handle["desi_targetid"][:].astype(np.int64))
            ys.append(y); slos.append(lo); shis.append(hi)
        tid = np.concatenate(tids)
        y = np.concatenate(ys); lo = np.concatenate(slos); hi = np.concatenate(shis)
        _, first = np.unique(tid, return_index=True)
        self.index = {int(t): int(i) for i, t in zip(first, tid[first])}
        self.y = torch.from_numpy(y.astype(np.float32))
        self.slo = torch.from_numpy(lo.astype(np.float32))
        self.shi = torch.from_numpy(hi.astype(np.float32))

    def batch(self, targetids: torch.Tensor, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = torch.tensor([self.index[int(t)] for t in targetids.cpu().tolist()], dtype=torch.long)
        return (self.y[rows].to(device), self.slo[rows].to(device), self.shi[rows].to(device))

    def values_for(self, targetids: np.ndarray) -> np.ndarray:
        rows = [self.index[int(t)] for t in targetids if int(t) in self.index]
        return self.y[rows].numpy()


def run_train_multi(args) -> None:
    """Train the 8-head read-only-CLS multi-target model."""
    import json
    import time

    try:
        from .attention_pooling_head import ComboSampler
        from .data_to_aion_embeddings import AIONTokenEncoder, build_dataloaders, write_json
        from .main import _OOM_ERROR  # noqa: F401  (import kept for parity)
        from .tracking import init_tracking
    except ImportError:
        from attention_pooling_head import ComboSampler
        from data_to_aion_embeddings import AIONTokenEncoder, build_dataloaders, write_json
        from tracking import init_tracking

    import pandas as pd

    configure_heads(tuple(args.drop_heads or ()))
    if args.drop_heads:
        print(f"[multi] dropped heads: {list(args.drop_heads)} -> {N_HEADS} heads", flush=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = Path(args.output_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Validation forwards are all-inputs at full length; cap their batch so the
    # attention transient (B x H x N^2) stays bounded regardless of loader size.
    eval_bs = min(args.batch_size, args.bucket_chunk if args.bucketed else 224)
    train_loader, val_loader, _ = build_dataloaders(
        staged_dir=Path(args.staged_dir), target_name="log_ml_flux_1",
        batch_size=args.batch_size, eval_batch_size=eval_bs,
        num_workers=args.num_workers, seed=args.seed,
        clean_split_csv=Path(args.clean_split_csv) if args.clean_split_csv else None,
    )
    # Train probe: a FIXED slice of the training split, scored with the exact
    # validation protocol, so train and val curves measure the same estimand
    # and their gap is the generalization gap.
    from torch.utils.data import DataLoader, Subset
    probe_n = min(int(args.train_probe_size), len(train_loader.dataset))
    # A SEEDED RANDOM subset, not the first N: the staged rows are ordered by
    # source_row, so a head slice is a biased sample of the sky and reads as a
    # systematic train/val offset rather than a generalization gap.
    probe_idx = np.random.default_rng(args.seed).choice(
        len(train_loader.dataset), size=probe_n, replace=False
    ).tolist() if probe_n > 0 else []
    probe_loader = DataLoader(
        Subset(train_loader.dataset, probe_idx),
        batch_size=eval_bs, shuffle=False, num_workers=args.num_workers, pin_memory=True,
    ) if probe_n > 0 else None
    lookup = MultiTargetLookup(Path(args.staged_dir), Path(args.extra_targets_csv) if args.extra_targets_csv else None)
    assign = pd.read_csv(args.clean_split_csv)
    train_tids = assign.loc[assign.split == "train", "targetid"].to_numpy(np.int64)
    train_y = lookup.values_for(train_tids)
    standardizers = []
    for j in range(N_TARGETS):
        vals = train_y[:, j][np.isfinite(train_y[:, j])]
        standardizers.append(TargetStandardizer.fit(vals))
        print(f"[multi] {MULTI_TARGETS[j]['name']}: {len(vals)} train values, "
              f"mean {standardizers[j].mean:.3f} std {standardizers[j].std:.3f}", flush=True)

    encoder = AIONTokenEncoder(
        freeze=False, cls_mode=True, cls_variant="readonly", num_cls=N_HEADS,
        grad_checkpoint=args.grad_checkpoint,
    ).to(device)
    head = SharedCLSHead().to(device)
    flows = MultiTargetFlows().to(device)
    # Separate LR for the zero-initialized read adapters: on a single LR they
    # move ~30x faster in update/weight terms than the standard-initialized
    # MLP and flows (|w| ~3 vs ~40), which left the flows within a few percent
    # of their initialization for a whole run.
    adapter_lr = args.adapter_lr if args.adapter_lr is not None else args.lr
    # BODY = the shared trunk every head reads through (CLS tokens, read
    # adapters, shared MLP). HEAD = the per-target flows. Splitting them lets
    # the flows be tuned without disturbing the representation, which is the
    # part all heads share. The adapters keep their own LR on top: they are
    # zero-initialised, so on a shared LR they move ~30x faster in
    # update/weight terms than the standard-initialised MLP and flows.
    head_lr = args.head_lr if getattr(args, "head_lr", None) is not None else args.lr
    joint_only = bool(getattr(args, "joint_only", False))
    # Quadrature resolution is a memory knob: rows missing the marginalisable
    # dimension cost nodes x inject_samples flow evaluations, so 48 nodes with
    # 50 draws is 2400x a single row. See --joint-quad-nodes for the measured
    # convergence.
    global JOINT_QUAD_NODES, JOINT_QUAD_SPAN
    if getattr(args, "joint_quad_nodes", None):
        JOINT_QUAD_NODES = int(args.joint_quad_nodes)
    if getattr(args, "joint_quad_span", None):
        JOINT_QUAD_SPAN = float(args.joint_quad_span)
    print(f"[multi] joint quadrature: {JOINT_QUAD_NODES} nodes, span +/-{JOINT_QUAD_SPAN}",
          flush=True)
    # joint_only: only the joint flow is optimized. The marginal flows still
    # exist (the joint indexes the same target matrix) but get no loss term.
    flow_params = list(flows.joint.parameters()) if joint_only else list(flows.parameters())
    param_groups = [
        {"params": list(head.parameters()), "lr": args.lr,
         "weight_decay": args.weight_decay},
        {"params": flow_params, "lr": head_lr,
         "weight_decay": args.weight_decay},
        {"params": [encoder.cls_token], "lr": args.lr, "weight_decay": 0.0},
        {"params": list(encoder.cls_read_adapters.parameters()), "lr": adapter_lr,
         "weight_decay": args.adapter_wd},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(args.beta1, 0.999))
    trainable = [p for g in param_groups for p in g["params"]]
    # Count the steps the SCHEDULER will actually take. With
    # --accumulate-buckets the buckets are accumulated into ONE optimizer step
    # per batch; without it each bucket (and each chunk of an oversized bucket)
    # steps separately. Multiplying by the bucket count in the accumulate case
    # overstates the horizon ~4x, so a cosine sized that way never anneals: it
    # reaches only a quarter of its schedule and ends near peak LR.
    if args.bucketed and getattr(args, "accumulate_buckets", False):
        steps_per_epoch = max(1, len(train_loader))
    elif args.bucketed:
        steps_per_epoch = max(1, len(train_loader)) * len(LENGTH_BUCKETS)
    else:
        steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = max(0, int(getattr(args, "warmup_steps", 0)))
    # Print the step budget. This run has FEW optimizer steps: ~20,160 train
    # rows over the batch size, so a large batch leaves only tens of steps per
    # epoch and a warmup sized in "typical" hundreds would swallow the run.
    print(f"[multi] {steps_per_epoch} steps/epoch x {args.epochs} epochs "
          f"= {total_steps} optimizer steps; warmup {warmup_steps} "
          f"({100.0 * warmup_steps / max(1, total_steps):.0f}%)", flush=True)
    if warmup_steps > 0.3 * total_steps:
        print(f"[multi] WARNING: warmup is {100.0 * warmup_steps / max(1, total_steps):.0f}% "
              f"of the whole run. Size it against steps/epoch, not a habit.", flush=True)
    scheduler = None
    if args.lr_schedule == "cosine" or warmup_steps > 0:
        # One LambdaLR for warmup AND cosine. LambdaLR scales each group's OWN
        # initial LR, so the per-part ratios set above (adapters vs flows vs
        # trunk) are preserved exactly -- which CosineAnnealingLR also does, but
        # composing the two schedulers would not.
        cosine = args.lr_schedule == "cosine"

        def lr_factor(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            if not cosine:
                return 1.0
            done = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return float(0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, done)))))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    # Per-head loss weight, multiplied on top of the detached-EMA balance. The
    # EMA equalises SCALE across heads; this is the manual lever for saying a
    # head matters more or less than that, e.g. holding back a head that
    # converges early so it stops dragging the shared trunk.
    head_weight = np.ones(N_HEADS)
    for spec in (args.head_loss_weight or []):
        name, _, val = spec.partition("=")
        if name not in HEAD_NAMES:
            raise SystemExit(f"--head-loss-weight: unknown head {name!r}; "
                             f"choose from {HEAD_NAMES}")
        head_weight[HEAD_NAMES.index(name)] = float(val)
    if not np.allclose(head_weight, 1.0):
        print("[multi] head loss weights: " + ", ".join(
            f"{n}={w:g}" for n, w in zip(HEAD_NAMES, head_weight) if w != 1.0), flush=True)

    sampler = ComboSampler.default()
    generator = torch.Generator(); generator.manual_seed(args.seed)
    ema = EMALossWeights()

    config = {
        "mode": "train-multi", "heads": HEAD_NAMES, "epochs": args.epochs,
        "batch_size": args.batch_size, "lr": args.lr, "lr_schedule": args.lr_schedule,
        "joint_only": joint_only, "warmup_steps": warmup_steps,
        "grad_clip": float(getattr(args, "grad_clip", 5.0)),
        "weight_decay": args.weight_decay, "adapter_wd": args.adapter_wd,
        "adapter_lr": adapter_lr, "head_lr": head_lr,
        "head_loss_weight": {n: float(w) for n, w in zip(HEAD_NAMES, head_weight)},
        "beta1": args.beta1,
        "inject": not args.no_inject, "inject_samples": int(args.inject_samples),
        "grad_checkpoint": args.grad_checkpoint,
        "bucketed": bool(args.bucketed),
        "accumulate_buckets": bool(args.accumulate_buckets),
        "drop_heads": list(args.drop_heads or []),
        "heads": HEAD_NAMES,
        "cls_variant": "readonly", "num_cls": N_HEADS, "adapter": "full-rank",
        "standardizers": [s.state_dict() for s in standardizers],
        "clean_split_csv": str(args.clean_split_csv), "extra_targets_csv": str(args.extra_targets_csv),
    }
    write_json(run_dir / "config.json", config)
    tracker = init_tracking(
        enabled=args.wandb, project=args.wandb_project, entity=args.wandb_entity,
        run_name=args.run_id, mode="online", config=config, run_dir=run_dir,
        tags=["multi-target", "v3b"],
    )

    def save(path, epoch, val_sum):
        torch.save({
            "epoch": epoch, "config": config,
            "encoder_trainable_state_dict": encoder.encoder_trainable_state(),
            "head_state_dict": head.state_dict(),
            "flows_state_dict": flows.state_dict(),
            "standardizers": [s.state_dict() for s in standardizers],
            "ema": ema.state_dict(), "val_multi_nll_sum": val_sum,
        }, path)

    grad_clip = float(getattr(args, "grad_clip", 5.0))
    snapshot_every = int(getattr(args, "snapshot_every", 0))
    snap_dir = run_dir / "snapshots"
    if snapshot_every > 0:
        snap_dir.mkdir(parents=True, exist_ok=True)
    diag_groups = param_diagnostic_groups(encoder, head, flows)
    prev_snapshot = None
    best = float("inf"); best_epoch = 0; global_step = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.monotonic()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        encoder.eval(); head.train(); flows.train()
        # With ONE head the EMA normaliser (weight = 1/EMA) becomes a
        # time-varying global scale on the only loss, i.e. a second learning
        # rate schedule fighting the cosine. Pin it.
        weights = (np.ones(N_HEADS) if joint_only
                   else ema.update_and_weights([None] * N_HEADS)) * head_weight
        n_seen = 0
        ep_sum = {n: 0.0 for n in HEAD_NAMES}; ep_cnt = {n: 0 for n in HEAD_NAMES}
        bk_sum: dict[tuple[str, str], float] = {}; bk_cnt: dict[tuple[str, str], int] = {}
        diag_extra: dict[str, float] = {}
        grad_norm = 0.0; last_raw = [None] * N_HEADS; last_loss = float("nan")
        for batch in train_loader:
            batch = tuple(t.to(device, non_blocking=True) for t in batch)
            y_all, slo_all, shi_all = lookup.batch(batch[7], device)
            if args.bucketed:
                # Length-bucketed packing: per-source combos, one forward AND
                # one optimizer step per bucket -- each bucket lands near the
                # calibrated per-forward size while the step count stays high.
                B = int(batch[6].shape[0])
                combos_ps = [sampler.sample(generator) for _ in range(B)]
                steps = []
                bucket_names = []
                for bucket, idx in bucket_assignments(combos_ps):
                    # Chunk cap: bucket shares are uneven (the heavy bucket is
                    # ~46% of the size-stratified mix), so cap each forward at
                    # the calibrated size instead of trusting the bucket split.
                    for lo in range(0, len(idx), args.bucket_chunk):
                        part = idx[lo : lo + args.bucket_chunk]
                        rows = torch.from_numpy(part).to(device)
                        sub = tuple(t[rows] for t in batch)
                        drop = {k: v.to(device) for k, v in
                                bucket_modality_dropout(bucket, combos_ps, part).items()}
                        steps.append((bucket["union"], sub, drop, rows))
                        bucket_names.append(bucket["name"])
            else:
                steps = [(sampler.sample(generator), batch, None, None)]
                bucket_names = ["all"]
            # With --accumulate-buckets every optimizer step sees the FULL combo
            # mix (gradients summed over all buckets) instead of alternating
            # between combo families: fewer, much lower-variance updates.
            accumulate = bool(args.accumulate_buckets)
            if accumulate:
                optimizer.zero_grad(set_to_none=True)
            n_steps = len(steps)
            want_diag = (global_step % args.diag_every == 0)
            for si, (combo, sub, drop, rows) in enumerate(steps):
                if rows is None:
                    y, slo, shi = y_all, slo_all, shi_all
                else:
                    y, slo, shi = y_all[rows], slo_all[rows], shi_all[rows]
                cls_seq, _ = encoder.encode_tokens(sub, tuple(combo), modality_dropout=drop)
                contexts = head(cls_seq)
                loss, raw, terms = multi_target_nll(
                    contexts=contexts, flows=flows, targets=y, sig_lo=slo, sig_hi=shi,
                    standardizers=standardizers, weights=weights, inject=not args.no_inject,
                    inject_samples=args.inject_samples, return_terms=True,
                    joint_only=joint_only,
                )
                if want_diag and si == 0:
                    diag_extra.update(head_influence(terms, contexts))
                if not accumulate:
                    optimizer.zero_grad(set_to_none=True)
                (loss / n_steps if accumulate else loss).backward()
                nb = int(sub[6].shape[0])
                bname = bucket_names[si] if si < len(bucket_names) else "mixed"
                for name, val in zip(HEAD_NAMES, raw):
                    if val is not None:
                        ep_sum[name] += val * nb; ep_cnt[name] += nb
                        bk_sum[(bname, name)] = bk_sum.get((bname, name), 0.0) + val * nb
                        bk_cnt[(bname, name)] = bk_cnt.get((bname, name), 0) + nb
                n_seen += nb
                if not accumulate:
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    if not joint_only:
                        weights = ema.update_and_weights(raw) * head_weight
                    global_step += 1
                last_raw, last_loss = raw, float(loss.item())
            if accumulate:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                if not joint_only:
                    weights = ema.update_and_weights(last_raw) * head_weight
                global_step += 1
            # NOT gated on tracker.enabled: tracker.log mirrors to
            # history.jsonl either way, so a run without wandb still records
            # the per-group movement the LR decision depends on.
            if want_diag:
                payload = {"train/weighted_loss": last_loss,
                           "train/grad_norm": float(grad_norm),
                           "train/lr": float(optimizer.param_groups[0]["lr"]), "epoch": epoch}
                payload.update(diag_extra)
                payload.update(group_metrics(diag_groups, prev_snapshot))
                prev_snapshot = group_snapshot(diag_groups)
                payload.update({f"weight/{n}": float(w) for n, w in zip(HEAD_NAMES, weights)})
                tracker.log(payload, step=global_step)
                diag_extra = {}

        def evaluate(loader):
            """Plain-LL NLL per head on the all-inputs combo (the val protocol)."""
            sums_ = np.zeros(N_HEADS); counts_ = np.zeros(N_HEADS)
            with torch.no_grad():
                for b in loader:
                    b = tuple(t.to(device, non_blocking=True) for t in b)
                    yv, slov, shiv = lookup.batch(b[7], device)
                    cs, _ = encoder.encode_tokens(b, ("spectra", "z", "wise", "image"))
                    _, raw_ = multi_target_nll(
                        contexts=head(cs), flows=flows, targets=yv, sig_lo=slov, sig_hi=shiv,
                        standardizers=standardizers, weights=np.ones(N_HEADS), inject=False,
                    )
                    nb = int(b[6].shape[0])
                    for jj, vv in enumerate(raw_):
                        if vv is not None:
                            sums_[jj] += vv * nb; counts_[jj] += nb
            return sums_ / np.maximum(counts_, 1)

        # validation: all-inputs combo, plain NLL (no injection) + posterior-mean
        # R^2 per scalar head so the review does not need a separate eval pass
        encoder.eval(); head.eval(); flows.eval()
        sums = np.zeros(N_HEADS); counts = np.zeros(N_HEADS)
        preds = [[] for _ in range(N_TARGETS)]; trues = [[] for _ in range(N_TARGETS)]
        with torch.no_grad():
            for batch in val_loader:
                batch = tuple(t.to(device, non_blocking=True) for t in batch)
                y, slo, shi = lookup.batch(batch[7], device)
                cls_seq, _ = encoder.encode_tokens(batch, ("spectra", "z", "wise", "image"))
                contexts = head(cls_seq)
                _, raw = multi_target_nll(
                    contexts=contexts, flows=flows, targets=y, sig_lo=slo, sig_hi=shi,
                    standardizers=standardizers, weights=np.ones(N_HEADS), inject=False,
                )
                n = int(batch[6].shape[0])
                for j, val in enumerate(raw):
                    if val is not None:
                        sums[j] += val * n; counts[j] += n
                for j in range(N_TARGETS):
                    mask = torch.isfinite(y[:, j])
                    if bool(mask.any()):
                        samp = flows.flows[j].sample(contexts[mask, j], num_samples=64)
                        preds[j].append(samp.mean(dim=0).cpu().numpy())
                        trues[j].append(standardizers[j].transform_tensor(y[mask, j]).cpu().numpy())
        val_r2 = np.full(N_TARGETS, np.nan)
        for j in range(N_TARGETS):
            if preds[j]:
                yp, yt = np.concatenate(preds[j]), np.concatenate(trues[j])
                ss = np.sum((yt - yt.mean()) ** 2)
                val_r2[j] = 1.0 - np.sum((yt - yp) ** 2) / ss if ss > 0 else np.nan
        val_nll = sums / np.maximum(counts, 1)
        val_sum = float(val_nll.sum())
        val_pair_mean = float(sums.sum() / max(counts.sum(), 1))
        dt = time.monotonic() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        print(f"epoch={epoch} val_sum={val_sum:.3f} " +
              " ".join(f"{n}={v:.3f}" for n, v in zip(HEAD_NAMES, val_nll)) +
              f" ({dt:.0f}s, {n_seen/dt:.1f}/s, {peak:.1f}GB)", flush=True)
        payload = {"epoch": epoch, "val/multi_nll_sum": val_sum,
                   "val/pair_mean_nll": val_pair_mean, "epoch_seconds": dt,
                   "throughput/samples_per_second": n_seen / dt, "vram/peak_gb": peak}
        payload.update({f"val/nll_{n}": float(v) for n, v in zip(HEAD_NAMES, val_nll)})
        # per-epoch train means (count-weighted): smooth convergence signal,
        # still on injected targets so NOT comparable to val
        payload.update({f"trainmean/nll_{n}": ep_sum[n] / max(ep_cnt[n], 1)
                        for n in HEAD_NAMES if ep_cnt[n]})
        # per-bucket series: makes the combo-composition effect explicit
        # instead of aliasing it into the step-level curve
        for (bname, hname), tot in bk_sum.items():
            c = bk_cnt[(bname, hname)]
            if c:
                payload[f"bucket/{bname}/nll_{hname}"] = tot / c
        # train probe: SAME protocol as val, so the gap is the generalization gap
        if probe_loader is not None:
            probe_nll = evaluate(probe_loader)
            payload.update({f"probe/nll_{n}": float(v) for n, v in zip(HEAD_NAMES, probe_nll)})
            payload.update({f"gap/nll_{n}": float(v - p_) for n, v, p_
                            in zip(HEAD_NAMES, val_nll, probe_nll)})
            payload["gap/pair_mean"] = float(np.mean(val_nll - probe_nll))
        payload.update({f"val/r2_{n}": float(v) for n, v in zip(HEAD_NAMES[:N_TARGETS], val_r2) if np.isfinite(v)})
        payload.update({f"ema/weight_{n}": float(w) for n, w in zip(HEAD_NAMES, weights)})
        tracker.log(payload, step=global_step)
        save(run_dir / "last.pt", epoch, val_pair_mean)
        # Periodic snapshots: phase 2 picks the BODY by post-refit validation,
        # which is not the same epoch as min joint val NLL. AION is frozen and
        # excluded, so these are small.
        if snapshot_every > 0 and epoch % snapshot_every == 0:
            save(snap_dir / f"epoch_{epoch:03d}.pt", epoch, val_pair_mean)
        if val_pair_mean < best:
            best = val_pair_mean
            best_epoch = epoch
            save(run_dir / "best.pt", epoch, val_pair_mean)
        if epoch - best_epoch >= args.early_stop_patience:
            print(f"[early-stop] no improvement for {args.early_stop_patience} epochs "
                  f"(best epoch {best_epoch})", flush=True)
            break
    tracker.summary({"val/best_pair_mean_nll": best, "best_epoch": best_epoch})
    tracker.finish()


# Length-bucketed combo packing (user design): combos grouped by padded token
# length, not identity. Each bucket encodes the UNION of its combos' modalities
# once; per-source masks drop the rest (padding waste <=1.5%). One optimizer
# step per bucket forward: with a large loader batch, each bucket lands at the
# calibrated per-forward size while keeping the step count of small batches.
LENGTH_BUCKETS: list[dict] = [
    {"name": "tiny", "union": ("z", "wise"),
     "combos": {("z",), ("wise",), ("z", "wise")}},
    {"name": "spectra", "union": ("spectra", "z", "wise"),
     "combos": {("spectra",), ("spectra", "z"), ("spectra", "wise"), ("spectra", "z", "wise")}},
    {"name": "image", "union": ("z", "wise", "image"),
     "combos": {("image",), ("z", "image"), ("wise", "image"), ("z", "wise", "image")}},
    {"name": "heavy", "union": ("spectra", "z", "wise", "image"),
     "combos": {("spectra", "image"), ("spectra", "z", "image"), ("spectra", "wise", "image"),
                ("spectra", "z", "wise", "image")}},
]


def bucket_assignments(
    combos_per_source: list[tuple[str, ...]],
) -> list[tuple[dict, np.ndarray]]:
    """Partition per-source combos into length buckets.

    Returns (bucket_spec, source_indices) for each non-empty bucket; every
    source lands in exactly one bucket.
    """
    out = []
    assigned = np.zeros(len(combos_per_source), dtype=bool)
    for bucket in LENGTH_BUCKETS:
        idx = np.array([i for i, c in enumerate(combos_per_source) if tuple(c) in bucket["combos"]],
                       dtype=np.int64)
        if len(idx):
            out.append((bucket, idx))
            assigned[idx] = True
    if not bool(assigned.all()):
        missing = [combos_per_source[i] for i in np.flatnonzero(~assigned)]
        raise ValueError(f"Combos not covered by any length bucket: {set(missing)}")
    return out


def bucket_modality_dropout(
    bucket: dict, combos: list[tuple[str, ...]], idx: np.ndarray
) -> dict[str, torch.Tensor]:
    """Per-source modality-dropout masks WITHIN a bucket (True = drop)."""
    return {
        group: torch.tensor([group not in combos[i] for i in idx], dtype=torch.bool)
        for group in bucket["union"]
    }
