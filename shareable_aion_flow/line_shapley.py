"""Shapley attribution of flux information over spectrum tokens.

Players: 7 emission-line windows (rest frame, z-dependent token positions) plus
fixed rest-frame continuum bins. Value: test log-likelihood of a spectra-only
flux model (identical to info gain up to the constant prior term). Removal:
masked spectrum tokens are replaced, at the token-code level before the AION
backbone, with codes from the source's own median-filtered continuum, so a
removed line reads as "same spectrum, continuum there".

Estimator: per-source randomized permutations (the test set is the Monte Carlo
ensemble). A FULL sweep walks each source's permutation start-to-end
(|P|+1 masked forwards, marginals for every player); a LINE-FOCUSED sweep
evaluates only the mask pairs bracketing the 7 line players (<=14 forwards),
buying extra line precision cheaply.
"""

from __future__ import annotations

import numpy as np
import torch

# Observed DESI grid and AION spectrum tokenization geometry.
GRID_LO, GRID_STEP, GRID_PIX = 3600.0, 0.8, 8704
PIX_PER_TOKEN = 32
N_SPEC_TOKENS = GRID_PIX // PIX_PER_TOKEN  # 272 wavelength tokens (+1 norm token in the dict)

LINE_PLAYERS = [  # (name, rest wavelength, half-window in rest-frame Angstrom)
    ("NeV", 3426.9, 30.0),
    ("OII", 3727.5, 30.0),
    ("HeII", 4686.0, 30.0),
    ("Hbeta", 4862.7, 80.0),
    ("OIII", 5008.2, 30.0),
    ("MgII", 2798.0, 80.0),
    ("Halpha", 6564.6, 80.0),
]
N_LINES = len(LINE_PLAYERS)


def token_obs_wavelengths() -> np.ndarray:
    """Observed-frame centre wavelength of each of the 272 spectrum tokens."""
    idx = np.arange(N_SPEC_TOKENS)
    return GRID_LO + GRID_STEP * PIX_PER_TOKEN * (idx + 0.5)


def continuum_bins(n_bins: int = 24) -> list[tuple[str, float, float]]:
    """Fixed rest-frame continuum bins (log-spaced over the reachable range)."""
    edges = np.geomspace(900.0, 9824.0, n_bins + 1)
    return [(f"cont_{int(lo)}_{int(hi)}", float(lo), float(hi)) for lo, hi in zip(edges[:-1], edges[1:])]


def player_catalog(n_cont_bins: int = 24) -> list[dict]:
    """Lines first (indices 0..6), then continuum bins."""
    players = [
        {"name": n, "kind": "line", "rest_lo": lam - hw, "rest_hi": lam + hw, "rest_center": lam}
        for n, lam, hw in LINE_PLAYERS
    ]
    players += [
        {"name": n, "kind": "cont", "rest_lo": lo, "rest_hi": hi, "rest_center": float(np.sqrt(lo * hi))}
        for n, lo, hi in continuum_bins(n_cont_bins)
    ]
    return players


def player_token_map(z: float, players: list[dict]) -> list[np.ndarray]:
    """Token indices (0..271, wavelength order) claimed by each player at this z.

    Line windows claim their tokens first; continuum bins get the remainder of
    their range. A player with no tokens is unavailable for this source.
    """
    rest = token_obs_wavelengths() / (1.0 + z)
    claimed = np.zeros(N_SPEC_TOKENS, dtype=bool)
    out: list[np.ndarray] = []
    for p in players:  # lines come first in the catalog, so they claim first
        sel = (rest >= p["rest_lo"]) & (rest <= p["rest_hi"]) & ~claimed
        idx = np.flatnonzero(sel)
        claimed[idx] = True
        out.append(idx.astype(np.int64))
    return out


def sample_training_mask(
    z_batch: np.ndarray, players: list[dict], rng: np.random.Generator
) -> np.ndarray:
    """Random-coalition mask for training augmentation.

    Per source: draw coalition size uniformly over {0..n_avail}, mask the
    complement. Returns bool [B, 272], True = replace with continuum code.
    """
    masks = np.zeros((len(z_batch), N_SPEC_TOKENS), dtype=bool)
    for b, z in enumerate(z_batch):
        tok = player_token_map(float(z), players)
        avail = [t for t in tok if len(t)]
        if not avail:
            continue
        k = rng.integers(0, len(avail) + 1)  # size of the KEPT coalition
        drop = rng.permutation(len(avail))[k:]
        for j in drop:
            masks[b, avail[j]] = True
    return masks


class ShapleyAccumulator:
    """Streaming mean/SE of per-source marginal contributions per player."""

    def __init__(self, n_players: int, n_z_bins: int = 4) -> None:
        self.sum = np.zeros(n_players)
        self.sumsq = np.zeros(n_players)
        self.count = np.zeros(n_players, dtype=np.int64)
        self.z_sum = np.zeros((n_z_bins, n_players))
        self.z_count = np.zeros((n_z_bins, n_players), dtype=np.int64)

    def add(self, player: int, marginal: float, z_bin: int) -> None:
        self.sum[player] += marginal
        self.sumsq[player] += marginal**2
        self.count[player] += 1
        self.z_sum[z_bin, player] += marginal
        self.z_count[z_bin, player] += 1

    def table(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = np.maximum(self.count, 1)
        phi = self.sum / n
        var = np.maximum(self.sumsq / n - phi**2, 0.0)
        se = np.sqrt(var / n)
        return phi, se, self.count

    def z_table(self) -> np.ndarray:
        return self.z_sum / np.maximum(self.z_count, 1)


@torch.no_grad()
def masked_log_prob(
    *, encoder, context_encoder, flow, batch, standardizer, spectrum_token_mask
) -> np.ndarray:
    """Per-source log p(y|x_spectra-masked). Plain likelihood (none/inject eval)."""
    target_std = standardizer.transform_tensor(batch[6])
    tokens, group_ids = encoder.encode_tokens(
        batch, ("spectra",), spectrum_token_mask=spectrum_token_mask
    )
    context = context_encoder(tokens, group_ids)
    return flow.log_prob(target_std, context).detach().cpu().numpy().ravel()


def run_sweeps(
    *,
    encoder,
    context_encoder,
    flow,
    loader,
    standardizer,
    players: list[dict],
    device,
    n_full_sweeps: int = 2,
    n_line_sweeps: int = 4,
    seed: int = 0,
    z_edges: tuple[float, ...] = (0.0, 0.5, 1.0, 1.7, 99.0),
) -> ShapleyAccumulator:
    """Per-source-randomized permutation sweeps over all batches."""
    rng = np.random.default_rng(seed)
    acc = ShapleyAccumulator(len(players), n_z_bins=len(z_edges) - 1)
    encoder.eval(); context_encoder.eval(); flow.eval()

    for sweep in range(n_full_sweeps + n_line_sweeps):
        line_only = sweep >= n_full_sweeps
        for batch in loader:
            batch = tuple(t.to(device, non_blocking=True) for t in batch)
            z_np = batch[3].detach().cpu().numpy().ravel()
            B = len(z_np)
            tok_maps = [player_token_map(float(z), players) for z in z_np]
            avail = [[j for j, t in enumerate(m) if len(t)] for m in tok_maps]
            z_bins = np.clip(np.searchsorted(z_edges, z_np, side="right") - 1, 0, len(z_edges) - 2)

            def eval_mask(mask_np):
                return masked_log_prob(
                    encoder=encoder, context_encoder=context_encoder, flow=flow,
                    batch=batch, standardizer=standardizer,
                    spectrum_token_mask=torch.from_numpy(mask_np).to(device),
                )

            if not line_only:
                # Full chain: per-source random permutation, |P|+1 batched steps.
                perms = [rng.permutation(a).tolist() if a else [] for a in avail]
                max_len = max((len(p) for p in perms), default=0)
                logp_prev = None
                for s_step in range(max_len + 1):
                    mask = np.zeros((B, N_SPEC_TOKENS), dtype=bool)
                    for b in range(B):
                        for j in perms[b][s_step:]:
                            mask[b, tok_maps[b][j]] = True
                    logp = eval_mask(mask)
                    if logp_prev is not None:
                        for b in range(B):
                            pos = s_step - 1
                            if pos < len(perms[b]):
                                acc.add(perms[b][pos], float(logp[b] - logp_prev[b]), int(z_bins[b]))
                    logp_prev = logp
            else:
                # Line-focused: Owen multilinear sampling. Per source draw
                # u ~ U(0,1); include each OTHER player independently w.p. u;
                # two batched forwards per line give unbiased Shapley marginals.
                u = rng.random(B)
                include = rng.random((B, len(players))) < u[:, None]
                for line_j in range(N_LINES):
                    base = np.zeros((B, N_SPEC_TOKENS), dtype=bool)
                    has_line = np.zeros(B, dtype=bool)
                    for b in range(B):
                        has_line[b] = len(tok_maps[b][line_j]) > 0
                        for j in avail[b]:
                            if j != line_j and not include[b, j]:
                                base[b, tok_maps[b][j]] = True
                    with_line = base.copy()
                    without = base.copy()
                    for b in range(B):
                        if has_line[b]:
                            without[b, tok_maps[b][line_j]] = True
                    lp_with = eval_mask(with_line)
                    lp_without = eval_mask(without)
                    for b in range(B):
                        if has_line[b]:
                            acc.add(line_j, float(lp_with[b] - lp_without[b]), int(z_bins[b]))
    return acc
