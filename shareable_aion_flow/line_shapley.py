"""Shapley attribution of flux information over spectrum tokens.

Players: 7 emission-line windows (rest frame, z-dependent token positions) plus
fixed rest-frame continuum bins. Value: test log-likelihood of a spectra-only
flux model (identical to info gain up to the constant prior term). Removal
("drop" mode, the default): masked pixel windows are filled with the source's
own median continuum BEFORE the codec (its ConvNeXt encoder has a multi-token
receptive field, so kept neighbouring tokens would otherwise still see the
line), and the masked tokens are then dropped from the AION encoder input the
native 4M way -- partial input is the backbone's pretraining task.

Token geometry follows the CODEC latent grid (lambda_min = 3500 A, 0.8 A/px,
8704 px, 32 px/token -> 272 wavelength tokens), NOT the DESI observed grid
(3600-9824 A): the two are offset by 100 A = 3.9 tokens. Only tokens fully
inside the observed range are usable.

Estimator: per-source randomized permutations (the test set is the Monte Carlo
ensemble). Sweep kinds:
- FULL: walk each source's permutation start-to-end (|P|+1 batched forwards,
  marginals for every player).
- LINE (Owen multilinear): include other players w.p. u~U(0,1); two forwards
  per line give unbiased line marginals cheaply.
- PAIR: same Owen base; evaluating base + 7 single-flips + 21 double-flips
  (29 forwards) yields unbiased pairwise Shapley interaction samples for all
  line pairs, plus extra single-line marginal samples for free.
"""

from __future__ import annotations

import numpy as np
import torch

# AION spectrum-codec latent grid (aion/codecs/spectrum.py: lambda_min=3500,
# resolution=0.8, num_pixels=8704) and the DESI observed coverage.
CODEC_LO, GRID_STEP, GRID_PIX = 3500.0, 0.8, 8704
PIX_PER_TOKEN = 32
N_SPEC_TOKENS = GRID_PIX // PIX_PER_TOKEN  # 272 codec wavelength tokens (+1 norm token)
OBS_LO, OBS_HI = 3600.0, 9824.0  # DESI spectral coverage in the observed frame
RAW_LO = 3600.0  # first pixel of the raw DESI grid carried in our batches

LINE_PLAYERS = [  # (name, rest wavelength, half-window in rest-frame Angstrom)
    # Broad-line windows are +-120 A: BLR FWHM up to ~10,000 km/s puts line
    # flux ~+-110 A from centre at Halpha; +-80 A left broad wings outside the
    # mask and biased those lines low.
    ("NeV", 3426.9, 30.0),
    ("OII", 3727.5, 30.0),
    ("HeII", 4686.0, 30.0),
    ("Hbeta", 4862.7, 120.0),
    ("OIII", 5008.2, 30.0),
    ("MgII", 2798.0, 120.0),
    ("Halpha", 6564.6, 120.0),
]
N_LINES = len(LINE_PLAYERS)


def token_obs_wavelength_edges() -> tuple[np.ndarray, np.ndarray]:
    """Observed-frame (lo, hi) wavelength edges of each codec wavelength token."""
    idx = np.arange(N_SPEC_TOKENS)
    lo = CODEC_LO + GRID_STEP * PIX_PER_TOKEN * idx
    return lo, lo + GRID_STEP * PIX_PER_TOKEN


def usable_tokens() -> np.ndarray:
    """Codec tokens fully inside the DESI observed range (bool [272])."""
    lo, hi = token_obs_wavelength_edges()
    return (lo >= OBS_LO) & (hi <= OBS_HI)


def token_to_raw_pixels(token: int) -> tuple[int, int]:
    """Raw-grid (3600 A based) pixel span [lo, hi) of one codec token."""
    lam_lo = CODEC_LO + GRID_STEP * PIX_PER_TOKEN * token
    p_lo = int(round((lam_lo - RAW_LO) / GRID_STEP))
    return p_lo, p_lo + PIX_PER_TOKEN


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


def player_token_map(
    z: float, players: list[dict], line_guard_tokens: int = 1
) -> list[np.ndarray]:
    """Codec-token indices (0..271) claimed by each player at this z.

    A token belongs to a player when its observed-wavelength CENTRE falls in
    the player's rest window; line players (first in the catalog) claim first.
    Line claims are dilated by ``line_guard_tokens`` on each side: the codec's
    ConvNeXt encoder has a multi-token effective receptive field, so codes of
    tokens adjacent to a line carry line information -- the guard band removes
    them together with the line (measured by scripts/codec_leakage_probe.py).
    Continuum bins get no guard (10-20 tokens wide; boundary blur is a small
    fractional effect there). Tokens outside DESI coverage are unusable.
    """
    lo, hi = token_obs_wavelength_edges()
    rest_center = (0.5 * (lo + hi)) / (1.0 + z)
    claimed = ~usable_tokens()
    out: list[np.ndarray] = []
    for p in players:
        sel = (rest_center >= p["rest_lo"]) & (rest_center <= p["rest_hi"]) & ~claimed
        idx = np.flatnonzero(sel)
        if len(idx) and p["kind"] == "line" and line_guard_tokens > 0:
            g = int(line_guard_tokens)
            widened = np.arange(idx.min() - g, idx.max() + g + 1)
            widened = widened[(widened >= 0) & (widened < N_SPEC_TOKENS)]
            idx = widened[~claimed[widened]]
        claimed[idx] = True
        out.append(idx.astype(np.int64))
    return out


def sample_training_mask(
    z_batch: np.ndarray, players: list[dict], rng: np.random.Generator
) -> np.ndarray:
    """Random-coalition mask for training augmentation.

    Per source: draw coalition size uniformly over {0..n_avail}, mask the
    complement. Returns bool [B, 272], True = remove this token.
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


class PairInteractionAccumulator:
    """Streaming mean/SE of pairwise Shapley interaction samples (lines only)."""

    def __init__(self, n_lines: int = N_LINES) -> None:
        self.n = n_lines
        self.sum = np.zeros((n_lines, n_lines))
        self.sumsq = np.zeros((n_lines, n_lines))
        self.count = np.zeros((n_lines, n_lines), dtype=np.int64)

    def add(self, i: int, j: int, delta: float) -> None:
        for a, b in ((i, j), (j, i)):
            self.sum[a, b] += delta
            self.sumsq[a, b] += delta**2
            self.count[a, b] += 1

    def table(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = np.maximum(self.count, 1)
        mean = self.sum / n
        var = np.maximum(self.sumsq / n - mean**2, 0.0)
        se = np.sqrt(var / n)
        return mean, se, self.count


@torch.no_grad()
def masked_log_prob(
    *, encoder, context_encoder, flow, batch, standardizer, spectrum_token_mask,
    mask_mode: str = "drop",
) -> np.ndarray:
    """Per-source log p(y|x_spectra-masked). Plain likelihood (none/inject eval)."""
    target_std = standardizer.transform_tensor(batch[6])
    tokens, group_ids = encoder.encode_tokens(
        batch, ("spectra",), spectrum_token_mask=spectrum_token_mask, mask_mode=mask_mode
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
    n_pair_sweeps: int = 3,
    mask_mode: str = "drop",
    line_guard_tokens: int = 1,
    seed: int = 0,
    z_edges: tuple[float, ...] = (0.0, 0.5, 1.0, 1.7, 99.0),
) -> tuple[ShapleyAccumulator, PairInteractionAccumulator]:
    """Per-source-randomized permutation/Owen sweeps over all batches."""
    rng = np.random.default_rng(seed)
    n_lines = sum(1 for p in players if p["kind"] == "line")
    acc = ShapleyAccumulator(len(players), n_z_bins=len(z_edges) - 1)
    pair_acc = PairInteractionAccumulator(n_lines)
    encoder.eval(); context_encoder.eval(); flow.eval()

    sweep_kinds = (["full"] * n_full_sweeps + ["line"] * n_line_sweeps + ["pair"] * n_pair_sweeps)
    for kind in sweep_kinds:
        for batch in loader:
            batch = tuple(t.to(device, non_blocking=True) for t in batch)
            z_np = batch[3].detach().cpu().numpy().ravel()
            B = len(z_np)
            tok_maps = [player_token_map(float(z), players, line_guard_tokens) for z in z_np]
            avail = [[j for j, t in enumerate(m) if len(t)] for m in tok_maps]
            z_bins = np.clip(np.searchsorted(z_edges, z_np, side="right") - 1, 0, len(z_edges) - 2)

            def eval_mask(mask_np):
                return masked_log_prob(
                    encoder=encoder, context_encoder=context_encoder, flow=flow,
                    batch=batch, standardizer=standardizer,
                    spectrum_token_mask=torch.from_numpy(mask_np).to(device),
                    mask_mode=mask_mode,
                )

            if kind == "full":
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
                continue

            # Owen base for line/pair sweeps: per source draw u ~ U(0,1) and
            # include each player independently w.p. u.
            u = rng.random(B)
            include = rng.random((B, len(players))) < u[:, None]

            def base_masks():
                """Mask of everything NOT included (lines handled by flips)."""
                base = np.zeros((B, N_SPEC_TOKENS), dtype=bool)
                for b in range(B):
                    for j in avail[b]:
                        if not include[b, j]:
                            base[b, tok_maps[b][j]] = True
                return base

            if kind == "line":
                for line_j in range(n_lines):
                    with_line = base_masks()
                    without = with_line.copy()
                    has_line = np.zeros(B, dtype=bool)
                    for b in range(B):
                        if len(tok_maps[b][line_j]):
                            has_line[b] = True
                            with_line[b, tok_maps[b][line_j]] = False
                            without[b, tok_maps[b][line_j]] = True
                    lp_with = eval_mask(with_line)
                    lp_without = eval_mask(without)
                    for b in range(B):
                        if has_line[b]:
                            acc.add(line_j, float(lp_with[b] - lp_without[b]), int(z_bins[b]))
                continue

            # kind == "pair": evaluate 1 base + 7 single-flips + 21 double-flips.
            # Flipping line l toggles its inclusion relative to `include`; the
            # four flip states of a pair generate all four presence combos, so
            # sign-corrected differences give interaction AND marginal samples.
            def config_mask(flips: tuple[int, ...]):
                mask = base_masks()
                for b in range(B):
                    for l in flips:
                        if len(tok_maps[b][l]):
                            mask[b, tok_maps[b][l]] = include[b, l]  # True = drop if it was included
                return mask

            lp: dict[tuple[int, ...], np.ndarray] = {(): eval_mask(config_mask(()))}
            for a in range(n_lines):
                lp[(a,)] = eval_mask(config_mask((a,)))
            for a in range(n_lines):
                for c in range(a + 1, n_lines):
                    lp[(a, c)] = eval_mask(config_mask((a, c)))
            for b in range(B):
                sign = {l: (1.0 if include[b, l] else -1.0) for l in range(n_lines)}
                have = [l for l in range(n_lines) if len(tok_maps[b][l])]
                for a in have:
                    # v(present) - v(absent) = sign_a * (v(base) - v(flip_a))
                    acc.add(a, float(sign[a] * (lp[()][b] - lp[(a,)][b])), int(z_bins[b]))
                for ai, a in enumerate(have):
                    for c in have[ai + 1:]:
                        key = (a, c) if a < c else (c, a)
                        delta = sign[a] * sign[c] * float(
                            lp[()][b] - lp[(a,)][b] - lp[(c,)][b] + lp[key][b]
                        )
                        pair_acc.add(a, c, delta)
    return acc, pair_acc
