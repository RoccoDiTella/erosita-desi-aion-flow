"""CPU verification of the line-Shapley estimator against exact analytic values.

A stub encoder/flow makes the per-source log-likelihood an explicit function of
which players are kept: v(S) = a*[L1 in S] + b*[L2 in S] + c*[both] + d*[C in S].
For two line players the exact values are phi(L1) = a + c/2, phi(L2) = b + c/2,
phi(C) = d, and the pairwise Shapley interaction I(L1, L2) = c. The sweeps
(full chains, Owen line sweeps, pair sweeps) must all agree with these.
"""

from __future__ import annotations

import numpy as np
import torch

from shareable_aion_flow.line_shapley import (
    N_SPEC_TOKENS,
    PairInteractionAccumulator,
    player_token_map,
    run_sweeps,
    usable_tokens,
)
from shareable_aion_flow.normalizing_flow import TargetStandardizer

A, B_, C, D = 1.0, 0.4, 0.5, 0.25


def two_line_catalog() -> list[dict]:
    """Two line players + one continuum player with disjoint z=0 token maps."""
    return [
        {"name": "L1", "kind": "line", "rest_lo": 4600.0, "rest_hi": 4700.0, "rest_center": 4650.0},
        {"name": "L2", "kind": "line", "rest_lo": 5000.0, "rest_hi": 5100.0, "rest_center": 5050.0},
        {"name": "C", "kind": "cont", "rest_lo": 6000.0, "rest_hi": 6500.0, "rest_center": 6250.0},
    ]


PLAYERS = two_line_catalog()
TOKEN_SETS = [set(t.tolist()) for t in player_token_map(0.0, PLAYERS)]


class MaskValueEncoder:
    """Encodes the coalition value into token position [b, 0, 0]."""

    def encode_tokens(self, batch, combo, spectrum_token_mask=None, mask_mode="drop"):
        assert mask_mode in ("drop", "replace")
        n = batch[0].shape[0]
        mask = (
            spectrum_token_mask.cpu().numpy()
            if spectrum_token_mask is not None
            else np.zeros((n, N_SPEC_TOKENS), dtype=bool)
        )
        values = torch.zeros(n)
        for b in range(n):
            dropped = set(np.flatnonzero(mask[b]).tolist())
            kept = [len(ts) > 0 and not (ts & dropped) for ts in TOKEN_SETS]
            v = A * kept[0] + B_ * kept[1] + D * kept[2]
            if kept[0] and kept[1]:
                v += C
            values[b] = v
        tokens = torch.zeros(n, 1, 8)
        tokens[:, 0, 0] = values
        return tokens, torch.zeros(n, 1, dtype=torch.long)

    def eval(self):
        return self


class PassThroughContext(torch.nn.Module):
    def forward(self, tokens, group_ids):
        return tokens[:, 0, :]


class ValueFlow(torch.nn.Module):
    def log_prob(self, target_std, context):
        return context[:, 0]


def test_player_token_maps_disjoint_and_nonempty() -> None:
    assert all(TOKEN_SETS), "every stub player must own tokens at z=0"
    assert not (TOKEN_SETS[0] & TOKEN_SETS[1]) and not (TOKEN_SETS[0] & TOKEN_SETS[2])
    usable = set(np.flatnonzero(usable_tokens()).tolist())
    for ts in TOKEN_SETS:
        assert ts <= usable


def test_run_sweeps_recovers_exact_shapley_and_interaction() -> None:
    n = 24
    batch = (
        torch.randn(n, 32), torch.ones(n, 32), torch.linspace(3600, 9800, 32).repeat(n, 1),
        torch.zeros(n),  # z = 0
        torch.randn(n, 3), torch.randn(n, 4, 8, 8),
        torch.randn(n), torch.arange(n, dtype=torch.int64),
        torch.zeros(n), torch.zeros(n),
    )
    standardizer = TargetStandardizer.fit(np.random.default_rng(0).normal(size=100))
    acc, pair_acc = run_sweeps(
        encoder=MaskValueEncoder(), context_encoder=PassThroughContext(), flow=ValueFlow(),
        loader=[batch], standardizer=standardizer, players=PLAYERS,
        device=torch.device("cpu"), n_full_sweeps=3, n_line_sweeps=3, n_pair_sweeps=4, seed=1,
    )
    phi, se, count = acc.table()
    assert count.min() > 0
    np.testing.assert_allclose(phi[0], A + C / 2, atol=0.06)
    np.testing.assert_allclose(phi[1], B_ + C / 2, atol=0.06)
    np.testing.assert_allclose(phi[2], D, atol=0.06)
    imean, ise, icount = pair_acc.table()
    assert icount[0, 1] > 0
    np.testing.assert_allclose(imean[0, 1], C, atol=1e-6)  # exact for 2 line players


def test_pair_accumulator_symmetry() -> None:
    acc = PairInteractionAccumulator(3)
    acc.add(0, 2, 0.5)
    mean, se, count = acc.table()
    assert mean[0, 2] == mean[2, 0] == 0.5
    assert count[1, 0] == 0
