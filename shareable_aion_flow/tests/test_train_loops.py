"""Smoke tests for batch_nll / validation_metrics with stub encoder + flow.

These exist because the training/validation plumbing can break in ways the
data-layer tests never see (a NameError inside validation_metrics killed two
GPU runs at epoch 1). The stubs stand in for AION and the NSF flow so every
error mode's full call path runs on CPU in milliseconds.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from shareable_aion_flow.main import batch_nll, validation_metrics
from shareable_aion_flow.normalizing_flow import TargetStandardizer


class StubEncoder:
    def encode_tokens(self, batch, combo, spectrum_token_mask=None, mask_mode="drop", modality_dropout=None):
        flux = batch[0]
        tokens = torch.zeros(flux.shape[0], 4, 8)
        group_ids = torch.zeros(flux.shape[0], 4, dtype=torch.long)
        return tokens, group_ids

    def eval(self):
        return self


class StubContext(torch.nn.Module):
    def forward(self, tokens, group_ids):
        return tokens.mean(dim=1)


class StubFlow(torch.nn.Module):
    """Standard-normal 'flow' with the ConditionalNSFFlow call surface."""

    def log_prob(self, target_std, context):
        return -0.5 * target_std**2 - 0.5 * float(np.log(2 * np.pi))

    def log_prob_draws(self, y_draws, context):
        return -0.5 * y_draws**2 - 0.5 * float(np.log(2 * np.pi))

    def log_prob_convolved(self, target_std, sig_lo, sig_hi, context):
        var = 1.0 + 0.5 * (sig_lo**2 + sig_hi**2)
        return -0.5 * target_std**2 / var - 0.5 * torch.log(2 * torch.pi * var)

    def sample(self, context, num_samples=8):
        return torch.zeros(num_samples, context.shape[0])


def make_batch(n=16, with_err=True):
    sig = torch.full((n,), 0.3 if with_err else 0.0)
    return (
        torch.randn(n, 32), torch.ones(n, 32), torch.linspace(3600, 9800, 32).repeat(n, 1),
        torch.rand(n), torch.randn(n, 3), torch.randn(n, 4, 8, 8),
        torch.randn(n), torch.arange(n, dtype=torch.int64), sig, sig,
    )


class Loader(list):
    pass


def parts():
    return StubEncoder(), StubContext(), StubFlow(), TargetStandardizer.fit(np.random.default_rng(0).normal(size=100))


def test_batch_nll_all_error_modes():
    encoder, ctx, flow, std = parts()
    for mode, k in (("none", 1), ("inject", 1), ("inject", 8), ("convolve", 1)):
        loss = batch_nll(
            encoder=encoder, context_encoder=ctx, flow=flow, batch=make_batch(),
            combo=("z", "spectra", "wise", "image"), standardizer=std,
            error_mode=mode, inject_samples=k,
        )
        assert torch.isfinite(loss), (mode, k)


class StubPrior:
    def log_prob_tensor(self, target_std):
        return torch.full_like(target_std, -1.2)


def test_validation_info_gain_sign():
    """IG must be (-nll) - prior_mean: a model matching the prior has IG ~ 0,
    not ~ -2*prior (the sign bug that shipped to two runs' live panels)."""
    encoder, ctx, flow, std = parts()
    loader = Loader([make_batch()])
    metrics = validation_metrics(
        encoder=encoder, context_encoder=ctx, flow=flow, val_loader=loader,
        standardizer=std, prior=StubPrior(), device=torch.device("cpu"), error_mode="none",
    )
    expected = -metrics["val/all_inputs_nll"] - (-1.2)
    assert abs(metrics["val/all_inputs_info_gain"] - expected) < 1e-5  # float32 prior accumulation
    assert metrics["val/all_inputs_info_gain"] > -1.0  # sane scale, not ~ -2.4


def test_validation_metrics_full_path_every_mode():
    """The exact call path that died on the GPU: every mode, with inject_samples."""
    encoder, ctx, flow, std = parts()
    loader = Loader([make_batch(), make_batch()])
    for mode in ("none", "inject", "convolve"):
        metrics = validation_metrics(
            encoder=encoder, context_encoder=ctx, flow=flow, val_loader=loader,
            standardizer=std, prior=None, device=torch.device("cpu"),
            error_mode=mode, inject_samples=4,
        )
        assert np.isfinite(metrics["val/all_inputs_nll"]), mode
        for modality in ("z", "spectra", "wise", "image"):
            assert f"val/{modality}_nll" in metrics
        assert "val/all_inputs_r2" in metrics


# ------------------------------------------------------------ presence in batch_nll
# batch_nll used to encode whatever the combo named. A source staged without a
# Legacy Survey cutout carries an all-zero image, so an image combo tokenized
# zeros as sky. Now every source is conditioned on `combo INTERSECT what it has`.


class SpyEncoder(StubEncoder):
    """Records what it was actually asked to encode."""

    def __init__(self):
        self.calls = []

    def encode_tokens(self, batch, combo, spectrum_token_mask=None, mask_mode="drop",
                      modality_dropout=None):
        self.calls.append({"n": int(batch[0].shape[0]), "combo": tuple(combo),
                           "dropout": modality_dropout})
        return super().encode_tokens(batch, combo, spectrum_token_mask, mask_mode,
                                     modality_dropout)


def batch_with_images(n, blank_rows):
    b = list(make_batch(n=n))
    b[5] = torch.rand(n, 4, 8, 8) + 0.1
    for row in blank_rows:
        b[5][row] = 0.0
    return tuple(b)


def test_batch_nll_drops_sources_that_have_none_of_the_combo():
    _e, ctx, flow, std = parts()
    encoder = SpyEncoder()
    batch = batch_with_images(8, blank_rows=[1, 3, 5, 6, 7])   # 3 real cutouts
    loss = batch_nll(encoder=encoder, context_encoder=ctx, flow=flow, batch=batch,
                     combo=("image",), standardizer=std)
    assert torch.isfinite(loss)
    assert encoder.calls[0]["n"] == 3, "only sources with a real cutout may be encoded"


def test_batch_nll_refuses_a_batch_with_nothing_to_condition_on():
    from shareable_aion_flow.main import NoConditionableRows

    _e, ctx, flow, std = parts()
    encoder = SpyEncoder()
    batch = batch_with_images(4, blank_rows=range(4))          # no cutouts at all
    with pytest.raises(NoConditionableRows):
        batch_nll(encoder=encoder, context_encoder=ctx, flow=flow, batch=batch,
                  combo=("image",), standardizer=std)
    assert not encoder.calls, "nothing may reach the encoder"


def test_batch_nll_masks_the_absent_modality_but_keeps_the_source():
    """A source with spectra but no cutout still trains -- on its spectra."""
    _e, ctx, flow, std = parts()
    encoder = SpyEncoder()
    batch = batch_with_images(4, blank_rows=[2])
    batch_nll(encoder=encoder, context_encoder=ctx, flow=flow, batch=batch,
              combo=("spectra", "image"), standardizer=std)
    call = encoder.calls[0]
    assert call["n"] == 4, "no source is dropped when spectra remain"
    assert call["dropout"]["image"].tolist() == [False, False, True, False]
    assert not call["dropout"]["spectra"].any()


def test_batch_nll_is_unchanged_when_every_modality_is_present():
    _e, ctx, flow, std = parts()
    encoder = SpyEncoder()
    batch = batch_with_images(4, blank_rows=[])
    b = list(batch)
    b[4] = torch.rand(4, 3) + 0.5                              # positive WISE flux
    batch_nll(encoder=encoder, context_encoder=ctx, flow=flow, batch=tuple(b),
              combo=("spectra", "z", "wise", "image"), standardizer=std)
    call = encoder.calls[0]
    assert call["n"] == 4 and call["dropout"] is None
