"""Tests for the split-normal error kernel and the convolution likelihood.

These validate the quadrature analytically without needing zuko: we feed a
standard-normal log-density, so the convolution has a closed form.
"""
from __future__ import annotations

import math

import torch

from shareable_aion_flow.normalizing_flow import convolve_logprob, split_normal_log_kernel


def _std_normal_logpdf(t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    # p(t) = N(t; 0, 1); ignores context (context only carries batch size here)
    return -0.5 * t**2 - 0.5 * math.log(2 * math.pi)


def test_split_normal_symmetric_equals_gaussian():
    r = torch.linspace(-3, 3, 200)
    s = torch.full_like(r, 0.7)
    got = split_normal_log_kernel(r, s, s)
    gauss = -0.5 * (r / 0.7) ** 2 - math.log(0.7 * math.sqrt(2 * math.pi))
    assert torch.allclose(got, gauss, atol=1e-5)


def test_split_normal_integrates_to_one():
    # both symmetric and asymmetric kernels are normalised
    grid = torch.linspace(-12, 12, 200001, dtype=torch.float64)
    dx = grid[1] - grid[0]
    for slo, shi in [(0.5, 0.5), (0.3, 1.2), (1.0, 0.2)]:
        dens = split_normal_log_kernel(
            grid, torch.tensor(slo, dtype=torch.float64), torch.tensor(shi, dtype=torch.float64)
        ).exp()
        assert abs(float(dens.sum() * dx) - 1.0) < 1e-3


def test_convolution_recovers_analytic_gaussian():
    # ∫ N(t;0,1) N(y;t,sig^2) dt = N(y; 0, 1+sig^2)
    ys = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    for sig in [0.1, 0.5, 1.0]:
        s = torch.full_like(ys, sig)
        ctx = torch.zeros(ys.shape[0], 4)  # batch-size carrier only
        got = convolve_logprob(_std_normal_logpdf, ys, s, s, ctx, n_nodes=61, span=6.0)
        var = 1.0 + sig**2
        expected = -0.5 * ys**2 / var - 0.5 * math.log(2 * math.pi * var)
        assert torch.allclose(got, expected, atol=2e-2), (sig, got, expected)


def test_tiny_error_recovers_plain_logprob():
    ys = torch.tensor([-1.5, 0.0, 1.3])
    s = torch.full_like(ys, 1e-4)
    ctx = torch.zeros(ys.shape[0], 4)
    got = convolve_logprob(_std_normal_logpdf, ys, s, s, ctx, n_nodes=41, span=5.0)
    plain = _std_normal_logpdf(ys, ctx)
    assert torch.allclose(got, plain, atol=1e-2), (got, plain)
