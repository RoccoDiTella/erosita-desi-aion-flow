"""Unit tests for the read-only CLS step (V3b) against a stub 4M block."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_to_aion_embeddings import CLSReadAdapter, cls_read_step  # noqa: E402

DIM, HEADS, B, N = 16, 4, 3, 7


class _StubAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_heads = HEADS
        self.scale = (DIM // HEADS) ** -0.5
        self.qkv = nn.Linear(DIM, DIM * 3, bias=True)
        self.proj = nn.Linear(DIM, DIM)


class _StubBlock(nn.Module):
    """Mirrors the 4M Block attributes cls_read_step touches."""

    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(DIM)
        self.norm2 = nn.LayerNorm(DIM)
        self.attn = _StubAttention()
        self.mlp = nn.Sequential(nn.Linear(DIM, DIM * 2), nn.GELU(), nn.Linear(DIM * 2, DIM))


def _setup(seed: int = 0):
    torch.manual_seed(seed)
    block = _StubBlock()
    for p in block.parameters():
        p.requires_grad = False
    adapter = CLSReadAdapter(DIM)
    x = torch.randn(B, N, DIM)
    c = torch.randn(B, 1, DIM, requires_grad=True)
    return block, adapter, x, c


def test_shapes_and_data_stream_untouched() -> None:
    block, adapter, x, c = _setup()
    x_before = x.clone()
    out = cls_read_step(block, adapter, x, c, None)
    assert out.shape == (B, 1, DIM)
    assert torch.isfinite(out).all()
    assert torch.equal(x, x_before)  # read-only: the data stream is never written


def test_gradients_reach_only_cls_and_adapters() -> None:
    block, adapter, x, c = _setup()
    out = cls_read_step(block, adapter, x, c, None)
    out.sum().backward()
    assert c.grad is not None and c.grad.abs().sum() > 0
    # Zero-init full-rank deltas receive gradients immediately; the frozen
    # block receives none by construction.
    assert adapter.q_delta.weight.grad is not None and adapter.q_delta.weight.grad.abs().sum() > 0
    assert adapter.v_delta.weight.grad is not None and adapter.v_delta.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in block.parameters())


def test_adapters_are_noop_at_init() -> None:
    block, _, x, c = _setup(seed=1)
    out_a = cls_read_step(block, CLSReadAdapter(DIM), x, c, None)
    out_b = cls_read_step(block, CLSReadAdapter(DIM), x, c, None)
    # Zero-init full-rank deltas: both adapters are exact no-ops at init.
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_masked_tokens_are_invisible_to_the_cls() -> None:
    block, adapter, x, c = _setup(seed=2)
    key_invalid = torch.zeros(B, 1, 1, N, dtype=torch.bool)
    key_invalid[..., -1] = True
    out = cls_read_step(block, adapter, x, c.detach(), key_invalid)
    x_perturbed = x.clone()
    x_perturbed[:, -1] += 100.0
    out_perturbed = cls_read_step(block, adapter, x_perturbed, c.detach(), key_invalid)
    assert torch.allclose(out, out_perturbed, atol=1e-5)


def test_multi_cls_streams_match_independent_singles() -> None:
    """K stacked CLS streams must equal K independent single-CLS reads."""
    block, adapter, x, _ = _setup(seed=3)
    c_multi = torch.randn(B, 3, DIM)
    out_multi = cls_read_step(block, adapter, x, c_multi, None)
    for j in range(3):
        out_single = cls_read_step(block, adapter, x, c_multi[:, j : j + 1], None)
        assert torch.allclose(out_multi[:, j : j + 1], out_single, atol=1e-5)
