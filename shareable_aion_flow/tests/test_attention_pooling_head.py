"""Unit tests for the attention-pooling head (shapes, presence ids, validation)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attention_pooling_head import (  # noqa: E402
    AIONAttentionContext,
    ComboSampler,
    ModalityAffine,
    PaperQ4L2AttentionPooler,
    batch_presence_ids,
    combo_presence_id,
)


def test_modality_affine_shape_and_direct_ids() -> None:
    module = ModalityAffine(embed_dim=8)
    tokens = torch.randn(2, 5, 8)
    group_ids = torch.tensor([[0, 0, 1, 2, 3], [3, 3, 2, 1, 0]])
    output = module(tokens, group_ids)
    assert output.shape == tokens.shape
    assert module.bias.shape == (4, 8)
    assert module.log_scale.shape == (4,)


def test_presence_ids_are_four_bit_modality_masks() -> None:
    group_ids = torch.tensor([[0, 0, 2], [1, 3, 3], [0, 1, 2]])
    assert batch_presence_ids(group_ids).tolist() == [
        combo_presence_id(("spectra", "wise")),
        combo_presence_id(("z", "image")),
        combo_presence_id(("spectra", "z", "wise")),
    ]


def test_paper_pooler_returns_flattened_four_query_features() -> None:
    pooler = PaperQ4L2AttentionPooler(embed_dim=16, num_heads=4, dropout=0.0)
    tokens = torch.randn(3, 7, 16)
    group_ids = torch.tensor(
        [
            [0, 0, 0, 1, 2, 2, 3],
            [3, 3, 3, 3, 2, 1, 0],
            [0, 1, 1, 2, 2, 2, 3],
        ]
    )
    output = pooler(tokens, group_ids)
    assert output.shape == (3, 4 * 16)


def test_paper_pooler_rejects_legacy_padding_or_invalid_ids() -> None:
    pooler = PaperQ4L2AttentionPooler(embed_dim=16, num_heads=4, dropout=0.0)
    tokens = torch.randn(1, 3, 16)
    with pytest.raises(ValueError, match="direct modality ids"):
        pooler(tokens, torch.tensor([[0, -1, 3]], dtype=torch.long))
    with pytest.raises(ValueError, match="direct modality ids"):
        pooler(tokens, torch.tensor([[0, 4, 3]], dtype=torch.long))


def test_context_encoder_returns_flow_context() -> None:
    model = AIONAttentionContext(dropout=0.0)
    tokens = torch.randn(2, 9, 768)
    group_ids = torch.tensor([[0, 0, 1, 2, 2, 3, 3, 3, 3], [3, 3, 3, 2, 2, 1, 0, 0, 0]])
    output = model(tokens, group_ids)
    assert output.shape == (2, 256)


def test_combo_sampler_samples_all_combo_sizes() -> None:
    sampler = ComboSampler.default()
    sizes = {len(combo) for combos in sampler.combos_by_size.values() for combo in combos}
    assert sizes == {1, 2, 3, 4}
