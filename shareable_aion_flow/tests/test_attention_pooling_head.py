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
    MODALITIES,
    AIONAttentionContext,
    ComboDrawStats,
    ComboSampler,
    ModalityAffine,
    PaperQ4L2AttentionPooler,
    available_matrix,
    batch_presence_ids,
    combo_presence_id,
    combo_supported,
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


def test_paper_pooler_rejects_invalid_ids_but_allows_padding() -> None:
    pooler = PaperQ4L2AttentionPooler(embed_dim=16, num_heads=4, dropout=0.0)
    tokens = torch.randn(1, 3, 16)
    with pytest.raises(ValueError, match="direct modality ids"):
        pooler(tokens, torch.tensor([[0, -2, 3]], dtype=torch.long))
    with pytest.raises(ValueError, match="direct modality ids"):
        pooler(tokens, torch.tensor([[0, 4, 3]], dtype=torch.long))
    # -1 is dropped-token padding and must be accepted
    out = pooler(tokens, torch.tensor([[0, -1, 3]], dtype=torch.long))
    assert out.shape == (1, 4 * 16)
    with pytest.raises(ValueError, match="non-padding token"):
        pooler(tokens, torch.tensor([[-1, -1, -1]], dtype=torch.long))


def test_padding_tokens_do_not_change_pooler_output() -> None:
    torch.manual_seed(0)
    pooler = PaperQ4L2AttentionPooler(embed_dim=16, num_heads=4, dropout=0.0).eval()
    tokens = torch.randn(2, 5, 16)
    ids = torch.tensor([[0, 0, 1, 2, 3], [3, 2, 1, 0, 0]])
    base = pooler(tokens, ids)
    garbage = 100.0 * torch.randn(2, 2, 16)
    padded_tokens = torch.cat([tokens, garbage], dim=1)
    padded_ids = torch.cat([ids, torch.full((2, 2), -1, dtype=torch.long)], dim=1)
    with_pad = pooler(padded_tokens, padded_ids)
    torch.testing.assert_close(base, with_pad, atol=1e-5, rtol=1e-4)


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


# --------------------------------------------------------- presence-aware sampling
# The defect these guard: ComboSampler.default() drew uniformly over all 15
# combos with no knowledge of what a source has, and staging writes a ZERO image
# for a source with no Legacy Survey cutout. On the merged sample only 35% of
# rows have a cutout, so two image-combo draws in three would have been zeros
# tokenized as sky.


def _avail(**flags) -> torch.Tensor:
    """[1, 4] availability row; unnamed modalities default to present."""
    return torch.tensor([[bool(flags.get(name, True)) for name in MODALITIES]])


def test_source_without_image_is_never_given_an_image_combo() -> None:
    sampler = ComboSampler.default()
    generator = torch.Generator(); generator.manual_seed(0)
    stats = ComboDrawStats()
    seen = set()
    for _ in range(4000):
        combo = sampler.sample_available(("spectra", "z", "wise"), generator, stats)
        assert "image" not in combo, combo
        seen.add(combo)
    # and it still gets the whole legal mix, not a degenerate corner
    assert seen == {c for c in
                    [("spectra",), ("z",), ("wise",), ("spectra", "z"), ("spectra", "wise"),
                     ("z", "wise"), ("spectra", "z", "wise")]}
    assert stats.draws == 4000
    # a quarter of draws ask for size 4 and get clamped to the available three
    assert 0.2 < stats.clamped / stats.draws < 0.3


def test_per_source_draws_never_name_an_absent_modality() -> None:
    sampler = ComboSampler.default()
    generator = torch.Generator(); generator.manual_seed(1)
    # row 0 has everything, row 1 has no image, row 2 has spectra only,
    # row 3 has nothing at all
    available = torch.tensor([
        [True, True, True, True],
        [True, True, True, False],
        [True, False, False, False],
        [False, False, False, False],
    ])
    stats = ComboDrawStats()
    for _ in range(500):
        combos, usable = sampler.sample_per_source_available(available, generator, stats)
        assert usable.tolist() == [True, True, True, False]
        for row, combo in enumerate(combos[:3]):
            for name in combo:
                assert bool(available[row, MODALITIES.index(name)]), (row, combo)
        assert combos[2] == ("spectra",)
        assert combos[3] == ()
    assert stats.dropped_missing == 500      # the empty row, every time


def test_size_stratified_marginal_is_untouched_when_everything_is_present() -> None:
    """Presence-awareness must be a no-op on a complete source -- draw for draw.

    Not just "the same distribution": the same generator consumption, so a run
    on the current dr2v2 sample (where has_image is exactly has_spectrum) is
    bit-identical to the presence-blind sampler it replaces.
    """
    sampler = ComboSampler.default()
    g_blind = torch.Generator(); g_blind.manual_seed(7)
    g_aware = torch.Generator(); g_aware.manual_seed(7)
    stats = ComboDrawStats()
    blind = [sampler.sample(g_blind) for _ in range(2000)]
    aware = [sampler.sample_available(MODALITIES, g_aware, stats) for _ in range(2000)]
    assert aware == blind
    assert stats.clamped == 0
    # and the marginal really is uniform-by-size: 4 sizes, 1/4 each
    by_size = {s: sum(1 for c in aware if len(c) == s) for s in (1, 2, 3, 4)}
    for size, count in by_size.items():
        assert abs(count / 2000 - 0.25) < 0.03, (size, count)


def test_fallback_is_a_clamp_and_both_candidate_rules_agree() -> None:
    """No-legal-combo-of-that-size falls back to the source's full available set.

    Legality is "subset of available", so the legal sizes are 1..len(available)
    with no gaps: a size is illegal only if it EXCEEDS what the source has, the
    nearest smaller legal size is len(available), and the only legal combo of
    that size is the full available set. The two candidate fallback rules name
    the same combo; this pins that.
    """
    sampler = ComboSampler.default()
    generator = torch.Generator(); generator.manual_seed(3)
    for available in (("z",), ("spectra", "wise"), ("spectra", "z", "image")):
        full = tuple(m for m in MODALITIES if m in available)
        draws = [sampler.sample_available(available, generator) for _ in range(3000)]
        assert all(set(c).issubset(set(full)) for c in draws)
        # every draw of a size > len(available) landed on the full set, so the
        # full set carries exactly the mass of the sizes at or above it
        expected = (len(MODALITIES) - len(full) + 1) / len(MODALITIES)
        assert abs(draws.count(full) / 3000 - expected) < 0.03, (available, expected)


def test_a_source_with_nothing_cannot_be_conditioned() -> None:
    sampler = ComboSampler.default()
    with pytest.raises(ValueError, match="no modality is available"):
        sampler.sample_available(())


def test_combo_supported_all_versus_any() -> None:
    available = torch.tensor([
        [True, True, True, True],
        [True, True, True, False],
        [False, False, False, True],
    ])
    strict = combo_supported(("spectra", "image"), available, require="all")
    loose = combo_supported(("spectra", "image"), available, require="any")
    assert strict.tolist() == [True, False, False]
    assert loose.tolist() == [True, True, True]
    with pytest.raises(ValueError, match="unknown require"):
        combo_supported(("spectra",), available, require="mostly")


def test_available_matrix_is_in_modalities_order() -> None:
    present = {"spectra": torch.tensor([True, False]),
               "z": torch.tensor([False, True]),
               "wise": torch.tensor([True, True]),
               "image": torch.tensor([False, False])}
    matrix = available_matrix(present)
    assert matrix.shape == (2, 4)
    for col, name in enumerate(MODALITIES):
        assert matrix[:, col].tolist() == present[name].tolist()


def test_cls_context_projects_cls_hidden() -> None:
    from attention_pooling_head import CLSContext

    head = CLSContext(embed_dim=768, context_dim=256)
    out = head(torch.randn(3, 1, 768), torch.zeros(3, 1, dtype=torch.long))
    assert out.shape == (3, 256)
    with pytest.raises(ValueError, match="CLS hidden"):
        head(torch.randn(3, 5, 768), torch.zeros(3, 5, dtype=torch.long))
