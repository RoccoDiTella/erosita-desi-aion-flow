"""Attention-pooling head for AION tokens (the clean q4/l2 paper model).

Pools AION's per-modality token sequence into a fixed-size flow context using a
learned modality-affine calibration, four global queries conditioned on a
modality-presence embedding, and two cross/self-attention blocks. This is the
simplified reference implementation of the architecture used in the PAI 2026
paper; see the repository README for how it relates to the full research code.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch
from torch import nn

MODALITIES: tuple[str, ...] = ("spectra", "z", "wise", "image")
MODALITY_TO_ID = {name: index for index, name in enumerate(MODALITIES)}


def all_nonempty_modality_combos() -> list[tuple[str, ...]]:
    combos: list[tuple[str, ...]] = []
    for size in range(1, len(MODALITIES) + 1):
        combos.extend(tuple(combo) for combo in itertools.combinations(MODALITIES, size))
    return combos


def combo_name(combo: tuple[str, ...] | list[str] | set[str]) -> str:
    ordered = [name for name in MODALITIES if name in set(combo)]
    return "+".join(ordered)


def combo_presence_id(combo: tuple[str, ...] | list[str] | set[str]) -> int:
    presence_id = 0
    for name in combo:
        presence_id |= 1 << MODALITY_TO_ID[name]
    return presence_id


def batch_presence_ids(group_ids: torch.Tensor) -> torch.Tensor:
    """Return one 4-bit modality-presence id per sample.

    ``group_ids`` is expected to contain only direct modality ids:
    0=spectra, 1=redshift, 2=WISE, 3=images. There is no padding modality in
    this clean implementation.
    """

    presence = torch.zeros(group_ids.shape[0], dtype=torch.long, device=group_ids.device)
    for group_idx in range(len(MODALITIES)):
        has_group = group_ids.eq(group_idx).any(dim=1)
        presence = presence + has_group.to(torch.long) * (1 << group_idx)
    return presence


def validate_direct_group_ids(tokens: torch.Tensor, group_ids: torch.Tensor) -> None:
    if tokens.dim() != 3:
        raise ValueError(f"Expected AION tokens with shape [B, T, D], got {tuple(tokens.shape)}.")
    if group_ids.shape != tokens.shape[:2]:
        raise ValueError(
            f"group_ids shape {tuple(group_ids.shape)} does not match token shape {tuple(tokens.shape[:2])}."
        )
    if group_ids.dtype != torch.long:
        raise TypeError(f"group_ids must use torch.long dtype, got {group_ids.dtype}.")
    if bool(group_ids.lt(0).any() or group_ids.ge(len(MODALITIES)).any()):
        raise ValueError("group_ids must contain only direct modality ids 0=spectra, 1=z, 2=wise, 3=image.")


class ModalityAffine(nn.Module):
    """Learned modality-specific affine calibration of AION tokens.

    For a token ``x`` with modality id ``m`` this applies

        ``LayerNorm(exp(log_scale[m]) * x + bias[m])``.

    The scale is intentionally one scalar per modality. That keeps the
    calibration close to the paper model while making the implementation
    explicit and readable.
    """

    def __init__(self, embed_dim: int = 768) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(len(MODALITIES)))
        self.bias = nn.Parameter(torch.empty(len(MODALITIES), embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.normal_(self.bias, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        validate_direct_group_ids(tokens, group_ids)
        scale = torch.exp(self.log_scale[group_ids].clamp(-2.0, 2.0)).unsqueeze(-1)
        calibrated = scale.to(tokens.dtype) * tokens + self.bias[group_ids].to(tokens.dtype)
        return self.norm(calibrated)


class AttentionPoolingBlock(nn.Module):
    """One q4/l2 paper attention-pooling block."""

    def __init__(
        self,
        *,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.05,
        ffn_mult: float = 2.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(round(embed_dim * ffn_mult))
        self.query_cross_norm = nn.LayerNorm(embed_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.query_self_norm = nn.LayerNorm(embed_dim)
        self.query_self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        cross_input = self.query_cross_norm(queries)
        cross_output, _ = self.cross_attention(cross_input, tokens, tokens, need_weights=False)
        queries = queries + self.dropout(cross_output)

        self_input = self.query_self_norm(queries)
        self_output, _ = self.query_self_attention(self_input, self_input, self_input, need_weights=False)
        queries = queries + self.dropout(self_output)

        return queries + self.dropout(self.ffn(self.ffn_norm(queries)))


class PaperQ4L2AttentionPooler(nn.Module):
    """The clean paper attention pooler.

    It uses one learned query tensor with four queries, not separate queries
    per modality combination. Presence conditioning is a learned 16-way offset
    added to those four global queries.
    """

    def __init__(
        self,
        *,
        embed_dim: int = 768,
        num_queries: int = 4,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.05,
        ffn_mult: float = 2.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.feature_dim = embed_dim * num_queries
        self.modality_affine = ModalityAffine(embed_dim=embed_dim)
        self.query = nn.Parameter(torch.empty(1, num_queries, embed_dim))
        self.presence_embedding = nn.Embedding(16, embed_dim)
        self.blocks = nn.ModuleList(
            [
                AttentionPoolingBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    ffn_mult=ffn_mult,
                )
                for _ in range(num_layers)
            ]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.LayerNorm(embed_dim),
        )
        nn.init.normal_(self.query, mean=0.0, std=0.02)
        nn.init.normal_(self.presence_embedding.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        validate_direct_group_ids(tokens, group_ids)
        tokens = self.modality_affine(tokens, group_ids)
        queries = self.query.expand(tokens.shape[0], -1, -1)
        queries = queries + self.presence_embedding(batch_presence_ids(group_ids)).unsqueeze(1)
        for block in self.blocks:
            queries = block(queries, tokens)
        pooled_queries = self.output(queries)
        return pooled_queries.flatten(start_dim=1)


class FlowContextMLP(nn.Module):
    """Adapter MLP from flattened attention queries to flow context."""

    def __init__(
        self,
        *,
        input_dim: int = 3072,
        hidden_dims: tuple[int, ...] = (512, 512),
        output_dim: int = 256,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        dims = (input_dim, *hidden_dims)
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.SiLU(),
                    nn.LayerNorm(out_dim),
                    nn.Dropout(dropout),
                ]
            )
        layers.extend([nn.Linear(dims[-1], output_dim), nn.LayerNorm(output_dim)])
        self.net = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class AIONAttentionContext(nn.Module):
    """AION-token attention pooler plus MLP context adapter.

    The pooler/adapter sizes are configurable so smaller heads can be tried
    without touching the paper defaults. ``num_queries=4, num_layers=2,
    context_hidden=(512, 512), context_dim=256`` reproduces the paper q4/l2
    head; e.g. ``num_queries=1, num_layers=1, context_hidden=(128,)`` is the
    minimal "V1" head. The adapter input width follows ``embed_dim * num_queries``
    automatically, so reducing the query count shrinks the MLP too.
    """

    def __init__(
        self,
        dropout: float = 0.05,
        *,
        num_queries: int = 4,
        num_layers: int = 2,
        num_heads: int = 8,
        context_hidden: tuple[int, ...] = (512, 512),
        context_dim: int = 256,
    ) -> None:
        super().__init__()
        self.pooler = PaperQ4L2AttentionPooler(
            num_queries=num_queries,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.adapter = FlowContextMLP(
            input_dim=self.pooler.feature_dim,
            hidden_dims=tuple(context_hidden),
            output_dim=context_dim,
            dropout=dropout,
        )
        self.context_dim = self.adapter.output_dim

    def forward(self, tokens: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        return self.adapter(self.pooler(tokens, group_ids))


@dataclass(frozen=True)
class ComboSampler:
    """Uniformly sample by combo size: singles, pairs, triples, all-input."""

    combos_by_size: dict[int, tuple[tuple[str, ...], ...]]

    @classmethod
    def default(cls) -> ComboSampler:
        combos_by_size: dict[int, list[tuple[str, ...]]] = {}
        for combo in all_nonempty_modality_combos():
            combos_by_size.setdefault(len(combo), []).append(combo)
        return cls({size: tuple(combos) for size, combos in combos_by_size.items()})

    def sample(self, generator: torch.Generator | None = None) -> tuple[str, ...]:
        sizes = tuple(sorted(self.combos_by_size))
        size_index = int(torch.randint(len(sizes), (1,), generator=generator).item())
        combos = self.combos_by_size[sizes[size_index]]
        combo_index = int(torch.randint(len(combos), (1,), generator=generator).item())
        return combos[combo_index]
