"""Attention-pooling head for AION tokens (the clean q4/l2 paper model).

Pools AION's per-modality token sequence into a fixed-size flow context using a
learned modality-affine calibration, four global queries conditioned on a
modality-presence embedding, and two cross/self-attention blocks. This is the
simplified reference implementation of the architecture used in the PAI 2026
paper; see the repository README for how it relates to the full research code.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
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


def available_matrix(present: dict[str, torch.Tensor]) -> torch.Tensor:
    """``[B, 4]`` bool availability matrix in ``MODALITIES`` order.

    ``present`` is what ``data_to_aion_embeddings.modality_presence`` returns:
    one boolean per source per modality. Column order is fixed by ``MODALITIES``
    so ``MODALITY_TO_ID`` indexes it.
    """
    return torch.stack([present[name].to(torch.bool) for name in MODALITIES], dim=1)


def combo_supported(
    combo: Iterable[str], available: torch.Tensor, *, require: str = "all"
) -> torch.Tensor:
    """Which rows of ``available`` can be conditioned on ``combo``.

    ``require="all"`` is the strict reading — the source has EVERY modality the
    combo names — and is what a pinned ``--fixed-combo`` needs, because a run
    that pins spectra+z is asking a question about spectra AND z.
    ``require="any"`` is the maskable reading: the source has at least one of
    them, so the ones it lacks can be dropped from the encoder input and the
    source still contributes something real. That is what a batch-level sampled
    combo needs, where the combo is a property of the step rather than a claim
    about the row.
    """
    cols = [MODALITY_TO_ID[name] for name in combo]
    if not cols:
        raise ValueError("combo is empty; there is nothing to condition on.")
    sub = available[:, cols]
    if require == "all":
        return sub.all(dim=1)
    if require == "any":
        return sub.any(dim=1)
    raise ValueError(f"unknown require {require!r}; use 'all' or 'any'.")


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
    if bool(group_ids.lt(-1).any() or group_ids.ge(len(MODALITIES)).any()):
        raise ValueError(
            "group_ids must contain only direct modality ids 0=spectra, 1=z, 2=wise, 3=image, "
            "or -1 for dropped-token padding."
        )
    if bool(group_ids.eq(-1).all(dim=1).any()):
        raise ValueError("Every sample needs at least one non-padding token.")


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
        # Padding tokens (-1) borrow modality 0's affine; their values are
        # irrelevant because attention masks them out downstream.
        ids = group_ids.clamp_min(0)
        scale = torch.exp(self.log_scale[ids].clamp(-2.0, 2.0)).unsqueeze(-1)
        calibrated = scale.to(tokens.dtype) * tokens + self.bias[ids].to(tokens.dtype)
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

    def forward(
        self,
        queries: torch.Tensor,
        tokens: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cross_input = self.query_cross_norm(queries)
        cross_output, _ = self.cross_attention(
            cross_input, tokens, tokens, need_weights=False, key_padding_mask=key_padding_mask
        )
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
        pad = group_ids.eq(-1)
        key_padding_mask = pad if bool(pad.any()) else None
        tokens = self.modality_affine(tokens, group_ids)
        queries = self.query.expand(tokens.shape[0], -1, -1)
        queries = queries + self.presence_embedding(batch_presence_ids(group_ids)).unsqueeze(1)
        for block in self.blocks:
            queries = block(queries, tokens, key_padding_mask=key_padding_mask)
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


class CLSContext(nn.Module):
    """V3 head: project the AION CLS hidden state to the flow context.

    Keeps the ``(tokens, group_ids)`` interface of ``AIONAttentionContext`` so
    the train/val/eval plumbing is untouched: ``tokens`` is the [B, 1, 768] CLS
    hidden returned by the cls-mode encoder; ``group_ids`` is ignored. The
    output feeds the SAME ConditionalNSFFlow as the attention heads, so V3 is
    an equivalent-flow comparison isolating pooling + encoder fine-tuning.
    """

    def __init__(self, embed_dim: int = 768, context_dim: int = 256) -> None:
        super().__init__()
        self.norm_in = nn.LayerNorm(embed_dim)
        self.project = nn.Linear(embed_dim, context_dim)
        self.norm_out = nn.LayerNorm(context_dim)
        self.context_dim = context_dim

    def forward(self, tokens: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 3 or tokens.shape[1] != 1:
            raise ValueError(f"CLSContext expects [B, 1, D] CLS hidden, got {tuple(tokens.shape)}.")
        return self.norm_out(self.project(self.norm_in(tokens[:, 0])))


@dataclass
class ComboDrawStats:
    """Counters for how often availability had to change a combo draw.

    A silent fix is only half a fix: a run has to be able to say what fraction
    of its conditioning was not what the sampler nominally asked for.

      draws            presence-aware draws requested
      clamped          draws whose SIZE was reduced because the source does not
                       have that many modalities
      masked           source-steps where a modality named by a batch-level
                       combo was removed for that source (the batch-level path
                       cannot restratify per source, so it masks instead)
      dropped_missing  sources excluded from a step because they have NO
                       modality the step could use
      dropped_pinned   sources excluded because they cannot honour --fixed-combo
    """

    draws: int = 0
    clamped: int = 0
    masked: int = 0
    dropped_missing: int = 0
    dropped_pinned: int = 0
    rows: int = 0

    def reset(self) -> None:
        for name in ("draws", "clamped", "masked", "dropped_missing",
                     "dropped_pinned", "rows"):
            setattr(self, name, 0)

    @property
    def adjusted(self) -> int:
        return self.clamped + self.masked

    @property
    def adjusted_fraction(self) -> float:
        return self.adjusted / self.rows if self.rows else 0.0

    def summary(self) -> str:
        pct = 100.0 * self.adjusted_fraction
        return (f"{self.rows:,} source-steps; combo adjusted for {self.adjusted:,} "
                f"({pct:.1f}%) [size-clamped {self.clamped:,}, modality-masked "
                f"{self.masked:,}]; dropped {self.dropped_pinned + self.dropped_missing:,} "
                f"(cannot honour the pin {self.dropped_pinned:,}, "
                f"no usable modality {self.dropped_missing:,})")


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

    def sample_per_source(
        self, n: int, generator: torch.Generator | None = None
    ) -> list[tuple[str, ...]]:
        """One combo per source, same size-stratified marginal as ``sample``.

        PRESENCE-BLIND. Only correct where every source has every modality; use
        ``sample_per_source_available`` anywhere real data is involved.
        """
        return [self.sample(generator) for _ in range(n)]

    def sample(self, generator: torch.Generator | None = None) -> tuple[str, ...]:
        sizes = tuple(sorted(self.combos_by_size))
        size_index = int(torch.randint(len(sizes), (1,), generator=generator).item())
        combos = self.combos_by_size[sizes[size_index]]
        combo_index = int(torch.randint(len(combos), (1,), generator=generator).item())
        return combos[combo_index]

    def sample_available(
        self,
        available: Iterable[str],
        generator: torch.Generator | None = None,
        stats: ComboDrawStats | None = None,
    ) -> tuple[str, ...]:
        """Draw a combo the source can actually honour.

        The size is drawn EXACTLY as in ``sample`` -- uniformly over the sizes
        this sampler knows -- and then CLAMPED to how many modalities the source
        has. Within the resulting size the choice is uniform over the combos that
        are subsets of ``available``. Two consequences worth stating:

        * when all four modalities are present this is ``sample`` exactly, draw
          for draw: it consumes the same two randints in the same order and
          returns the same combo, so the size-stratified marginal is not merely
          approximated, it is untouched;
        * for a source with ``k`` modalities, sizes 1..k-1 keep their uniform
          within-size marginal and all draws of size >= k collapse onto the one
          legal combo of size k.

        THE FALLBACK, and why there is nothing to choose. Legality here is
        "subset of what the source has", so the legal sizes are exactly
        1..len(available) with no gaps. A drawn size is illegal only when it
        EXCEEDS len(available), the nearest smaller legal size is then
        len(available) itself, and the only legal combo of that size is the
        source's full available set. "Fall back to a smaller size" and "fall
        back to the source's full available set" therefore name the same combo.
        Clamping implements both. It is counted as ``clamped`` so a run can
        report how much of its mix was decided by availability rather than by
        the sampler.

        Raises ``ValueError`` when ``available`` is empty: a source with no
        input at all cannot be conditioned on anything, and inventing a combo
        for it is precisely the bug this function exists to remove. Callers
        drop such rows (see ``sample_per_source_available``).
        """
        wanted = {name for name in available}
        avail = tuple(name for name in MODALITIES if name in wanted)
        if not avail:
            raise ValueError(
                "no modality is available for this source, so no combo is legal; "
                "the caller must exclude the row rather than condition it on "
                "inputs that do not exist.")
        sizes = tuple(sorted(self.combos_by_size))
        drawn_size = sizes[int(torch.randint(len(sizes), (1,), generator=generator).item())]
        aset = set(avail)
        legal: tuple[tuple[str, ...], ...] = ()
        size = min(drawn_size, len(avail))
        while size >= 1:
            legal = tuple(c for c in self.combos_by_size.get(size, ())
                          if aset.issuperset(c))
            if legal:
                break
            size -= 1
        if not legal:
            raise ValueError(
                f"no combo in this sampler is a subset of {avail}; the sampler's "
                f"combo set and the modality set have diverged.")
        combo = legal[int(torch.randint(len(legal), (1,), generator=generator).item())]
        if stats is not None:
            stats.draws += 1
            stats.clamped += int(size != drawn_size)
        return combo

    def sample_per_source_available(
        self,
        available: torch.Tensor,
        generator: torch.Generator | None = None,
        stats: ComboDrawStats | None = None,
    ) -> tuple[list[tuple[str, ...]], torch.Tensor]:
        """Per-source presence-aware combos for a ``[B, 4]`` availability matrix.

        Returns ``(combos, usable)``. A row with no modality at all gets an
        EMPTY combo and ``usable=False`` so the caller can drop it; it is never
        handed a combo naming an input it does not have.
        """
        if available.dim() != 2 or available.shape[1] != len(MODALITIES):
            raise ValueError(
                f"available must be [B, {len(MODALITIES)}] in MODALITIES order, "
                f"got {tuple(available.shape)}.")
        flags = available.to(torch.bool).cpu().tolist()
        combos: list[tuple[str, ...]] = []
        usable: list[bool] = []
        for row in flags:
            names = [name for name, ok in zip(MODALITIES, row) if ok]
            if not names:
                combos.append(())
                usable.append(False)
                if stats is not None:
                    stats.dropped_missing += 1
                continue
            combos.append(self.sample_available(names, generator, stats))
            usable.append(True)
        return combos, torch.tensor(usable, dtype=torch.bool, device=available.device)
