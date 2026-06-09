"""Clean AION attention-flow package for the eROSITA/DESI flux model."""

from .attention_pooling_head import (
    MODALITIES,
    AIONAttentionContext,
    all_nonempty_modality_combos,
)
from .normalizing_flow import ConditionalNSFFlow, KDEPrior, TargetStandardizer

__version__ = "0.1.0"

__all__ = [
    "AIONAttentionContext",
    "ConditionalNSFFlow",
    "KDEPrior",
    "TargetStandardizer",
    "MODALITIES",
    "all_nonempty_modality_combos",
    "__version__",
]
