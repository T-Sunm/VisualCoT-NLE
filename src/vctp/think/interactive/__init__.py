"""Interactive attention module for Visual CoT."""

from .attention_strategy import (
    AttentionStrategy,
    LLMAttentionStrategy,
    RandomAttentionStrategy,
    OracleAttentionStrategy,
    AllRegionsAttentionStrategy,
)
from .interactive_attention import InteractiveAttention
from .interactive_loop import InteractiveLoop

__all__ = [
    # Strategies
    "AttentionStrategy",
    "LLMAttentionStrategy",
    "RandomAttentionStrategy",
    "OracleAttentionStrategy",
    "AllRegionsAttentionStrategy",
    # Core
    "InteractiveAttention",
    "InteractiveLoop",
]
