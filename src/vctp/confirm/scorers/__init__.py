"""
Scorers Module
Provides various scoring functions for answer/thought quality
"""

from typing import Optional

# Import individual scorers
try:
    from . import clip_scorer
    from . import rule_based
    from . import blip2_scorer
except ImportError:
    pass

# Import ensemble scorer
try:
    from .ensemble_scorer import (
        EnsembleScorer,
        AdaptiveEnsembleScorer,
        create_clip_blip2_ensemble,
    )
except ImportError:
    EnsembleScorer = None
    AdaptiveEnsembleScorer = None
    create_clip_blip2_ensemble = None


__all__ = [
    "clip_scorer",
    "rule_based",
    "blip2_scorer",
    "EnsembleScorer",
    "AdaptiveEnsembleScorer",
    "create_clip_blip2_ensemble",
]
