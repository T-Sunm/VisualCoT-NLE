"""
Confirmation Module
Verifies reasoning outputs against image evidence

Main confirmation strategies:
1. VisualConsistencyConfirmer: Verify thoughts against visual evidence
2. AnswerConsistencyConfirmer: Verify answer against choices
"""

from typing import Optional

# Import confirmer classes
try:
    from .confirmer import (
        NoOpConfirmer,
        VisualConsistencyConfirmer,
        AnswerConsistencyConfirmer,
    )
except ImportError:
    NoOpConfirmer = None
    VisualConsistencyConfirmer = None
    AnswerConsistencyConfirmer = None

# Import verifiers (for advanced usage)
try:
    from .verifiers import (
        CLIPThoughtVerifier,
        BLIP2ThoughtVerifier,
        ChoiceAnswerVerifier,
    )
except ImportError:
    CLIPThoughtVerifier = None
    BLIP2ThoughtVerifier = None
    ChoiceAnswerVerifier = None

# Import scorers (for custom verification)
try:
    from .scorers import clip_scorer, blip2_scorer, rule_based
except ImportError:
    pass


__all__ = [
    # Main Confirmer Classes
    "NoOpConfirmer",
    "VisualConsistencyConfirmer",
    "AnswerConsistencyConfirmer",
    # Verifier Classes (advanced usage)
    "CLIPThoughtVerifier",
    "BLIP2ThoughtVerifier",
    "ChoiceAnswerVerifier",
]
