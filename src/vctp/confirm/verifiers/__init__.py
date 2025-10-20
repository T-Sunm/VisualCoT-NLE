"""
Verifiers Module
Provides various verification strategies for thoughts and answers
"""

from typing import Optional

# Import base verifier
try:
    from .base_verifier import BaseVerifier
except ImportError:
    BaseVerifier = None

# Import thought verifiers
try:
    from .thought_verifier import CLIPThoughtVerifier, BLIP2ThoughtVerifier
except ImportError:
    CLIPThoughtVerifier = None
    BLIP2ThoughtVerifier = None

# Import answer verifier
try:
    from .answer_verifier import ChoiceAnswerVerifier
except ImportError:
    ChoiceAnswerVerifier = None


__all__ = [
    "BaseVerifier",
    "CLIPThoughtVerifier",
    "BLIP2ThoughtVerifier",
    "ChoiceAnswerVerifier",
]
