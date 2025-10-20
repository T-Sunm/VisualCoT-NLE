"""Base interface for verification modules."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class BaseVerifier(ABC):
    """Base class for all verifiers."""

    @abstractmethod
    def verify(
        self,
        candidate: str,
        image_embedding: Optional[np.ndarray] = None,
        image_path: Optional[str] = None,
        **kwargs,
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Verify a candidate (thought/answer) against image.

        Args:
            candidate: Text to verify
            image_embedding: Pre-computed image embedding
            image_path: Path to image
            **kwargs: Additional arguments

        Returns:
            Tuple of (is_valid, confidence_score, corrected_text_or_none)
        """
        raise NotImplementedError
