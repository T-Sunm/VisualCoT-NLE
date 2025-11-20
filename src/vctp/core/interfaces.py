"""Interfaces (ABCs) for VCTP modules."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .types import ConfirmationOutput, EvidenceBundle, ReasoningOutput


class PerceptionModule(ABC):
    @abstractmethod
    def run(self, image_path: str, question: str, **kwargs: Dict[str, Any]) -> EvidenceBundle:
        """Produce visual evidence bundle from image and question."""
        raise NotImplementedError


class ReasoningModule(ABC):
    @abstractmethod
    def run(
        self, evidence: EvidenceBundle, question: str, **kwargs: Dict[str, Any]
    ) -> ReasoningOutput:
        """Generate candidate answer and rationale given evidence and question."""
        raise NotImplementedError


class ConfirmationModule(ABC):
    @abstractmethod
    def run(
        self,
        question: str,
        candidate: ReasoningOutput,
        evidence: EvidenceBundle,
        image_path: Optional[str] = None,
        **kwargs: Dict[str, Any],
    ) -> ConfirmationOutput:
        """Verify candidate answer using evidence and optionally retrieval."""
        raise NotImplementedError
