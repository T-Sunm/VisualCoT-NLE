from typing import Any, Dict

from vctp.core.interfaces import ReasoningModule
from vctp.core.registry import register_think
from vctp.core.types import EvidenceBundle, ReasoningOutput


@register_think("noop-think")
class NoOpReasoner(ReasoningModule):
    """Minimal reasoner returning a constant answer and rationale."""

    def run(
        self, evidence: EvidenceBundle, question: str, **kwargs: Dict[str, Any]
    ) -> ReasoningOutput:
        return ReasoningOutput(
            candidate_answer="unknown", cot_rationale="placeholder rationale", used_concepts=[]
        )
