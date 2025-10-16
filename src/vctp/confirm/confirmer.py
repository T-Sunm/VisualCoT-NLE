from typing import Any, Dict

from vctp.core.interfaces import ConfirmationModule
from vctp.core.registry import register_confirm
from vctp.core.types import ConfirmationOutput, EvidenceBundle, ReasoningOutput


@register_confirm("noop-confirm")
class NoOpConfirmer(ConfirmationModule):
    """Minimal confirmer that always confirms with score 0.0."""

    def run(
        self,
        question: str,
        candidate: ReasoningOutput,
        evidence: EvidenceBundle,
        **kwargs: Dict[str, Any],
    ) -> ConfirmationOutput:
        return ConfirmationOutput(is_confirmed=True, score=0.0, rationale="placeholder confirm")
