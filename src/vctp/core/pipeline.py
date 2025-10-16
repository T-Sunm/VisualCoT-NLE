"""VCTP pipeline orchestration (skeleton)."""

from typing import Any, Dict, Iterable, List

from .interfaces import ConfirmationModule, PerceptionModule, ReasoningModule
from .types import EvidenceBundle, ReasoningOutput


class VCTPPipeline:
    def __init__(
        self, see: PerceptionModule, think: ReasoningModule, confirm: ConfirmationModule
    ) -> None:
        self.see = see
        self.think = think
        self.confirm = confirm

    def run(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        evidence: EvidenceBundle = self.see.run(sample["image_path"], sample["question"])  # type: ignore[index]
        reasoning: ReasoningOutput = self.think.run(evidence, sample["question"])  # type: ignore[index]
        confirmation = self.confirm.run(sample["question"], reasoning, evidence)
        return {
            "image_id": sample.get("image_id"),
            "question_id": sample.get("question_id"),
            "answer": reasoning.candidate_answer,
            "confirmed": confirmation.is_confirmed,
            "score": confirmation.score,
            "rationale": reasoning.cot_rationale,
        }

    def run_dataset(self, dataset: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.run(sample) for sample in dataset]
