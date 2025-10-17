from typing import Any, Dict, List, Optional

from vctp.core.interfaces import ReasoningModule
from vctp.core.registry import register_think
from vctp.core.types import EvidenceBundle, ReasoningOutput

from .llm import BaseLLMAdapter
from .reasoning import QuestionAnswerer, ThoughtVerifier
from .prompts import FewShotExamplesManager, compute_vqa_score


@register_think("noop-think")
class NoOpReasoner(ReasoningModule):
    """Minimal reasoner returning a constant answer and rationale."""

    def run(
        self, evidence: EvidenceBundle, question: str, **kwargs: Dict[str, Any]
    ) -> ReasoningOutput:
        return ReasoningOutput(
            candidate_answer="unknown", cot_rationale="placeholder rationale", used_concepts=[]
        )


@register_think("visualcot-reasoner")
class VisualCoTReasoner(ReasoningModule):
    """Visual Chain-of-Thought reasoner with iterative refinement."""

    def __init__(
        self,
        llm: BaseLLMAdapter,
        examples_manager: FewShotExamplesManager,
        engine: str = "gpt3",
        chain_of_thoughts: bool = True,
        choice_only: bool = False,
        n_ensemble: int = 1,
        use_thought_verification: bool = False,
        thought_verifier: Optional[ThoughtVerifier] = None,
        ablation_visual: bool = False,
        ablation_reason: bool = False,
        debug: bool = False,
    ):
        """
        Initialize Visual CoT reasoner.

        Args:
            llm: LLM adapter
            examples_manager: Few-shot examples manager
            engine: Engine type
            chain_of_thoughts: Use chain-of-thought
            choice_only: Multiple choice mode
            n_ensemble: Number of ensemble runs
            use_thought_verification: Verify thoughts with image
            thought_verifier: ThoughtVerifier instance
            ablation_visual: Ablation: remove visual evidence
            ablation_reason: Ablation: remove reasoning
            debug: Debug mode
        """
        self.llm = llm
        self.examples_manager = examples_manager
        self.engine = engine
        self.chain_of_thoughts = chain_of_thoughts
        self.choice_only = choice_only
        self.n_ensemble = n_ensemble
        self.use_thought_verification = use_thought_verification
        self.thought_verifier = thought_verifier
        self.ablation_visual = ablation_visual
        self.ablation_reason = ablation_reason
        self.debug = debug

        # Question answerer
        self.answerer = QuestionAnswerer(
            llm=llm,
            engine=engine,
            chain_of_thoughts=chain_of_thoughts,
            choice_only=choice_only,
            n_ensemble=n_ensemble,
            debug=debug,
        )

    def run(
        self,
        evidence: EvidenceBundle,
        question: str,
        context_caption: str = "",
        choices: Optional[List[str]] = None,
        example_keys: Optional[List[str]] = None,
        accumulated_thoughts: Optional[List[str]] = None,
        reference_answer: Optional[List[str]] = None,
        **kwargs: Dict[str, Any],
    ) -> ReasoningOutput:
        """
        Run Visual CoT reasoning.

        Args:
            evidence: Evidence bundle from perception
            question: Question to answer
            context_caption: Global image caption
            choices: Multiple choice options
            example_keys: Keys for few-shot examples
            accumulated_thoughts: Thoughts from previous rounds
            reference_answer: Reference answer for scoring
            **kwargs: Additional arguments

        Returns:
            ReasoningOutput with answer, rationale, and metadata
        """
        # Get few-shot examples
        examples = []
        if example_keys:
            examples = self.examples_manager.format_examples_batch(
                example_keys,
                include_rationale=self.chain_of_thoughts,
                include_choices=self.choice_only,
            )

        # Format scene graph
        scene_graph_text = self._format_scene_graph(evidence)

        # Apply ablations
        if self.ablation_visual:
            scene_graph_text = ""

        thoughts_to_use = [] if self.ablation_reason else (accumulated_thoughts or [])

        # Answer question
        answer, rationale, logprob = self.answerer.answer(
            question=question,
            context=context_caption,
            scene_graph_text=scene_graph_text,
            choices=choices,
            examples=examples,
            thoughts=thoughts_to_use,
        )

        # Verify thoughts if enabled
        verified_rationale = rationale
        all_rationale = rationale

        if self.chain_of_thoughts and self.use_thought_verification and rationale:
            if self.thought_verifier:
                image_emb = kwargs.get("image_embedding")
                image_path = kwargs.get("image_path")

                verified_rationale, all_rationale, sim_scores = (
                    self.thought_verifier.verify_thoughts(
                        thoughts=rationale, image_embedding=image_emb, image_path=image_path
                    )
                )

        # Compute accuracy if reference provided
        accuracy = 0.0
        if reference_answer is not None:
            if self.choice_only and choices:
                # Multiple choice accuracy
                accuracy = 1.0 if answer in choices and answer == choices[reference_answer] else 0.0
            else:
                # VQA-style accuracy
                accuracy = compute_vqa_score(answer, reference_answer)

        # Get used concepts from evidence
        used_concepts = []
        if hasattr(evidence, "detected_objects"):
            used_concepts = [obj.label for obj in evidence.detected_objects]

        return ReasoningOutput(
            candidate_answer=answer,
            cot_rationale=verified_rationale or "",
            used_concepts=used_concepts,
            confidence=logprob,
            metadata={
                "all_rationale": all_rationale,
                "logprob": logprob,
                "accuracy": accuracy,
                "n_ensemble": self.n_ensemble,
            },
        )

    def _format_scene_graph(self, evidence: EvidenceBundle) -> str:
        """
        Format scene graph from evidence.

        Args:
            evidence: Evidence bundle

        Returns:
            Formatted scene graph text
        """
        if not hasattr(evidence, "detected_objects"):
            return ""

        lines = []
        for obj in evidence.detected_objects:
            # Format: "object is attribute1, attribute2"
            if obj.attributes:
                attr_str = ", ".join(obj.attributes)
                lines.append(f"{obj.label} is {attr_str}")
            else:
                lines.append(obj.label)

        return "\n".join(lines)
