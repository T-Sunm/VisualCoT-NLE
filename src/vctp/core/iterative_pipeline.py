from typing import Any, Dict, Iterable, List, Optional

from .interfaces import ConfirmationModule, PerceptionModule, ReasoningModule
from .types import EvidenceBundle, ReasoningOutput
from ..think.interactive import InteractiveLoop, InteractiveAttention
from ..think.interactive.attention_strategy import LLMAttentionStrategy
from ..think.context import InteractiveContextManager
from ..think.reasoning import ObjectSelector, QuestionAnswerer
from ..think.prompts import FewShotExamplesManager


class IterativeVCTPPipeline:
    """
    Visual CoT Pipeline with iterative multi-round reasoning.
    """

    def __init__(
        self,
        see: PerceptionModule,
        think: ReasoningModule,
        confirm: ConfirmationModule,
        # Iterative config
        max_rounds: int = 3,
        stop_on_convergence: bool = True,
        # Context config
        context_manager: Optional[InteractiveContextManager] = None,
        examples_manager: Optional[FewShotExamplesManager] = None,
        llm_engine_name: Optional[str] = None,
        # Other
        debug: bool = False,
    ) -> None:
        """
        Initialize iterative pipeline.

        Args:
            see: Perception module
            think: Reasoning module (VisualCoTReasoner)
            confirm: Confirmation module
            max_rounds: Maximum attention rounds (default: 3)
            stop_on_convergence: Stop early if answer converges
            context_manager: Manager for interactive few-shot examples
            examples_manager: Manager for QA prompt examples
            llm_engine_name: Name of the LLM engine to use
            debug: Enable debug mode
        """
        self.see = see
        self.think = think
        self.confirm = confirm
        self.max_rounds = max_rounds
        self.stop_on_convergence = stop_on_convergence
        self.debug = debug

        # Get components from reasoner
        self.llm = getattr(think, "llm", None)
        self.answerer = getattr(think, "answerer", None)
        self.thought_verifier = getattr(think, "thought_verifier", None)

        if not self.llm or not self.answerer:
            raise ValueError(
                "IterativeVCTPPipeline requires VisualCoTReasoner with LLM and QuestionAnswerer"
            )

        # Context managers
        self.context_manager = context_manager
        self.examples_manager = examples_manager or getattr(think, "examples_manager", None)

        # Build LLM-based attention strategy (like original Visual CoT)
        object_selector = ObjectSelector(
            model=llm_engine_name,
            use_attributes=False,
            use_captions=False,
            debug=self.debug,
        )

        attention_strategy = LLMAttentionStrategy(
            object_selector=object_selector,
            context_manager=self.context_manager,
            n_shot=8,  # Default from original paper
        )

        # Build interactive attention
        self.interactive_attention = InteractiveAttention(
            attention_strategy=attention_strategy,
            max_rounds=max_rounds,
            stop_on_convergence=stop_on_convergence,
            use_blip2=False,  # Handled by SEE module
            debug=debug,
        )

        # Build interactive loop
        self.interactive_loop = InteractiveLoop(
            interactive_attention=self.interactive_attention,
            question_answerer=self.answerer,
            context_manager=context_manager,
            examples_manager=self.examples_manager,
            thought_verifier=self.thought_verifier,
            n_shot_qa=16,
            n_ensemble=getattr(think, "n_ensemble", 1),
            chain_of_thoughts=getattr(think, "chain_of_thoughts", True),
            choice_only=getattr(think, "choice_only", False),
            debug=debug,
        )

    def run(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run iterative pipeline on a sample.

        Follows Visual CoT algorithm:
        1. SEE: Extract all visual evidence once
        2. For each round (up to max_rounds):
        a. ATTEND: Select most important object using LLM
        b. THINK: Generate answer with accumulated context
        c. Stop if answer converges
        3. CONFIRM: Verify final answer

        Args:
            sample: Sample dict with image_path, question, etc.

        Returns:
            Result dict with answer, rationale, rounds info
        """
        question = sample["question"]
        image_path = sample["image_path"]
        choices = sample.get("choices", [])  
        reference_answer = sample.get("answer", [])

        print(f"\n[Pipeline] Q: {question[:60]}...")
        print(f"[SEE] ", end="")
        evidence: EvidenceBundle = self.see.run(image_path, question)
        print(f"Evidence: {len(evidence.detected_objects)} objects, CLIP embed {'✓' if evidence.clip_image_embed is not None else '✗'}")

        # Convert evidence to objects format
        objects = self._evidence_to_objects(evidence)
        global_caption = evidence.global_caption or ""

        
        # STEP 2-3: ATTEND + THINK - Iterative reasoning with LLM attention
        if objects:
            # Generate query key
            query_key = f"{sample.get('image_id', '0')}<->{sample.get('question_id', '0')}"

            # Run interactive loop (LLM selects objects, accumulates thoughts)
            loop_result = self.interactive_loop.run_single_sample(
                query_key=query_key,
                question=question,
                objects=objects,
                global_caption=global_caption,
                choices=choices,
                reference_answer=reference_answer,
                image_embedding=evidence.clip_image_embed,
                image_path=image_path,
            )
            print(f"[THINK] Answer: {loop_result}")

            answer = loop_result["answer"]
            rationale = loop_result["rationale"]
            all_thoughts = loop_result.get("all_thoughts", [])
            num_rounds = loop_result.get("num_rounds", 1)

        else:
            reasoning: ReasoningOutput = self.think.run(evidence, question, choices=choices)  # ✅ Đã định nghĩa
            answer = reasoning.candidate_answer
            rationale = reasoning.cot_rationale
            all_thoughts = [rationale] if rationale else []
            num_rounds = 1

        # Create reasoning output for confirmation
        reasoning_output = ReasoningOutput(
            candidate_answer=answer,
            cot_rationale=rationale,
            used_concepts=[],
        )

        # STEP 4: CONFIRM - Verify answer
        confirmation = self.confirm.run(question, reasoning_output, evidence)

        # Return results
        return {
            "image_id": sample.get("image_id"),
            "question_id": sample.get("question_id"),
            "question": question,
            "answer": answer,
            "confirmed": confirmation.is_confirmed,
            "score": confirmation.score,
            "rationale": rationale,
            "all_thoughts": all_thoughts,
            "num_rounds": num_rounds,
            
            "ground_truth": {
                "answer": sample.get("answer", []), 
                "choices": sample.get("choices", []),  
                "image_path": sample.get("image_path", ""),
                "dataset": sample.get("dataset", "vivqax")
            },
        }

    def _evidence_to_objects(self, evidence: EvidenceBundle) -> List[List]:
        """
        Convert EvidenceBundle to objects format for interactive loop.

        Format matches original Visual CoT:
        [[confidence, name, attributes, caption, ocr], ...]
        """
        objects = []

        if evidence.detected_objects:
            for obj in evidence.detected_objects:
                obj_entry = [
                    obj.score,  # [0] confidence score (DetectedObject uses 'score')
                    obj.name,  # [1] object name (DetectedObject uses 'name')
                    obj.attributes or [],  # [2] attributes list
                    getattr(obj, "caption", ""),  # [3] caption (if available)
                    "",  # [4] ocr text (placeholder)
                ]
                objects.append(obj_entry)

        return objects

    def run_dataset(self, dataset: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run pipeline on entire dataset."""
        return [self.run(sample) for sample in dataset]
