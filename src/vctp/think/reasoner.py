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
        
        # Initialize debug info collector
        debug_info = {
            "few_shot_examples": [],
            "similarity_scores": {},
            "confidence": 0.0,
            "intermediate_thoughts": [],
            "prompt_used": "",
            "llm_response": "",
            "processing_steps": [],
            "errors": []
        }

        debug_info["processing_steps"].append("Starting reasoning process")
        
        # Get few-shot examples if available
        if example_keys:
            debug_info["few_shot_examples"] = example_keys[:5]  # Top 5 only
            debug_info["processing_steps"].append(f"Using {len(example_keys)} few-shot examples")
        
        # Prepare sample dict
        sample_dict = {
            "question": question,
            "choices": choices,
            "key": kwargs.get("query_key", ""),
            "train_context": kwargs.get("train_context", {}),
        }

        # Prepare visual context
        scene_graph_text = self._format_scene_graph(evidence.detected_objects)
        visual_context_list = []
        if context_caption:
            visual_context_list.append(context_caption)
        if scene_graph_text:
            visual_context_list.append(scene_graph_text)

        debug_info["processing_steps"].append("Formatted visual context")

        # Call answerer
        try:
            result = self.answerer.answer(
                sample=sample_dict,
                visual_context=visual_context_list,
                thoughts=accumulated_thoughts or [],
            )
            
            # Extract results
            answer = result.get("answer", "")
            rationale = result.get("rationale", "")
            confidence = result.get("confidence", 0.0)
            
            # Store debug info
            debug_info["confidence"] = confidence
            debug_info["prompt_used"] = result.get("prompt", "")[:200] + "..." if len(result.get("prompt", "")) > 200 else result.get("prompt", "")
            debug_info["llm_response"] = result.get("response", "")[:200] + "..." if len(result.get("response", "")) > 200 else result.get("response", "")
            debug_info["intermediate_thoughts"] = accumulated_thoughts or []
            
            debug_info["processing_steps"].append("Generated answer and rationale")
            
        except Exception as e:
            debug_info["errors"].append(f"Answer generation failed: {str(e)}")
            answer = "Error"
            rationale = ""
            confidence = 0.0

        # Compute accuracy
        accuracy = 0.0
        if reference_answer is not None:
            if self.choice_only and choices:
                accuracy = 1.0 if answer in choices and answer == choices[reference_answer] else 0.0
            else:
                from vctp.think.prompts.formatters import compute_vqa_score
                accuracy = compute_vqa_score(answer, reference_answer)

        # Get used concepts
        used_concepts = []
        if hasattr(evidence, "detected_objects"):
            used_concepts = [obj.name for obj in evidence.detected_objects[:10]]  # Top 10 only

        return ReasoningOutput(
            candidate_answer=answer,
            cot_rationale=rationale,
            logprob=0.0,
            accuracy=accuracy,
            used_concepts=used_concepts,
            debug_info=debug_info  # Thêm debug info
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
                lines.append(f"{obj.name} is {attr_str}")
            else:
                lines.append(obj.name)

        return "\n".join(lines)


# if __name__ == "__main__":
#     """Test VisualCoTReasoner with real AOKVQA data."""
#     import os
#     import json
#     from pathlib import Path

#     # Load environment
#     try:
#         from dotenv import load_dotenv

#         project_root = Path(__file__).parent.parent.parent.parent
#         env_path = project_root / ".env"
#         if env_path.exists():
#             load_dotenv(env_path)
#             print(f"✓ Loaded environment from: {env_path}")
#     except ImportError:
#         pass

#     from .llm import create_llm_adapter
#     from .prompts import FewShotExamplesManager
#     from vctp.core.types import EvidenceBundle, DetectedObject, ReasoningOutput
#     from vctp.data.loader import load_vivqax_annotations, build_vivqax_dicts, load_coco_captions

#     print("=" * 80)
#     print("TESTING VISUAL COT REASONER WITH REAL AOKVQA DATA")
#     print("=" * 80)

#     # Check API key
#     if not os.getenv("GROQ_API_KEY"):
#         print("❌ Error: GROQ_API_KEY not set")
#         exit(1)

#     # Define paths
#     project_root = Path(__file__).parent.parent.parent.parent
#     train_annotations_path = (
#         project_root / "data/raw/aokvqa_annotations/aokvqa_v1p0_train_from_hf.json"
#     )
#     train_captions_path = project_root / "data/processed/captions_train2017.json"

#     # 1. Load training data for few-shot examples
#     print("\n--- Loading Training Data ---")
#     if not train_annotations_path.exists():
#         print(f"❌ Training annotations not found at: {train_annotations_path}")
#         print("Please download AOKVQA data first.")
#         exit(1)

#     train_annotations = load_vivqax_annotations(str(train_annotations_path))
#     print(f"✓ Loaded {len(train_annotations)} training samples")

#     # Build dicts
#     train_answers, train_questions, train_rationales, train_choices = build_aokvqa_dicts(
#         train_annotations, choice_only=False
#     )
#     print(f"✓ Built training dictionaries")

#     # Load captions
#     train_captions = {}
#     if train_captions_path.exists():
#         train_captions = load_coco_captions(str(train_captions_path))
#         print(f"✓ Loaded {len(train_captions)} training captions")
#     else:
#         print(f"⚠ Training captions not found at: {train_captions_path}")

#     # 2. Create FewShotExamplesManager with REAL data
#     examples_manager = FewShotExamplesManager(
#         train_questions=train_questions,
#         train_answers=train_answers,
#         train_rationales=train_rationales,
#         train_choices=train_choices,
#         train_captions=train_captions,
#     )
#     print(f"✓ Created FewShotExamplesManager with {len(train_questions)} examples")

#     # 3. Create LLM
#     llm = create_llm_adapter(
#         engine="groq",
#         engine_name="llama-3.1-8b-instant",
#         temperature=0.0,
#         max_tokens=150,
#         debug=False,
#     )
#     print("✓ LLM adapter created")

#     # 4. Create Reasoner
#     reasoner = VisualCoTReasoner(
#         llm=llm,
#         examples_manager=examples_manager,
#         engine="groq",
#         chain_of_thoughts=True,
#         choice_only=False,
#         n_ensemble=1,
#         use_thought_verification=False,
#         debug=True,
#     )
#     print("✓ Reasoner initialized")

#     # 5. Get real few-shot examples
#     print("\n--- Testing Few-Shot Examples ---")
#     # Get random example keys
#     example_keys = examples_manager.get_random_examples(n_shot=2)
#     print(f"Selected example keys: {example_keys}")

#     # Format examples
#     formatted_examples = examples_manager.format_examples_batch(
#         example_keys, include_rationale=True, include_choices=False
#     )
#     print(f"\nFormatted examples:")
#     for i, ex in enumerate(formatted_examples):
#         print(f"  Example {i+1}:")
#         print(f"    Q: {ex['question']}")
#         print(f"    A: {ex['answer']}")
#         print(f"    R: {ex.get('rationale', 'N/A')[:50]}...")

#     # 6. Prepare test data - EvidenceBundle
#     evidence = EvidenceBundle(
#         image_id="test_001",
#         global_caption="A man playing tennis on clay court",
#         detected_objects=[
#             DetectedObject(
#                 name="man", bbox=[10, 20, 100, 200], score=0.95, attributes=["standing"]
#             ),
#             DetectedObject(name="racket", bbox=[50, 80, 70, 120], score=0.92, attributes=["green"]),
#             DetectedObject(
#                 name="court", bbox=[0, 150, 300, 250], score=0.88, attributes=["red", "clay"]
#             ),
#         ],
#         attributes={},
#         relations=[],
#         clip_image_embed=None,
#         region_captions=[],
#     )

#     question = "What is the man doing?"
#     choices = ["tennis", "soccer", "basketball"]
#     accumulated_thoughts = ["The man holds a racket"]

#     print("\n--- Input ---")
#     print(f"Question: {question}")
#     print(f"Choices: {choices}")
#     print(f"Global Caption: {evidence.global_caption}")
#     print(f"Detected Objects: {[obj.name for obj in evidence.detected_objects]}")
#     print(f"Accumulated Thoughts: {accumulated_thoughts}")
#     print(f"Using {len(example_keys)} few-shot examples from training data")

#     # 7. Run reasoning
#     print("\n--- Running Reasoner ---")
#     result = reasoner.run(
#         evidence=evidence,
#         question=question,
#         context_caption=evidence.global_caption,
#         choices=choices,
#         example_keys=example_keys,  # Use real training examples
#         accumulated_thoughts=accumulated_thoughts,
#         reference_answer=["tennis", "playing tennis"],
#     )

#     # 8. Display output
#     print("\n--- Output ---")
#     print(f"Type: {type(result)}")
#     print(f"Candidate Answer: {result.candidate_answer}")
#     print(f"CoT Rationale: {result.cot_rationale}")
#     print(f"Used Concepts: {result.used_concepts}")
#     print(f"Confidence: {result.confidence}")
#     print(f"Metadata: {result.metadata}")

#     # 9. Validation
#     print("\n" + "=" * 80)
#     print("VALIDATION")
#     print("=" * 80)

#     # Check type
#     assert isinstance(result, ReasoningOutput), f"❌ Expected ReasoningOutput, got {type(result)}"
#     print("✓ Result is ReasoningOutput")

#     # Check required fields
#     assert result.candidate_answer, "❌ candidate_answer should not be empty"
#     print(f"✓ candidate_answer: '{result.candidate_answer}'")

#     assert result.cot_rationale, "❌ cot_rationale should not be empty"
#     print(f"✓ cot_rationale: '{result.cot_rationale[:50]}...'")

#     assert isinstance(result.used_concepts, list), "❌ used_concepts should be a list"
#     assert len(result.used_concepts) > 0, "❌ used_concepts should not be empty"
#     print(f"✓ used_concepts: {result.used_concepts}")

#     # Check metadata
#     assert result.metadata is not None, "❌ metadata should not be None"
#     assert "accuracy" in result.metadata, "❌ metadata should have 'accuracy'"
#     print(f"✓ metadata contains accuracy: {result.metadata.get('accuracy')}")

#     print("\n" + "=" * 80)
#     print("ALL TESTS PASSED - VISUAL COT REASONER WORKS WITH REAL DATA!")
#     print("=" * 80)
