"""
Confirmation Module
Verifies reasoning outputs against image evidence

Based on main_aokvqa.py, there are 2 main confirmation strategies:
1. Visual Consistency (lines 1080-1136): Verify thoughts against visual evidence
2. Answer Consistency (lines 1149-1159): Verify answer against choices
"""

from typing import Any, Dict, List, Optional
import numpy as np

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
        return ConfirmationOutput(
            is_confirmed=True, score=0.0, rationale="placeholder confirm"
        )


@register_confirm("visual-consistency")
class VisualConsistencyConfirmer(ConfirmationModule):
    """
    Verify thoughts/rationales against visual evidence.
    Based on lines 1080-1136 in main_aokvqa.py

    Supports multiple verification methods:
    - "clip": CLIP image-text similarity (lines 1096-1116)
    - "blip2": BLIP2 VQA verification (lines 1081-1094)
    - "oracle": Ground-truth rationale (lines 1123-1126, ablation)
    - "random": Random rationale (lines 1117-1122, ablation)
    """

    def __init__(
        self,
        method: str = "clip",
        verify_threshold: float = 0.0,
        blip2_captioner=None,
        rationale_dict: Optional[Dict[str, List[str]]] = None,
        device: Optional[str] = None,
        debug: bool = False,
    ):
        """
        Initialize visual consistency confirmer.

        Args:
            method: Verification method ("clip", "blip2", "oracle", "random")
            verify_threshold: Similarity threshold for CLIP (default 0.0)
            blip2_captioner: BLIP2Captioner instance (required for method="blip2")
            rationale_dict: Dict of rationales (required for method="oracle" or "random")
            device: Device to run on
            debug: Enable debug mode
        """
        self.method = method
        self.verify_threshold = verify_threshold
        self.rationale_dict = rationale_dict
        self.debug = debug

        # Initialize verifier based on method
        self.verifier = None
        if method == "clip":
            from vctp.confirm.verifiers.thought_verifier import CLIPThoughtVerifier

            self.verifier = CLIPThoughtVerifier(
                threshold=verify_threshold, device=device
            )

        elif method == "blip2":
            if blip2_captioner is None:
                raise ValueError("BLIP2 captioner required for method='blip2'")
            from vctp.confirm.verifiers.thought_verifier import BLIP2ThoughtVerifier

            self.verifier = BLIP2ThoughtVerifier(
                blip2_captioner=blip2_captioner, debug=debug
            )

        elif method in ["oracle", "random"]:
            if rationale_dict is None:
                raise ValueError(f"rationale_dict required for method='{method}'")

        else:
            raise ValueError(f"Unknown method: {method}")

    def run(
        self,
        question: str,
        candidate: ReasoningOutput,
        evidence: EvidenceBundle,
        query_key: Optional[str] = None,
        image_path: Optional[str] = None,
        **kwargs: Dict[str, Any],
    ) -> ConfirmationOutput:
        """
        Verify thoughts against visual evidence.

        Args:
            question: Question text
            candidate: Reasoning output to verify
            evidence: Evidence bundle
            query_key: Query key for oracle/random methods
            image_path: Path to image (for BLIP2)
            **kwargs: Additional arguments

        Returns:
            ConfirmationOutput with verification result
        """
        
        # No rationale to verify
        if not candidate.cot_rationale:
            return ConfirmationOutput(
                is_confirmed=True, score=1.0, rationale="No rationale to verify"
            )

        if self.method == "clip":
            if evidence.clip_image_embed is None:
                return ConfirmationOutput(...)

            print(f"[DEBUG CONFIRM] Rationale to verify: '{candidate.cot_rationale}'")
            print(f"[DEBUG CONFIRM] Threshold: {self.verify_threshold}")
                
            filtered, all_thoughts, scores = self.verifier.verify_and_filter(
                    thoughts=candidate.cot_rationale,
                    image_embedding=evidence.clip_image_embed,
                )
                

            print(f"[DEBUG CONFIRM] Thoughts found: {len(all_thoughts)}")
            print(f"[DEBUG CONFIRM] Thoughts passed: {len(filtered)}")
            print(f"[DEBUG CONFIRM] Scores: {scores}")
                
            # Only calculate average of scores that passed the threshold
            filtered_scores = [s for s in scores if s > self.verify_threshold]
            avg_score = np.mean(filtered_scores) if filtered_scores else 0.0
            is_confirmed = len(filtered) > 0
            rationale = (
                f"CLIP verified {len(filtered)}/{len(scores)} thoughts "
                f"(avg similarity: {avg_score:.3f})"
            )
            print(f"[CONFIRM] CLIP similarity: {avg_score:.4f} ({len(filtered)}/{len(scores)} passed) {'✓' if is_confirmed else '✗'}")

            return ConfirmationOutput(
                is_confirmed=is_confirmed, score=avg_score, rationale=rationale
            )

        elif self.method == "blip2":
            if image_path is None:
                return ConfirmationOutput(
                    is_confirmed=True, score=1.0, rationale="No image path available"
                )

            # Verify thoughts using BLIP2 VQA
            filtered, all_thoughts, scores = self.verifier.verify_and_filter(
                thoughts=candidate.cot_rationale, image_path=image_path
            )

            avg_score = np.mean(scores) if scores else 0.0
            is_confirmed = len(filtered) > 0

            rationale = f"BLIP2 verified {len(filtered)} valid thoughts (avg score: {avg_score:.3f})"
            print(f"[CONFIRM] BLIP2 similarity: {avg_score:.4f} ({len(filtered)}/{len(scores)} passed) {'✓' if is_confirmed else '✗'}")

            return ConfirmationOutput(
                is_confirmed=is_confirmed, score=avg_score, rationale=rationale
            )

        # Method: Oracle (lines 1123-1126) - Ablation study
        elif self.method == "oracle":
            if query_key is None or query_key not in self.rationale_dict:
                return ConfirmationOutput(
                    is_confirmed=True,
                    score=1.0,
                    rationale="No oracle rationale available",
                )

            # Use ground-truth rationale
            oracle_rationale = self.rationale_dict[query_key][0]

            if self.debug:
                print(f"Visual Consistency (Oracle):")
                print(f"  Predicted: {candidate.cot_rationale}")
                print(f"  Oracle: {oracle_rationale}")

            return ConfirmationOutput(
                is_confirmed=True, score=1.0, rationale=f"Oracle: {oracle_rationale}"
            )

        # Method: Random (lines 1117-1122) - Ablation study
        elif self.method == "random":
            import random

            if not self.rationale_dict:
                return ConfirmationOutput(
                    is_confirmed=True, score=0.5, rationale="No rationales available"
                )

            # Select random rationale
            random_key = random.choice(list(self.rationale_dict.keys()))
            random_rationale = random.choice(self.rationale_dict[random_key])

            if self.debug:
                print(f"Visual Consistency (Random):")
                print(f"  Predicted: {candidate.cot_rationale}")
                print(f"  Random: {random_rationale}")

            return ConfirmationOutput(
                is_confirmed=True, score=0.5, rationale=f"Random: {random_rationale}"
            )

        else:
            raise ValueError(f"Unknown method: {self.method}")


@register_confirm("answer-consistency")
class AnswerConsistencyConfirmer(ConfirmationModule):
    """
    Verify answer against available choices using CLIP text similarity.
    Based on lines 1149-1159 in main_aokvqa.py

    If the predicted answer is not in choices, find the closest match
    using CLIP text similarity.
    """

    def __init__(
        self,
        correct_answer: bool = True,
        device: Optional[str] = None,
        debug: bool = False,
    ):
        """
        Initialize answer consistency confirmer.

        Args:
            correct_answer: Whether to correct answer to closest choice
            device: Device to run on
            debug: Enable debug mode
        """
        from vctp.confirm.verifiers.answer_verifier import ChoiceAnswerVerifier

        self.verifier = ChoiceAnswerVerifier(device=device)
        self.correct_answer = correct_answer
        self.debug = debug

    def run(
        self,
        question: str,
        candidate: ReasoningOutput,
        evidence: EvidenceBundle,
        choices: Optional[List[str]] = None,
        **kwargs: Dict[str, Any],
    ) -> ConfirmationOutput:
        """
        Verify answer against choices.

        Args:
            question: Question text
            candidate: Reasoning output with candidate answer
            evidence: Evidence bundle
            choices: List of valid answer choices
            **kwargs: Additional arguments

        Returns:
            ConfirmationOutput with verification result
        """
        if choices is None or len(choices) == 0:
            return ConfirmationOutput(
                is_confirmed=True, score=1.0, rationale="No choices to verify against"
            )

        # Verify answer (lines 1149-1159)
        is_in_choices, score, corrected = self.verifier.verify(
            candidate=candidate.candidate_answer, choices=choices
        )

        if is_in_choices:
            # Answer is directly in choices
            rationale = f"Answer '{candidate.candidate_answer}' is valid"
            confirmed_answer = candidate.candidate_answer
        else:
            # Answer not in choices, find closest match
            rationale = (
                f"Answer '{candidate.candidate_answer}' not in choices. "
                f"Closest match: '{corrected}' (similarity: {score:.3f})"
            )
            confirmed_answer = (
                corrected if self.correct_answer else candidate.candidate_answer
            )

        if self.debug:
            print(f"Answer Consistency: {rationale}")
            print(f"  Choices: {choices}")
            print(f"  Confirmed: {confirmed_answer}")

        return ConfirmationOutput(
            is_confirmed=is_in_choices or self.correct_answer,
            score=score,
            rationale=rationale,
        )


# def main():
#     """
#     Clean test function for confirmation module.
#     Tests all three confirmation strategies with mock data.
#     """
#     import numpy as np

#     print("=" * 60)
#     print("TESTING CONFIRMATION MODULE")
#     print("=" * 60)

#     # ============================================================
#     # Setup mock data
#     # ============================================================

#     # Mock evidence bundle
#     evidence = EvidenceBundle(
#         image_id="test_image_001",
#         global_caption="A cat sitting on a couch",
#         detected_objects=[],
#         attributes={},
#         relations=[],
#         clip_image_embed=np.random.randn(512).astype(np.float32),  # Mock CLIP embedding
#         region_captions=None,
#     )

#     # Mock reasoning output
#     candidate = ReasoningOutput(
#         candidate_answer="cat",
#         cot_rationale="There is a cat. The cat is sitting on furniture.",
#         used_concepts=["cat", "furniture"],
#     )

#     question = "What animal is in the image?"
#     choices = ["cat", "dog", "bird", "fish"]

#     # ============================================================
#     # Test 1: NoOp Confirmer
#     # ============================================================
#     print("\n[TEST 1] NoOp Confirmer")
#     print("-" * 60)

#     noop = NoOpConfirmer()
#     result = noop.run(question=question, candidate=candidate, evidence=evidence)

#     print(f"✓ Is Confirmed: {result.is_confirmed}")
#     print(f"✓ Score: {result.score}")
#     print(f"✓ Rationale: {result.rationale}")
#     assert isinstance(result, ConfirmationOutput)
#     assert result.is_confirmed == True
#     assert result.score == 0.0

#     # ============================================================
#     # Test 2: Visual Consistency Confirmer (CLIP)
#     # ============================================================
#     print("\n[TEST 2] Visual Consistency Confirmer (CLIP)")
#     print("-" * 60)

#     try:
#         visual_clip = VisualConsistencyConfirmer(method="clip", verify_threshold=0.0, debug=True)
#         result = visual_clip.run(question=question, candidate=candidate, evidence=evidence)

#         print(f"✓ Is Confirmed: {result.is_confirmed}")
#         print(f"✓ Score: {result.score:.4f}")
#         print(f"✓ Rationale: {result.rationale}")
#         assert isinstance(result, ConfirmationOutput)
#         assert isinstance(result.score, (float, np.floating))

#     except Exception as e:
#         print(f"⚠ CLIP test skipped: {e}")

#     # ============================================================
#     # Test 3: Answer Consistency Confirmer
#     # ============================================================
#     print("\n[TEST 3] Answer Consistency Confirmer")
#     print("-" * 60)

#     try:
#         answer_checker = AnswerConsistencyConfirmer(correct_answer=True, debug=True)

#         # Test 3a: Answer in choices
#         result = answer_checker.run(
#             question=question, candidate=candidate, evidence=evidence, choices=choices
#         )

#         print(f"✓ Is Confirmed: {result.is_confirmed}")
#         print(f"✓ Score: {result.score:.4f}")
#         print(f"✓ Rationale: {result.rationale}")
#         assert isinstance(result, ConfirmationOutput)

#         # Test 3b: Answer NOT in choices
#         print("\n  [3b] Testing out-of-choice answer...")
#         candidate_wrong = ReasoningOutput(
#             candidate_answer="feline",  # Similar to "cat" but not in choices
#             cot_rationale="There is a feline animal",
#             used_concepts=["feline"],
#         )

#         result = answer_checker.run(
#             question=question, candidate=candidate_wrong, evidence=evidence, choices=choices
#         )

#         print(f"Is Confirmed: {result.is_confirmed}")
#         print(f"Score: {result.score:.4f}")
#         print(f"Rationale: {result.rationale}")

#     except Exception as e:
#         print(f"Answer consistency test skipped: {e}")

#     # ============================================================
#     # Summary
#     # ============================================================
#     print("\n" + "=" * 60)
#     print("ALL TESTS COMPLETED")
#     print("=" * 60)
#     print("\nExpected Output Format:")
#     print("  ConfirmationOutput(")
#     print("      is_confirmed=True,")
#     print("      score=0.78,")
#     print("      rationale='Answer is consistent with visual evidence'")
#     print("  )")
#     print("=" * 60)


# if __name__ == "__main__":
#     main()
