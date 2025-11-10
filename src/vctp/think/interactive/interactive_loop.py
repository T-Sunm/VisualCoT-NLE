"""Interactive loop combining See and Think modules."""

from typing import Any, Dict, List, Optional, Tuple
import os
import json

from .interactive_attention import InteractiveAttention
from ..context import InteractiveContextManager
from ..prompts import FewShotExamplesManager


class InteractiveLoop:
    """
    Full Visual CoT pipeline with interactive attention.

    Orchestrates: See (perception) -> Select (attention) -> Think (reasoning)
    across multiple rounds with iterative refinement.
    """

    def __init__(
        self,
        interactive_attention: InteractiveAttention,
        question_answerer,
        context_manager: Optional[InteractiveContextManager] = None,
        examples_manager: Optional[FewShotExamplesManager] = None,
        thought_verifier=None,
        n_shot_qa: int = 16,
        n_ensemble: int = 1,
        chain_of_thoughts: bool = True,
        choice_only: bool = False,
        ablation_visual: bool = False,
        ablation_reason: bool = False,
        debug: bool = False,
    ):
        """
        Initialize interactive loop.

        Args:
            interactive_attention: InteractiveAttention instance
            question_answerer: QuestionAnswerer instance
            context_manager: Context manager for examples
            examples_manager: Examples manager for formatting
            thought_verifier: ThoughtVerifier instance
            n_shot_qa: Number of few-shot examples for QA
            n_ensemble: Number of ensemble runs
            chain_of_thoughts: Use chain-of-thought
            choice_only: Multiple choice mode
            ablation_visual: Ablation: remove visual evidence
            ablation_reason: Ablation: remove reasoning
            debug: Debug mode
        """
        self.interactive_attention = interactive_attention
        self.question_answerer = question_answerer
        self.context_manager = context_manager
        self.examples_manager = examples_manager
        self.thought_verifier = thought_verifier
        self.n_shot_qa = n_shot_qa
        self.n_ensemble = n_ensemble
        self.chain_of_thoughts = chain_of_thoughts
        self.choice_only = choice_only
        self.ablation_visual = ablation_visual
        self.ablation_reason = ablation_reason
        self.debug = debug

    def run_single_sample(
        self,
        query_key: str,
        question: str,
        objects: List[List],
        global_caption: str = "",
        choices: Optional[List[str]] = None,
        reference_answer: Optional[List[str]] = None,
        image_embedding: Optional[Any] = None,
        image_path: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run full pipeline for a single sample.

        Args:
            query_key: Query key (e.g., "image_id<->question_id")
            question: Question to answer
            objects: Available objects from scene graph
            global_caption: Global image caption
            choices: Multiple choice options
            reference_answer: Ground-truth answer for evaluation
            image_embedding: CLIP image embedding for verification
            image_path: Path to image
            **kwargs: Additional arguments

        Returns:
            Result dict with answer, rationale, and metadata
        """
        print(f"\n[InteractiveLoop] Processing: {query_key}")
        print(f"[InteractiveLoop] Question: {question}")

        # Get few-shot examples for QA
        qa_examples = self._get_qa_examples(query_key)

        # Create reasoning callback for interactive rounds
        def reasoning_callback(selected_objects, accumulated_thoughts):
            return self._reasoning_step(
                question=question,
                global_caption=global_caption,
                selected_objects=selected_objects,
                accumulated_thoughts=accumulated_thoughts,
                choices=choices,
                qa_examples=qa_examples,
                image_embedding=image_embedding,
                image_path=image_path,
            )

        # Run interactive attention rounds
        round_results, accumulated_thoughts = self.interactive_attention.run_rounds(
            question=question,
            objects=objects,
            reasoning_callback=reasoning_callback,
            query_key=query_key,
            image_path=image_path,
        )

        # Get final result from last round
        if round_results:
            final_result = round_results[-1]
        else:
            final_result = {
                "answer": "unknown",
                "rationale": "",
                "confidence": 0.0,
            }

        # Compute accuracy if reference provided
        accuracy = 0.0
        if reference_answer is not None:
            accuracy = self._compute_accuracy(
                pred_answer=final_result["answer"],
                ref_answer=reference_answer,
                choices=choices,
            )

        # Prepare output
        output = {
            "key": query_key,
            "question": question,
            "answer": final_result["answer"],
            "rationale": final_result.get("rationale", ""),
            "all_thoughts": accumulated_thoughts,
            "confidence": final_result.get("confidence", 0.0),
            "accuracy": accuracy,
            "rounds": len(round_results),
            "round_results": round_results,
            "global_caption": global_caption,
            "selected_objects": [r.get("selected_objects", []) for r in round_results],
        }

        print(f"[THINK] Answer: {output['answer']} | Accuracy: {accuracy:.2f} | Rounds: {len(round_results)}")

        return output

    def _reasoning_step(
        self,
        question: str,
        global_caption: str,
        selected_objects: List[List],
        accumulated_thoughts: List[str],
        choices: Optional[List[str]],
        qa_examples: List[Dict],
        image_embedding: Optional[Any],
        image_path: Optional[str],
    ) -> Dict[str, Any]:
        """
        Single reasoning step.

        Args:
            question: Question
            global_caption: Global caption
            selected_objects: Selected objects so far
            accumulated_thoughts: Thoughts from previous rounds
            choices: Multiple choice options
            qa_examples: Few-shot examples
            image_embedding: Image embedding
            image_path: Image path

        Returns:
            Result dict
        """
        # Format scene graph from selected objects
        scene_graph_text = self._format_scene_graph(selected_objects)

        # Apply ablations
        if self.ablation_visual:
            scene_graph_text = ""

        thoughts_to_use = [] if self.ablation_reason else accumulated_thoughts

        # Prepare context (global caption + accumulated thoughts)
        context = global_caption
        if thoughts_to_use:
            filtered_thoughts = [t for t in thoughts_to_use if t]
            if filtered_thoughts:
                context += "\n" + " ".join(filtered_thoughts)

        # Answer question
        answer, rationale, confidence = self.question_answerer._answer_single(
            question=question,
            context=context,
            scene_graph_text=scene_graph_text,
            choices=choices,
            examples=qa_examples,
            thoughts=thoughts_to_use,
        )

        # Verify thoughts if enabled
        verified_rationale = rationale
        all_rationale = rationale

        if self.chain_of_thoughts and self.thought_verifier and rationale:
            verified_rationale, all_rationale, _ = self.thought_verifier.verify_thoughts(
                thoughts=rationale, image_embedding=image_embedding, image_path=image_path
            )

        return {
            "answer": answer,
            "rationale": verified_rationale,
            "all_rationale": all_rationale,
            "confidence": confidence,
            "selected_objects": [obj[1] for obj in selected_objects],  # Object names
        }

    def _get_qa_examples(self, query_key: str) -> List[Dict]:
        """Get few-shot examples for QA."""
        if not self.context_manager or not self.examples_manager:
            return []

        # Get example keys
        example_keys = self.context_manager.get_qa_context_examples(
            query_key=query_key, n_shot=self.n_shot_qa * self.n_ensemble
        )

        if not example_keys:
            return []

        # Format examples
        return self.examples_manager.format_examples_batch(
            example_keys=example_keys,
            include_rationale=self.chain_of_thoughts,
            include_choices=self.choice_only,
        )

    def _format_scene_graph(self, objects: List[List]) -> str:
        """Format scene graph from objects."""
        lines = []
        for obj in objects:
            obj_name = obj[1]

            # Check if we have caption (index 3) or attributes (index 2)
            if len(obj) > 3 and obj[3]:
                # Use caption if available
                lines.append(obj[3])
            elif len(obj) > 2 and obj[2]:
                # Use attributes
                attr_str = ", ".join(obj[2])
                lines.append(f"{obj_name} is {attr_str}")
            else:
                # Just object name
                lines.append(obj_name)

        return "\n".join(lines)

    def _compute_accuracy(
        self,
        pred_answer: str,
        ref_answer: List[str],
        choices: Optional[List[str]] = None,
    ) -> float:
        """Compute accuracy."""
        if self.choice_only and choices:
            # Multiple choice
            if isinstance(ref_answer, int):
                return 1.0 if pred_answer == choices[ref_answer] else 0.0
            else:
                return 1.0 if pred_answer in ref_answer else 0.0
        else:
            # VQA-style accuracy
            counter = 0
            for ans in ref_answer:
                if pred_answer == ans:
                    counter += 1
            return min(1.0, float(counter) * 0.3)

    def run_batch(
        self,
        samples: List[Dict[str, Any]],
        save_path: Optional[str] = None,
        save_every: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Run on a batch of samples.

        Args:
            samples: List of sample dicts with keys:
                - query_key
                - question
                - objects
                - global_caption
                - choices (optional)
                - reference_answer (optional)
                - image_embedding (optional)
                - image_path (optional)
            save_path: Path to save results
            save_every: Save every N samples

        Returns:
            List of result dicts
        """
        results = []

        for idx, sample in enumerate(samples):
            if self.debug:
                print(f"\n[InteractiveLoop] Sample {idx + 1}/{len(samples)}")

            result = self.run_single_sample(**sample)
            results.append(result)

            # Save intermediate results
            if save_path and (idx + 1) % save_every == 0:
                self._save_results(results, save_path, idx + 1)

            # Print running accuracy
            if idx > 0:
                running_acc = sum(r["accuracy"] for r in results) / len(results)
                print(f"[InteractiveLoop] Running Accuracy: {running_acc * 100:.2f}%")

        # Save final results
        if save_path:
            self._save_results(results, save_path, len(samples))

        return results

    def _save_results(self, results: List[Dict[str, Any]], save_path: str, n_samples: int):
        """Save results to file."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save full results
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)

        # Also save format compatible with evaluation
        format_results = []
        for r in results:
            question_id = r["key"].split("<->")[-1]
            format_results.append(
                {
                    "question_id": question_id,
                    "answer": r["answer"],
                    "rationale": r.get("all_thoughts", []),
                }
            )

        format_path = save_path.replace(".json", "_format.json")
        with open(format_path, "w") as f:
            json.dump(format_results, f, indent=2)

        if self.debug:
            print(f"[InteractiveLoop] Saved {n_samples} results to {save_path}")
