"""Interactive attention mechanism for Visual CoT."""

from typing import Dict, List, Optional, Tuple, Any

from .attention_strategy import AttentionStrategy, AllRegionsAttentionStrategy


class InteractiveAttention:
    """
    Manage interactive attention rounds.

    Iteratively selects objects to attend to, gathers visual evidence,
    and accumulates reasoning thoughts across multiple rounds.
    """

    def __init__(
        self,
        attention_strategy: AttentionStrategy,
        max_rounds: int = 3,
        stop_on_convergence: bool = True,
        use_blip2: bool = False,
        blip2_captioner=None,
        debug: bool = False,
    ):
        """
        Initialize interactive attention.

        Args:
            attention_strategy: Strategy for selecting objects
            max_rounds: Maximum number of attention rounds
            stop_on_convergence: Stop if answer doesn't change
            use_blip2: Use BLIP2 for local captions
            blip2_captioner: BLIP2Captioner instance
            debug: Debug mode
        """
        self.attention_strategy = attention_strategy
        self.max_rounds = max_rounds
        self.stop_on_convergence = stop_on_convergence
        self.use_blip2 = use_blip2
        self.blip2_captioner = blip2_captioner
        self.debug = debug

    def run_rounds(
        self,
        question: str,
        objects: List[List],
        reasoning_callback,
        query_key: Optional[str] = None,
        image_path: Optional[str] = None,
        **kwargs,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Run interactive attention rounds.

        Args:
            question: Question to answer
            objects: Available objects [conf, name, attrs, caption, ...]
            reasoning_callback: Function to call for reasoning each round
                Should accept (selected_objects, accumulated_thoughts) and return result dict
            query_key: Query key for context
            image_path: Path to image (for BLIP2)
            **kwargs: Additional arguments

        Returns:
            Tuple of (round_results, accumulated_thoughts)
            - round_results: List of result dicts from each round
            - accumulated_thoughts: List of thoughts accumulated across rounds
        """
        # Check for all regions mode
        all_regions_mode = isinstance(self.attention_strategy, AllRegionsAttentionStrategy)

        if all_regions_mode:
            return self._run_all_regions(
                question=question,
                objects=objects,
                reasoning_callback=reasoning_callback,
                query_key=query_key,
                image_path=image_path,
                **kwargs,
            )

        # Standard iterative mode
        round_results = []
        accumulated_thoughts = []
        noticed_objects = []
        remaining_objects = objects.copy()
        previous_answer = None

        for round_idx in range(self.max_rounds):
            # Check if we have objects left
            if len(remaining_objects) == 0:
                print("[InteractiveAttention] No more objects to attend to")
                break

            # Select object to attend to
            selected_idx = self.attention_strategy.select_object(
                question=question, objects=remaining_objects, query_key=query_key, **kwargs
            )

            # Validate index
            if selected_idx >= len(remaining_objects):
                selected_idx = selected_idx % len(remaining_objects)

            selected_object = remaining_objects[selected_idx]
            
            # IN RA SAU KHI ĐÃ CÓ selected_object
            print(f"[Round {round_idx + 1}/{self.max_rounds}] → {selected_object[1]}", end="")

            # Enhance with BLIP2 if needed
            if self.use_blip2 and self.blip2_captioner and image_path:
                local_caption = self.blip2_captioner.query_local_caption(
                    object_name=selected_object[1], question=question, image_path=image_path
                )
                # Update object with local caption
                enhanced_obj = [
                    selected_object[0],  # confidence
                    selected_object[1],  # name
                    [],  # attributes (empty for BLIP2)
                    local_caption,  # local caption
                    "",  # ocr text
                ]
                noticed_objects.append(enhanced_obj)
            else:
                noticed_objects.append(selected_object)

            # Run reasoning with current evidence
            result = reasoning_callback(
                selected_objects=noticed_objects.copy(),
                accumulated_thoughts=accumulated_thoughts.copy(),
            )

            round_results.append(result)

            # Extract thought from result
            if "rationale" in result:
                thought = result["rationale"]
                if thought and thought not in accumulated_thoughts:
                    accumulated_thoughts.append(thought)

            # Check for convergence
            current_answer = result.get("answer", "")
            if self.stop_on_convergence and current_answer == previous_answer:
                print(" (converged)")
                break
            elif round_idx < self.max_rounds - 1:  # Không in | ở round cuối
                print(" | ", end="")
            else:
                print()  # Xuống dòng ở round cuối

            previous_answer = current_answer

            # Remove selected object for next round
            remaining_objects = (
                remaining_objects[:selected_idx] + remaining_objects[selected_idx + 1 :]
            )

        return round_results, accumulated_thoughts

    def _run_all_regions(
        self,
        question: str,
        objects: List[List],
        reasoning_callback,
        query_key: Optional[str] = None,
        image_path: Optional[str] = None,
        **kwargs,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Run with all regions at once (no iterative selection).

        Args:
            question: Question to answer
            objects: All objects
            reasoning_callback: Reasoning function
            query_key: Query key
            image_path: Image path
            **kwargs: Additional arguments

        Returns:
            Tuple of (round_results, accumulated_thoughts)
        """
        if self.debug:
            print("[InteractiveAttention] All regions mode - using all objects at once")

        # Use all objects
        noticed_objects = objects.copy()

        # Single round with all objects
        result = reasoning_callback(selected_objects=noticed_objects, accumulated_thoughts=[])

        # Extract thought
        accumulated_thoughts = []
        if "rationale" in result:
            thought = result["rationale"]
            if thought:
                accumulated_thoughts.append(thought)

        return [result], accumulated_thoughts

    def get_round_summary(
        self, round_results: List[Dict[str, Any]], accumulated_thoughts: List[str]
    ) -> Dict[str, Any]:
        """
        Get summary of all rounds.

        Args:
            round_results: Results from each round
            accumulated_thoughts: Accumulated thoughts

        Returns:
            Summary dict with final answer and metadata
        """
        if not round_results:
            return {
                "answer": "unknown",
                "rationale": "",
                "rounds": 0,
                "confidence": 0.0,
            }

        # Use last round as final result
        final_result = round_results[-1]

        return {
            "answer": final_result.get("answer", "unknown"),
            "rationale": final_result.get("rationale", ""),
            "all_thoughts": accumulated_thoughts,
            "rounds": len(round_results),
            "confidence": final_result.get("confidence", 0.0),
            "round_results": round_results,
        }


class SingleRoundAttention(InteractiveAttention):
    """Single round attention (no iteration)."""

    def __init__(
        self,
        attention_strategy: AttentionStrategy,
        use_blip2: bool = False,
        blip2_captioner=None,
        debug: bool = False,
    ):
        """
        Initialize single round attention.

        Args:
            attention_strategy: Strategy for selecting object
            use_blip2: Use BLIP2
            blip2_captioner: BLIP2Captioner
            debug: Debug mode
        """
        super().__init__(
            attention_strategy=attention_strategy,
            max_rounds=1,
            stop_on_convergence=False,
            use_blip2=use_blip2,
            blip2_captioner=blip2_captioner,
            debug=debug,
        )
