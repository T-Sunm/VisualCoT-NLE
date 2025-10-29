"""Attention strategies for interactive object selection."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import random


class AttentionStrategy(ABC):
    """Base class for attention strategies."""

    @abstractmethod
    def select_object(
        self,
        question: str,
        objects: List[List],
        query_key: Optional[str] = None,
        **kwargs,
    ) -> int:
        """
        Select object to attend to.

        Args:
            question: Question to answer
            objects: List of objects [conf, name, attrs, ...]
            query_key: Query key for context
            **kwargs: Additional arguments

        Returns:
            Index of selected object
        """
        pass


class LLMAttentionStrategy(AttentionStrategy):
    """Use LLM to select most relevant object."""

    def __init__(
        self,
        object_selector,
        context_manager,
        n_shot: int = 8,
    ):
        """
        Initialize LLM attention strategy.

        Args:
            object_selector: ObjectSelector instance
            context_manager: InteractiveContextManager instance
            n_shot: Number of few-shot examples
        """
        self.object_selector = object_selector
        self.context_manager = context_manager
        self.n_shot = n_shot

    def select_object(
        self,
        question: str,
        objects: List[List],
        query_key: Optional[str] = None,
        **kwargs,
    ) -> int:
        """Select using LLM with few-shot examples."""
        if len(objects) == 0:
            return 0

        # Get examples with object selection
        examples = []
        if query_key and self.context_manager:
            example_keys, rel_objs = (
                self.context_manager.get_interactive_context_examples(
                    query_key=query_key, n_shot=self.n_shot
                )
            )

            # Format examples
            for key, rel_obj_dict in zip(example_keys, rel_objs):
                # Get most relevant object
                if rel_obj_dict:
                    best_obj = max(rel_obj_dict.items(), key=lambda x: x[1])[0]
                    examples.append(
                        {
                            "question": self.context_manager.train_questions.get(
                                key, ""
                            ),
                            "selected_object": best_obj,
                        }
                    )

        # Select object
        return self.object_selector.select_object(
            question=question, objects=objects, examples=examples
        )


class RandomAttentionStrategy(AttentionStrategy):
    """Randomly select object (ablation study)."""

    def select_object(
        self,
        question: str,
        objects: List[List],
        query_key: Optional[str] = None,
        **kwargs,
    ) -> int:
        """Randomly select an object."""
        if len(objects) == 0:
            return 0
        return random.randint(0, len(objects) - 1)


class OracleAttentionStrategy(AttentionStrategy):
    """Use oracle (ground-truth) object relevance."""

    def __init__(self, oracle_attend_dict: Dict[str, Dict[str, float]]):
        """
        Initialize oracle attention.

        Args:
            oracle_attend_dict: Dict mapping key -> {object_name: relevance_score}
        """
        self.oracle_attend_dict = oracle_attend_dict

    def select_object(
        self,
        question: str,
        objects: List[List],
        query_key: Optional[str] = None,
        **kwargs,
    ) -> int:
        """Select using oracle scores."""
        if len(objects) == 0 or not query_key:
            return 0

        if query_key not in self.oracle_attend_dict:
            return 0

        oracle_scores = self.oracle_attend_dict[query_key]

        # Get scores for each object
        obj_scores = []
        for obj in objects:
            obj_name = obj[1]  # Object name at index 1
            score = oracle_scores.get(obj_name, 0.0)
            obj_scores.append(score)

        if not obj_scores:
            return 0

        # Return index of max score
        return obj_scores.index(max(obj_scores))


class AllRegionsAttentionStrategy(AttentionStrategy):
    """Attend to all regions at once (no selection)."""

    def select_object(
        self,
        question: str,
        objects: List[List],
        query_key: Optional[str] = None,
        **kwargs,
    ) -> int:
        """
        Return None to indicate all regions should be used.

        Note: This requires special handling in the caller.
        """
        return -1  # Special value indicating "all regions"
