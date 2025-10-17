"""Unified context manager for example retrieval."""

from typing import Dict, List, Optional, Tuple

from .similarity_retriever import SimilarityRetriever
from .object_similarity import ObjectSimilarityComputer


class ContextManager:
    """Manage context retrieval for both QA and interactive phases."""

    def __init__(
        self,
        similarity_path: str,
        sg_dir: str,
        sg_attr_dir: str,
        train_questions: Dict[str, str],
        train_answers: Dict[str, List[str]],
        train_rationales: Optional[Dict[str, List[str]]] = None,
        dataset_name: str = "aokvqa",
        split: str = "val",
        use_object_similarity: bool = True,
    ):
        """
        Initialize context manager.

        Args:
            similarity_path: Path to CLIP features
            sg_dir: Scene graph directory
            sg_attr_dir: Scene graph attributes directory
            train_questions: Training questions
            train_answers: Training answers
            train_rationales: Training rationales
            dataset_name: Dataset name (aokvqa, okvqa)
            split: Data split (val, test)
            use_object_similarity: Whether to use object similarity
        """
        # Similarity retriever
        self.similarity_retriever = SimilarityRetriever(
            similarity_path=similarity_path,
            dataset_name=dataset_name,
            split=split,
        )

        # Object similarity computer
        self.object_similarity = None
        if use_object_similarity:
            self.object_similarity = ObjectSimilarityComputer(
                sg_dir=sg_dir,
                sg_attr_dir=sg_attr_dir,
                train_questions=train_questions,
                train_answers=train_answers,
                train_rationales=train_rationales,
                use_clip=True,
            )

        self.train_questions = train_questions
        self.train_answers = train_answers
        self.train_rationales = train_rationales

    def load_features(self, metric: str = "imagequestion"):
        """
        Load CLIP features for similarity computation.

        Args:
            metric: Similarity metric (question, imagequestion)
        """
        self.similarity_retriever.load_features(metric)

    def get_qa_context_examples(
        self,
        query_key: str,
        metric: str = "imagequestion",
        n_shot: int = 16,
    ) -> List[str]:
        """
        Get context examples for QA.

        Args:
            query_key: Query key (image_id<->question_id)
            metric: Similarity metric
            n_shot: Number of examples

        Returns:
            List of example keys
        """
        return self.similarity_retriever.get_similar_examples(
            query_key=query_key, metric=metric, n_shot=n_shot
        )

    def get_interactive_context_examples(
        self,
        query_key: str,
        metric: str = "imagequestion",
        n_shot: int = 8,
        object_sim_metric: str = "answer",
    ) -> Tuple[List[str], List[Dict[str, float]]]:
        """
        Get context examples for interactive object selection.

        Args:
            query_key: Query key
            metric: Similarity metric for example retrieval
            n_shot: Number of examples
            object_sim_metric: Metric for object similarity (answer, rationale)

        Returns:
            Tuple of (example_keys, object_similarity_dicts)
        """
        # Get similarity scores for all examples
        similarity_scores = self.similarity_retriever.get_similar_with_scores(
            query_key=query_key, metric=metric
        )

        # Sort by similarity
        sorted_examples = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)

        # Get examples with valid object selections
        example_keys = []
        object_sims = []

        for example_key, score in sorted_examples:
            if example_key == query_key:
                continue

            # Get object similarity for this example
            if self.object_similarity:
                _, _, obj_sim_dict = self.object_similarity.compute_object_similarity(
                    example_key, metric=object_sim_metric
                )

                # Only include examples with at least one object
                if obj_sim_dict:
                    example_keys.append(example_key)
                    object_sims.append(obj_sim_dict)

            if len(example_keys) >= n_shot:
                break

        return example_keys, object_sims

    def get_object_ranking(
        self,
        example_key: str,
        metric: str = "answer",
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Get ranked objects for an example.

        Args:
            example_key: Example key
            metric: Object similarity metric
            top_k: Number of top objects

        Returns:
            List of (object_name, score) tuples
        """
        if not self.object_similarity:
            return []

        return self.object_similarity.get_ranked_objects(
            example_key=example_key, metric=metric, top_k=top_k
        )

    def validate_example(self, example_key: str) -> bool:
        """
        Check if an example is valid (has question and answer).

        Args:
            example_key: Example key

        Returns:
            True if valid
        """
        if example_key not in self.train_questions:
            return False

        if example_key not in self.train_answers:
            return False

        if not self.train_answers[example_key]:
            return False

        if len(self.train_questions[example_key]) == 0:
            return False

        if len(self.train_answers[example_key][0]) == 0:
            return False

        return True

    def filter_valid_examples(self, example_keys: List[str]) -> List[str]:
        """
        Filter out invalid examples.

        Args:
            example_keys: List of example keys

        Returns:
            List of valid example keys
        """
        return [key for key in example_keys if self.validate_example(key)]


class InteractiveContextManager(ContextManager):
    """
    Specialized context manager for interactive object selection.

    This is used in the See phase to select relevant objects.
    """

    def get_examples_with_object_selection(
        self,
        query_key: str,
        n_shot: int = 8,
        metric: str = "imagequestion",
        object_sim_metric: str = "answer",
    ) -> List[Dict[str, str]]:
        """
        Get examples formatted for object selection prompts.

        Args:
            query_key: Query key
            n_shot: Number of examples
            metric: Similarity metric
            object_sim_metric: Object similarity metric

        Returns:
            List of examples with question and selected_object
        """
        example_keys, object_sims = self.get_interactive_context_examples(
            query_key=query_key,
            metric=metric,
            n_shot=n_shot,
            object_sim_metric=object_sim_metric,
        )

        examples = []
        for example_key, obj_sim_dict in zip(example_keys, object_sims):
            if example_key not in self.train_questions:
                continue

            question = self.train_questions[example_key]

            # Get most relevant object
            if obj_sim_dict:
                selected_object = max(obj_sim_dict.items(), key=lambda x: x[1])[0]
                examples.append(
                    {
                        "question": question,
                        "selected_object": selected_object,
                        "example_key": example_key,
                    }
                )

        return examples
