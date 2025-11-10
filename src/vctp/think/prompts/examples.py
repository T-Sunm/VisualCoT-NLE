"""Few-shot examples manager for in-context learning."""

from typing import List, Dict, Optional, Tuple
import random


class FewShotExamplesManager:
    """Manage few-shot examples for in-context learning."""

    def __init__(
        self,
        train_questions: Dict[str, str],
        train_answers: Dict[str, List[str]],
        train_rationales: Optional[Dict[str, List[str]]] = None,
        train_choices: Optional[Dict[str, List[str]]] = None,
        train_captions: Optional[Dict[int, List[str]]] = None,
    ):
        """
        Initialize examples manager.

        Args:
            train_questions: Dictionary of question_id -> question
            train_answers: Dictionary of question_id -> list of answers
            train_rationales: Dictionary of question_id -> list of rationales
            train_choices: Dictionary of question_id -> list of choices
            train_captions: Dictionary of image_id -> list of captions
        """
        self.train_questions = train_questions
        self.train_answers = train_answers
        self.train_rationales = train_rationales or {}
        self.train_choices = train_choices or {}
        self.train_captions = train_captions or {}

        self.train_keys = list(train_questions.keys())

    def get_random_examples(
        self, n_shot: int, exclude_keys: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get random few-shot example keys.

        Args:
            n_shot: Number of examples to retrieve
            exclude_keys: Keys to exclude from selection

        Returns:
            List of example keys
        """
        available_keys = self.train_keys.copy()

        if exclude_keys:
            available_keys = [k for k in available_keys if k not in exclude_keys]

        if len(available_keys) < n_shot:
            return available_keys

        return random.sample(available_keys, n_shot)

    def get_examples_by_similarity(
        self, query_key: str, similarity_scores: Dict[str, float], n_shot: int
    ) -> List[str]:
        """
        Get examples based on similarity scores.

        Args:
            query_key: Query key (usually image_id<->question_id)
            similarity_scores: Dictionary of key -> similarity score
            n_shot: Number of examples to retrieve

        Returns:
            List of example keys sorted by similarity
        """
        # Sort by similarity
        sorted_keys = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)

        # Get top-k
        example_keys = []
        for key, score in sorted_keys:
            if key == query_key:
                continue
            if key in self.train_keys:
                example_keys.append(key)
            if len(example_keys) >= n_shot:
                break

        return example_keys

    def format_example(
        self,
        example_key: str,
        include_rationale: bool = True,
        include_choices: bool = False,
        random_caption: bool = False,
    ) -> Dict[str, str]:
        """
        Format a single example for prompt.

        Args:
            example_key: Example key (image_id<->question_id)
            include_rationale: Include rationale/reasoning
            include_choices: Include multiple choices
            random_caption: Use random caption instead of matched one

        Returns:
            Dictionary with formatted example fields
        """
        # Extract image ID
        img_id = int(example_key.split("<->")[0])

        # Get caption
        if random_caption and self.train_captions:
            caption_list = random.choice(list(self.train_captions.values()))
            caption = random.choice(caption_list)
        elif img_id in self.train_captions:
            caption = random.choice(self.train_captions[img_id])
        else:
            caption = ""

        # Get question and answer
        question = self.train_questions[example_key]
        answer = self.train_answers[example_key][0] if self.train_answers[example_key] else ""

        # Get rationale
        rationale = ""
        if include_rationale and example_key in self.train_rationales:
            rationale_list = self.train_rationales[example_key]
            rationale = random.choice(rationale_list)
        # Get choices
        choices = None
        if include_choices and example_key in self.train_choices:
            choices = self.train_choices[example_key]

        return {
            "context": caption,
            "question": question,
            "answer": answer,
            "rationale": rationale,
            "choices": choices,
        }

    def format_examples_batch(
        self,
        example_keys: List[str],
        include_rationale: bool = True,
        include_choices: bool = False,
        random_caption: bool = False,
    ) -> List[Dict[str, str]]:
        """
        Format multiple examples for prompt.

        Args:
            example_keys: List of example keys
            include_rationale: Include rationales
            include_choices: Include multiple choices
            random_caption: Use random captions

        Returns:
            List of formatted examples
        """
        examples = []

        for key in example_keys:
            # Skip invalid examples
            if key not in self.train_questions:
                continue
            if not self.train_answers.get(key):
                continue

            example = self.format_example(
                key,
                include_rationale=include_rationale,
                include_choices=include_choices,
                random_caption=random_caption,
            )

            examples.append(example)

        return examples


class ObjectSelectionExamplesManager:
    """Manage examples for object selection prompts."""

    def __init__(
        self,
        train_questions: Dict[str, str],
        train_object_similarity: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """
        Initialize object selection examples manager.

        Args:
            train_questions: Dictionary of question_id -> question
            train_object_similarity: Dictionary of key -> {object: similarity}
        """
        self.train_questions = train_questions
        self.train_object_similarity = train_object_similarity or {}
        self.train_keys = list(train_questions.keys())

    def get_examples_with_object_selection(self, example_keys: List[str]) -> List[Dict[str, str]]:
        """
        Get examples with selected objects.

        Args:
            example_keys: List of example keys

        Returns:
            List of examples with question and selected_object
        """
        examples = []

        for key in example_keys:
            if key not in self.train_questions:
                continue

            question = self.train_questions[key]

            # Get most relevant object
            selected_object = None
            if key in self.train_object_similarity:
                obj_scores = self.train_object_similarity[key]
                if obj_scores:
                    # Get object with highest score
                    selected_object = max(obj_scores.items(), key=lambda x: x[1])[0]

            if selected_object:
                examples.append({"question": question, "selected_object": selected_object})

        return examples

    def filter_examples_with_valid_selections(
        self, example_keys: List[str], min_examples: int = 1
    ) -> Tuple[List[str], List[Dict[str, Dict[str, float]]]]:
        """
        Filter examples that have valid object selections.

        Args:
            example_keys: Candidate example keys
            min_examples: Minimum number of examples needed

        Returns:
            Tuple of (filtered_keys, object_similarities)
        """
        filtered_keys = []
        object_sims = []

        for key in example_keys:
            if key in self.train_object_similarity:
                obj_sim = self.train_object_similarity[key]
                if obj_sim:  # Has at least one object
                    filtered_keys.append(key)
                    object_sims.append(obj_sim)

            if len(filtered_keys) >= min_examples:
                break

        return filtered_keys, object_sims
