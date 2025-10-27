from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel
from groq import Groq
import json
import os

from ..prompts import ObjectSelectionPromptBuilder


class ObjectSelection(BaseModel):
    """Structured response for object selection."""

    selected_object: str
    reasoning: str


class ObjectSelector:
    """Select most relevant object from scene graph for a question."""

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        use_attributes: bool = False,
        use_captions: bool = False,
        debug: bool = False,
    ):
        """
        Initialize object selector with GroqCloud.

        Args:
            model: Groq model name (e.g., "llama-3.3-70b-versatile", "moonshotai/kimi-k2-instruct-0905")
            use_attributes: Include object attributes in selection
            use_captions: Include object captions in selection
            debug: Enable debug mode
        """
        self.model = model
        self.use_attributes = use_attributes
        self.use_captions = use_captions
        self.debug = debug

        # Initialize Groq client
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        # Prompt builder (use "chat" engine for Groq)
        self.prompt_builder = ObjectSelectionPromptBuilder(
            engine="chat", use_attributes=use_attributes, use_captions=use_captions
        )

    def select_object(
        self,
        question: str,
        objects: List[List],
        examples: Optional[List[Dict]] = None,
    ) -> int:
        """
        Select most relevant object for a question using Groq structured output.

        Args:
            question: Question to answer
            objects: List of objects [conf, name, attrs, caption, ...]
            examples: Few-shot examples with question and selected_object

        Returns:
            Index of selected object in objects list
        """
        if len(objects) == 0:
            return 0

        # Format object list for prompt
        object_names = [obj[1] for obj in objects]

        # Build prompt - this returns the user message content
        user_prompt = self.prompt_builder.build(
            question=question, object_list=object_names, examples=examples
        )

        if self.debug:
            print(f"[ObjectSelector] Object choices: {object_names}")
            print(f"[ObjectSelector] User Prompt:\n{user_prompt}")

        try:
            # Call Groq with structured output
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing visual scenes and selecting the most relevant object for answering questions. Choose the single most important object from the provided list that would help answer the question.",
                    },
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "object_selection",
                        "schema": ObjectSelection.model_json_schema(),
                    },
                },
                temperature=0.1,  # Low temperature for consistent selection
            )

            # Parse structured response
            result = ObjectSelection.model_validate(json.loads(response.choices[0].message.content))

            if self.debug:
                print(f"[ObjectSelector] Selected: {result.selected_object}")
                print(f"[ObjectSelector] Reasoning: {result.reasoning}")

            # Find index of selected object
            selected_name = result.selected_object.strip().lower()
            for idx, obj_name in enumerate(object_names):
                if obj_name.lower() == selected_name or selected_name in obj_name.lower():
                    return idx

            # Fallback: try partial matching
            for idx, obj_name in enumerate(object_names):
                if obj_name.lower() in selected_name or selected_name in obj_name.lower():
                    return idx

            if self.debug:
                print(
                    f"[ObjectSelector] Warning: Could not find exact match for '{result.selected_object}', using first object"
                )

            return 0

        except Exception as e:
            if self.debug:
                print(f"[ObjectSelector] Error during selection: {e}")
            return 0

    def select_multiple_rounds(
        self,
        question: str,
        objects: List[List],
        examples: Optional[List[Dict]] = None,
        n_rounds: int = 3,
    ) -> List[Tuple[int, List]]:
        """
        Select objects for multiple reasoning rounds.

        Args:
            question: Question to answer
            objects: List of objects
            examples: Few-shot examples
            n_rounds: Number of rounds

        Returns:
            List of (selected_index, selected_object) tuples
        """
        selected = []
        remaining_objects = objects.copy()

        for round_num in range(n_rounds):
            if len(remaining_objects) == 0:
                break

            if self.debug:
                print(f"[ObjectSelector] Round {round_num + 1}/{n_rounds}")

            # Select next object
            idx = self.select_object(question, remaining_objects, examples)

            # Store selection
            selected.append((idx, remaining_objects[idx]))

            # Remove from remaining
            remaining_objects = remaining_objects[:idx] + remaining_objects[idx + 1 :]

        return selected


class RandomObjectSelector:
    """Random object selector for ablation studies."""

    def select_object(
        self, question: str, objects: List[List], examples: Optional[List[Dict]] = None
    ) -> int:
        """Randomly select an object."""
        import random

        return random.randint(0, len(objects) - 1) if objects else 0


class OracleObjectSelector:
    """Oracle object selector using ground-truth similarity."""

    def __init__(self, oracle_attend_dict: Dict[str, Dict[str, float]]):
        """
        Initialize oracle selector.

        Args:
            oracle_attend_dict: Dict mapping key -> {object_name: score}
        """
        self.oracle_attend_dict = oracle_attend_dict

    def select_object(
        self,
        question: str,
        objects: List[List],
        query_key: str,
        examples: Optional[List[Dict]] = None,
    ) -> int:
        """
        Select object using oracle scores.

        Args:
            question: Question (not used)
            objects: List of objects
            query_key: Query key for oracle lookup
            examples: Examples (not used)

        Returns:
            Index of best object according to oracle
        """
        if query_key not in self.oracle_attend_dict:
            return 0

        oracle_scores = self.oracle_attend_dict[query_key]

        # Get scores for current objects
        obj_scores = []
        for obj in objects:
            obj_name = obj[1]
            score = oracle_scores.get(obj_name, 0.0)
            obj_scores.append(score)

        if not obj_scores:
            return 0

        # Return index of max score
        return obj_scores.index(max(obj_scores))
