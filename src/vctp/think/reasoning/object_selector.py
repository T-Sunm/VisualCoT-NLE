"""Interactive object selector for Visual CoT."""

from typing import Dict, List, Optional, Tuple

from ..llm import BaseLLMAdapter
from ..prompts import ObjectSelectionPromptBuilder


class ObjectSelector:
    """Select most relevant object from scene graph for a question."""

    def __init__(
        self,
        llm: BaseLLMAdapter,
        engine: str = "gpt3",
        use_attributes: bool = False,
        use_captions: bool = False,
        debug: bool = False,
    ):
        """
        Initialize object selector.

        Args:
            llm: LLM adapter for selection
            engine: Engine type (for prompt building)
            use_attributes: Include object attributes in selection
            use_captions: Include object captions in selection
            debug: Enable debug mode
        """
        self.llm = llm
        self.engine = engine
        self.use_attributes = use_attributes
        self.use_captions = use_captions
        self.debug = debug

        # Prompt builder
        self.prompt_builder = ObjectSelectionPromptBuilder(
            engine=engine, use_attributes=use_attributes, use_captions=use_captions
        )

    def select_object(
        self,
        question: str,
        objects: List[List],
        examples: Optional[List[Dict]] = None,
    ) -> int:
        """
        Select most relevant object for a question.

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

        # Build prompt
        prompt = self.prompt_builder.build(
            question=question, object_list=object_names, examples=examples
        )

        if self.debug:
            print(f"[ObjectSelector] Prompt:\n{prompt}")

        # Select based on engine type
        if self.engine in ["ada", "babbage", "curie", "davinci", "codex", "instruct", "gpt3"]:
            return self._select_with_gpt3(prompt, object_names)
        elif self.engine == "chat":
            return self._select_with_chat(prompt, object_names)
        elif self.engine in ["opt", "llama"]:
            return self._select_with_local_model(prompt, object_names)
        elif self.engine == "chat-test":
            return 0  # Test mode
        else:
            return 0

    def _select_with_gpt3(self, prompt: str, object_names: List[str]) -> int:
        """Select using GPT-3 Completion API with logit bias."""
        # Get token IDs for object names
        try:
            from transformers import GPT2Tokenizer
        except ImportError:
            return 0

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        obj_token_ids = [tokenizer.encode(f" {obj}")[0] for obj in object_names]

        # Create logit bias to force selection from objects
        logit_bias = {str(tok_id): 100 for tok_id in obj_token_ids}

        # Generate with logit bias
        response = self.llm.generate(
            prompt=prompt, max_tokens=4, logit_bias=logit_bias, stop_tokens=["\n", "<|endoftext|>"]
        )

        # Parse result
        tokenizer_response = GPT2Tokenizer.from_pretrained("gpt2")
        result_tokens = tokenizer_response.encode(response.text)

        if len(result_tokens) > 0 and result_tokens[0] in obj_token_ids:
            return obj_token_ids.index(result_tokens[0])

        return 0

    def _select_with_chat(self, prompt: str, object_names: List[str]) -> int:
        """Select using ChatGPT API with logit bias."""
        try:
            import tiktoken

            tokenizer = tiktoken.encoding_for_model(self.llm.config.engine_name)
        except ImportError:
            return 0

        # Get token IDs for object names
        obj_token_ids = [tokenizer.encode(f" {obj}")[0] for obj in object_names]

        # Create logit bias
        logit_bias = {str(tok_id): 25 for tok_id in obj_token_ids}

        # Get system prompt from builder
        system_prompt = self.prompt_builder.OBJECT_SELECTION_SYSTEM_PROMPT

        # Generate
        response = self.llm.generate(
            prompt=prompt, max_tokens=5, logit_bias=logit_bias, system_prompt=system_prompt
        )

        # Parse result
        result_tokens = tokenizer.encode(response.text)

        if len(result_tokens) > 0 and result_tokens[0] in obj_token_ids:
            return obj_token_ids.index(result_tokens[0])

        return 0

    def _select_with_local_model(self, prompt: str, object_names: List[str]) -> int:
        """Select using local models (OPT, LLaMA)."""
        # Get tokenizer from LLM
        if not hasattr(self.llm, "tokenizer"):
            return 0

        tokenizer = self.llm.tokenizer

        # Get token IDs for object names
        obj_token_ids = [tokenizer.encode(f" {obj}")[1] for obj in object_names]

        # Use special method if available
        if hasattr(self.llm, "generate_with_object_selection"):
            return self.llm.generate_with_object_selection(prompt, obj_token_ids)

        # Otherwise generate and parse
        response = self.llm.generate(prompt=prompt, max_tokens=5)

        # Try to match response text with object names
        result_str = response.text.split("\n")[0].strip()
        result_str = result_str[:-1] if result_str.endswith(".") else result_str

        for obj_id, obj_name in enumerate(object_names):
            if result_str in obj_name or obj_name in result_str:
                return obj_id

        # Default: use first token scores
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

        for _ in range(n_rounds):
            if len(remaining_objects) == 0:
                break

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
