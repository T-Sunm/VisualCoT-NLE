"""Prompt builders for constructing few-shot prompts."""

from typing import List, Dict, Optional, Tuple
import random

from . import templates


class ObjectSelectionPromptBuilder:
    """Builder for object selection prompts (interactive phase)."""

    def __init__(
        self, engine: str = "chat", use_attributes: bool = False, use_captions: bool = False
    ):
        """
        Initialize object selection prompt builder.

        Args:
            engine: LLM engine type (chat, gpt3, opt, etc.)
            use_attributes: Whether to include object attributes in options
            use_captions: Whether to include captions in options
        """
        self.engine = engine
        self.use_attributes = use_attributes
        self.use_captions = use_captions
        self.is_chat = engine in ["chat", "chat-test"]

    def build(
        self,
        question: str,
        object_list: List[str],
        examples: Optional[List[Dict]] = None,
    ) -> str:
        """
        Build object selection prompt.

        Args:
            question: The question to answer
            object_list: List of object names to choose from
            examples: Few-shot examples with 'question' and 'selected_object'

        Returns:
            Formatted prompt string
        """
        if self.is_chat:
            return self._build_chat_prompt(question, object_list, examples)
        else:
            return self._build_completion_prompt(question, object_list, examples)

    def _build_chat_prompt(
        self, question: str, object_list: List[str], examples: Optional[List[Dict]]
    ) -> str:
        """Build prompt for chat models."""
        prompt = templates.OBJECT_SELECTION_SYSTEM_PROMPT
        prompt += "===\n"

        # Add few-shot examples
        if examples:
            for example in examples:
                prompt += templates.OBJECT_SELECTION_EXAMPLE_CHAT.format(
                    question=example["question"], object_name=example["selected_object"]
                )

        # Add query
        prompt += templates.OBJECT_SELECTION_QUERY_CHAT.format(
            question=question, object_list=", ".join(object_list)
        )

        return prompt

    def _build_completion_prompt(
        self, question: str, object_list: List[str], examples: Optional[List[Dict]]
    ) -> str:
        """Build prompt for completion models."""
        prompt = templates.OBJECT_SELECTION_COMPLETION_PROMPT

        # Add few-shot examples
        if examples:
            for example in examples:
                prompt += templates.OBJECT_SELECTION_EXAMPLE_COMPLETION.format(
                    question=example["question"], object_name=example["selected_object"]
                )

        # Add query with options
        prompt += templates.OBJECT_SELECTION_QUERY_COMPLETION.format(question=question)

        return prompt

    def format_object_options(
        self, objects: List[List], use_attributes: bool = False, use_captions: bool = False
    ) -> List[str]:
        """
        Format object list for display in prompt.

        Args:
            objects: List of objects [conf, name, attrs, caption, ...]
            use_attributes: Include attributes
            use_captions: Include captions

        Returns:
            List of formatted object strings
        """
        formatted = []

        for obj in objects:
            if use_captions and len(obj) >= 4:
                # Format: "object_name: caption"
                formatted.append(f"{obj[1]}: {obj[3]}")
            elif use_attributes and len(obj) >= 3 and obj[2]:
                # Format: "object_name: attr1 attr2 object_name"
                formatted.append(f"{obj[1]}: {' '.join(obj[2])} {obj[1]}")
            else:
                # Just object name
                formatted.append(obj[1])

        return formatted


class QuestionAnsweringPromptBuilder:
    """Builder for question answering prompts with chain-of-thought."""

    def __init__(
        self,
        engine: str = "chat",
        chain_of_thoughts: bool = True,
        choice_only: bool = False,
        remove_caption: bool = False,
    ):
        """
        Initialize QA prompt builder.

        Args:
            engine: LLM engine type
            chain_of_thoughts: Whether to use chain-of-thought reasoning
            choice_only: Whether to use multiple choice format
            remove_caption: Whether to remove caption from context
        """
        self.engine = engine
        self.chain_of_thoughts = chain_of_thoughts
        self.choice_only = choice_only
        self.remove_caption = remove_caption
        self.is_chat = engine in ["chat", "chat-test"]

    def build(
        self,
        question: str,
        context: str,
        scene_graph_text: str,
        choices: Optional[List[str]] = None,
        examples: Optional[List[Dict]] = None,
        thoughts: Optional[List[str]] = None,
    ) -> str:
        """
        Build question answering prompt.

        Args:
            question: The question to answer
            context: Global image context/caption
            scene_graph_text: Scene graph description
            choices: Multiple choice options
            examples: Few-shot examples
            thoughts: Previous thoughts for iterative reasoning

        Returns:
            Formatted prompt string
        """
        if self.is_chat:
            prompt = templates.QA_SYSTEM_PROMPT_CHAT + "===\n"
            answer_prefix = templates.QA_ANSWER_PREFIX_CHAT
        else:
            prompt = templates.QA_COMPLETION_PROMPT
            answer_prefix = templates.QA_ANSWER_PREFIX_COMPLETION

        # Add few-shot examples
        if examples:
            for example in examples:
                prompt += self._format_example(
                    context=example.get("context", ""),
                    question=example["question"],
                    answer=example["answer"],
                    rationale=example.get("rationale", ""),
                    choices=example.get("choices"),
                    answer_prefix=answer_prefix,
                )

        # Prepare current context
        current_context = "" if self.remove_caption else context

        # Add thoughts to context
        if thoughts and len(thoughts) > 0:
            valid_thoughts = [t for t in thoughts if t]
            if valid_thoughts:
                current_context += "\n" + " ".join(valid_thoughts)

        # Format choices
        choice_text = ""
        if self.choice_only and choices:
            choice_text = f"\nChoices: {', '.join(choices)}."

        # Add query
        if scene_graph_text:
            full_context = f"{current_context}\n{scene_graph_text}"
        else:
            full_context = current_context

        prompt += f"Context: {full_context}\n===\n"
        prompt += f"Question: {question}{choice_text}\n{answer_prefix}"

        return prompt

    def _format_example(
        self,
        context: str,
        question: str,
        answer: str,
        rationale: str,
        choices: Optional[List[str]],
        answer_prefix: str,
    ) -> str:
        """Format a single few-shot example."""
        choice_text = ""
        if self.choice_only and choices:
            choice_text = f"\nChoices: {', '.join(choices)}."

        if self.chain_of_thoughts and rationale:
            return templates.QA_EXAMPLE_WITH_COT.format(
                context=context,
                question=question,
                choices=choice_text,
                answer_prefix=answer_prefix,
                answer=answer,
                rationale=rationale,
            )
        else:
            return templates.QA_EXAMPLE_WITHOUT_COT.format(
                context=context,
                question=question,
                choices=choice_text,
                answer_prefix=answer_prefix,
                answer=answer,
            )


class BLIP2PromptBuilder:
    """Builder for BLIP2-specific prompts."""

    @staticmethod
    def build_global_caption_prompt(question: str) -> Tuple[str, str]:
        """
        Build prompts for global image captioning.

        Args:
            question: Question context

        Returns:
            Tuple of (general_caption_prompt, question_caption_prompt)
        """
        general_prompt = templates.BLIP2_GLOBAL_CAPTION_PROMPT
        question_prompt = templates.BLIP2_GLOBAL_CAPTION_QUESTION_PROMPT.format(question=question)
        return general_prompt, question_prompt

    @staticmethod
    def build_local_caption_prompt(object_name: str, question: str) -> str:
        """
        Build prompt for local object captioning.

        Args:
            object_name: Name of object to caption
            question: Question context

        Returns:
            Caption prompt
        """
        return templates.BLIP2_LOCAL_CAPTION_PROMPT.format(object_name=object_name)

    @staticmethod
    def build_followup_question_prompt(
        object_name: str, observation: str, main_question: str, engine: str = "chat"
    ) -> str:
        """
        Build prompt for generating follow-up questions.

        Args:
            object_name: Object being examined
            observation: Initial observation about the object
            main_question: Main question being answered
            engine: LLM engine type

        Returns:
            Follow-up question generation prompt
        """
        if engine == "chat":
            return templates.BLIP2_FOLLOWUP_USER_PROMPT.format(
                object_name=object_name, observation=observation, main_question=main_question
            )
        else:
            return templates.BLIP2_FOLLOWUP_COMPLETION_PROMPT.format(
                object_name=object_name, observation=observation, main_question=main_question
            )

    @staticmethod
    def build_verify_thought_prompt(thought: str) -> str:
        """
        Build prompt to verify if thought matches image.

        Args:
            thought: Reasoning/thought to verify

        Returns:
            Verification prompt
        """
        return templates.BLIP2_VERIFY_THOUGHT_PROMPT.format(thought=thought)

    @staticmethod
    def build_correct_thought_prompt(thought: str) -> str:
        """
        Build prompt to correct a thought based on image.

        Args:
            thought: Thought to correct

        Returns:
            Correction prompt
        """
        return templates.BLIP2_CORRECT_THOUGHT_PROMPT.format(thought=thought)

    @staticmethod
    def build_detect_object_prompt(existing_objects: Optional[List[str]] = None) -> str:
        """
        Build prompt for object detection.

        Args:
            existing_objects: Objects already detected

        Returns:
            Object detection prompt
        """
        if not existing_objects:
            return templates.BLIP2_DETECT_OBJECT_FIRST_PROMPT
        else:
            # Format existing objects list
            obj_str = ""
            for i, obj in enumerate(existing_objects):
                obj_str += f" {obj}"
                if i < len(existing_objects) - 1:
                    obj_str += ","
                else:
                    obj_str += "?"

            return templates.BLIP2_DETECT_OBJECT_BESIDES_PROMPT.format(existing_objects=obj_str)
