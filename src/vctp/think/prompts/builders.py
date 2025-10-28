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

    def build_structured(
        self,
        question: str,
        object_list: List[str],
        examples: Optional[List[Dict]] = None,
    ) -> Tuple[str, str]:
        """
        Build object selection prompt for structured output (separate system & user messages).

        Args:
            question: The question to answer
            object_list: List of object names to choose from
            examples: Few-shot examples with 'question' and 'selected_object'

        Returns:
            Tuple of (system_message, user_message)
        """
        # System message - chỉ lấy phần instruction
        system_msg = templates.OBJECT_SELECTION_SYSTEM_PROMPT.strip()

        # User message - bao gồm examples và query
        user_msg = ""

        # Add few-shot examples nếu có
        if examples:
            for example in examples:
                user_msg += templates.OBJECT_SELECTION_EXAMPLE_CHAT.format(
                    question=example["question"], object_name=example["selected_object"]
                )

        # Add query - format cho structured output
        user_msg += f"""Question: {question}
        ===
        Options: {', '.join(object_list)}

        Please respond in JSON format with:
        - selected_object: the name of the most related object from the options
        - reasoning: brief explanation of why this object is most relevant to answer the question"""

        return system_msg, user_msg


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
        self.is_chat = engine in ["chat", "chat-test", "groq"]

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
                    confidence=example.get("confidence", round(random.random(), 2)),
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
        confidence: float,
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
                confidence=confidence,
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


# if __name__ == "__main__":

#     import sys
#     from pathlib import Path

#     # Thêm thư mục src vào Python path
#     PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
#     sys.path.insert(0, str(PROJECT_ROOT / "src"))
#     print("=" * 80)
#     print("TESTING PROMPT BUILDERS")
#     print("=" * 80)

#     # ========================================================================
#     # Test 1: Object Selection Prompt Builder
#     # ========================================================================
#     print("\n" + "=" * 80)
#     print("TEST 1: Object Selection Prompt Builder (Chat Mode)")
#     print("=" * 80)

#     obj_builder_chat = ObjectSelectionPromptBuilder(engine="chat", use_attributes=False)

#     test_question = "What is the man doing?"
#     test_objects = ["man", "racket", "court", "ball", "net"]
#     test_examples = [
#         {"question": "What color is the car?", "selected_object": "car"},
#         {"question": "Where is the dog sitting?", "selected_object": "dog"},
#     ]

#     prompt_chat = obj_builder_chat.build(
#         question=test_question, object_list=test_objects, examples=test_examples
#     )

#     print("\n--- Input ---")
#     print(f"Question: {test_question}")
#     print(f"Objects: {test_objects}")
#     print(f"Examples: {len(test_examples)} examples")

#     print("\n--- Output Prompt ---")
#     print(prompt_chat)

#     # ========================================================================
#     # Test 2: Object Selection with Attributes
#     # ========================================================================
#     print("\n" + "=" * 80)
#     print("TEST 2: Object Selection with Attributes")
#     print("=" * 80)

#     obj_builder_attr = ObjectSelectionPromptBuilder(engine="chat", use_attributes=True)

#     # Format: [confidence, name, attributes, caption]
#     test_objects_full = [
#         [0.95, "man", ["standing", "playing"], "a man holding a racket"],
#         [0.87, "racket", ["green", "tennis"], "a green tennis racket"],
#         [0.82, "court", ["red", "clay"], "a clay tennis court"],
#     ]

#     formatted_objects = obj_builder_attr.format_object_options(
#         test_objects_full, use_attributes=True, use_captions=False
#     )

#     print("\n--- Input Objects (with attributes) ---")
#     for obj in test_objects_full:
#         print(f"  {obj}")

#     print("\n--- Formatted Options ---")
#     for i, opt in enumerate(formatted_objects):
#         print(f"  {i+1}. {opt}")

#     # ========================================================================
#     # Test 3: Question Answering Prompt Builder (Chat Mode with CoT)
#     # ========================================================================
#     print("\n" + "=" * 80)
#     print("TEST 3: Question Answering Prompt Builder (Chat + CoT)")
#     print("=" * 80)

#     qa_builder = QuestionAnsweringPromptBuilder(
#         engine="chat", chain_of_thoughts=True, choice_only=True
#     )

#     test_context = "A man playing tennis on clay court"
#     test_scene_graph = "man is standing. racket is green. court is red."
#     test_choices = ["playing tennis", "running", "sitting"]
#     test_thoughts = ["The man holds a racket"]
#     test_qa_examples = [
#         {
#             "question": "What sport is being played?",
#             "answer": "tennis",
#             "context": "A player on a court with a racket",
#             "rationale": "The presence of a racket and court indicates tennis.",
#             "choices": ["tennis", "badminton", "squash"],
#         }
#     ]

#     qa_prompt = qa_builder.build(
#         question=test_question,
#         context=test_context,
#         scene_graph_text=test_scene_graph,
#         choices=test_choices,
#         examples=test_qa_examples,
#         thoughts=test_thoughts,
#     )

#     print("\n--- Input ---")
#     print(f"Question: {test_question}")
#     print(f"Context: {test_context}")
#     print(f"Scene Graph: {test_scene_graph}")
#     print(f"Choices: {test_choices}")
#     print(f"Thoughts: {test_thoughts}")
#     print(f"Examples: {len(test_qa_examples)} examples")

#     print("\n--- Output Prompt ---")
#     print(qa_prompt)

#     # ========================================================================
#     # Test 4: BLIP2 Prompt Builder
#     # ========================================================================
#     print("\n" + "=" * 80)
#     print("TEST 4: BLIP2 Prompt Builder")
#     print("=" * 80)

#     print("\n--- 4.1: Global Caption Prompts ---")
#     general_prompt, question_prompt = BLIP2PromptBuilder.build_global_caption_prompt(test_question)
#     print(f"General: {general_prompt}")
#     print(f"Question-aware: {question_prompt}")

#     print("\n--- 4.2: Local Caption Prompt ---")
#     local_prompt = BLIP2PromptBuilder.build_local_caption_prompt("man", test_question)
#     print(f"Local: {local_prompt}")

#     print("\n--- 4.3: Follow-up Question Prompt ---")
#     followup_prompt = BLIP2PromptBuilder.build_followup_question_prompt(
#         object_name="racket",
#         observation="it is green",
#         main_question=test_question,
#         engine="chat",
#     )
#     print(f"Follow-up: {followup_prompt}")

#     print("\n--- 4.4: Verify Thought Prompt ---")
#     verify_prompt = BLIP2PromptBuilder.build_verify_thought_prompt("the man is playing tennis")
#     print(f"Verify: {verify_prompt}")

#     print("\n--- 4.5: Object Detection Prompts ---")
#     detect_first = BLIP2PromptBuilder.build_detect_object_prompt()
#     print(f"Detect (first): {detect_first}")

#     detect_besides = BLIP2PromptBuilder.build_detect_object_prompt(["man", "racket"])
#     print(f"Detect (besides): {detect_besides}")

#     # ========================================================================
#     # Test 5: Completion Mode (non-chat)
#     # ========================================================================
#     print("\n" + "=" * 80)
#     print("TEST 5: Completion Mode (GPT-3 style)")
#     print("=" * 80)

#     obj_builder_completion = ObjectSelectionPromptBuilder(engine="gpt3")
#     completion_prompt = obj_builder_completion.build(
#         question=test_question, object_list=test_objects, examples=test_examples[:1]
#     )

#     print("\n--- Output Prompt (Completion) ---")
#     print(completion_prompt)

#     # ========================================================================
#     # Summary
#     # ========================================================================
#     print("\n" + "=" * 80)
#     print("EXPECTED OUTPUT VALIDATION")
#     print("=" * 80)
#     print(
#         """
# ✓ Object Selection Prompt: Should contain system prompt + examples + query
# ✓ Formatted Options: Should include attributes when use_attributes=True
# ✓ QA Prompt: Should include context + scene graph + choices + thoughts
# ✓ BLIP2 Prompts: Should format correctly with object names and questions
# ✓ Chat vs Completion: Different formats for different engines

# All prompts should be well-formatted strings ready for LLM input.
#     """
#     )

#     print("\n" + "=" * 80)
#     print("TEST COMPLETE")
#     print("=" * 80)
