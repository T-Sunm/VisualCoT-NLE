"""Prompt engineering utilities for Visual CoT."""

from .templates import (
    # Object Selection
    OBJECT_SELECTION_SYSTEM_PROMPT,
    OBJECT_SELECTION_COMPLETION_PROMPT,
    # Question Answering
    QA_SYSTEM_PROMPT_CHAT,
    QA_COMPLETION_PROMPT,
    QA_ANSWER_PREFIX_CHAT,
    QA_ANSWER_PREFIX_COMPLETION,
    # BLIP2
    BLIP2_GLOBAL_CAPTION_PROMPT,
    BLIP2_LOCAL_CAPTION_PROMPT,
    BLIP2_VERIFY_THOUGHT_PROMPT,
    BLIP2_DETECT_OBJECT_FIRST_PROMPT,
)

from .builders import (
    ObjectSelectionPromptBuilder,
    QuestionAnsweringPromptBuilder,
    BLIP2PromptBuilder,
)

from .formatters import (
    process_answer,
    parse_sentence,
    parse_object_name,
    extract_answer_and_rationale,
    extract_logprobs_until_stop,
    make_choices_text,
    compute_vqa_score,
    filter_thoughts_by_similarity,
)

from .examples import (
    FewShotExamplesManager,
    ObjectSelectionExamplesManager,
)

__all__ = [
    # Templates
    "OBJECT_SELECTION_SYSTEM_PROMPT",
    "OBJECT_SELECTION_COMPLETION_PROMPT",
    "QA_SYSTEM_PROMPT_CHAT",
    "QA_COMPLETION_PROMPT",
    "QA_ANSWER_PREFIX_CHAT",
    "QA_ANSWER_PREFIX_COMPLETION",
    "BLIP2_GLOBAL_CAPTION_PROMPT",
    "BLIP2_LOCAL_CAPTION_PROMPT",
    "BLIP2_VERIFY_THOUGHT_PROMPT",
    "BLIP2_DETECT_OBJECT_FIRST_PROMPT",
    # Builders
    "ObjectSelectionPromptBuilder",
    "QuestionAnsweringPromptBuilder",
    "BLIP2PromptBuilder",
    # Formatters
    "process_answer",
    "parse_sentence",
    "parse_object_name",
    "extract_answer_and_rationale",
    "extract_logprobs_until_stop",
    "make_choices_text",
    "compute_vqa_score",
    "filter_thoughts_by_similarity",
    # Examples
    "FewShotExamplesManager",
    "ObjectSelectionExamplesManager",
]
