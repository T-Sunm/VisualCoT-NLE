"""Think module - Reasoning and inference for Visual CoT."""

# LLM Adapters
from .llm import (
    BaseLLMAdapter,
    LLMResponse,
    ChatMessage,
    LLMConfig,
    OpenAIGPT3Adapter,
    OpenAIChatAdapter,
    OPTAdapter,
    LLaMAAdapter,
    create_llm_adapter,
    create_llm_from_args,
)

# Prompts
from .prompts import (
    ObjectSelectionPromptBuilder,
    QuestionAnsweringPromptBuilder,
    BLIP2PromptBuilder,
    FewShotExamplesManager,
    ObjectSelectionExamplesManager,
    process_answer,
    extract_answer_and_rationale,
    compute_vqa_score,
)

# Context Retrieval
from .context import (
    SimilarityRetriever,
    ObjectSimilarityComputer,
    ContextManager,
    InteractiveContextManager,
)

# Reasoning Components
from .reasoning import (
    ObjectSelector,
    RandomObjectSelector,
    OracleObjectSelector,
    QuestionAnswerer,
    EnsembleQuestionAnswerer,
    ThoughtVerifier,
    OracleThoughtVerifier,
    RandomThoughtVerifier,
)

# Interactive Attention
from .interactive import (
    AttentionStrategy,
    LLMAttentionStrategy,
    RandomAttentionStrategy,
    OracleAttentionStrategy,
    AllRegionsAttentionStrategy,
    InteractiveAttention,
    InteractiveLoop,
)

# Main Reasoner
from .reasoner import NoOpReasoner, VisualCoTReasoner

__all__ = [
    # LLM
    "BaseLLMAdapter",
    "LLMResponse",
    "ChatMessage",
    "LLMConfig",
    "OpenAIGPT3Adapter",
    "OpenAIChatAdapter",
    "OPTAdapter",
    "LLaMAAdapter",
    "create_llm_adapter",
    "create_llm_from_args",
    # Prompts
    "ObjectSelectionPromptBuilder",
    "QuestionAnsweringPromptBuilder",
    "BLIP2PromptBuilder",
    "FewShotExamplesManager",
    "ObjectSelectionExamplesManager",
    "process_answer",
    "extract_answer_and_rationale",
    "compute_vqa_score",
    # Context
    "SimilarityRetriever",
    "ObjectSimilarityComputer",
    "ContextManager",
    "InteractiveContextManager",
    # Reasoning
    "ObjectSelector",
    "RandomObjectSelector",
    "OracleObjectSelector",
    "QuestionAnswerer",
    "EnsembleQuestionAnswerer",
    "ThoughtVerifier",
    "OracleThoughtVerifier",
    "RandomThoughtVerifier",
    # Interactive
    "AttentionStrategy",
    "LLMAttentionStrategy",
    "RandomAttentionStrategy",
    "OracleAttentionStrategy",
    "AllRegionsAttentionStrategy",
    "InteractiveAttention",
    "InteractiveLoop",
    # Reasoner
    "NoOpReasoner",
    "VisualCoTReasoner",
]
