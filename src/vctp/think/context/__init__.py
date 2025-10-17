"""Context retrieval for few-shot learning."""

from .similarity_retriever import SimilarityRetriever
from .object_similarity import ObjectSimilarityComputer
from .context_manager import ContextManager, InteractiveContextManager

__all__ = [
    "SimilarityRetriever",
    "ObjectSimilarityComputer",
    "ContextManager",
    "InteractiveContextManager",
]
