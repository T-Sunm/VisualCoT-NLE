"""Base LLM adapter interface."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from .types import LLMResponse, ChatMessage, LLMConfig


class BaseLLMAdapter(ABC):
    """Base class for all LLM adapters."""

    def __init__(self, config: LLMConfig):
        """
        Initialize LLM adapter.

        Args:
            config: LLM configuration
        """
        self.config = config

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_tokens: Optional[List[str]] = None,
        logit_bias: Optional[Dict[int, float]] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate (overrides config)
            temperature: Sampling temperature (overrides config)
            stop_tokens: Stop sequences (overrides config)
            logit_bias: Token bias dictionary
            **kwargs: Additional engine-specific arguments

        Returns:
            LLM response with generated text and metadata
        """
        pass

    def chat(
        self,
        messages: List[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate response in chat format (for chat-based models).

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional arguments

        Returns:
            LLM response
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support chat mode")
