import time
from typing import Dict, List, Optional
from groq import Groq

from .base_adapter import BaseLLMAdapter
from .types import LLMConfig, LLMResponse


class GroqAdapter(BaseLLMAdapter):
    """
    Adapter for Groq Cloud API.

    Supported models:
    - openai/gpt-oss-20b
    """

    DEFAULT_MODEL = "openai/gpt-oss-20b"

    # Model aliases for convenience
    MODEL_ALIASES = {
        "gpt-oss": "openai/gpt-oss-20b",
    }

    def __init__(self, config: LLMConfig):
        """
        Initialize Groq adapter.

        Args:
            config: LLM configuration with api_keys
        """
        super().__init__(config)

        # Get API key
        if not config.api_keys or len(config.api_keys) == 0:
            import os

            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError(
                    "Groq API key required. Set GROQ_API_KEY environment variable "
                    "or provide in config.api_keys"
                )
            self.api_keys = [api_key]
        else:
            self.api_keys = config.api_keys

        # Initialize Groq client
        self.current_key_idx = 0
        self.client = Groq(api_key=self.api_keys[self.current_key_idx])

        # Resolve model name
        model_name = config.engine_name or self.DEFAULT_MODEL
        self.model_name = self.MODEL_ALIASES.get(model_name, model_name)

        if config.debug:
            print(f"GroqAdapter initialized:")
            print(f"  Model: {self.model_name}")
            print(f"  API keys: {len(self.api_keys)}")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_tokens: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate completion using Groq API.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop_tokens: Stop sequences
            system_prompt: System message (for chat models)
            **kwargs: Additional generation parameters

        Returns:
            LLMResponse with generated text and metadata
        """
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature if temperature is not None else self.config.temperature

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Generate with retry logic
        response = None
        retry_count = 0
        max_retries = 3

        while response is None and retry_count < max_retries:
            try:
                # Call Groq API
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=512,
                    temperature=temperature,
                    stop=stop_tokens,
                    **kwargs,
                )
                # Extract response
                text = completion.choices[0].message.content

                logprobs = None
                tokens = None

                if self.config.debug:
                    print(f"\n[GroqAdapter] Generated:")
                    print(f"  Tokens used: {completion.usage.total_tokens}")
                    print(f"  Response: {text[:100]}...")

                response = LLMResponse(
                    text=text,
                    tokens=tokens,
                    logprobs=logprobs,
                    raw_response={
                        "model": self.model_name,
                        "usage": {
                            "prompt_tokens": completion.usage.prompt_tokens,
                            "completion_tokens": completion.usage.completion_tokens,
                            "total_tokens": completion.usage.total_tokens,
                        },
                        "finish_reason": completion.choices[0].finish_reason,
                    },
                )

            except Exception as e:
                retry_count += 1
                print(f"Groq API error (attempt {retry_count}/{max_retries}): {e}")

                if retry_count < max_retries:
                    # Try next API key if available
                    if len(self.api_keys) > 1:
                        self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
                        self.client = Groq(api_key=self.api_keys[self.current_key_idx])
                        print(f"Switched to API key {self.current_key_idx}")

                    # Wait before retry
                    wait_time = 2**retry_count  # Exponential backoff
                    print(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Failed to generate after {max_retries} attempts") from e

        return response


class GroqChatAdapter(GroqAdapter):
    """
    Specialized Groq adapter for chat-optimized models.
    Uses chat formatting for better conversation handling.
    """

    DEFAULT_MODEL = "openai/gpt-oss-20b"

    def __init__(self, config: LLMConfig):
        """Initialize chat adapter."""
        super().__init__(config)

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_tokens: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate with conversation context.

        Args:
            prompt: Current user message
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            stop_tokens: Stop sequences
            system_prompt: System message
            conversation_history: Previous conversation turns
            **kwargs: Additional parameters

        Returns:
            LLMResponse
        """
        # Build conversation
        messages = []

        # Add system prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add history
        if conversation_history:
            messages.extend(conversation_history)

        # Add current prompt
        messages.append({"role": "user", "content": prompt})

        # Generate
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature if temperature is not None else self.config.temperature

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop_tokens,
                **kwargs,
            )

            text = completion.choices[0].message.content

            return LLMResponse(
                text=text,
                tokens=None,
                logprobs=None,
                raw_response={
                    "model": self.model_name,
                    "usage": {
                        "prompt_tokens": completion.usage.prompt_tokens,
                        "completion_tokens": completion.usage.completion_tokens,
                        "total_tokens": completion.usage.total_tokens,
                    },
                    "finish_reason": completion.choices[0].finish_reason,
                },
            )

        except Exception as e:
            raise RuntimeError(f"Groq chat generation failed: {e}") from e
