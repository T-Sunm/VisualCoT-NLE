"""OpenAI API adapters for GPT-3 and ChatGPT."""

import time
from typing import Dict, List, Optional

import openai

from .base_adapter import BaseLLMAdapter
from .types import LLMResponse, ChatMessage, LLMConfig


class OpenAIGPT3Adapter(BaseLLMAdapter):
    """Adapter for OpenAI GPT-3 models using Completion API."""

    def __init__(self, config: LLMConfig):
        """
        Initialize OpenAI GPT-3 adapter.

        Args:
            config: LLM configuration with API keys
        """
        super().__init__(config)
        self.api_key_idx = 0
        if not config.api_keys:
            raise ValueError("API keys required for OpenAI adapter")
        openai.api_key = config.api_keys[self.api_key_idx]

        # Map engine names
        self.engine_map = {
            "codex": "code-davinci-002",
            "instruct": "davinci-instruct-beta",
            "gpt3": "text-davinci-001",
        }

    def _get_engine_name(self) -> str:
        """Get the actual engine name for API calls."""
        if self.config.engine in self.engine_map:
            return self.engine_map[self.config.engine]
        return self.config.engine_name

    def _sleep(self, sleep_time: Optional[float] = None, switch_key: bool = False):
        """Sleep between API calls and optionally switch API key."""
        if self.config.engine == "codex":
            sleep_time = 0.1
        else:
            sleep_time = sleep_time or self.config.sleep_time

        if switch_key:
            self.api_key_idx = (self.api_key_idx + 1) % len(self.config.api_keys)
            openai.api_key = self.config.api_keys[self.api_key_idx]

        time.sleep(sleep_time)

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
        Generate text using OpenAI Completion API.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop_tokens: Stop sequences
            logit_bias: Token bias dictionary
            **kwargs: Additional arguments

        Returns:
            LLM response with generated text and logprobs
        """
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature if temperature is not None else self.config.temperature
        stop_tokens = stop_tokens or self.config.stop_tokens or ["\n", "<|endoftext|>"]

        successful = False
        response = None

        while not successful:
            try:
                self._sleep()
                response = openai.Completion.create(
                    engine=self._get_engine_name(),
                    prompt=prompt,
                    max_tokens=max_tokens,
                    logprobs=1,
                    temperature=temperature,
                    stream=False,
                    stop=stop_tokens,
                    logit_bias=logit_bias or {},
                )
                successful = True
            except Exception as e:
                if self.config.debug:
                    print(f"OpenAI API error: {e}")
                self._sleep(switch_key=True)

        # Extract text and logprobs
        text = response["choices"][0]["text"]
        tokens = response["choices"][0]["logprobs"]["tokens"]
        token_logprobs = response["choices"][0]["logprobs"]["token_logprobs"]

        return LLMResponse(
            text=text,
            logprobs=token_logprobs,
            tokens=tokens,
            total_logprob=sum([lp for lp in token_logprobs if lp is not None]),
            raw_response=response,
        )


class OpenAIChatAdapter(BaseLLMAdapter):
    """Adapter for OpenAI ChatGPT models using ChatCompletion API."""

    def __init__(self, config: LLMConfig):
        """
        Initialize OpenAI ChatGPT adapter.

        Args:
            config: LLM configuration with API keys
        """
        super().__init__(config)
        self.api_key_idx = 0
        if not config.api_keys:
            raise ValueError("API keys required for OpenAI adapter")
        openai.api_key = config.api_keys[self.api_key_idx]

        # For tokenizer
        try:
            import tiktoken

            self.tokenizer = tiktoken.encoding_for_model(config.engine_name)
        except ImportError:
            self.tokenizer = None

    def _sleep(self, sleep_time: Optional[float] = None, switch_key: bool = False):
        """Sleep between API calls and optionally switch API key."""
        sleep_time = sleep_time or self.config.sleep_time

        if switch_key:
            self.api_key_idx = (self.api_key_idx + 1) % len(self.config.api_keys)
            openai.api_key = self.config.api_keys[self.api_key_idx]

        time.sleep(sleep_time)

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_tokens: Optional[List[str]] = None,
        logit_bias: Optional[Dict[int, float]] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate text using OpenAI ChatCompletion API.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop_tokens: Stop sequences (not used in chat API)
            logit_bias: Token bias dictionary
            system_prompt: System prompt for chat
            **kwargs: Additional arguments

        Returns:
            LLM response with generated text
        """
        messages = [{"role": "user", "content": prompt}]

        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        return self._chat_internal(messages, max_tokens, temperature, logit_bias)

    def chat(
        self,
        messages: List[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        logit_bias: Optional[Dict[int, float]] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate response using ChatCompletion API.

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            logit_bias: Token bias dictionary
            **kwargs: Additional arguments

        Returns:
            LLM response
        """
        msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
        return self._chat_internal(msg_dicts, max_tokens, temperature, logit_bias)

    def _chat_internal(
        self,
        messages: List[Dict],
        max_tokens: Optional[int],
        temperature: Optional[float],
        logit_bias: Optional[Dict[int, float]],
    ) -> LLMResponse:
        """Internal method for chat API calls."""
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature if temperature is not None else self.config.temperature

        successful = False
        response = None
        current_bias = 25  # For logit bias adjustment on failure

        while not successful:
            try:
                self._sleep()
                response = openai.ChatCompletion.create(
                    model=self.config.engine_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=False,
                    logit_bias=logit_bias or {},
                )
                successful = True
            except Exception as e:
                if self.config.debug:
                    print(f"OpenAI Chat API error: {e}")
                # Reduce bias if it's too high
                if logit_bias:
                    current_bias = int(0.8 * current_bias)
                    logit_bias = {k: current_bias for k in logit_bias}
                self._sleep(switch_key=True)

        text = response["choices"][0]["message"]["content"]

        return LLMResponse(
            text=text,
            logprobs=None,  # Chat API doesn't return logprobs
            tokens=None,
            total_logprob=0.0,
            raw_response=response,
        )
