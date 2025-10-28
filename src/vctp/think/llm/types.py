"""Types and data classes for LLM adapters."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class LLMResponse:
    """Response from an LLM generation call."""

    text: str
    """Generated text"""

    logprobs: Optional[List[float]] = None
    """Log probabilities for each token"""

    tokens: Optional[List[str]] = None
    """Generated tokens"""

    total_logprob: Optional[float] = None
    """Sum of log probabilities"""

    raw_response: Optional[Dict[str, Any]] = None
    """Raw response from API/model"""


@dataclass
class ChatMessage:
    """A single message in a chat conversation."""

    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class LLMConfig:
    """Configuration for LLM adapters."""

    engine: str = "gpt3"
    """Engine type: gpt3, chat, opt, llama, bloom"""

    engine_name: Optional[str] = None
    """Specific model name"""

    api_keys: List[str] = field(default_factory=list)
    """List of API keys for rotation"""

    temperature: float = 0.0
    """Sampling temperature"""

    max_tokens: int = 41
    """Maximum tokens to generate"""

    stop_tokens: Optional[List[str]] = None
    """Stop sequences"""

    sleep_time: float = 1.5
    """Sleep time between API calls"""

    device: str = "auto"
    """Device for local models"""

    model_path: Optional[str] = None
    """Path to local model weights"""

    debug: bool = False
    """Enable debug mode"""
