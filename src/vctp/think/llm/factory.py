"""Factory for creating LLM adapters."""

from typing import List, Optional

from .base_adapter import BaseLLMAdapter
from .types import LLMConfig
from .openai_adapter import OpenAIGPT3Adapter, OpenAIChatAdapter
from .opt_adapter import OPTAdapter
from .llama_adapter import LLaMAAdapter


def create_llm_adapter(
    engine: str,
    engine_name: Optional[str] = None,
    api_keys: Optional[List[str]] = None,
    model_path: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 41,
    device: str = "auto",
    debug: bool = False,
    **kwargs,
) -> BaseLLMAdapter:
    """
    Create an LLM adapter based on engine type.

    Args:
        engine: Engine type (gpt3, chat, opt, llama, bloom, etc.)
        engine_name: Specific model name
        api_keys: List of API keys for OpenAI models
        model_path: Path to local model weights (for llama)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        device: Device for local models
        debug: Enable debug mode
        **kwargs: Additional configuration

    Returns:
        Configured LLM adapter

    Raises:
        ValueError: If engine type is not supported
    """
    # Default engine names
    if engine_name is None:
        engine_name_map = {
            "gpt3": "text-davinci-001",
            "davinci": "text-davinci-003",
            "chat": "gpt-3.5-turbo",
            "codex": "code-davinci-002",
            "instruct": "davinci-instruct-beta",
            "opt": "facebook/opt-1.3b",
            "llama": None,  # Requires model_path
        }
        engine_name = engine_name_map.get(engine, engine)

    config = LLMConfig(
        engine=engine,
        engine_name=engine_name,
        api_keys=api_keys or [],
        temperature=temperature,
        max_tokens=max_tokens,
        device=device,
        model_path=model_path,
        debug=debug,
        **kwargs,
    )

    # Create adapter based on engine type
    if engine in ["ada", "babbage", "curie", "davinci", "codex", "instruct", "gpt3"]:
        return OpenAIGPT3Adapter(config)

    elif engine == "chat":
        return OpenAIChatAdapter(config)

    elif engine == "opt":
        return OPTAdapter(config)

    elif engine == "llama":
        return LLaMAAdapter(config)

    else:
        raise ValueError(f"Unsupported engine type: {engine}")


def create_llm_from_args(args) -> BaseLLMAdapter:
    """
    Create LLM adapter from argparse arguments.

    This is a convenience function for compatibility with the original code.

    Args:
        args: Parsed arguments with engine, engine_name, etc.

    Returns:
        Configured LLM adapter
    """
    # Get API keys
    api_keys = []
    if hasattr(args, "apikey_file") and args.apikey_file:
        with open(args.apikey_file, "r") as f:
            api_keys = [line.strip() for line in f if line.strip()]
    elif hasattr(args, "apikey") and args.apikey:
        api_keys = [args.apikey]

    # Get model path for LLaMA
    model_path = None
    if hasattr(args, "llama_path"):
        model_path = args.llama_path

    # Get device configuration
    device = "auto"
    if hasattr(args, "with_six_gpus") and args.with_six_gpus:
        device = "auto"  # Multi-GPU handled in adapter
    elif hasattr(args, "with_one_gpu") and args.with_one_gpu:
        device = "auto"

    return create_llm_adapter(
        engine=args.engine,
        engine_name=getattr(args, "engine_name", None),
        api_keys=api_keys,
        model_path=model_path,
        temperature=0.0,  # Default from original code
        max_tokens=41,
        device=device,
        debug=getattr(args, "debug", False),
    )
