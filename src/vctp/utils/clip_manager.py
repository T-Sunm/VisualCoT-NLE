"""
CLIP Model Manager - Singleton pattern for efficient CLIP model loading
Prevents multiple CLIP model instances to save VRAM
"""

import torch
from typing import Optional, Dict, Tuple, Any
from transformers import CLIPModel, CLIPTextModel, CLIPProcessor, CLIPTokenizer


class CLIPModelManager:
    """
    Singleton manager for CLIP models to avoid multiple initializations.
    Caches models and reuses them across the application.
    """

    _instance = None
    _models: Dict[str, Any] = {}
    _processors: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self._models = {}
            self._processors = {}
            self._default_device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_clip_model(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        model_type: str = "full",  # "full", "text", "vision"
        device: Optional[str] = None,
        use_safetensors: bool = True,
    ) -> Tuple[Any, Any]:
        """
        Get CLIP model and processor (cached).

        Args:
            model_name: HuggingFace model name
            model_type: Type of model ("full", "text", "vision")
            device: Device to load model on
            use_safetensors: Whether to use safetensors format

        Returns:
            Tuple of (model, processor)
        """
        device = device or self._default_device
        cache_key = f"{model_name}_{model_type}_{device}"

        # Return cached model if exists
        if cache_key in self._models:
            return self._models[cache_key], self._processors[cache_key]

        # Load new model
        print(f"[CLIPManager] Loading {model_type} CLIP model: {model_name} on {device}")

        if model_type == "text":
            model = CLIPTextModel.from_pretrained(model_name, use_safetensors=use_safetensors)
            processor = CLIPProcessor.from_pretrained(model_name, use_safetensors=use_safetensors)
        elif model_type == "full":
            model = CLIPModel.from_pretrained(model_name, use_safetensors=use_safetensors)
            processor = CLIPProcessor.from_pretrained(model_name, use_safetensors=use_safetensors)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Move to device
        model = model.to(device)
        model.eval()  # Set to eval mode by default

        # Cache
        self._models[cache_key] = model
        self._processors[cache_key] = processor

        print(f"[CLIPManager] ✓ Model cached: {cache_key}")
        return model, processor

    def get_clip_tokenizer(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        device: Optional[str] = None,
    ) -> Tuple[Any, Any]:
        """
        Get CLIP text model with tokenizer (for text-only tasks).

        Args:
            model_name: HuggingFace model name
            device: Device to load model on

        Returns:
            Tuple of (text_model, tokenizer)
        """
        device = device or self._default_device
        cache_key = f"{model_name}_tokenizer_{device}"

        # Return cached if exists
        if cache_key in self._models:
            return self._models[cache_key], self._processors[cache_key]

        print(f"[CLIPManager] Loading CLIP with tokenizer: {model_name} on {device}")

        model = CLIPTextModel.from_pretrained(model_name, use_safetensors=True)
        tokenizer = CLIPTokenizer.from_pretrained(model_name)

        model = model.to(device)
        model.eval()

        # Cache
        self._models[cache_key] = model
        self._processors[cache_key] = tokenizer

        print(f"[CLIPManager] ✓ Model cached: {cache_key}")
        return model, tokenizer

    def clear_cache(self, model_name: Optional[str] = None):
        """
        Clear cached models to free VRAM.

        Args:
            model_name: Specific model to clear, or None to clear all
        """
        if model_name is None:
            # Clear all
            count = len(self._models)
            self._models.clear()
            self._processors.clear()
            torch.cuda.empty_cache()
            print(f"[CLIPManager] Cleared {count} cached models")
        else:
            # Clear specific model
            keys_to_remove = [k for k in self._models.keys() if model_name in k]
            for key in keys_to_remove:
                del self._models[key]
                del self._processors[key]
            torch.cuda.empty_cache()
            print(f"[CLIPManager] Cleared {len(keys_to_remove)} models matching '{model_name}'")


# Global singleton instance
_clip_manager = CLIPModelManager()


def get_clip_model(
    model_name: str = "openai/clip-vit-base-patch16",
    model_type: str = "full",
    device: Optional[str] = None,
    use_safetensors: bool = True,
) -> Tuple[Any, Any]:
    """
    Convenience function to get CLIP model.

    Args:
        model_name: HuggingFace model name
        model_type: "full" or "text"
        device: Device to load on
        use_safetensors: Use safetensors format

    Returns:
        Tuple of (model, processor)

    Example:
        >>> model, processor = get_clip_model(model_type="text", device="cuda")
    """
    return _clip_manager.get_clip_model(model_name, model_type, device, use_safetensors)


def get_clip_tokenizer(
    model_name: str = "openai/clip-vit-base-patch16",
    device: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Convenience function to get CLIP with tokenizer.

    Args:
        model_name: HuggingFace model name
        device: Device to load on

    Returns:
        Tuple of (text_model, tokenizer)
    """
    return _clip_manager.get_clip_tokenizer(model_name, device)


def clear_clip_cache(model_name: Optional[str] = None):
    """Clear CLIP model cache to free VRAM."""
    _clip_manager.clear_cache(model_name)


def get_clip_memory_stats() -> Dict[str, Any]:
    """Get CLIP memory usage statistics."""
    return _clip_manager.get_memory_usage()
