"""OPT model adapter for local inference."""

from typing import Dict, List, Optional

import torch
from transformers import OPTForCausalLM, GPT2Tokenizer

from .base_adapter import BaseLLMAdapter
from .types import LLMResponse, LLMConfig


class OPTAdapter(BaseLLMAdapter):
    """Adapter for Meta's OPT models."""

    def __init__(self, config: LLMConfig):
        """
        Initialize OPT adapter.

        Args:
            config: LLM configuration
        """
        super().__init__(config)

        # Determine model size and device mapping
        if config.engine_name == "facebook/opt-66b":
            self._init_opt_66b(config)
        else:
            # Default: smaller models like opt-1.3b
            self._init_opt_small(config)

    def _init_opt_small(self, config: LLMConfig):
        """Initialize smaller OPT models (e.g., 1.3B)."""
        model_name = config.engine_name or "facebook/opt-1.3b"
        self.model = OPTForCausalLM.from_pretrained(
            model_name, device_map=config.device, torch_dtype=torch.float16
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    def _init_opt_66b(self, config: LLMConfig):
        """Initialize OPT-66B with multi-GPU device mapping."""
        num_gpus = torch.cuda.device_count()

        if num_gpus >= 8:
            num_layers_per_gpu = 8  # 64 layers / 8 GPUs
        elif num_gpus >= 6:
            num_layers_per_gpu = 11  # ceil(64 / 6)
        else:
            raise ValueError(f"OPT-66B requires at least 6 GPUs, got {num_gpus}")

        # Create device map
        device_map = {
            "model.decoder.embed_tokens": 0,
            "lm_head": 0,
            "model.decoder.embed_positions": 0,
            "model.decoder.final_layer_norm": 0,
            "model.decoder.layers.0": 0,
        }

        for layer in range(64):
            device = min(layer // num_layers_per_gpu, num_gpus - 1)
            device_map[f"model.decoder.layers.{layer}"] = device

        self.model = OPTForCausalLM.from_pretrained(
            "facebook/opt-66b", device_map=device_map, torch_dtype=torch.float16
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-66b")

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
        Generate text using OPT model.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (not used, greedy decoding)
            stop_tokens: Stop sequences
            logit_bias: Token bias (used for constrained decoding)
            **kwargs: Additional arguments

        Returns:
            LLM response with generated text and logprobs
        """
        max_tokens = max_tokens or self.config.max_tokens

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_length = len(inputs.input_ids[0])

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids.to(torch.cuda.current_device()),
                max_length=input_length + max_tokens,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Decode generated text
        generated_ids = outputs["sequences"][0][input_length:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)

        # Extract logprobs
        logprobs = []
        for i, token_id in enumerate(generated_ids):
            if i < len(outputs["scores"]):
                scores = torch.log_softmax(outputs["scores"][i][0], dim=-1)
                logprobs.append(scores[token_id].item())

        # Extract tokens
        tokens = [self.tokenizer.decode([tid]) for tid in generated_ids]

        return LLMResponse(
            text=text,
            logprobs=logprobs,
            tokens=tokens,
            total_logprob=sum(logprobs) if logprobs else 0.0,
            raw_response={"sequences": outputs["sequences"], "scores": outputs["scores"]},
        )

    def generate_with_object_selection(
        self, prompt: str, object_tokens: List[int], max_tokens: int = 5
    ) -> int:
        """
        Generate with constrained decoding to select from object tokens.

        This is used in the interactive object selection phase.

        Args:
            prompt: Input prompt
            object_tokens: List of token IDs for objects to choose from
            max_tokens: Maximum tokens to generate

        Returns:
            Index of selected object in object_tokens list
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_length = len(inputs.input_ids[0])

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids.to(torch.cuda.current_device()),
                max_length=input_length + max_tokens,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Get first token scores and find best match
        if len(outputs["scores"]) > 0:
            scores = outputs["scores"][0][0]
            obj_scores = scores[object_tokens]
            result_idx = obj_scores.argmax().item()
            return result_idx

        return 0
