"""LLaMA model adapter for local inference."""

from typing import Dict, List, Optional

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

from .base_adapter import BaseLLMAdapter
from .types import LLMResponse, LLMConfig


class LLaMAAdapter(BaseLLMAdapter):
    """Adapter for Meta's LLaMA models."""

    def __init__(self, config: LLMConfig):
        """
        Initialize LLaMA adapter.

        Args:
            config: LLM configuration with model_path
        """
        super().__init__(config)

        if not config.model_path:
            raise ValueError("model_path required for LLaMA adapter")

        # Load LLaMA model
        self.model = LlamaForCausalLM.from_pretrained(
            config.model_path, device_map=config.device, torch_dtype=torch.float16
        )
        self.tokenizer = LlamaTokenizer.from_pretrained(config.model_path)

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
        Generate text using LLaMA model.

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
