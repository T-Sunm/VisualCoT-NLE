"""Similarity metrics for object selection."""

from abc import ABC, abstractmethod
from typing import Dict, List
import torch
from vctp.utils.clip_manager import get_clip_tokenizer


class SimilarityMetric(ABC):
    """Base class for similarity metrics."""

    @abstractmethod
    def compute(
        self, obj_list: List[str], answers: List[str] = None, rationales: List[str] = None, **kwargs
    ) -> Dict[str, float]:
        """Compute similarity scores for objects."""
        raise NotImplementedError


class AnswerSimilarityMetric(SimilarityMetric):
    """CLIP-based similarity between objects and answers."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch16"):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.tokenizer = get_clip_tokenizer(model_name=model_name, device=self.device)

    def compute(self, obj_list: List[str], answers: List[str] = None, **kwargs) -> Dict[str, float]:
        """Compute CLIP similarity between objects and answers."""
        if not answers or not obj_list:
            return {}

        similarity_dict = {}

        with torch.no_grad():
            # Encode answers
            inputs = self.tokenizer(text=answers, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            ans_emb = self.model(**inputs)["pooler_output"].mean(dim=0, keepdim=True)

            # Encode objects
            inputs = self.tokenizer(text=obj_list, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            obj_emb = self.model(**inputs)["pooler_output"]

            # Normalize and compute similarity
            ans_emb = ans_emb / ans_emb.norm(dim=-1, keepdim=True)
            obj_emb = obj_emb / obj_emb.norm(dim=-1, keepdim=True)

            sim_scores = obj_emb @ ans_emb.T

            for idx, obj_name in enumerate(obj_list):
                similarity_dict[obj_name] = sim_scores[idx, 0].cpu().item()

        return similarity_dict


class RationaleSimilarityMetric(SimilarityMetric):
    """Count-based similarity using rationale mentions."""

    def compute(
        self, obj_list: List[str], rationales: List[str] = None, **kwargs
    ) -> Dict[str, float]:
        """Count object mentions in rationales."""
        if not rationales or not obj_list:
            return {}

        similarity_dict = {}

        for obj in obj_list:
            count = sum(1 for r in rationales if obj in r)
            if count > 0:
                similarity_dict[obj] = float(count)

        return similarity_dict
