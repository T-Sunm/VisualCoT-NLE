"""Similarity metrics for object selection."""

from abc import ABC, abstractmethod
from typing import Dict, List
import torch

# Sửa import để dùng CLIPClient có sẵn
from src.utils.models.clip import CLIPClient


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
        # Khởi tạo CLIPClient từ file clip.py
        self.client = CLIPClient(model_type=model_name, device=self.device)

    def compute(self, obj_list: List[str], answers: List[str] = None, **kwargs) -> Dict[str, float]:
        """Compute CLIP similarity between objects and answers."""
        if not answers or not obj_list:
            return {}

        similarity_dict = {}
        
        # Gộp tất cả câu trả lời thành một văn bản để so sánh (hoặc so sánh từng cái, tùy logic bài toán)
        # Ở đây mình giữ logic cũ: so sánh embeddings
        
        with torch.no_grad():
            # 1. Encode answers
            # Tokenize và lấy features
            ans_inputs = self.client.processor(
                text=answers, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(self.device)
            
            # Lấy text embeddings từ model
            ans_emb = self.client.model.get_text_features(**ans_inputs)
            ans_emb = ans_emb.mean(dim=0, keepdim=True) # Average các answers

            # 2. Encode objects
            obj_inputs = self.client.processor(
                text=obj_list, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(self.device)
            
            obj_emb = self.client.model.get_text_features(**obj_inputs)

            # 3. Normalize và tính cosine similarity
            ans_emb = ans_emb / ans_emb.norm(dim=-1, keepdim=True)
            obj_emb = obj_emb / obj_emb.norm(dim=-1, keepdim=True)

            # Dot product
            sim_scores = obj_emb @ ans_emb.T

            for idx, obj_name in enumerate(obj_list):
                similarity_dict[obj_name] = sim_scores[idx, 0].item()

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
