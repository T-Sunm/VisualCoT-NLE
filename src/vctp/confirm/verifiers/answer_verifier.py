"""
Answer Verifier
Verifies predicted answers against multiple-choice options
Based on lines 1149-1159 from main_aokvqa.py
"""

from typing import List, Optional, Tuple
import torch
import numpy as np

from .base_verifier import BaseVerifier


class ChoiceAnswerVerifier(BaseVerifier):
    """Verify answer against multiple choices using CLIP."""

    def __init__(
        self,
        clip_model=None,
        clip_processor=None,
        device: Optional[str] = None,
    ):
        """
        Initialize choice answer verifier.

        Args:
            clip_model: CLIP text model
            clip_processor: CLIP processor
            device: Device to run on
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize CLIP if not provided
        if clip_model is None or clip_processor is None:
            self._init_clip()
        else:
            self.clip_model = clip_model
            self.clip_processor = clip_processor

    def _init_clip(self):
        """Initialize CLIP model."""
        from vctp.utils.clip_manager import get_clip_model

        self.clip_model, self.clip_processor = get_clip_model(model_type="text", device=self.device)

    def verify(
        self,
        candidate: str,
        image_embedding: Optional[np.ndarray] = None,
        image_path: Optional[str] = None,
        choices: Optional[List[str]] = None,
        **kwargs,
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Verify if candidate answer is in choices.
        If not, find closest choice.

        Based on lines 1149-1159 from main_aokvqa.py

        Args:
            candidate: Predicted answer
            choices: List of valid choices

        Returns:
            (is_in_choices, similarity, corrected_answer_or_none)
        """
        if choices is None or len(choices) == 0:
            return True, 1.0, None

        # Check if answer is in choices
        if candidate in choices:
            return True, 1.0, None

        # Find closest choice using CLIP
        with torch.no_grad():
            # Add candidate to choices for similarity computation
            all_texts = choices + [candidate]

            inputs = self.clip_processor(text=all_texts, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.clip_model(**inputs)
            text_emb = outputs["pooler_output"]
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

            # Compute similarity between candidate and choices
            candidate_emb = text_emb[-1].unsqueeze(0)
            choices_emb = text_emb[:-1]

            similarities = (candidate_emb @ choices_emb.T).squeeze()

            # Find best matching choice
            best_idx = similarities.argmax().item()
            best_score = similarities[best_idx].item()
            best_choice = choices[best_idx]

        return False, best_score, best_choice
