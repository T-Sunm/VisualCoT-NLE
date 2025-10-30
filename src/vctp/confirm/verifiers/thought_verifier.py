"""
Thought/Rationale Verifier
Verifies if reasoning steps match image content
Based on lines 1080-1131 and 671-683 from main_aokvqa.py
"""

from typing import List, Optional, Tuple
import numpy as np
import torch

from .base_verifier import BaseVerifier


class CLIPThoughtVerifier(BaseVerifier):
    """Verify thoughts using CLIP image-text similarity."""

    def __init__(
        self,
        clip_model=None,
        clip_processor=None,
        threshold: float = 0.0,
        device: Optional[str] = None,
    ):
        """
        Initialize CLIP thought verifier.

        Args:
            clip_model: CLIP text model
            clip_processor: CLIP processor
            threshold: Similarity threshold for filtering
            device: Device to run on
        """
        self.threshold = threshold
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

        self.clip_model, self.clip_processor = get_clip_model(
            model_type="text", device=self.device, use_safetensors=True
        )

    def verify(
        self,
        candidate: str,
        image_embedding: Optional[np.ndarray] = None,
        image_path: Optional[str] = None,
        **kwargs,
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Verify a single thought sentence.

        Returns:
            (is_valid, similarity_score, None)
        """
        if image_embedding is None:
            raise ValueError("Image embedding required for CLIP verification")

        # Split into sentences if needed
        thoughts = [candidate] if "." not in candidate else candidate.split(".")
        thoughts = [t.strip() for t in thoughts if t.strip()]

        with torch.no_grad():
            # Encode thoughts
            inputs = self.clip_processor(
                text=thoughts, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.clip_model(**inputs)
            thought_emb = outputs["pooler_output"]

            # Normalize
            thought_emb = thought_emb / thought_emb.norm(dim=-1, keepdim=True)

            # Prepare image embedding
            img_emb = torch.from_numpy(image_embedding).to(self.device).float()
            if len(img_emb.shape) == 1:
                img_emb = img_emb.unsqueeze(0)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

            # Compute similarity
            similarities = (img_emb @ thought_emb.T).squeeze()

            if len(thoughts) == 1:
                score = similarities.item()
                is_valid = score > self.threshold
            else:
                # Average similarity for multiple thoughts
                score = similarities.mean().item()
                is_valid = score > self.threshold

        return is_valid, score, None

    def verify_and_filter(
        self, thoughts: str, image_embedding: Optional[np.ndarray] = None, **kwargs
    ) -> Tuple[str, str, List[float]]:
        """
        Verify and filter thoughts based on similarity.
        Returns filtered thoughts and all thoughts with scores.

        Based on lines 1096-1131 from main_aokvqa.py

        Args:
            thoughts: Thoughts string (sentences separated by .)
            image_embedding: Image CLIP embedding

        Returns:
            Tuple of (filtered_thoughts, all_thoughts, similarity_scores)
        """
        # Split thoughts into sentences
        thought_list = [t.strip() for t in thoughts.split(".") if t.strip()]

        if not thought_list:
            return "", "", []

        with torch.no_grad():
            # Encode thoughts
            inputs = self.clip_processor(
                text=thought_list, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.clip_model(**inputs)
            thought_emb = outputs["pooler_output"]
            thought_emb = thought_emb / thought_emb.norm(dim=-1, keepdim=True)

            # Prepare image embedding
            img_emb = torch.from_numpy(image_embedding).to(self.device).float()
            if len(img_emb.shape) == 1:
                img_emb = img_emb.unsqueeze(0)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

            # Compute similarities
            sim_scores = (img_emb @ thought_emb.T).squeeze()
            if sim_scores.dim() == 0:  # If 0-d tensor (single thought)
                sim_scores = sim_scores.unsqueeze(0)  # Make it 1-d

            # Filter thoughts
            filtered_thoughts = []
            all_thoughts = []
            scores = []

        # ← THÊM DEBUG CHI TIẾT
        print(f"\n[CLIPThoughtVerifier] Similarity Analysis:")
        for i, (thought, sim) in enumerate(zip(thought_list, sim_scores)):
            score = sim.item()
            scores.append(score)
            all_thoughts.append(thought)

            passed = "✓ PASS" if score > self.threshold else "✗ FAIL"
            print(f"  [{i+1}] Score: {score:.4f} {passed}")
            print(f"      Text: {thought[:60]}...")

            if score > self.threshold and len(thought) > 0:
                filtered_thoughts.append(thought)

        print(f"\n[CLIPThoughtVerifier] Summary:")
        print(f"  Total: {len(thought_list)} thoughts")
        print(f"  Passed: {len(filtered_thoughts)} thoughts")
        print(f"  Average similarity: {np.mean(scores):.4f}")

        filtered_text = ". ".join(filtered_thoughts).strip()
        if filtered_text:
            filtered_text += "."

        all_text = ". ".join(all_thoughts).strip()
        if all_text:
            all_text += "."

        return filtered_text, all_text, scores


class BLIP2ThoughtVerifier(BaseVerifier):
    """Verify thoughts using BLIP2 VQA."""

    def __init__(
        self,
        blip2_captioner,
        debug: bool = False,
    ):
        """
        Initialize BLIP2 thought verifier.

        Args:
            blip2_captioner: BLIP2Captioner instance
            debug: Enable debug mode
        """
        self.captioner = blip2_captioner
        self.debug = debug

    def verify(
        self,
        candidate: str,
        image_embedding: Optional[np.ndarray] = None,
        image_path: Optional[str] = None,
        **kwargs,
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Verify thought using BLIP2.

        Based on lines 671-683 from main_aokvqa.py

        Returns:
            (is_valid, confidence, corrected_text_or_none)
        """
        # Ask BLIP2 if sentence matches image
        prompt = (
            f"Question: Does this sentence match the facts in the picture? "
            f"Please answer yes or no. Sentence: In this picture, {candidate} Answer:"
        )

        blip2_answer = self.captioner.query_basic(prompt=prompt)[0].lower()

        if self.debug:
            print(f"BLIP2 Verification: {blip2_answer} for '{candidate}'")

        if "no" in blip2_answer:
            # Request correction
            correction_prompt = (
                f"Question: Please correct the following sentence according to "
                f"the image. Sentence: {candidate}"
            )
            correction = self.captioner.query_basic(prompt=correction_prompt)[0]
            return False, 0.0, correction
        else:
            return True, 1.0, None

    def verify_and_filter(
        self, thoughts: str, image_path: Optional[str] = None, **kwargs
    ) -> Tuple[str, str, List[float]]:
        """
        Verify and optionally correct multiple thoughts.

        Based on lines 1081-1094 from main_aokvqa.py
        """
        thought_list = [t.strip() for t in thoughts.split(".") if t.strip()]

        filtered_thoughts = []
        all_thoughts = []
        scores = []

        for thought in thought_list:
            is_valid, score, correction = self.verify(thought, image_path=image_path)

            all_thoughts.append(thought)
            scores.append(score)

            if correction:
                filtered_thoughts.append(correction)
            elif is_valid:
                filtered_thoughts.append(thought)

        filtered_text = ". ".join(filtered_thoughts).strip()
        if filtered_text:
            filtered_text += "."

        all_text = ". ".join(all_thoughts).strip()
        if all_text:
            all_text += "."

        return filtered_text, all_text, scores
