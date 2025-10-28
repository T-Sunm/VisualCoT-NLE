from typing import List, Tuple, Optional
import numpy as np
import torch
from vctp.utils.clip_manager import get_clip_model


class ThoughtVerifier:
    """Verify if thoughts/reasoning match image content."""

    def __init__(
        self,
        use_clip: bool = True,
        use_blip2: bool = False,
        clip_model=None,
        clip_processor=None,
        blip2_captioner=None,
        threshold: float = 0.0,
        debug: bool = False,
    ):
        """
        Initialize thought verifier.

        Args:
            use_clip: Use CLIP for verification
            use_blip2: Use BLIP2 for verification
            clip_model: CLIP text model
            clip_processor: CLIP processor
            blip2_captioner: BLIP2Captioner instance
            threshold: Similarity threshold for filtering
            debug: Debug mode
        """
        self.use_clip = use_clip
        self.use_blip2 = use_blip2
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.blip2_captioner = blip2_captioner
        self.threshold = threshold
        self.debug = debug

        # Initialize CLIP if needed
        if use_clip and (clip_model is None or clip_processor is None):
            self._init_clip()

    def _init_clip(self):
        self.clip_model, self.clip_processor = get_clip_model(model_type="text", device="cuda")

    def verify_thoughts(
        self,
        thoughts: str,
        image_embedding: Optional[np.ndarray] = None,
        image_path: Optional[str] = None,
    ) -> Tuple[str, str, List[float]]:
        """
        Verify thoughts against image.

        Args:
            thoughts: Reasoning text to verify
            image_embedding: Pre-computed CLIP image embedding
            image_path: Path to image (for BLIP2)

        Returns:
            Tuple of (filtered_thoughts, all_thoughts, similarity_scores)
        """
        if self.use_blip2 and self.blip2_captioner:
            return self._verify_with_blip2(thoughts, image_path)
        elif self.use_clip:
            return self._verify_with_clip(thoughts, image_embedding)
        else:
            # No verification
            return thoughts, thoughts, []

    def _verify_with_clip(
        self, thoughts: str, image_embedding: Optional[np.ndarray]
    ) -> Tuple[str, str, List[float]]:
        """Verify thoughts using CLIP similarity."""
        if image_embedding is None or not self.use_clip:
            return thoughts, thoughts, []

        # Split thoughts into sentences
        thought_list = thoughts.split(".")
        thought_list = [t.strip() for t in thought_list if t.strip()]

        if not thought_list:
            return "", "", []

        with torch.no_grad():
            # Encode thoughts
            inputs = self.clip_processor(text=thought_list, return_tensors="pt", padding=True)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            outputs = self.clip_model(**inputs)
            thought_emb = outputs["pooler_output"]

            # Normalize
            thought_emb = thought_emb / thought_emb.norm(dim=-1, keepdim=True)

            # Convert image embedding to tensor
            img_emb = torch.from_numpy(image_embedding).cuda().float().unsqueeze(0)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

            # Compute similarities
            similarities = (img_emb @ thought_emb.T)[0].cpu().numpy()

        # Filter by threshold
        filtered_thoughts = []
        all_thoughts = []
        similarity_scores = []

        for thought, sim in zip(thought_list, similarities):
            similarity_scores.append(float(sim))
            all_thoughts.append(thought)

            if sim > self.threshold:
                filtered_thoughts.append(thought)

        # Join thoughts
        filtered_text = ". ".join(filtered_thoughts).strip() + "." if filtered_thoughts else ""
        all_text = ". ".join(all_thoughts).strip() + "." if all_thoughts else ""

        if self.debug:
            print(f"[ThoughtVerifier] Original: {thoughts}")
            print(f"[ThoughtVerifier] Filtered: {filtered_text}")
            print(f"[ThoughtVerifier] Similarities: {similarity_scores}")

        return filtered_text, all_text, similarity_scores

    def _verify_with_blip2(
        self, thoughts: str, image_path: Optional[str]
    ) -> Tuple[str, str, List[float]]:
        """Verify thoughts using BLIP2."""
        if not self.blip2_captioner or not image_path:
            return thoughts, thoughts, []

        # Split thoughts into sentences
        thought_list = thoughts.split(".")
        thought_list = [t.strip() for t in thought_list if t.strip()]

        if not thought_list:
            return "", "", []

        filtered_thoughts = []
        all_thoughts = []

        for thought in thought_list:
            all_thoughts.append(thought)

            # Verify with BLIP2
            verified_thought = self.blip2_captioner.verify_thought_with_image(thought, image_path)

            if verified_thought:
                filtered_thoughts.append(verified_thought)

        # Join thoughts
        filtered_text = ". ".join(filtered_thoughts).strip() + "." if filtered_thoughts else ""
        all_text = ". ".join(all_thoughts).strip() + "." if all_thoughts else ""

        if self.debug:
            print(f"[ThoughtVerifier] Original: {thoughts}")
            print(f"[ThoughtVerifier] Verified: {filtered_text}")

        return filtered_text, all_text, []


class OracleThoughtVerifier:
    """Use ground-truth rationales (oracle mode)."""

    def __init__(self, rationale_dict: dict):
        """
        Initialize oracle verifier.

        Args:
            rationale_dict: Dict mapping keys to rationales
        """
        self.rationale_dict = rationale_dict

    def verify_thoughts(
        self,
        thoughts: str,
        query_key: str,
        image_embedding: Optional[np.ndarray] = None,
        image_path: Optional[str] = None,
    ) -> Tuple[str, str, List[float]]:
        """
        Return oracle rationale.

        Args:
            thoughts: Generated thoughts (ignored)
            query_key: Query key for oracle lookup
            image_embedding: Image embedding (ignored)
            image_path: Image path (ignored)

        Returns:
            Tuple of (oracle_rationale, oracle_rationale, [])
        """
        if query_key in self.rationale_dict:
            oracle_rationale = self.rationale_dict[query_key][0]
            return oracle_rationale, oracle_rationale, []

        return thoughts, thoughts, []


class RandomThoughtVerifier:
    """Use random rationales (ablation study)."""

    def __init__(self, rationale_dict: dict):
        """
        Initialize random verifier.

        Args:
            rationale_dict: Dict of all rationales
        """
        self.rationale_dict = rationale_dict

    def verify_thoughts(
        self,
        thoughts: str,
        image_embedding: Optional[np.ndarray] = None,
        image_path: Optional[str] = None,
    ) -> Tuple[str, str, List[float]]:
        """
        Return random rationale.

        Args:
            thoughts: Generated thoughts (ignored)
            image_embedding: Image embedding (ignored)
            image_path: Image path (ignored)

        Returns:
            Tuple of (random_rationale, random_rationale, [])
        """
        import random

        random_key = random.choice(list(self.rationale_dict.keys()))
        random_rationale = random.choice(self.rationale_dict[random_key])

        return random_rationale, random_rationale, []
