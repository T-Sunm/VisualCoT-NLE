"""
CLIP Feature Extraction Module
Extracted and refactored from main_aokvqa.py
Provides image and text feature extraction using CLIP
"""

import torch
import numpy as np
from typing import Union, List, Optional, Tuple
from PIL import Image
from pathlib import Path


class CLIPFeatureExtractor:
    """
    CLIP-based feature extractor for images and text.
    Supports similarity computation and verification tasks.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        device: Optional[str] = None,
        use_text_model_only: bool = False,
    ):
        """
        Initialize CLIP feature extractor.

        Args:
            model_name: HuggingFace model name for CLIP
            device: Device to run on (cuda/cpu)
            use_text_model_only: If True, only load text model (lighter)
        """
        from transformers import CLIPProcessor, CLIPModel, CLIPTextModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.use_text_model_only = use_text_model_only

        # Load processor
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Load model
        if use_text_model_only:
            self.model = CLIPTextModel.from_pretrained(model_name)
        else:
            self.model = CLIPModel.from_pretrained(model_name)

        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"CLIP model loaded: {model_name} on {self.device}")

    def extract_image_features(
        self,
        images: Union[str, Image.Image, List[Union[str, Image.Image]]],
        normalize: bool = True,
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Extract CLIP features from image(s).

        Args:
            images: Single image or list of images (path or PIL Image)
            normalize: Whether to L2-normalize features
            return_numpy: Return numpy array instead of torch tensor

        Returns:
            Image features [batch_size, feature_dim]
        """
        # Handle single image
        if not isinstance(images, list):
            images = [images]

        # Load images
        image_list = []
        for img in images:
            if isinstance(img, str):
                image_list.append(Image.open(img).convert("RGB"))
            else:
                image_list.append(img)

        # Process images
        inputs = self.processor(images=image_list, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract features
        with torch.no_grad():
            if self.use_text_model_only:
                raise ValueError("Text-only model cannot extract image features")

            outputs = self.model.get_image_features(**inputs)
            features = outputs

        # Normalize if requested
        if normalize:
            features = features / features.norm(dim=-1, keepdim=True)

        # Convert to numpy if requested
        if return_numpy:
            return features.cpu().numpy()

        return features

    def extract_text_features(
        self, texts: Union[str, List[str]], normalize: bool = True, return_numpy: bool = False
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Extract CLIP features from text(s).

        Args:
            texts: Single text or list of texts
            normalize: Whether to L2-normalize features
            return_numpy: Return numpy array instead of torch tensor

        Returns:
            Text features [batch_size, feature_dim]
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]

        # Process texts
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract features
        with torch.no_grad():
            if self.use_text_model_only:
                outputs = self.model(**inputs)
                features = outputs.pooler_output
            else:
                features = self.model.get_text_features(**inputs)

        # Normalize if requested
        if normalize:
            features = features / features.norm(dim=-1, keepdim=True)

        # Convert to numpy if requested
        if return_numpy:
            return features.cpu().numpy()

        return features

    def compute_similarity(
        self,
        images: Union[str, Image.Image, List[Union[str, Image.Image]], torch.Tensor, np.ndarray],
        texts: Union[str, List[str], torch.Tensor, np.ndarray],
        return_numpy: bool = True,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Compute similarity between images and texts.

        Args:
            images: Images or pre-computed image features
            texts: Texts or pre-computed text features
            return_numpy: Return numpy array

        Returns:
            Similarity matrix [num_images, num_texts]
        """
        # Get image features
        if isinstance(images, (torch.Tensor, np.ndarray)):
            if isinstance(images, np.ndarray):
                img_features = torch.from_numpy(images).to(self.device)
            else:
                img_features = images.to(self.device)

            # Normalize if not already
            if img_features.norm(dim=-1).mean() > 1.1:
                img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        else:
            img_features = self.extract_image_features(images, normalize=True)

        # Get text features
        if isinstance(texts, (torch.Tensor, np.ndarray)):
            if isinstance(texts, np.ndarray):
                text_features = torch.from_numpy(texts).to(self.device)
            else:
                text_features = texts.to(self.device)

            # Normalize if not already
            if text_features.norm(dim=-1).mean() > 1.1:
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        else:
            text_features = self.extract_text_features(texts, normalize=True)

        # Compute similarity
        similarity = img_features @ text_features.T

        if return_numpy:
            return similarity.cpu().numpy()

        return similarity

    def verify_text_with_image(
        self,
        image: Union[str, Image.Image, torch.Tensor, np.ndarray],
        texts: Union[str, List[str]],
        threshold: float = 0.0,
    ) -> Tuple[List[bool], List[float]]:
        """
        Verify if texts match the image content.

        Args:
            image: Image to verify against
            texts: Text(s) to verify
            threshold: Similarity threshold for verification

        Returns:
            Tuple of (verification results, similarity scores)
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False

        # Compute similarities
        similarities = self.compute_similarity(image, texts, return_numpy=True)

        # Get scores (first image, all texts)
        if len(similarities.shape) == 2:
            scores = similarities[0].tolist()
        else:
            scores = similarities.tolist()

        # Verify based on threshold
        verifications = [score > threshold for score in scores]

        if single_text:
            return verifications[0], scores[0]

        return verifications, scores

    def select_best_match(
        self, image: Union[str, Image.Image, torch.Tensor, np.ndarray], candidates: List[str]
    ) -> Tuple[int, str, float]:
        """
        Select best matching text from candidates for an image.

        Args:
            image: Image to match
            candidates: List of candidate texts

        Returns:
            Tuple of (best_index, best_text, best_score)
        """
        similarities = self.compute_similarity(image, candidates, return_numpy=True)

        # Get best match
        if len(similarities.shape) == 2:
            scores = similarities[0]
        else:
            scores = similarities

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_text = candidates[best_idx]

        return best_idx, best_text, best_score

    def filter_thoughts_by_image(
        self,
        image: Union[str, Image.Image, torch.Tensor, np.ndarray],
        thoughts: List[str],
        threshold: float = 0.0,
    ) -> Tuple[List[str], List[str], List[float]]:
        """
        Filter thoughts/reasoning steps based on image relevance.

        Args:
            image: Image to verify against
            thoughts: List of thoughts/reasoning steps
            threshold: Similarity threshold

        Returns:
            Tuple of (filtered_thoughts, all_thoughts, scores)
        """
        # Split thoughts if given as single string
        if len(thoughts) == 1 and "." in thoughts[0]:
            thought_list = [t.strip() for t in thoughts[0].split(".") if t.strip()]
        else:
            thought_list = thoughts

        # Compute similarities
        verifications, scores = self.verify_text_with_image(image, thought_list, threshold)

        # Filter thoughts
        filtered_thoughts = [
            thought for thought, verified in zip(thought_list, verifications) if verified
        ]

        return filtered_thoughts, thought_list, scores


class CLIPSimilarityComputer:
    """
    Helper class for computing similarities between pre-extracted features.
    Used for efficient batch processing (like in main_aokvqa.py).
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialize similarity computer.

        Args:
            device: Device for computation
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def compute_question_similarity(
        self, train_features: np.ndarray, val_features: np.ndarray, top_k: int = 16
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Compute question similarity for context selection.

        Args:
            train_features: Training question features [N, D]
            val_features: Validation question features [M, D] or [D]
            top_k: Number of top similar samples to return

        Returns:
            Tuple of (similarity_scores, top_k_indices)
        """
        # Handle single validation feature
        if len(val_features.shape) == 1:
            val_features = val_features[np.newaxis, :]

        # Compute similarity
        similarity = np.matmul(train_features, val_features.T)

        # Get top-k
        if similarity.shape[1] == 1:
            similarity = similarity[:, 0]

        top_k_indices = similarity.argsort()[-top_k:][::-1]
        top_k_scores = similarity[top_k_indices]

        return top_k_scores, top_k_indices.tolist()

    def compute_image_question_similarity(
        self,
        train_q_features: np.ndarray,
        val_q_features: np.ndarray,
        train_img_features: np.ndarray,
        val_img_features: np.ndarray,
        top_k: int = 16,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Compute combined image+question similarity.

        Args:
            train_q_features: Training question features
            val_q_features: Validation question features
            train_img_features: Training image features
            val_img_features: Validation image features
            top_k: Number of top similar samples

        Returns:
            Tuple of (combined_scores, top_k_indices)
        """
        # Handle single validation feature
        if len(val_q_features.shape) == 1:
            val_q_features = val_q_features[np.newaxis, :]
            val_img_features = val_img_features[np.newaxis, :]

        # Compute question similarity
        q_similarity = np.matmul(train_q_features, val_q_features.T)

        # Compute image similarity
        img_similarity = np.matmul(train_img_features, val_img_features.T)

        # Combine
        combined_similarity = q_similarity + img_similarity

        if combined_similarity.shape[1] == 1:
            combined_similarity = combined_similarity[:, 0]

        # Get top-k
        top_k_indices = combined_similarity.argsort()[-top_k:][::-1]
        top_k_scores = combined_similarity[top_k_indices]

        return top_k_scores, top_k_indices.tolist()


def extract_clip_image_embedding(
    image_path: str,
    model_name: str = "openai/clip-vit-base-patch16",
    normalize: bool = True,
    **kwargs,
) -> np.ndarray:
    """
    Convenience function to extract CLIP image embedding.

    Args:
        image_path: Path to image
        model_name: CLIP model name
        normalize: Whether to normalize features
        **kwargs: Additional arguments for CLIPFeatureExtractor

    Returns:
        Image embedding as numpy array
    """
    extractor = CLIPFeatureExtractor(model_name=model_name, **kwargs)
    return extractor.extract_image_features(image_path, normalize=normalize, return_numpy=True)


def extract_clip_text_embedding(
    text: Union[str, List[str]],
    model_name: str = "openai/clip-vit-base-patch16",
    normalize: bool = True,
    **kwargs,
) -> np.ndarray:
    """
    Convenience function to extract CLIP text embedding.

    Args:
        text: Text or list of texts
        model_name: CLIP model name
        normalize: Whether to normalize features
        **kwargs: Additional arguments for CLIPFeatureExtractor

    Returns:
        Text embedding as numpy array
    """
    extractor = CLIPFeatureExtractor(model_name=model_name, **kwargs)
    return extractor.extract_text_features(text, normalize=normalize, return_numpy=True)
