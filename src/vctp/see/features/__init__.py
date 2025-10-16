"""
Feature Extraction Module
Provides CLIP and other visual feature extractors
"""

from typing import Optional

try:
    from .clip_extractor import (
        CLIPFeatureExtractor,
        CLIPSimilarityComputer,
        extract_clip_image_embedding,
        extract_clip_text_embedding,
    )
except ImportError:
    CLIPFeatureExtractor = None
    CLIPSimilarityComputer = None
    extract_clip_image_embedding = None
    extract_clip_text_embedding = None

try:
    from .feature_loader import FeatureLoader
except ImportError:
    FeatureLoader = None


__all__ = [
    "CLIPFeatureExtractor",
    "CLIPSimilarityComputer",
    "FeatureLoader",
    "extract_clip_image_embedding",
    "extract_clip_text_embedding",
    "get_feature_extractor",
]


def get_feature_extractor(extractor_type: str = "clip", **kwargs):
    """
    Factory function to get feature extractor.

    Args:
        extractor_type: Type of extractor ('clip')
        **kwargs: Arguments for extractor initialization

    Returns:
        Feature extractor instance
    """
    if extractor_type == "clip":
        if CLIPFeatureExtractor is None:
            raise ImportError("CLIP extractor not available")
        return CLIPFeatureExtractor(**kwargs)

    else:
        raise ValueError(f"Unknown extractor type: {extractor_type}")
