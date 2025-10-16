"""
Scene Graph Module
Provides scene graph building, processing, and utilities
"""

from typing import Optional, Dict, List

try:
    from .scene_graph_builder import (
        SceneGraphBuilder,
        SceneGraphProcessor,
        SceneGraphVisualizer,
        build_scene_graph,
    )
except ImportError:
    SceneGraphBuilder = None
    SceneGraphProcessor = None
    SceneGraphVisualizer = None
    build_scene_graph = None

try:
    from .scene_graph_utils import (
        compute_iou,
        compute_distance,
        normalize_rect,
        denormalize_rect,
        rect_area,
        get_object_relationships,
        find_objects_by_class,
        get_top_k_objects,
    )
except ImportError:
    compute_iou = None
    compute_distance = None
    normalize_rect = None
    denormalize_rect = None
    rect_area = None
    get_object_relationships = None
    find_objects_by_class = None
    get_top_k_objects = None


__all__ = [
    "SceneGraphBuilder",
    "SceneGraphProcessor",
    "SceneGraphVisualizer",
    "build_scene_graph",
    "compute_iou",
    "compute_distance",
    "normalize_rect",
    "denormalize_rect",
    "rect_area",
    "get_object_relationships",
    "find_objects_by_class",
    "get_top_k_objects",
    "get_processor",
]


def get_processor(strategy: str = "caption", **kwargs):
    """
    Factory function to get scene graph processor.

    Args:
        strategy: Processing strategy ("sg", "caption")
        **kwargs: Additional arguments

    Returns:
        SceneGraphProcessor instance
    """
    if SceneGraphProcessor is None:
        raise ImportError("SceneGraphProcessor not available")

    return SceneGraphProcessor(strategy=strategy, **kwargs)
