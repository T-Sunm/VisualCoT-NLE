"""
Scene Graph Utilities
Helper functions for scene graph operations
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


def compute_iou(rect1: List[float], rect2: List[float]) -> float:
    """
    Compute IoU between two bounding boxes.

    Args:
        rect1: [x1, y1, x2, y2]
        rect2: [x1, y1, x2, y2]

    Returns:
        IoU score
    """
    x1_min, y1_min, x1_max, y1_max = rect1
    x2_min, y2_min, x2_max, y2_max = rect2

    # Check if boxes overlap
    if x1_min >= x2_max or x1_max <= x2_min or y1_min >= y2_max or y1_max <= y2_min:
        return 0.0

    # Compute intersection
    inter_width = min(x1_max, x2_max) - max(x1_min, x2_min)
    inter_height = min(y1_max, y2_max) - max(y1_min, y2_min)
    intersection = inter_width * inter_height

    # Compute union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def compute_distance(rect1: List[float], rect2: List[float]) -> float:
    """
    Compute Euclidean distance between centers of two boxes.

    Args:
        rect1: [x1, y1, x2, y2]
        rect2: [x1, y1, x2, y2]

    Returns:
        Distance between centers
    """
    c1_x = (rect1[0] + rect1[2]) / 2
    c1_y = (rect1[1] + rect1[3]) / 2
    c2_x = (rect2[0] + rect2[2]) / 2
    c2_y = (rect2[1] + rect2[3]) / 2

    return np.sqrt((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2)


def normalize_rect(rect: List[float], image_size: Tuple[int, int]) -> List[float]:
    """
    Normalize bounding box coordinates to [0, 1].

    Args:
        rect: [x1, y1, x2, y2]
        image_size: (width, height)

    Returns:
        Normalized rect
    """
    width, height = image_size
    return [rect[0] / width, rect[1] / height, rect[2] / width, rect[3] / height]


def denormalize_rect(rect: List[float], image_size: Tuple[int, int]) -> List[float]:
    """
    Denormalize bounding box coordinates from [0, 1] to pixels.

    Args:
        rect: Normalized [x1, y1, x2, y2]
        image_size: (width, height)

    Returns:
        Denormalized rect
    """
    width, height = image_size
    return [rect[0] * width, rect[1] * height, rect[2] * width, rect[3] * height]


def rect_area(rect: List[float]) -> float:
    """
    Compute area of bounding box.

    Args:
        rect: [x1, y1, x2, y2]

    Returns:
        Area in pixels
    """
    return (rect[2] - rect[0]) * (rect[3] - rect[1])


def get_object_relationships(scene_graph: Dict, object_id: int) -> List[Dict]:
    """
    Get all relationships involving a specific object.

    Args:
        scene_graph: Scene graph dictionary
        object_id: ID of object

    Returns:
        List of relationships
    """
    relationships = []

    for edge in scene_graph.get("edges", []):
        if edge["subject"] == object_id or edge["object"] == object_id:
            relationships.append(edge)

    return relationships


def find_objects_by_class(scene_graph: Dict, class_name: str) -> List[Dict]:
    """
    Find all objects of a specific class in scene graph.

    Args:
        scene_graph: Scene graph dictionary
        class_name: Object class name

    Returns:
        List of matching nodes
    """
    return [node for node in scene_graph["nodes"] if node["class"].lower() == class_name.lower()]


def get_top_k_objects(scene_graph: Dict, k: int = 10, by: str = "confidence") -> List[Dict]:
    """
    Get top-k objects from scene graph.

    Args:
        scene_graph: Scene graph dictionary
        k: Number of objects to return
        by: Sorting criterion ("confidence", "area")

    Returns:
        List of top-k nodes
    """
    nodes = scene_graph["nodes"]

    if by == "confidence":
        sorted_nodes = sorted(nodes, key=lambda x: x["conf"], reverse=True)
    elif by == "area":
        sorted_nodes = sorted(nodes, key=lambda x: rect_area(x["rect"]), reverse=True)
    else:
        sorted_nodes = nodes

    return sorted_nodes[:k]
