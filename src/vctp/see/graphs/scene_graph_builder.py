"""
Scene Graph Builder
Constructs scene graphs from object detections and relationships
Extracted and refactored from main_aokvqa.py
"""

import json
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path


class SceneGraphBuilder:
    """
    Builder for constructing scene graphs from object detections.
    Supports multiple formats and relationship extraction.
    """

    def __init__(
        self,
        include_attributes: bool = True,
        include_relationships: bool = False,
        confidence_threshold: float = 0.0,
    ):
        """
        Initialize Scene Graph Builder.

        Args:
            include_attributes: Whether to include object attributes
            include_relationships: Whether to extract relationships
            confidence_threshold: Minimum confidence for objects
        """
        self.include_attributes = include_attributes
        self.include_relationships = include_relationships
        self.confidence_threshold = confidence_threshold

    def build_from_detections(
        self, detections: List[Dict], image_size: Optional[Tuple[int, int]] = None
    ) -> Dict:
        """
        Build scene graph from object detections.

        Args:
            detections: List of object detections with format:
                [{'class': str, 'conf': float, 'rect': [x1,y1,x2,y2], 'attr': []}]
            image_size: Optional (width, height) of image

        Returns:
            Scene graph dictionary with nodes and edges
        """
        # Filter by confidence
        filtered_detections = [
            det for det in detections if det.get("conf", 0.0) >= self.confidence_threshold
        ]

        # Build nodes (objects)
        nodes = []
        for i, det in enumerate(filtered_detections):
            node = {
                "id": i,
                "class": det["class"],
                "conf": det["conf"],
                "rect": det.get("rect", [0, 0, 0, 0]),
            }

            if self.include_attributes and "attr" in det:
                node["attributes"] = det["attr"]

            nodes.append(node)

        # Build edges (relationships)
        edges = []
        if self.include_relationships:
            edges = self._extract_relationships(nodes, image_size)

        return {"nodes": nodes, "edges": edges, "image_size": image_size}

    def _extract_relationships(
        self, nodes: List[Dict], image_size: Optional[Tuple[int, int]] = None
    ) -> List[Dict]:
        """
        Extract spatial relationships between objects.

        Args:
            nodes: List of object nodes
            image_size: Image dimensions

        Returns:
            List of relationship edges
        """
        edges = []

        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i >= j:
                    continue

                # Compute spatial relationship
                rel = self._compute_spatial_relation(node1["rect"], node2["rect"], image_size)

                if rel:
                    edges.append({"subject": i, "object": j, "relation": rel})

        return edges

    def _compute_spatial_relation(
        self, rect1: List[float], rect2: List[float], image_size: Optional[Tuple[int, int]] = None
    ) -> Optional[str]:
        """
        Compute spatial relationship between two bounding boxes.

        Args:
            rect1: [x1, y1, x2, y2]
            rect2: [x1, y1, x2, y2]
            image_size: Image dimensions

        Returns:
            Relationship string or None
        """
        x1_c = (rect1[0] + rect1[2]) / 2
        y1_c = (rect1[1] + rect1[3]) / 2
        x2_c = (rect2[0] + rect2[2]) / 2
        y2_c = (rect2[1] + rect2[3]) / 2

        # Vertical relationship
        if y1_c < y2_c - 20:
            v_rel = "above"
        elif y1_c > y2_c + 20:
            v_rel = "below"
        else:
            v_rel = None

        # Horizontal relationship
        if x1_c < x2_c - 20:
            h_rel = "left of"
        elif x1_c > x2_c + 20:
            h_rel = "right of"
        else:
            h_rel = None

        # Combine relationships
        if v_rel and h_rel:
            return f"{v_rel} and {h_rel}"
        elif v_rel:
            return v_rel
        elif h_rel:
            return h_rel
        else:
            return "near"

    def build_from_json(self, json_path: str) -> Dict:
        """
        Build scene graph from JSON file.

        Args:
            json_path: Path to scene graph JSON

        Returns:
            Scene graph dictionary
        """
        with open(json_path, "r") as f:
            data = json.load(f)

        # Handle different JSON formats
        if isinstance(data, list):
            detections = data[0] if len(data) > 0 else []
        elif isinstance(data, dict):
            detections = data.get("objects", data.get("detections", []))
        else:
            detections = []

        return self.build_from_detections(detections)


class SceneGraphProcessor:
    """
    Processor for scene graphs - decoding, formatting, filtering.
    Based on decode_scene_graph() from main_aokvqa.py
    """

    def __init__(self, strategy: str = "caption", include_ocr: bool = False):
        """
        Initialize Scene Graph Processor.

        Args:
            strategy: Processing strategy ("sg", "caption")
            include_ocr: Whether to include OCR text
        """
        self.strategy = strategy
        self.include_ocr = include_ocr

    def decode(
        self, scene_graph_attrs: List[List], format_type: str = "text"
    ) -> Union[str, List[str], Dict]:
        """
        Decode scene graph attributes to various formats.

        Args:
            scene_graph_attrs: List of attribute lists
                Format: [[conf, class, [attrs], caption, ocr_text], ...]
            format_type: Output format ("text", "list", "dict")

        Returns:
            Decoded scene graph in specified format
        """
        attr_list = []

        for attr in scene_graph_attrs:
            if self.strategy == "sg":
                # Format: "object is attr1, attr2, ..."
                if len(attr) >= 3 and attr[2]:
                    text = f"{attr[1]} is {', '.join(attr[2])}"
                else:
                    text = attr[1]
                attr_list.append(text)

            elif self.strategy == "caption":
                # Use pre-generated captions
                if len(attr) >= 4:
                    attr_list.append(attr[3])

                    # Add OCR if available and requested
                    if self.include_ocr and len(attr) >= 5 and attr[4]:
                        attr_list.append(attr[4])

        # Format output
        if format_type == "text":
            return "\n".join(attr_list)
        elif format_type == "list":
            return attr_list
        elif format_type == "dict":
            return {
                "objects": [
                    {
                        "class": attr[1],
                        "conf": attr[0],
                        "attributes": attr[2] if len(attr) > 2 else [],
                        "caption": attr[3] if len(attr) > 3 else "",
                        "ocr": attr[4] if len(attr) > 4 else "",
                    }
                    for attr in scene_graph_attrs
                ]
            }
        else:
            return "\n".join(attr_list)

    def filter_by_confidence(
        self, scene_graph_attrs: List[List], threshold: float = 0.3
    ) -> List[List]:
        """
        Filter scene graph attributes by confidence.

        Args:
            scene_graph_attrs: List of attribute lists
            threshold: Minimum confidence

        Returns:
            Filtered attributes
        """
        return [attr for attr in scene_graph_attrs if attr[0] >= threshold]

    def filter_by_objects(
        self, scene_graph_attrs: List[List], object_names: List[str]
    ) -> List[List]:
        """
        Filter scene graph to include only specified objects.

        Args:
            scene_graph_attrs: List of attribute lists
            object_names: List of object names to keep

        Returns:
            Filtered attributes
        """
        return [attr for attr in scene_graph_attrs if attr[1] in object_names]

    def sort_by_confidence(self, scene_graph_attrs: List[List], reverse: bool = True) -> List[List]:
        """
        Sort scene graph attributes by confidence.

        Args:
            scene_graph_attrs: List of attribute lists
            reverse: Sort in descending order

        Returns:
            Sorted attributes
        """
        return sorted(scene_graph_attrs, key=lambda x: x[0], reverse=reverse)

    def get_object_list(self, scene_graph_attrs: List[List], unique: bool = True) -> List[str]:
        """
        Extract list of object names from scene graph.

        Args:
            scene_graph_attrs: List of attribute lists
            unique: Return only unique object names

        Returns:
            List of object names
        """
        objects = [attr[1] for attr in scene_graph_attrs]

        if unique:
            return list(dict.fromkeys(objects))  # Preserve order

        return objects

    def merge_duplicate_objects(self, scene_graph_attrs: List[List]) -> List[List]:
        """
        Merge attributes for duplicate objects.

        Args:
            scene_graph_attrs: List of attribute lists

        Returns:
            Merged attributes
        """
        merged = {}

        for attr in scene_graph_attrs:
            obj_name = attr[1]

            if obj_name not in merged:
                merged[obj_name] = attr.copy()
            else:
                # Keep higher confidence
                if attr[0] > merged[obj_name][0]:
                    merged[obj_name][0] = attr[0]

                # Merge attributes
                if len(attr) > 2 and len(merged[obj_name]) > 2:
                    existing_attrs = set(merged[obj_name][2])
                    new_attrs = set(attr[2])
                    merged[obj_name][2] = list(existing_attrs | new_attrs)

        return list(merged.values())


class SceneGraphVisualizer:
    """
    Visualizer for scene graphs (optional, for debugging).
    """

    def __init__(self):
        """Initialize visualizer."""
        pass

    def to_dot(self, scene_graph: Dict) -> str:
        """
        Convert scene graph to DOT format for visualization.

        Args:
            scene_graph: Scene graph dictionary

        Returns:
            DOT format string
        """
        lines = ["digraph SceneGraph {"]

        # Add nodes
        for node in scene_graph["nodes"]:
            label = f"{node['class']} ({node['conf']:.2f})"
            lines.append(f'  {node["id"]} [label="{label}"];')

        # Add edges
        for edge in scene_graph.get("edges", []):
            lines.append(
                f'  {edge["subject"]} -> {edge["object"]} ' f'[label="{edge["relation"]}"];'
            )

        lines.append("}")
        return "\n".join(lines)

    def to_text_summary(self, scene_graph: Dict) -> str:
        """
        Create text summary of scene graph.

        Args:
            scene_graph: Scene graph dictionary

        Returns:
            Text summary
        """
        lines = []

        # Objects
        lines.append("Objects:")
        for node in scene_graph["nodes"]:
            attrs = node.get("attributes", [])
            attr_str = f" ({', '.join(attrs)})" if attrs else ""
            lines.append(f"  - {node['class']}{attr_str} [conf: {node['conf']:.2f}]")

        # Relationships
        if scene_graph.get("edges"):
            lines.append("\nRelationships:")
            nodes_by_id = {n["id"]: n for n in scene_graph["nodes"]}
            for edge in scene_graph["edges"]:
                subj = nodes_by_id[edge["subject"]]["class"]
                obj = nodes_by_id[edge["object"]]["class"]
                lines.append(f"  - {subj} {edge['relation']} {obj}")

        return "\n".join(lines)


def build_scene_graph(image_path: str, detections: Optional[List[Dict]] = None, **kwargs) -> Dict:
    """
    Convenience function to build scene graph.

    Args:
        image_path: Path to image (for context)
        detections: Object detections (if None, will attempt to load)
        **kwargs: Additional arguments for SceneGraphBuilder

    Returns:
        Scene graph dictionary
    """
    builder = SceneGraphBuilder(**kwargs)

    if detections is None:
        # Try to load from corresponding JSON file
        json_path = Path(image_path).with_suffix(".json")
        if json_path.exists():
            return builder.build_from_json(str(json_path))
        else:
            return {"nodes": [], "edges": []}

    return builder.build_from_detections(detections)
