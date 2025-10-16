"""
Mock tests for Scene Graph components
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
import tempfile
import os


def test_scene_graph_processor():
    """Test SceneGraphProcessor basic functionality"""
    from vctp.see.graphs import SceneGraphProcessor

    processor = SceneGraphProcessor(strategy="sg", include_ocr=False)

    # Test data: [[conf, class, attrs, caption, ocr]]
    scene_graph_attrs = [
        [0.9, "table", ["wooden", "brown"], "A wooden brown table", ""],
        [0.8, "chair", ["metal"], "A metal chair", ""],
        [0.7, "book", [], "A book on the table", ""],
    ]

    # Test decode to text with "sg" strategy
    text_output = processor.decode(scene_graph_attrs, format_type="text")
    print("SG Strategy Output:")
    print(text_output)
    assert "table is wooden, brown" in text_output
    assert "chair is metal" in text_output

    # Test decode to list
    list_output = processor.decode(scene_graph_attrs, format_type="list")
    assert isinstance(list_output, list)
    assert len(list_output) == 3

    # Test decode to dict
    dict_output = processor.decode(scene_graph_attrs, format_type="dict")
    assert "objects" in dict_output
    assert len(dict_output["objects"]) == 3

    # Test with caption strategy
    processor_caption = SceneGraphProcessor(strategy="caption", include_ocr=False)
    text_caption = processor_caption.decode(scene_graph_attrs, format_type="text")
    print("\nCaption Strategy Output:")
    print(text_caption)
    assert "wooden brown table" in text_caption.lower()

    # Test filter by confidence
    filtered = processor.filter_by_confidence(scene_graph_attrs, threshold=0.75)
    assert len(filtered) == 2  # Only table and chair

    # Test get object list
    objects = processor.get_object_list(scene_graph_attrs, unique=True)
    assert objects == ["table", "chair", "book"]

    print("\n✓ SceneGraphProcessor tests passed!")


def test_scene_graph_builder():
    """Test SceneGraphBuilder"""
    from vctp.see.graphs import SceneGraphBuilder

    builder = SceneGraphBuilder(
        include_attributes=True, include_relationships=True, confidence_threshold=0.5
    )

    # Mock detections
    detections = [
        {"class": "person", "conf": 0.9, "rect": [100, 100, 200, 300], "attr": ["standing"]},
        {"class": "dog", "conf": 0.85, "rect": [250, 200, 350, 350], "attr": ["brown", "small"]},
        {"class": "low_conf_object", "conf": 0.3, "rect": [0, 0, 50, 50], "attr": []},
    ]

    # Build scene graph
    scene_graph = builder.build_from_detections(detections, image_size=(640, 480))

    print("\nScene Graph:")
    print(f"Nodes: {len(scene_graph['nodes'])}")
    print(f"Edges: {len(scene_graph['edges'])}")

    # Should filter out low confidence object
    assert len(scene_graph["nodes"]) == 2
    assert scene_graph["nodes"][0]["class"] == "person"
    assert scene_graph["nodes"][1]["class"] == "dog"

    # Check attributes
    assert "standing" in scene_graph["nodes"][0]["attributes"]
    assert "brown" in scene_graph["nodes"][1]["attributes"]

    # Check relationships
    assert len(scene_graph["edges"]) > 0
    assert "relation" in scene_graph["edges"][0]

    print("✓ SceneGraphBuilder tests passed!")


def test_scene_graph_detector():
    """Test SceneGraphDetector with mock files"""
    from vctp.see.detectors.scene_graph_detector import SceneGraphDetector

    # Create temporary directory for mock data
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock scene graph file
        sg_data = [
            [
                {
                    "class": "car",
                    "conf": 0.95,
                    "rect": [100, 100, 300, 200],
                    "attr": ["red", "sports"],
                },
                {
                    "class": "tree",
                    "conf": 0.88,
                    "rect": [400, 50, 500, 400],
                    "attr": ["green", "tall"],
                },
            ]
        ]

        sg_file = os.path.join(tmpdir, "000000000001.json")
        with open(sg_file, "w") as f:
            json.dump(sg_data, f)

        # Initialize detector
        detector = SceneGraphDetector(sg_dir=tmpdir, sg_attr_dir=tmpdir, iterative_strategy="sg")

        # Load scene graph
        attrs = detector.load_scene_graph(image_id=1, include_attributes=True)

        print(f"\nLoaded {len(attrs)} objects")
        assert len(attrs) == 2
        assert attrs[0][1] == "car"  # class name
        assert attrs[0][0] == 0.95  # confidence
        assert "red" in attrs[0][2]  # attributes

        # Test decode
        text = detector.decode_scene_graph(attrs)
        print("Decoded text:")
        print(text)
        assert "car is red, sports" in text

        print("✓ SceneGraphDetector tests passed!")


def test_scene_graph_utils():
    """Test scene graph utility functions"""
    from vctp.see.graphs import compute_iou, compute_distance, rect_area

    # Test IoU
    box1 = [0, 0, 100, 100]
    box2 = [50, 50, 150, 150]
    iou = compute_iou(box1, box2)
    print(f"\nIoU: {iou:.3f}")
    assert 0 < iou < 1

    # Test no overlap
    box3 = [200, 200, 300, 300]
    iou_no_overlap = compute_iou(box1, box3)
    assert iou_no_overlap == 0

    # Test distance
    distance = compute_distance(box1, box2)
    print(f"Distance: {distance:.2f}")
    assert distance > 0

    # Test area
    area = rect_area(box1)
    assert area == 10000  # 100 * 100

    print("✓ Scene graph utils tests passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Scene Graph Components")
    print("=" * 60)

    test_scene_graph_processor()
    test_scene_graph_builder()
    test_scene_graph_detector()
    test_scene_graph_utils()

    print("\n" + "=" * 60)
    print("All Scene Graph tests passed! ✓")
    print("=" * 60)
