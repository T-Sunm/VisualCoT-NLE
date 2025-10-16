"""
Mock tests for Perception module
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import tempfile
import json
import numpy as np


def test_noop_perception():
    """Test NoOpPerception"""
    from vctp.see.perception import NoOpPerception

    print("\n" + "=" * 60)
    print("Testing NoOp Perception")
    print("=" * 60)

    perception = NoOpPerception()

    result = perception.run(image_path="/fake/path/image.jpg", question="What is in the image?")

    assert result.image_id == "/fake/path/image.jpg"
    assert result.global_caption == "placeholder caption"
    assert len(result.detected_objects) == 1
    assert result.detected_objects[0].name == "object"

    print("✓ NoOpPerception works")
    print(f"  Image ID: {result.image_id}")
    print(f"  Caption: {result.global_caption}")
    print(f"  Objects: {len(result.detected_objects)}")


def test_visualcot_perception_basic():
    """Test VisualCoTPerception with scene graph only"""
    from vctp.see.perception import VisualCoTPerception

    print("\n" + "=" * 60)
    print("Testing VisualCoT Perception (Scene Graph Only)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock scene graph
        sg_data = [
            [
                {
                    "class": "person",
                    "conf": 0.95,
                    "rect": [100, 100, 200, 300],
                    "attr": ["standing", "wearing hat"],
                },
                {"class": "bicycle", "conf": 0.88, "rect": [250, 200, 400, 350], "attr": ["red"]},
            ]
        ]

        sg_file = Path(tmpdir) / "test_image.json"
        with open(sg_file, "w") as f:
            json.dump(sg_data, f)

        # Initialize perception (without BLIP2 and CLIP for simplicity)
        perception = VisualCoTPerception(
            sg_dir=tmpdir,
            sg_attr_dir=tmpdir,
            iterative_strategy="sg",
            use_blip2=False,
            use_clip_features=False,
            debug=True,
        )

        # Run perception
        result = perception.run(
            image_path=str(sg_file.with_suffix(".jpg")),
            question="What is the person doing?",
            max_objects=10,
        )

        print(f"\n✓ Perception completed")
        print(f"  Image ID: {result.image_id}")
        print(f"  Detected objects: {len(result.detected_objects)}")
        print(f"  Global caption: {result.global_caption[:100]}")
        print(f"  Attributes: {result.attributes}")

        assert len(result.detected_objects) == 2
        assert result.detected_objects[0].name == "person"
        assert "standing" in result.detected_objects[0].attributes
        assert result.image_id == "test_image"


def test_visualcot_perception_caption_strategy():
    """Test with caption strategy"""
    from vctp.see.perception import VisualCoTPerception

    print("\n" + "=" * 60)
    print("Testing VisualCoT Perception (Caption Strategy)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create scene graph with captions
        sg_data = [
            [{"class": "dog", "conf": 0.92, "rect": [50, 50, 200, 200], "attr": ["furry", "brown"]}]
        ]

        sg_file = Path(tmpdir) / "dog_image.json"
        with open(sg_file, "w") as f:
            json.dump(sg_data, f)

        # Create caption file
        caption_dir = Path(tmpdir) / "captions"
        caption_dir.mkdir()
        captions = ["A brown furry dog sitting"]

        caption_file = caption_dir / "dog_image.json"
        with open(caption_file, "w") as f:
            json.dump(captions, f)

        perception = VisualCoTPerception(
            sg_dir=tmpdir,
            sg_attr_dir=tmpdir,
            sg_caption_dir=str(caption_dir),
            iterative_strategy="caption",
            use_blip2=False,
            use_clip_features=False,
            debug=True,
        )

        result = perception.run(
            image_path=str(sg_file.with_suffix(".jpg")),
            question="What animal is this?",
            max_objects=5,
        )

        print(f"\n✓ Caption strategy works")
        print(f"  Objects: {len(result.detected_objects)}")
        print(f"  Caption includes: {result.global_caption[:50]}")


def test_perception_module_integration():
    """Test that all components work together"""
    from vctp.see.perception import VisualCoTPerception
    from vctp.see.graphs import SceneGraphProcessor
    from vctp.see.detectors.scene_graph_detector import SceneGraphDetector

    print("\n" + "=" * 60)
    print("Testing Module Integration")
    print("=" * 60)

    # Test that components can be imported and initialized
    processor = SceneGraphProcessor(strategy="sg")
    print("✓ SceneGraphProcessor initialized")

    # Test with mock data
    mock_attrs = [
        [0.9, "cat", ["fluffy", "white"], "A fluffy white cat", ""],
        [0.8, "sofa", ["leather"], "A leather sofa", ""],
    ]

    text = processor.decode(mock_attrs, format_type="text")
    print(f"✓ Processed scene graph: {text[:50]}...")

    # Test filtering
    filtered = processor.filter_by_confidence(mock_attrs, threshold=0.85)
    assert len(filtered) == 1
    print(f"✓ Filtering works: {len(filtered)} objects above threshold")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Perception Module")
    print("=" * 60)

    test_noop_perception()
    test_visualcot_perception_basic()
    test_visualcot_perception_caption_strategy()
    test_perception_module_integration()

    print("\n" + "=" * 60)
    print("All Perception tests passed! ✓")
    print("=" * 60)
