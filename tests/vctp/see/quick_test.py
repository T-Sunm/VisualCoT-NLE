"""
Quick smoke test - minimal dependencies
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def quick_test():
    """Quick test that modules can be imported"""
    print("Quick Smoke Test - Checking imports...")
    print(f"Project root: {project_root}")
    print("-" * 60)

    try:
        from vctp.see.graphs import SceneGraphProcessor, SceneGraphBuilder

        print("✓ Scene Graph modules imported")
    except Exception as e:
        print(f"✗ Scene Graph import failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    try:
        from vctp.see.detectors.scene_graph_detector import SceneGraphDetector

        print("✓ Scene Graph Detector imported")
    except Exception as e:
        print(f"✗ Detector import failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    try:
        from vctp.see.perception import NoOpPerception, VisualCoTPerception

        print("✓ Perception modules imported")
    except Exception as e:
        print(f"✗ Perception import failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Quick functionality test
    try:
        processor = SceneGraphProcessor(strategy="sg")
        mock_data = [[0.9, "cat", ["fluffy"], "a cat", ""]]
        result = processor.decode(mock_data, format_type="text")
        assert "cat" in result
        print("✓ Basic functionality works")
        print(f"  Result: {result}")
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("-" * 60)
    print("✓ All quick tests passed!")
    return True


if __name__ == "__main__":
    success = quick_test()
    exit(0 if success else 1)
