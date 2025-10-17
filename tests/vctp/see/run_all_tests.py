"""
Run all mock tests for the SEE module
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import traceback


def run_test_file(test_name, test_func):
    """Run a single test file"""
    print(f"\n{'=' * 70}")
    print(f"Running: {test_name}")
    print("=" * 70)

    try:
        test_func()
        print(f"\n✓ {test_name} PASSED")
        return True
    except Exception as e:
        print(f"\n✗ {test_name} FAILED")
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("VISUAL COT SEE MODULE - MOCK TEST SUITE")
    print("=" * 70)

    results = {}

    # Test 1: Scene Graph
    try:
        from test_scene_graph import (
            test_scene_graph_processor,
            test_scene_graph_builder,
            test_scene_graph_detector,
            test_scene_graph_utils,
        )

        results["scene_graph_processor"] = run_test_file(
            "Scene Graph Processor", test_scene_graph_processor
        )
        results["scene_graph_builder"] = run_test_file(
            "Scene Graph Builder", test_scene_graph_builder
        )
        results["scene_graph_detector"] = run_test_file(
            "Scene Graph Detector", test_scene_graph_detector
        )
        results["scene_graph_utils"] = run_test_file("Scene Graph Utils", test_scene_graph_utils)
    except Exception as e:
        print(f"Failed to import scene graph tests: {e}")
        results["scene_graph"] = False

    # Test 2: Features
    try:
        from test_features import test_feature_loader, test_clip_similarity_computer

        results["feature_loader"] = run_test_file("Feature Loader", test_feature_loader)
        results["clip_similarity"] = run_test_file(
            "CLIP Similarity Computer", test_clip_similarity_computer
        )
    except Exception as e:
        print(f"Failed to import feature tests: {e}")
        results["features"] = False

    # Test 3: Perception
    try:
        from test_perception import (
            test_noop_perception,
            test_visualcot_perception_basic,
            test_visualcot_perception_caption_strategy,
            test_perception_module_integration,
        )

        results["noop_perception"] = run_test_file("NoOp Perception", test_noop_perception)
        results["visualcot_basic"] = run_test_file(
            "VisualCoT Basic", test_visualcot_perception_basic
        )
        results["visualcot_caption"] = run_test_file(
            "VisualCoT Caption Strategy", test_visualcot_perception_caption_strategy
        )
        results["module_integration"] = run_test_file(
            "Module Integration", test_perception_module_integration
        )
    except Exception as e:
        print(f"Failed to import perception tests: {e}")
        results["perception"] = False

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print("=" * 70)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 70)

    if passed == total:
        print("\n All tests passed! Refactoring is correct!")
        return 0
    else:
        print(f"\n  {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
