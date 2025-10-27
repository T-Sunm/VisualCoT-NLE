"""
Test script to verify if we can run a single AOKVQA sample end-to-end.
This tests the minimal viable pipeline without requiring full preprocessing.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_data_loading():
    """Test if we can load AOKVQA data."""
    print("\n=== Test 1: Data Loading ===")
    try:
        from vctp.data.loader import build_dataset

        # Mock dataset config
        dataset_cfg = {
            "name": "aokvqa",
            "annotations_file": "data/raw/aokvqa_annotations/aokvqa_v1p0_train_from_hf.json",
            "images_root": "data/raw/aokvqa_images",
        }

        dataset = build_dataset(dataset_cfg, "train")
        sample = next(iter(dataset))

        print(f"✓ Loaded sample: {sample['question'][:50]}...")
        print(f"  Image ID: {sample['image_id']}")
        print(f"  Question ID: {sample['question_id']}")
        print(f"  Image Path: {sample['image_path']}")
        return True, sample
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        import traceback

        traceback.print_exc()
        return False, None


def test_perception_module(sample):
    """Test if perception module can be initialized."""
    print("\n=== Test 2: Perception Module ===")
    try:
        from vctp.see.perception import NoOpPerception

        perception = NoOpPerception()
        evidence = perception.run(sample["image_path"], sample["question"])

        print(f"✓ Perception module works")
        print(f"  Global caption: {evidence.global_caption}")
        print(f"  Detected objects: {len(evidence.detected_objects)}")
        return True, evidence
    except Exception as e:
        print(f"✗ Perception module failed: {e}")
        import traceback

        traceback.print_exc()
        return False, None


def test_reasoning_module(evidence, question):
    """Test if reasoning module can be initialized."""
    print("\n=== Test 3: Reasoning Module ===")
    try:
        from vctp.think.reasoner import NoOpReasoner

        reasoner = NoOpReasoner()
        reasoning = reasoner.run(evidence, question)

        print(f"✓ Reasoning module works")
        print(f"  Answer: {reasoning.candidate_answer}")
        print(f"  Rationale: {reasoning.cot_rationale}")
        return True, reasoning
    except Exception as e:
        print(f"✗ Reasoning module failed: {e}")
        import traceback

        traceback.print_exc()
        return False, None


def test_confirmation_module(question, reasoning, evidence):
    """Test if confirmation module can be initialized."""
    print("\n=== Test 4: Confirmation Module ===")
    try:
        from vctp.confirm.confirmer import NoOpConfirmer

        confirmer = NoOpConfirmer()
        confirmation = confirmer.run(question, reasoning, evidence)

        print(f"✓ Confirmation module works")
        print(f"  Is confirmed: {confirmation.is_confirmed}")
        print(f"  Score: {confirmation.score}")
        return True, confirmation
    except Exception as e:
        print(f"✗ Confirmation module failed: {e}")
        import traceback

        traceback.print_exc()
        return False, None


def test_full_pipeline():
    """Test if full pipeline can run."""
    print("\n=== Test 5: Full Pipeline ===")
    try:
        from vctp.core.pipeline import VCTPPipeline
        from vctp.see.perception import NoOpPerception
        from vctp.think.reasoner import NoOpReasoner
        from vctp.confirm.confirmer import NoOpConfirmer

        pipeline = VCTPPipeline(see=NoOpPerception(), think=NoOpReasoner(), confirm=NoOpConfirmer())

        sample = {
            "image_id": "0",
            "image_path": "data/raw/aokvqa_images/000000000000.jpg",
            "question": "What is in the image?",
            "question_id": "test_0",
        }

        result = pipeline.run(sample)

        print(f"✓ Full pipeline works")
        print(f"  Answer: {result['answer']}")
        print(f"  Confirmed: {result['confirmed']}")
        print(f"  Score: {result['score']}")
        return True
    except Exception as e:
        print(f"✗ Full pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_preprocessing_scripts():
    """Test if preprocessing scripts are available."""
    print("\n=== Test 6: Preprocessing Scripts ===")
    try:
        # Check if preprocessing modules exist
        from vctp.data.preprocess import make_clip_features, make_line2sample

        print(f"✓ make_clip_features: Available")
        print(f"✓ make_line2sample: Available")

        # Check object similarity
        from vctp.data.preprocess.object_similarity import (
            ObjectSimilarityBuilder,
            AnswerSimilarityMetric,
        )

        print(f"✓ ObjectSimilarityBuilder: Available")
        print(f"✓ AnswerSimilarityMetric: Available")

        return True
    except Exception as e:
        print(f"✗ Preprocessing scripts check failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_api_utilities():
    """Test if API utilities are available."""
    print("\n=== Test 7: API Utilities ===")
    try:
        from vctp.utils import openai_complete, blip_completev2

        print(f"✓ openai_complete: Available")
        print(f"✓ blip_completev2: Available")
        return True
    except Exception as e:
        print(f"✗ API utilities check failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("TESTING: Can we run 1 AOKVQA sample?")
    print("=" * 70)

    results = {}

    # Test 1: Data Loading
    success, sample = test_data_loading()
    results["data_loading"] = success
    if not success:
        print("\n⚠ Skipping further tests due to data loading failure")
        print_summary(results)
        return

    # Test 2: Perception
    success, evidence = test_perception_module(sample)
    results["perception"] = success

    # Test 3: Reasoning
    if evidence:
        success, reasoning = test_reasoning_module(evidence, sample["question"])
        results["reasoning"] = success
    else:
        results["reasoning"] = False
        reasoning = None

    # Test 4: Confirmation
    if reasoning and evidence:
        success, confirmation = test_confirmation_module(sample["question"], reasoning, evidence)
        results["confirmation"] = success
    else:
        results["confirmation"] = False

    # Test 5: Full Pipeline
    success = test_full_pipeline()
    results["full_pipeline"] = success

    # Test 6: Preprocessing
    success = test_preprocessing_scripts()
    results["preprocessing"] = success

    # Test 7: API Utilities
    success = test_api_utilities()
    results["api_utilities"] = success

    # Print summary
    print_summary(results)


def print_summary(results):
    """Print test summary."""
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    total = len(results)
    passed = sum(results.values())

    for test_name, passed_flag in results.items():
        status = "✓ PASS" if passed_flag else "✗ FAIL"
        print(f"  {status}: {test_name}")

    print("-" * 70)
    print(f"  Total: {passed}/{total} tests passed")
    print("=" * 70)

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! You can run 1 AOKVQA sample.")
    elif passed >= total * 0.7:
        print("\n⚠ MOSTLY WORKING! Some components need attention.")
    else:
        print("\n❌ NEEDS WORK! Several components are missing or broken.")


if __name__ == "__main__":
    main()
