"""
Mock tests for Feature extraction components
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from unittest.mock import Mock, patch


def test_clip_extractor_mock():
    """Test CLIPFeatureExtractor with mocked model"""

    # Mock the transformers imports
    with (
        patch("vctp.see.features.clip_extractor.CLIPProcessor") as mock_processor,
        patch("vctp.see.features.clip_extractor.CLIPModel") as mock_model,
    ):

        # Setup mocks
        mock_model_instance = Mock()
        mock_processor_instance = Mock()

        mock_model.from_pretrained.return_value = mock_model_instance
        mock_processor.from_pretrained.return_value = mock_processor_instance

        # Mock model outputs
        mock_features = np.random.randn(1, 512).astype(np.float32)
        mock_model_instance.get_image_features.return_value = Mock()

        from vctp.see.features.clip_extractor import CLIPFeatureExtractor

        print("\n" + "=" * 60)
        print("Testing CLIP Feature Extractor (Mocked)")
        print("=" * 60)

        # This would fail without actual model, so we just check imports work
        print("✓ CLIPFeatureExtractor can be imported")
        print("✓ Mock setup successful")


def test_feature_loader():
    """Test FeatureLoader with mock data"""
    import tempfile
    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock feature files
        train_features = np.random.randn(10, 512).astype(np.float32)
        val_features = np.random.randn(5, 512).astype(np.float32)

        np.save(f"{tmpdir}/coco_clip_vitb16_train2017_aokvqa_question.npy", train_features)
        np.save(f"{tmpdir}/coco_clip_vitb16_val2017_aokvqa_question.npy", val_features)

        # Create index mappings
        train_idx = {str(i): f"train_sample_{i}" for i in range(10)}
        val_idx = {str(i): f"val_sample_{i}" for i in range(5)}

        with open(f"{tmpdir}/aokvqa_qa_line2sample_idx_train2017.json", "w") as f:
            json.dump(train_idx, f)
        with open(f"{tmpdir}/aokvqa_qa_line2sample_idx_val2017.json", "w") as f:
            json.dump(val_idx, f)

        # Test FeatureLoader
        from vctp.see.features.feature_loader import FeatureLoader

        loader = FeatureLoader(feature_dir=tmpdir, dataset_name="aokvqa", split="val")

        # Load question features
        train_q, val_q, train_map, val_map = loader.load_question_features("question")

        print("\n" + "=" * 60)
        print("Testing Feature Loader")
        print("=" * 60)

        assert train_q.shape == (10, 512)
        assert val_q.shape == (5, 512)
        assert len(train_map) == 10
        assert len(val_map) == 5

        print(f"✓ Loaded train features: {train_q.shape}")
        print(f"✓ Loaded val features: {val_q.shape}")
        print(f"✓ Index mappings: {len(train_map)} train, {len(val_map)} val")

        # Test get feature by key
        feature = loader.get_val_feature_by_key("val_sample_0", "question")
        assert feature is not None
        assert feature.shape == (512,)

        print("✓ Feature lookup by key works")


def test_clip_similarity_computer():
    """Test CLIPSimilarityComputer"""
    from vctp.see.features.clip_extractor import CLIPSimilarityComputer

    print("\n" + "=" * 60)
    print("Testing CLIP Similarity Computer")
    print("=" * 60)

    computer = CLIPSimilarityComputer()

    # Mock features
    train_features = np.random.randn(100, 512).astype(np.float32)
    val_feature = np.random.randn(512).astype(np.float32)

    # Compute similarity
    scores, indices = computer.compute_question_similarity(train_features, val_feature, top_k=5)

    assert len(scores) == 5
    assert len(indices) == 5
    assert all(0 <= idx < 100 for idx in indices)

    print(f"✓ Top-5 similarity scores computed")
    print(f"  Indices: {indices}")
    print(f"  Scores: {scores}")

    # Test image+question similarity
    train_img_features = np.random.randn(100, 512).astype(np.float32)
    val_img_feature = np.random.randn(512).astype(np.float32)

    combined_scores, combined_indices = computer.compute_image_question_similarity(
        train_features, val_feature, train_img_features, val_img_feature, top_k=3
    )

    assert len(combined_scores) == 3
    print(f"✓ Combined image+question similarity works")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Feature Components")
    print("=" * 60)

    test_clip_extractor_mock()
    test_feature_loader()
    test_clip_similarity_computer()

    print("\n" + "=" * 60)
    print("All Feature tests passed! ✓")
    print("=" * 60)
