"""
Test module See với data thực tế - Ngắn gọn và clean
Không mock - Sử dụng data có sẵn trong data/raw/
"""
import json
from pathlib import Path

def test_see_noop():
    """Test NoOpPerception - Baseline đơn giản nhất"""
    from vctp.see.perception import NoOpPerception
    
    # Load 1 sample từ data thực
    data_file = Path("data/raw/aokvqa_annotations/aokvqa_v1p0_train_from_hf.json")
    with open(data_file) as f:
        samples = json.load(f)
    
    sample = samples[0]  # Sample đầu tiên
    image_path = f"data/raw/aokvqa_images/{sample['image_id']:012d}.jpg"
    
    # Test See module
    see = NoOpPerception()
    result = see.run(
        image_path=image_path,
        question=sample['question']
    )
    
    # Verify output structure
    print(f"✓ NoOpPerception")
    print(f"  Image: {result.image_id}")
    print(f"  Caption: {result.global_caption}")
    print(f"  Objects: {len(result.detected_objects)}")
    
    assert result.image_id is not None
    assert result.global_caption is not None
    assert len(result.detected_objects) > 0


def test_see_visualcot_minimal():
    """Test VisualCoTPerception - Chỉ với scene graph (không BLIP2, không CLIP)"""
    from vctp.see.perception import VisualCoTPerception
    
    # Load sample
    data_file = Path("data/raw/aokvqa_annotations/aokvqa_v1p0_train_from_hf.json")
    with open(data_file) as f:
        samples = json.load(f)
    
    sample = samples[0]
    image_id = f"{sample['image_id']:012d}"
    image_path = f"data/raw/aokvqa_images/{image_id}.jpg"
    
    # Kiểm tra scene graph có tồn tại không
    sg_dir = Path("VisualCoT/input_text/scene_graph_text/scene_graph_coco17")
    sg_file = sg_dir / f"{image_id}.json"
    
    if not sg_file.exists():
        print(f"⚠ Scene graph không tồn tại: {sg_file}")
        print("  → Bỏ qua test này")
        return
    
    # Initialize See module (tối giản)
    see = VisualCoTPerception(
        sg_dir=str(sg_dir),
        iterative_strategy="sg",
        use_blip2=False,
        use_clip_features=False,
        debug=True
    )
    
    # Run perception
    result = see.run(
        image_path=image_path,
        question=sample['question'],
        max_objects=10
    )
    
    # Verify
    print(f"\n✓ VisualCoTPerception (Scene Graph Only)")
    print(f"  Question: {sample['question']}")
    print(f"  Image: {result.image_id}")
    print(f"  Objects detected: {len(result.detected_objects)}")
    if result.detected_objects:
        print(f"  First object: {result.detected_objects[0].name}")
    print(f"  Caption: {result.global_caption[:80]}...")
    print(f"  Attributes: {list(result.attributes.keys())[:3]}")
    
    assert len(result.detected_objects) > 0
    assert result.global_caption is not None


def test_see_with_clip():
    """Test VisualCoTPerception với CLIP features"""
    from vctp.see.perception import VisualCoTPerception
    
    # Load sample
    data_file = Path("data/raw/aokvqa_annotations/aokvqa_v1p0_train_from_hf.json")
    with open(data_file) as f:
        samples = json.load(f)
    
    sample = samples[0]
    image_id = f"{sample['image_id']:012d}"
    image_path = f"data/raw/aokvqa_images/{image_id}.jpg"
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"⚠ Image không tồn tại: {image_path}")
        return
    
    sg_dir = Path("VisualCoT/input_text/scene_graph_text/scene_graph_coco17")
    
    # Initialize với CLIP
    see = VisualCoTPerception(
        sg_dir=str(sg_dir) if sg_dir.exists() else None,
        use_clip_features=True,
        clip_model_name="openai/clip-vit-base-patch16",
        device="cpu",  # Use CPU for testing
        debug=True
    )
    
    # Run
    result = see.run(
        image_path=image_path,
        question=sample['question']
    )
    
    # Verify
    print(f"\n✓ VisualCoTPerception (with CLIP)")
    print(f"  Image: {result.image_id}")
    print(f"  Objects: {len(result.detected_objects)}")
    print(f"  CLIP features shape: {result.clip_features.shape if result.clip_features is not None else 'None'}")
    
    assert result.clip_features is not None, "CLIP features should be extracted"
    assert result.clip_features.shape[0] > 0


def test_see_pipeline_integration():
    """Test See module trong context của pipeline hoàn chỉnh"""
    from vctp.see.perception import VisualCoTPerception
    
    # Load 3 samples
    data_file = Path("data/raw/aokvqa_annotations/aokvqa_v1p0_train_from_hf.json")
    with open(data_file) as f:
        samples = json.load(f)[:3]
    
    sg_dir = Path("VisualCoT/input_text/scene_graph_text/scene_graph_coco17")
    
    # Initialize
    see = VisualCoTPerception(
        sg_dir=str(sg_dir) if sg_dir.exists() else None,
        use_clip_features=False,
        debug=False  # Tắt debug cho batch
    )
    
    results = []
    print(f"\n✓ Processing {len(samples)} samples...")
    
    for i, sample in enumerate(samples):
        image_id = f"{sample['image_id']:012d}"
        image_path = f"data/raw/aokvqa_images/{image_id}.jpg"
        
        try:
            result = see.run(
                image_path=image_path,
                question=sample['question']
            )
            results.append(result)
            print(f"  [{i+1}] {sample['question'][:50]}... → {len(result.detected_objects)} objects")
        except Exception as e:
            print(f"  [{i+1}] ERROR: {e}")
    
    print(f"\n  Total processed: {len(results)}/{len(samples)}")
    assert len(results) > 0, "Should process at least one sample"


if __name__ == "__main__":
    print("="*70)
    print("TEST MODULE SEE - VỚI DATA THỰC TẾ")
    print("="*70)
    
    # Test 1: NoOp baseline
    test_see_noop()
    
    # Test 2: Scene graph only
    test_see_visualcot_minimal()
    
    # Test 3: With CLIP
    test_see_with_clip()
    
    # Test 4: Batch processing
    test_see_pipeline_integration()
    
    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED")
    print("="*70)


# cd /home/research/workspace/VisualCoT-NLE
# PYTHONPATH=/home/research/workspace/VisualCoT-NLE/src python tests/vctp/see/quick_test.py