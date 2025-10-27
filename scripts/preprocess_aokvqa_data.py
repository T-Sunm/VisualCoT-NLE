"""
Preprocess AOKVQA data for the refactored pipeline.
Based on VisualCoT/preprocess/ scripts.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def reorganize_captions(coco_dir: str, output_dir: str):
    """Reorganize COCO captions from list to dict format."""
    from vctp.data.preprocess import reorganize_captions as reorganize

    print("Reorganizing captions...")
    reorganize(coco_dir, output_dir)
    print("✓ Done")


def make_clip_features(
    annotations_file: str,
    images_dir: str,
    output_file: str,
    model_name: str = "openai/clip-vit-base-patch16",
):
    """Extract CLIP features for all images."""
    from vctp.data.preprocess import make_clip_features as extract_features

    print(f"Extracting CLIP features with {model_name}...")
    extract_features(
        annotations_file=annotations_file,
        images_dir=images_dir,
        output_file=output_file,
        model_name=model_name,
    )
    print("✓ Done")


def make_line2sample_mapping(annotations_file: str, output_file: str):
    """Create line-to-sample index mapping."""
    from vctp.data.preprocess import make_line2sample

    print("Creating line-to-sample mapping...")
    make_line2sample(annotations_file=annotations_file, output_file=output_file)
    print("✓ Done")


def build_object_similarity(
    annotations_file: str, captions_file: str, output_file: str, metric: str = "answer"
):
    """Build object similarity index."""
    from vctp.data.preprocess.object_similarity import ObjectSimilarityBuilder

    print(f"Building object similarity (metric: {metric})...")
    builder = ObjectSimilarityBuilder(
        annotations_file=annotations_file, captions_file=captions_file, metric=metric
    )
    builder.build_and_save(output_file)
    print("✓ Done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["captions", "clip", "line2sample", "object_similarity", "all"],
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument(
        "--annotations",
        type=str,
        default="data/raw/aokvqa_annotations/aokvqa_v1p0_train_from_hf.json",
    )
    parser.add_argument("--images", type=str, default="data/raw/aokvqa_images")
    parser.add_argument("--output_dir", type=str, default="data/processed")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.task in ["captions", "all"]:
        reorganize_captions(
            coco_dir=os.path.join(args.data_dir, "raw/coco_annotations"),
            output_dir=os.path.join(args.output_dir),
        )

    if args.task in ["clip", "all"]:
        make_clip_features(
            annotations_file=args.annotations,
            images_dir=args.images,
            output_file=os.path.join(args.output_dir, "clip_features.npy"),
        )

    if args.task in ["line2sample", "all"]:
        make_line2sample_mapping(
            annotations_file=args.annotations,
            output_file=os.path.join(args.output_dir, "line2sample.json"),
        )

    if args.task in ["object_similarity", "all"]:
        build_object_similarity(
            annotations_file=args.annotations,
            captions_file=os.path.join(args.output_dir, "captions_train2017.json"),
            output_file=os.path.join(args.output_dir, "object_similarity.pkl"),
        )

    print("\n✓ All preprocessing tasks completed!")


if __name__ == "__main__":
    main()
