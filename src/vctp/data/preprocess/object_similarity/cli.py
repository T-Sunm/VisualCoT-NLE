"""CLI for building object similarity scores."""

import argparse
from pathlib import Path

from .similarity_builder import ObjectSimilarityBuilder
from .metrics import AnswerSimilarityMetric, RationaleSimilarityMetric
from .aokvqa_processor import AOKVQASimilarityProcessor


def main():
    parser = argparse.ArgumentParser(description="Build object similarity scores")
    parser.add_argument("--dataset", choices=["aokvqa"], required=True)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--sg_path", type=str, required=True)
    parser.add_argument("--annotations_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--metric", choices=["answer", "rationale"], default="answer")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Initialize metric
    if args.metric == "answer":
        metric = AnswerSimilarityMetric()
    else:
        metric = RationaleSimilarityMetric()

    # Initialize processor
    if args.dataset == "aokvqa":
        processor = AOKVQASimilarityProcessor(args.annotations_dir)

    # Load data
    questions, answers, rationales = processor.load_split(args.split)

    # Build similarity
    builder = ObjectSimilarityBuilder(
        sg_dir=Path(args.sg_path) / "scene_graph_coco17",
        sg_attr_dir=Path(args.sg_path) / "scene_graph_coco17_attr",
        metric=metric,
    )

    builder.build(
        questions=questions,
        answers=answers,
        rationales=rationales,
        output_path=args.output_path,
    )

    print(f"✓ Done! Saved to {args.output_path}")


if __name__ == "__main__":
    main()
