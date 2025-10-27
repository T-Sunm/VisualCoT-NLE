"""AOKVQA-specific processor for object similarity."""

import json
from pathlib import Path
from typing import Dict, List, Tuple


class AOKVQASimilarityProcessor:
    """Process AOKVQA annotations for object similarity."""

    def __init__(self, annotations_dir: str, captions_dir: str = None):
        """
        Initialize processor.

        Args:
            annotations_dir: Directory with AOKVQA annotations
            captions_dir: Directory with COCO captions (optional)
        """
        self.annotations_dir = Path(annotations_dir)
        self.captions_dir = Path(captions_dir) if captions_dir else None

    def load_split(
        self, split: str = "train"
    ) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Load AOKVQA split.

        Args:
            split: 'train' or 'val'

        Returns:
            Tuple of (questions, answers, rationales) dicts
        """
        # Load annotations
        anno_file = self.annotations_dir / f"aokvqa_v1p0_{split}_from_hf.json"
        with open(anno_file) as f:
            annotations = json.load(f)

        questions = {}
        answers = {}
        rationales = {}

        for sample in annotations:
            key = f"{sample['image_id']}<->{sample['question_id']}"
            questions[key] = sample["question"]
            answers[key] = sample.get("direct_answers", [])
            rationales[key] = sample.get("rationales", [])

        return questions, answers, rationales
