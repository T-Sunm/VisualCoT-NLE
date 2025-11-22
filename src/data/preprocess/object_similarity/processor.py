import json
from pathlib import Path
from typing import Dict, List, Tuple


class Processor:
    """Process annotations for object similarity."""

    def __init__(self, annotations_dir: str, captions_dir: str = None):
        """
        Initialize processor.

        Args:
            annotations_dir: Directory with annotations
            captions_dir: Directory with COCO captions (optional)
        """
        self.annotations_dir = Path(annotations_dir)

    def load_split(
        self, split: str = "train"
    ) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Load split.

        Args:
            split: 'train' or 'val' or 'test'

        Returns:
            Tuple of (questions, answers, rationales) dicts
        """
        # Load annotations
        anno_file = self.annotations_dir / f"{split}.json"
        with open(anno_file) as f:
            annotations = json.load(f)

        questions = {}
        answers = {}
        rationales = {}

        for sample in annotations:
            key = f"{sample['image_id']}<->{sample['question_id']}"
            questions[key] = sample["question"]
            
            answer_text = sample.get("answer", "")
            answers[key] = [answer_text] if answer_text else []
            
            rationales[key] = sample.get("explanation", [])

        return questions, answers, rationales