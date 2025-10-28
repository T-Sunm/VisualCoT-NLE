import json
import os
import csv
from typing import Dict, Iterable, Iterator, List, Optional


def load_aokvqa_annotations(ann_path: str) -> List[Dict]:
    """Load AOKVQA annotations file."""
    with open(ann_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_coco_captions(caption_path: str) -> Dict[int, List[str]]:
    """Load COCO captions and return dict: image_id -> [captions]."""
    with open(caption_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both formats: {"annotations": [...]} or [...]
    if isinstance(data, dict):
        annotations = data.get("annotations", [])
    else:
        annotations = data

    caption_dict = {}
    for sample in annotations:
        image_id = sample["image_id"]
        caption = sample["caption"]
        if image_id not in caption_dict:
            caption_dict[image_id] = [caption]
        else:
            caption_dict[image_id].append(caption)

    return caption_dict


def load_vinvl_captions(vinvl_tsv_path: str) -> Dict[int, List[str]]:
    """Load VinVL cached captions from TSV file."""
    caption_dict = {}

    if not os.path.exists(vinvl_tsv_path):
        print(f"Warning: VinVL caption file not found: {vinvl_tsv_path}")
        return caption_dict

    with open(vinvl_tsv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            try:
                image_id = int(row[0])
                # Parse caption from JSON-like string
                # Format: '{"caption": "text here", "conf": 0.9}'
                caption = row[1].split('caption": "')[1].split('", "conf"')[0]

                if image_id not in caption_dict:
                    caption_dict[image_id] = [caption]
                else:
                    caption_dict[image_id].append(caption)
            except (IndexError, ValueError) as e:
                continue

    return caption_dict


def build_aokvqa_dicts(annotations: List[Dict], choice_only: bool = False) -> tuple:
    """
    Build dictionaries from AOKVQA annotations.

    Returns:
        tuple: (answer_dict, question_dict, rationale_dict, choices_dict)
        where keys are "image_id<->question_id"
    """
    answer_dict = {}
    question_dict = {}
    rationale_dict = {}
    choices_dict = {}

    for sample in annotations:
        key = f"{sample['image_id']}<->{sample['question_id']}"

        # Question
        question_dict[key] = sample["question"]

        # Answer
        if choice_only:
            if "correct_choice_idx" in sample:
                answer_dict[key] = sample["correct_choice_idx"]
            else:
                answer_dict[key] = 0
        else:
            if "direct_answers" in sample:
                answer_dict[key] = sample["direct_answers"]
            else:
                answer_dict[key] = [""]

        # Rationale
        if "rationales" in sample:
            rationale_dict[key] = sample["rationales"]
        else:
            rationale_dict[key] = [""]

        # Choices
        if "choices" in sample:
            choices_dict[key] = sample["choices"]
        else:
            choices_dict[key] = []

    return answer_dict, question_dict, rationale_dict, choices_dict


class AOKVQADataset:
    """
    Full AOKVQA dataset with training context for few-shot examples.

    This loads all necessary data that was in the original VisualCoT implementation:
    - Val questions, answers, rationales, choices
    - Train questions, answers, rationales, choices (for few-shot context)
    - Captions (COCO or VinVL)
    """

    def __init__(self, dataset_cfg: Dict, split: str = "val"):
        self.dataset_cfg = dataset_cfg
        self.split = split
        self.choice_only = dataset_cfg.get("choice_only", False)

        # Load val/test annotations
        val_ann_path = dataset_cfg["annotations_file"]
        print(f"Loading {split} annotations from {val_ann_path}...")
        val_annotations = load_aokvqa_annotations(val_ann_path)

        self.answer_dict, self.question_dict, self.rationale_dict, self.choices_dict = (
            build_aokvqa_dicts(val_annotations, self.choice_only)
        )

        # Store val keys
        self.val_keys = list(self.question_dict.keys())

        # Load captions for val images
        vinvl_path = dataset_cfg.get("valcaption_file")
        if vinvl_path and os.path.exists(vinvl_path):
            print(f"Loading val captions from {vinvl_path}...")
            self.inputtext_dict = load_vinvl_captions(vinvl_path)
        else:
            print("Warning: No val caption file specified")
            self.inputtext_dict = {}

        # Load training annotations for few-shot context
        train_ann_path = dataset_cfg.get("train_annotations_file")
        if train_ann_path and os.path.exists(train_ann_path):
            print(f"Loading training annotations from {train_ann_path}...")
            train_annotations = load_aokvqa_annotations(train_ann_path)

            (
                self.traincontext_answer_dict,
                self.traincontext_question_dict,
                self.traincontext_rationale_dict,
                self.traincontext_choices_dict,
            ) = build_aokvqa_dicts(train_annotations, self.choice_only)

            self.train_keys = list(self.traincontext_question_dict.keys())
        else:
            print("Warning: No training annotations specified - few-shot examples disabled")
            self.traincontext_answer_dict = {}
            self.traincontext_question_dict = {}
            self.traincontext_rationale_dict = {}
            self.traincontext_choices_dict = {}
            self.train_keys = []

        # Load training captions for few-shot context
        train_caption_path = dataset_cfg.get("train_caption_file")
        if train_caption_path and os.path.exists(train_caption_path):
            print(f"Loading training captions from {train_caption_path}...")
            self.traincontext_caption_dict = load_coco_captions(train_caption_path)
        else:
            print("Warning: No training caption file specified")
            self.traincontext_caption_dict = {}

        # Image paths
        self.images_root = dataset_cfg["images_root"]

        print(f"Dataset loaded:")
        print(f"  Val samples: {len(self.val_keys)}")
        print(f"  Train samples: {len(self.train_keys)}")
        print(f"  Val captions: {len(self.inputtext_dict)}")
        print(f"  Train captions: {len(self.traincontext_caption_dict)}")

    def __iter__(self) -> Iterator[Dict]:
        """Iterate over validation/test samples."""
        for key in self.val_keys:
            image_id_str, question_id = key.split("<->")
            image_id = int(image_id_str)

            # Build image path
            filename = f"{image_id:012d}.jpg"
            image_path = os.path.join(self.images_root, filename)

            # Get data for this sample
            sample = {
                "key": key,
                "image_id": image_id_str,
                "question_id": question_id,
                "image_path": image_path,
                "question": self.question_dict[key],
                "answer": self.answer_dict.get(key, []),
                "rationale": self.rationale_dict.get(key, []),
                "choices": self.choices_dict.get(key, []),
                "caption": self.inputtext_dict.get(image_id, []),
            }

            # Add training context (for few-shot examples)
            sample["train_context"] = {
                "questions": self.traincontext_question_dict,
                "answers": self.traincontext_answer_dict,
                "rationales": self.traincontext_rationale_dict,
                "choices": self.traincontext_choices_dict,
                "captions": self.traincontext_caption_dict,
                "keys": self.train_keys,
            }

            yield sample

    def __len__(self) -> int:
        return len(self.val_keys)


def _iter_aokvqa(dataset_cfg: Dict) -> Iterator[Dict]:
    """Legacy simple iterator - replaced by AOKVQADataset."""
    ann_path = dataset_cfg["annotations_file"]
    images_root = dataset_cfg["images_root"]
    with open(ann_path, "r", encoding="utf-8") as f:
        ann: List[Dict] = json.load(f)
    for sample in ann[:3]:  # small subset for smoke
        image_id = sample["image_id"]
        qid = sample["question_id"]
        # files are typically zero-padded COCO style
        filename = f"{int(image_id):012d}.jpg"
        image_path = os.path.join(images_root, filename)
        yield {
            "image_id": str(image_id),
            "image_path": image_path,
            "question": sample["question"],
            "question_id": str(qid),
        }


def build_dataset(dataset_cfg: Dict, split: str) -> Iterable[Dict]:
    """
    Build dataset iterator with full context.

    Args:
        dataset_cfg: Dataset configuration
        split: Dataset split ("train", "val", "test")

    Returns:
        Iterator yielding samples with full context
    """
    name = dataset_cfg.get("name") or dataset_cfg.get("dataset", {}).get("name")

    if name == "aokvqa":
        # Use full dataset with training context
        return AOKVQADataset(dataset_cfg.get("dataset", dataset_cfg), split)

    # Fallback: one-sample iterator
    class _One(Iterator[Dict]):
        def __init__(self) -> None:
            self._done = False

        def __iter__(self) -> "_One":
            return self

        def __next__(self) -> Dict:
            if self._done:
                raise StopIteration
            self._done = True
            return {
                "image_id": "placeholder",
                "image_path": "data/raw/placeholder.jpg",
                "question": "?",
                "question_id": "0",
            }

    return _One()
