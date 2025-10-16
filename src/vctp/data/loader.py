import json
import os
from typing import Dict, Iterable, Iterator, List


def _iter_aokvqa(dataset_cfg: Dict) -> Iterator[Dict]:
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
    name = dataset_cfg.get("name") or dataset_cfg.get("dataset", {}).get("name")
    if name == "aokvqa":
        return _iter_aokvqa(dataset_cfg.get("dataset", dataset_cfg))

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
