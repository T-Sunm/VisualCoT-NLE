import json
import os
import csv
from typing import Dict, Iterable, Iterator, List, Optional


def load_vivqax_annotations(ann_path: str) -> List[Dict]:
    """Load ViVQA-X annotations file."""
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


def build_vivqax_dicts(annotations: List[Dict]) -> tuple:
    """
    Build dictionaries from ViVQA-X annotations.
    
    Args:
        annotations: List of ViVQA-X samples
    
    Returns:
        tuple: (answer_dict, question_dict, rationale_dict)
        where keys are "image_id<->question_id"
    """
    answer_dict = {}
    question_dict = {}
    rationale_dict = {}
    
    for sample in annotations:
        key = f"{sample['image_id']}<->{sample['question_id']}"
        
        # Question
        question_dict[key] = sample["question"]
        
        # Answer (convert to list for consistency)
        answer = sample.get("answer", "")
        answer_dict[key] = [answer] if answer else [""]
        
        # Rationale (explanation field)
        explanation = sample.get("explanation", "")
        rationale_dict[key] = explanation if explanation else [""]
    
    return answer_dict, question_dict, rationale_dict


class ViVQAXDataset:
    """ViVQA-X dataset with AOKVQA-compatible format."""
    
    def __init__(self, ann_file: str, images_root: str, 
                 train_ann_file: str = None, split: str = "val"):
        self.split = split
        self.images_root = images_root
        
        # Load val/test annotations
        print(f"Loading {split} annotations...")
        with open(ann_file, 'r') as f:
            self.val_data = json.load(f)
        
        # Load train annotations for context (if provided)
        self.train_data = []
        if train_ann_file and os.path.exists(train_ann_file):
            print(f"Loading train annotations for context...")
            with open(train_ann_file, 'r') as f:
                self.train_data = json.load(f)
        
        # Build train context dicts
        self._build_train_context()
        
        print(f"✓ Val samples: {len(self.val_data)}")
        print(f"✓ Train samples: {len(self.train_data)}")
    
    def _build_train_context(self):
        """Build training context dictionaries."""
        self.train_context = {
            "questions": {},
            "answers": {},
            "rationales": {},
            "choices": {},  
            "captions": {},  
            "keys": []
        }
        
        for sample in self.train_data:
            key = f"{sample['image_id']}<->{sample['question_id']}"
            self.train_context["keys"].append(key)
            self.train_context["questions"][key] = sample['question']
            self.train_context["answers"][key] = [sample.get('answer', '')]
            self.train_context["rationales"][key] = [sample.get('explanation', '')]
    
    def __iter__(self) -> Iterator[Dict]:
        """Iterate with AOKVQA-compatible format."""
        for sample in self.val_data:
            # Build key
            key = f"{sample['image_id']}<->{sample['question_id']}"
            
            # Determine image folder
            image_name = sample['image_name']
            if 'train2014' in image_name:
                folder = 'train2014'
            elif 'val2014' in image_name:
                folder = 'val2014'
            else:
                folder = 'test2014'
            
            image_path = os.path.join(self.images_root, folder, image_name)
            
            # Map to AOKVQA format
            yield {
                # Core fields
                "key": key,
                "image_id": str(sample['image_id']),
                "question_id": str(sample['question_id']),
                "image_path": image_path,
                "question": sample['question'],
                
                # Answer data (convert to list format)
                "answer": [sample.get('answer', '')],  
                "rationale": [sample.get('explanation', '')],  # rationale
                "choices": [],  
                
                # Visual data
                "caption": [],  
                
                # Training context
                "train_context": self.train_context
            }
    
    def __len__(self) -> int:
        return len(self.val_data)


def build_dataset(dataset_cfg: Dict, split: str) -> ViVQAXDataset:
    """
    Build dataset with AOKVQA-compatible format.
    
    Config format:
        {
            "annotations_file": "path/to/val.json",
            "train_annotations_file": "path/to/train.json",  # optional
            "images_root": "path/to/coco/images"
        }
    """
    return ViVQAXDataset(
        ann_file=dataset_cfg["annotations_file"],
        images_root=dataset_cfg["images_root"],
        train_ann_file=dataset_cfg.get("train_annotations_file"),
        split=split
    )
