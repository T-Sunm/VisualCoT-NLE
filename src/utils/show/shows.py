from typing import Dict
from pathlib import Path
import json


def show_sample(sample: Dict):
    """Show sample info."""
    print(f"\n{'='*60}")
    print(f"Image ID: {sample['image_id']}")
    print(f"Question: {sample['question']}")
    print(f"Answer: {sample['answer']}")
    
    sg_path = Path(sample['sg_attr_path'])
    if not sg_path.exists():
        print(f"{'='*60}\n")
        return
    
    with open(sg_path) as f:
        sg_data = json.load(f)

    print(f"\n--- Objects (top 10) ---")
    objects = sg_data[0] if isinstance(sg_data, list) else sg_data
    objects_sorted = sorted(objects, key=lambda x: x.get('conf', 0), reverse=True)[:10]
    
    for i, obj in enumerate(objects_sorted, 1):
        attrs = ", ".join(obj.get('attr', []))
        conf = obj.get('conf', 0)
        print(f"{i}. {obj['class']}: {attrs} (conf: {conf:.3f})")
    
    sg_cap_path = sg_path.parent.parent / "scene_graph_coco14_caption" / f"{str(sample['image_id']).zfill(12)}.json"
    if sg_cap_path.exists():
        with open(sg_cap_path) as f:
            captions = json.load(f)
        
        print(f"\n--- Captions (top 5) ---")
        if isinstance(captions, list):
            for i, cap in enumerate(captions[:5], 1):
                print(f"{i}. {cap}")
        elif isinstance(captions, dict):
            for i, (name, cap) in enumerate(list(captions.items())[:5], 1):
                print(f"{i}. {name}: {cap}")
    
    print(f"{'='*60}\n")