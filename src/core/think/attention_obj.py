"""
Object Attention Module - Detect & Select objects
"""
import pickle
import json
from pathlib import Path
from typing import List, Dict, Optional
import yaml


class ObjectAttentionSelector:
    """Quản lý việc detect và select objects"""
    
    def __init__(self, similarity_path: str = None):
        self.similarity_dict = self._load_similarity(similarity_path)
    
    def _load_similarity(self, path: str) -> Dict:
        """Load precomputed similarity scores"""
        if not path or not Path(path).exists():
            return {}
        
        print(f"Loading similarity scores from {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
            
    def detect_objects(self, sg_attr_path: str) -> List[Dict]:
        """
        Load objects từ scene graph file
        
        Args:
            sg_attr_path: Path đến file scene graph attribute JSON
            
        Returns:
            List objects đã normalized
        """
        if not Path(sg_attr_path).exists():
            return []
            
        with open(sg_attr_path) as f:
            sg_data = json.load(f)
            
        # Handle format list of lists
        raw_objects = sg_data[0] if isinstance(sg_data, list) else sg_data
        
        # Normalize
        objects = []
        for obj in raw_objects:
            objects.append({
                "name": obj["class"],
                "confidence": obj.get("conf", 1.0),
                "attributes": obj.get("attr", []),
                "box": obj.get("rect", [])
            })
            
        # Sort by confidence
        return sorted(objects, key=lambda x: x["confidence"], reverse=True)
    
    def select(self, objects: List[Dict], key: str) -> int:
        """
        Chọn object tốt nhất dựa trên similarity score
        """
        if not objects:
            return None
            
        if key not in self.similarity_dict:
            return 0
            
        obj_scores = self.similarity_dict[key]
        scores = []
        
        for obj in objects:
            sim_score = obj_scores.get(obj["name"], 0.0)
            final_score = sim_score * obj["confidence"]
            scores.append(final_score)
            
        return scores.index(max(scores))


def load_object_selector_from_config(config_path: str) -> ObjectAttentionSelector:
    """Factory function load từ config"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    dataset_name = config["experiment"]["dataset"]
    dataset_cfg_path = config["datasets"][dataset_name]
    
    with open(dataset_cfg_path) as f:
        dataset_cfg = yaml.safe_load(f)
    
    sim_path = dataset_cfg["dataset"].get("precomputed_object_similarity_path")
    
    return ObjectAttentionSelector(sim_path)