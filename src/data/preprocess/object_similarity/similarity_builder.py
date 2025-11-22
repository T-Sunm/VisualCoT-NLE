"""Main builder for object similarity preprocessing."""

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

from .metrics import SimilarityMetric


class ObjectSimilarityBuilder:
    """Build object similarity scores for dataset."""

    def __init__(
        self,
        sg_dir: str,
        sg_attr_dir: str,
        metric: SimilarityMetric,
    ):
        """
        Initialize builder.

        Args:
            sg_dir: Scene graph directory
            sg_attr_dir: Scene graph attributes directory
            metric: Similarity metric to use
        """
        self.sg_dir = Path(sg_dir)
        self.sg_attr_dir = Path(sg_attr_dir)
        self.metric = metric

    def load_objects(self, img_id: int) -> Tuple[List[str], List[float]]:
        """Load objects from scene graphs."""
        obj_list = []
        conf_list = []

        # Load basic scene graph
        sg_path = self.sg_dir / f"{str(img_id).zfill(12)}.json"
        if sg_path.exists():
            with open(sg_path) as f:
                scene_graph = json.load(f)
            for obj in scene_graph[0]:
                if obj["class"] not in obj_list:
                    obj_list.append(obj["class"])
                    conf_list.append(obj["conf"])

        # Load attributes scene graph
        sg_attr_path = self.sg_attr_dir / f"{str(img_id).zfill(12)}.json"
        if sg_attr_path.exists():
            with open(sg_attr_path) as f:
                scene_graph_attr = json.load(f)
            for obj in scene_graph_attr[0]:
                if obj["class"] not in obj_list:
                    obj_list.append(obj["class"])
                    conf_list.append(obj["conf"])

        return obj_list, conf_list

    def build(
        self,
        questions: Dict[str, str],
        answers: Dict[str, List[str]],
        rationales: Dict[str, List[str]] = None,
        output_path: str = None,
        show_progress: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Build similarity scores for all examples.

        Args:
            questions: Dict of {key: question}
            answers: Dict of {key: [answers]}
            rationales: Dict of {key: [rationales]}
            output_path: Path to save pickle file
            show_progress: Show progress bar

        Returns:
            Dict of {key: {object: score}}
        """
        similarity_dict = {}

        keys = list(questions.keys())
        iterator = tqdm(keys) if show_progress else keys

        for key in iterator:
            # Extract image ID
            img_id = int(key.split("<->")[0])

            # Load objects
            obj_list, conf_list = self.load_objects(img_id)

            if not obj_list:
                continue

            # Compute similarity
            scores = self.metric.compute(
                obj_list=obj_list,
                answers=answers.get(key, []),
                rationales=rationales.get(key, []) if rationales else None,
            )

            similarity_dict[key] = scores

        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                pickle.dump(similarity_dict, f)
            print(f"Saved similarity scores to {output_path}")

        return similarity_dict
