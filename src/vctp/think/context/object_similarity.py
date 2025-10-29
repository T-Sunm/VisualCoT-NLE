"""Object similarity computation for interactive selection."""

import json
import os
from typing import Dict, List, Optional, Tuple
import torch
from vctp.utils.clip_manager import get_clip_tokenizer


class ObjectSimilarityComputer:
    """Compute similarity between objects and answers/rationales."""

    def __init__(
        self,
        sg_dir: str,
        sg_attr_dir: str,
        train_questions: Dict[str, str],
        train_answers: Dict[str, List[str]],
        train_rationales: Optional[Dict[str, List[str]]] = None,
        use_clip: bool = True,
    ):
        """
        Initialize object similarity computer.

        Args:
            sg_dir: Scene graph directory
            sg_attr_dir: Scene graph attributes directory
            train_questions: Training questions
            train_answers: Training answers
            train_rationales: Training rationales
            use_clip: Whether to use CLIP for similarity
        """
        self.sg_dir = sg_dir
        self.sg_attr_dir = sg_attr_dir
        self.train_questions = train_questions
        self.train_answers = train_answers
        self.train_rationales = train_rationales or {}
        self.use_clip = use_clip

        # CLIP model for answer-based similarity
        self.clip_model = None
        self.clip_processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if use_clip:
            self._init_clip()

    def _init_clip(self):
        """Initialize CLIP model for text similarity."""
        self.clip_model, self.clip_processor = get_clip_tokenizer(device=self.device)

    def compute_object_similarity(
        self, example_key: str, metric: str = "answer"
    ) -> Tuple[List[str], List[float], Dict[str, float]]:
        """
        Compute object similarity for an example.

        Args:
            example_key: Example key (image_id<->question_id)
            metric: Similarity metric (answer, rationale)

        Returns:
            Tuple of (object_list, confidence_list, similarity_dict)
        """
        # Get image ID
        img_id = int(example_key.split("<->")[0])

        # Load scene graph
        obj_list, conf_list = self._load_objects(img_id)

        # Compute similarity based on metric
        if metric == "rationale":
            similarity_dict = self._compute_rationale_similarity(example_key, obj_list)
        elif metric == "answer":
            similarity_dict = self._compute_answer_similarity(example_key, obj_list)
        else:
            similarity_dict = {}

        return obj_list, conf_list, similarity_dict

    def _load_objects(self, img_id: int) -> Tuple[List[str], List[float]]:
        """
        Load objects from scene graph.

        Args:
            img_id: Image ID

        Returns:
            Tuple of (object_list, confidence_list)
        """
        # Load scene graphs
        sg_path = os.path.join(self.sg_dir, f"{str(img_id).zfill(12)}.json")
        sg_attr_path = os.path.join(self.sg_attr_dir, f"{str(img_id).zfill(12)}.json")

        obj_list = []
        conf_list = []

        # Load basic scene graph
        if os.path.exists(sg_path):
            scene_graph = json.load(open(sg_path))
            for obj in scene_graph[0]:
                if obj["class"] not in obj_list:
                    obj_list.append(obj["class"])
                    conf_list.append(obj["conf"])

        # Load attributes scene graph
        if os.path.exists(sg_attr_path):
            scene_graph_attr = json.load(open(sg_attr_path))
            for obj in scene_graph_attr[0]:
                if obj["class"] not in obj_list:
                    obj_list.append(obj["class"])
                    conf_list.append(obj["conf"])

        return obj_list, conf_list

    def _compute_rationale_similarity(
        self, example_key: str, obj_list: List[str]
    ) -> Dict[str, float]:
        """
        Compute similarity based on rationale mentions.

        Args:
            example_key: Example key
            obj_list: List of objects

        Returns:
            Dictionary of object -> count
        """
        if example_key not in self.train_rationales:
            return {}

        rationales = self.train_rationales[example_key]
        similarity_dict = {}

        for obj in obj_list:
            count = 0
            for rationale in rationales:
                if obj in rationale:
                    count += 1
            if count > 0:
                similarity_dict[obj] = float(count)

        return similarity_dict

    def _compute_answer_similarity(
        self, example_key: str, obj_list: List[str]
    ) -> Dict[str, float]:
        """
        Compute similarity based on answer using CLIP.

        Args:
            example_key: Example key
            obj_list: List of objects

        Returns:
            Dictionary of object -> similarity score
        """
        if not self.use_clip or example_key not in self.train_answers:
            return {}

        answer_list = self.train_answers[example_key]
        if not answer_list:
            return {}

        similarity_dict = {}

        with torch.no_grad():
            # Encode answers
            inputs = self.clip_processor(
                text=answer_list, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.clip_model(**inputs)
            ans_text_emb = outputs["pooler_output"].mean(dim=0).unsqueeze(dim=0)

            # Encode objects
            inputs = self.clip_processor(
                text=obj_list, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.clip_model(**inputs)
            obj_text_emb = outputs["pooler_output"]

            # Normalize
            ans_text_emb /= ans_text_emb.norm(dim=-1, keepdim=True)
            obj_text_emb /= obj_text_emb.norm(dim=-1, keepdim=True)

            # Compute similarity
            sim_scores = obj_text_emb @ ans_text_emb.T

            for idx, obj_name in enumerate(obj_list):
                similarity_dict[obj_name] = sim_scores[idx, 0].to(self.device).item()

        return similarity_dict

    def get_most_relevant_object(
        self, example_key: str, metric: str = "answer"
    ) -> Optional[str]:
        """
        Get the most relevant object for an example.

        Args:
            example_key: Example key
            metric: Similarity metric

        Returns:
            Most relevant object name or None
        """
        obj_list, conf_list, similarity_dict = self.compute_object_similarity(
            example_key, metric
        )

        if not similarity_dict:
            return None

        # Return object with highest similarity
        return max(similarity_dict.items(), key=lambda x: x[1])[0]

    def get_ranked_objects(
        self, example_key: str, metric: str = "answer", top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get ranked list of objects by relevance.

        Args:
            example_key: Example key
            metric: Similarity metric
            top_k: Number of top objects to return

        Returns:
            List of (object_name, similarity_score) tuples
        """
        obj_list, conf_list, similarity_dict = self.compute_object_similarity(
            example_key, metric
        )

        if not similarity_dict:
            # Return by confidence
            ranked = sorted(zip(obj_list, conf_list), key=lambda x: x[1], reverse=True)
            return ranked[:top_k]

        # Sort by similarity
        ranked = sorted(similarity_dict.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
