"""Similarity-based context retrieval using CLIP features."""

import json
import os
from typing import Dict, List, Optional

import numpy as np


class SimilarityRetriever:
    """Retrieve similar examples using pre-computed CLIP features."""

    def __init__(
        self,
        similarity_path: str,
        dataset_name: str = "vivqax",
        split: str = "val",
    ):
        """
        Initialize similarity retriever.

        Args:
            similarity_path: Path to directory containing CLIP features
            dataset_name: Dataset name (vivqax)
            split: Data split (val, test)
        """
        self.similarity_path = similarity_path
        self.dataset_name = dataset_name
        self.split = split

        # Features and indices
        self.train_feature = None
        self.val_feature = None
        self.image_train_feature = None
        self.image_val_feature = None
        self.train_idx = None
        self.valkey2idx = None

    def load_features(self, metric: str = "imagequestion"):
        """
        Load pre-computed CLIP features.

        Args:
            metric: Similarity metric (question, imagequestion)
        """
        print(f"Loading {self.dataset_name} features from {self.similarity_path}...")
        
        # Load val key mapping
        val_idx_file = os.path.join(
            self.similarity_path, f"{self.dataset_name}_qa_line2sample_idx_{self.split}.json"
        )
        print(f"  Loading val index: {val_idx_file}")
        val_idx = json.load(open(val_idx_file, "r"))

        self.valkey2idx = {}
        for ii in val_idx:
            self.valkey2idx[val_idx[ii]] = int(ii) # --> {"image_id<->question_id": line_index}
        # ------- Example -------
        # query_key = "262284<->262284001"

        # # Bước 1: Tìm line index
        # lineid = self.valkey2idx[query_key]  # → 0

        # # Bước 2: Lấy embedding từ numpy array
        # query_embedding = self.val_feature[lineid, :]  # → val_feature[0, :]

        # # Bước 3: Tính similarity với tất cả training samples
        # similarity = np.matmul(self.train_feature, query_embedding)
        # -----------------------------
        
        # Load question features
        if metric in ["question", "imagequestion"]:
            train_q_file = os.path.join(
                self.similarity_path, f"coco_clip_vitb16_train_{self.dataset_name}_question.npy"
            )
            val_q_file = os.path.join(
                self.similarity_path, f"coco_clip_vitb16_{self.split}_{self.dataset_name}_question.npy"
            )
            train_idx_file = os.path.join(
                self.similarity_path, f"{self.dataset_name}_qa_line2sample_idx_train.json"
            )
            
            print(f"  Loading train question features: {train_q_file}")
            self.train_feature = np.load(train_q_file)
            
            print(f"  Loading {self.split} question features: {val_q_file}")
            self.val_feature = np.load(val_q_file)
            
            print(f"  Loading train index: {train_idx_file}")
            self.train_idx = json.load(open(train_idx_file, "r"))

        # Load image features
        if metric == "imagequestion":
            train_img_file = os.path.join(
                self.similarity_path,
                f"coco_clip_vitb16_train_{self.dataset_name}_convertedidx_image.npy",
            )
            val_img_file = os.path.join(
                self.similarity_path,
                f"coco_clip_vitb16_{self.split}_{self.dataset_name}_convertedidx_image.npy",
            )
            
            print(f"  Loading train image features: {train_img_file}")
            self.image_train_feature = np.load(train_img_file)
            
            print(f"  Loading {self.split} image features: {val_img_file}")
            self.image_val_feature = np.load(val_img_file)
        
        print(f"✓ Loaded features successfully!")
        print(f"  Train samples: {len(self.train_idx) if self.train_idx else 0}")
        print(f"  Val samples: {len(self.valkey2idx)}")

    def get_similar_examples(
        self, query_key: str, metric: str = "imagequestion", n_shot: int = 16
    ) -> List[str]:
        """
        Get similar training examples for a query.

        Args:
            query_key: Query key (image_id<->question_id)
            metric: Similarity metric (question, imagequestion)
            n_shot: Number of examples to retrieve

        Returns:
            List of similar example keys
        """
        if metric == "question":
            return self._get_similar_by_question(query_key, n_shot)
        elif metric == "imagequestion":
            return self._get_similar_by_image_question(query_key, n_shot)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def _get_similar_by_question(self, query_key: str, n_shot: int) -> List[str]:
        """Get similar examples based on question similarity only."""
        if query_key not in self.valkey2idx:
            print(f"Warning: Query key '{query_key}' not found in valkey2idx")
            return []

        lineid = self.valkey2idx[query_key]

        # Compute similarity: (N_train,) = (N_train, D) @ (D,)
        similarity = np.matmul(self.train_feature, self.val_feature[lineid, :])
        
        # Get top-k indices
        index = similarity.argsort()[-n_shot:][::-1]
        
        # Map to keys
        return [self.train_idx[str(x)] for x in index]

    def _get_similar_by_image_question(self, query_key: str, n_shot: int) -> List[str]:
        """Get similar examples based on image + question similarity."""
        if query_key not in self.valkey2idx:
            print(f"Warning: Query key '{query_key}' not found in valkey2idx")
            return []

        lineid = self.valkey2idx[query_key]

        # Compute question similarity
        question_similarity = np.matmul(self.train_feature, self.val_feature[lineid, :])
        
        # Compute image similarity
        image_similarity = np.matmul(
            self.image_train_feature, self.image_val_feature[lineid, :]
        )
        
        # Combined similarity
        similarity = question_similarity + image_similarity
        
        # Get top-k indices
        index = similarity.argsort()[-n_shot:][::-1]
        
        # Map to keys
        return [self.train_idx[str(x)] for x in index]

    def get_similar_with_scores(
        self, query_key: str, metric: str = "imagequestion"
    ) -> Dict[str, float]:
        """
        Get similarity scores for all training examples.

        Args:
            query_key: Query key
            metric: Similarity metric

        Returns:
            Dictionary mapping example keys to similarity scores
        """
        if query_key not in self.valkey2idx:
            print(f"Warning: Query key '{query_key}' not found in valkey2idx")
            return {}

        lineid = self.valkey2idx[query_key]
        scores = {}

        if metric == "question":
            similarity = np.matmul(self.train_feature, self.val_feature[lineid, :])
            for i, score in enumerate(similarity):
                key = self.train_idx[str(i)]
                scores[key] = float(score)

        elif metric == "imagequestion":
            question_sim = np.matmul(self.train_feature, self.val_feature[lineid, :])
            image_sim = np.matmul(self.image_train_feature, self.image_val_feature[lineid, :])
            similarity = question_sim + image_sim
            
            for i, score in enumerate(similarity):
                key = self.train_idx[str(i)]
                scores[key] = float(score)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return scores