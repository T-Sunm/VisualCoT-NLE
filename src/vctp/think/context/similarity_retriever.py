"""Similarity-based context retrieval using CLIP features."""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


class SimilarityRetriever:
    """Retrieve similar examples using pre-computed CLIP features."""

    def __init__(
        self,
        similarity_path: str,
        dataset_name: str = "aokvqa",
        split: str = "val",
    ):
        """
        Initialize similarity retriever.

        Args:
            similarity_path: Path to directory containing CLIP features
            dataset_name: Dataset name (aokvqa, okvqa)
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

        # For OKVQA with different train/val sets
        self.train_ok_feature = None
        self.image_train_ok_feature = None
        self.train_ok_idx = None

    def load_features(self, metric: str = "imagequestion"):
        """
        Load pre-computed CLIP features.

        Args:
            metric: Similarity metric (question, imagequestion)
        """
        if self.dataset_name == "aokvqa":
            self._load_aokvqa_features(metric)
        elif self.dataset_name == "okvqa":
            self._load_okvqa_features(metric)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def _load_aokvqa_features(self, metric: str):
        """Load A-OKVQA features."""
        # Load val key mapping
        val_idx_file = os.path.join(
            self.similarity_path, f"aokvqa_qa_line2sample_idx_{self.split}2017.json"
        )
        val_idx = json.load(open(val_idx_file, "r"))

        self.valkey2idx = {}
        for ii in val_idx:
            self.valkey2idx[val_idx[ii]] = int(ii)

        # Load question features
        if metric in ["question", "imagequestion"]:
            self.train_feature = np.load(
                os.path.join(self.similarity_path, "coco_clip_vitb16_train2017_aokvqa_question.npy")
            )
            self.val_feature = np.load(
                os.path.join(
                    self.similarity_path, f"coco_clip_vitb16_{self.split}2017_aokvqa_question.npy"
                )
            )
            self.train_idx = json.load(
                open(
                    os.path.join(self.similarity_path, "aokvqa_qa_line2sample_idx_train2017.json"),
                    "r",
                )
            )

        # Load image features
        if metric == "imagequestion":
            self.image_train_feature = np.load(
                os.path.join(
                    self.similarity_path,
                    "coco_clip_vitb16_train2017_aokvqa_convertedidx_image.npy",
                )
            )
            self.image_val_feature = np.load(
                os.path.join(
                    self.similarity_path,
                    f"coco_clip_vitb16_{self.split}2017_aokvqa_convertedidx_image.npy",
                )
            )

    def _load_okvqa_features(self, metric: str):
        """Load OK-VQA features."""
        # Load val key mapping
        val_idx_file = os.path.join(
            self.similarity_path, f"okvqa_qa_line2sample_idx_{self.split}2014.json"
        )
        val_idx = json.load(open(val_idx_file, "r"))

        self.valkey2idx = {}
        for ii in val_idx:
            self.valkey2idx[val_idx[ii]] = int(ii)

        # Load question features
        if metric in ["question", "imagequestion"]:
            # Train features (A-OKVQA)
            self.train_feature = np.load(
                os.path.join(self.similarity_path, "coco_clip_vitb16_train2017_aokvqa_question.npy")
            )
            # OK-VQA train features
            self.train_ok_feature = np.load(
                os.path.join(self.similarity_path, "coco_clip_vitb16_train2014_okvqa_question.npy")
            )
            # Val features
            self.val_feature = np.load(
                os.path.join(
                    self.similarity_path, f"coco_clip_vitb16_{self.split}2014_okvqa_question.npy"
                )
            )

            # Indices
            self.train_idx = json.load(
                open(
                    os.path.join(self.similarity_path, "aokvqa_qa_line2sample_idx_train2017.json"),
                    "r",
                )
            )
            self.train_ok_idx = json.load(
                open(
                    os.path.join(self.similarity_path, "okvqa_qa_line2sample_idx_train2014.json"),
                    "r",
                )
            )

        # Load image features
        if metric == "imagequestion":
            self.image_train_feature = np.load(
                os.path.join(
                    self.similarity_path,
                    "coco_clip_vitb16_train2017_aokvqa_convertedidx_image.npy",
                )
            )
            self.image_train_ok_feature = np.load(
                os.path.join(
                    self.similarity_path,
                    "coco_clip_vitb16_train2014_okvqa_convertedidx_image.npy",
                )
            )
            self.image_val_feature = np.load(
                os.path.join(
                    self.similarity_path,
                    f"coco_clip_vitb16_{self.split}2014_okvqa_convertedidx_image.npy",
                )
            )

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
            return None

    def _get_similar_by_question(self, query_key: str, n_shot: int) -> List[str]:
        """Get similar examples based on question similarity only."""
        if query_key not in self.valkey2idx:
            return []

        lineid = self.valkey2idx[query_key]

        # Compute similarity
        if self.dataset_name == "okvqa" and self.train_ok_feature is not None:
            # Use OK-VQA train features
            similarity = np.matmul(self.train_ok_feature, self.val_feature[lineid, :])
            index = similarity.argsort()[-n_shot:][::-1]
            return [self.train_ok_idx[str(x)] for x in index]
        else:
            similarity = np.matmul(self.train_feature, self.val_feature[lineid, :])
            index = similarity.argsort()[-n_shot:][::-1]
            return [self.train_idx[str(x)] for x in index]

    def _get_similar_by_image_question(self, query_key: str, n_shot: int) -> List[str]:
        """Get similar examples based on image + question similarity."""
        if query_key not in self.valkey2idx:
            return []

        lineid = self.valkey2idx[query_key]

        # Compute question similarity
        if self.dataset_name == "okvqa" and self.train_ok_feature is not None:
            question_similarity = np.matmul(self.train_ok_feature, self.val_feature[lineid, :])
            image_similarity = np.matmul(
                self.image_train_ok_feature, self.image_val_feature[lineid, :]
            )
            similarity = question_similarity + image_similarity
            index = similarity.argsort()[-n_shot:][::-1]
            return [self.train_ok_idx[str(x)] for x in index]
        else:
            question_similarity = np.matmul(self.train_feature, self.val_feature[lineid, :])
            image_similarity = np.matmul(
                self.image_train_feature, self.image_val_feature[lineid, :]
            )
            similarity = question_similarity + image_similarity
            index = similarity.argsort()[-n_shot:][::-1]
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
            return {}

        lineid = self.valkey2idx[query_key]
        scores = {}

        if metric == "question":
            if self.dataset_name == "okvqa" and self.train_ok_feature is not None:
                similarity = np.matmul(self.train_ok_feature, self.val_feature[lineid, :])
                for i, score in enumerate(similarity):
                    key = self.train_ok_idx[str(i)]
                    scores[key] = float(score)
            else:
                similarity = np.matmul(self.train_feature, self.val_feature[lineid, :])
                for i, score in enumerate(similarity):
                    key = self.train_idx[str(i)]
                    scores[key] = float(score)

        elif metric == "imagequestion":
            if self.dataset_name == "okvqa" and self.train_ok_feature is not None:
                question_sim = np.matmul(self.train_ok_feature, self.val_feature[lineid, :])
                image_sim = np.matmul(
                    self.image_train_ok_feature, self.image_val_feature[lineid, :]
                )
                similarity = question_sim + image_sim
                for i, score in enumerate(similarity):
                    key = self.train_ok_idx[str(i)]
                    scores[key] = float(score)
            else:
                question_sim = np.matmul(self.train_feature, self.val_feature[lineid, :])
                image_sim = np.matmul(self.image_train_feature, self.image_val_feature[lineid, :])
                similarity = question_sim + image_sim
                for i, score in enumerate(similarity):
                    key = self.train_idx[str(i)]
                    scores[key] = float(score)

        return scores
