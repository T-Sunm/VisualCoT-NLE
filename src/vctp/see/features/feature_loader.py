"""
Feature Loader for pre-computed CLIP features
Based on the feature loading logic from main_aokvqa.py
"""

import json
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path


class FeatureLoader:
    """
    Loader for pre-computed CLIP features.
    Handles question features, image features, and index mappings.
    """

    def __init__(self, feature_dir: str, dataset_name: str = "aokvqa", split: str = "val"):
        """
        Initialize feature loader.

        Args:
            feature_dir: Directory containing feature files
            dataset_name: Name of dataset (aokvqa, okvqa, etc.)
            split: Data split (train, val, test)
        """
        self.feature_dir = Path(feature_dir)
        self.dataset_name = dataset_name
        self.split = split

        # Feature caches
        self.train_q_features = None
        self.val_q_features = None
        self.train_img_features = None
        self.val_img_features = None
        self.train_idx_map = None
        self.val_idx_map = None
        self.valkey2idx = None

    def load_question_features(
        self, metric: str = "question"
    ) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
        """
        Load question CLIP features.

        Args:
            metric: Similarity metric type

        Returns:
            Tuple of (train_features, val_features, train_idx, val_idx)
        """
        if metric == "question" or metric == "imagequestion":
            # Load training features
            train_feat_path = (
                self.feature_dir / f"coco_clip_vitb16_train2017_{self.dataset_name}_question.npy"
            )
            self.train_q_features = np.load(str(train_feat_path))

            # Load validation features
            val_feat_path = (
                self.feature_dir
                / f"coco_clip_vitb16_{self.split}2017_{self.dataset_name}_question.npy"
            )
            self.val_q_features = np.load(str(val_feat_path))

            # Load index mappings
            train_idx_path = (
                self.feature_dir / f"{self.dataset_name}_qa_line2sample_idx_train2017.json"
            )
            self.train_idx_map = json.load(open(train_idx_path, "r"))

            val_idx_path = (
                self.feature_dir / f"{self.dataset_name}_qa_line2sample_idx_{self.split}2017.json"
            )
            val_idx = json.load(open(val_idx_path, "r"))

            # Create reverse mapping for validation
            self.valkey2idx = {}
            for idx, key in val_idx.items():
                self.valkey2idx[key] = int(idx)

            return (self.train_q_features, self.val_q_features, self.train_idx_map, self.valkey2idx)

        return None, None, None, None

    def load_image_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load image CLIP features.

        Returns:
            Tuple of (train_features, val_features)
        """
        # Load training image features
        train_img_path = (
            self.feature_dir
            / f"coco_clip_vitb16_train2017_{self.dataset_name}_convertedidx_image.npy"
        )
        self.train_img_features = np.load(str(train_img_path))

        # Load validation image features
        val_img_path = (
            self.feature_dir
            / f"coco_clip_vitb16_{self.split}2017_{self.dataset_name}_convertedidx_image.npy"
        )
        self.val_img_features = np.load(str(val_img_path))

        return self.train_img_features, self.val_img_features

    def load_all_features(self, metric: str = "imagequestion") -> Dict[str, np.ndarray]:
        """
        Load all features at once.

        Args:
            metric: Similarity metric to use

        Returns:
            Dictionary containing all features and mappings
        """
        # Load question features
        train_q, val_q, train_idx, val_idx = self.load_question_features(metric)

        # Load image features if needed
        if metric == "imagequestion":
            train_img, val_img = self.load_image_features()
        else:
            train_img, val_img = None, None

        return {
            "train_question_features": train_q,
            "val_question_features": val_q,
            "train_image_features": train_img,
            "val_image_features": val_img,
            "train_idx_map": train_idx,
            "val_idx_map": val_idx,
        }

    def get_val_feature_by_key(
        self, key: str, feature_type: str = "question"
    ) -> Optional[np.ndarray]:
        """
        Get validation feature by sample key.

        Args:
            key: Sample key (e.g., "image_id<->question_id")
            feature_type: Type of feature ("question" or "image")

        Returns:
            Feature vector or None
        """
        if key not in self.valkey2idx:
            return None

        idx = self.valkey2idx[key]

        if feature_type == "question":
            if self.val_q_features is None:
                self.load_question_features()
            return self.val_q_features[idx]

        elif feature_type == "image":
            if self.val_img_features is None:
                self.load_image_features()
            return self.val_img_features[idx]

        return None
