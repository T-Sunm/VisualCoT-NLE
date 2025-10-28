"""
Scene Graph Detector - Loads and parses scene graph data
Extracted from main_aokvqa.py
"""

import json
import os
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path


class SceneGraphDetector:
    """
    Detector that loads pre-computed scene graph data.
    Compatible with VinVL scene graphs and custom formats.
    """

    def __init__(
        self,
        sg_dir: str,
        sg_attr_dir: Optional[str] = None,
        sg_caption_dir: Optional[str] = None,
        ocr_train_file: Optional[str] = None,
        ocr_val_file: Optional[str] = None,
        ocr_threshold: float = 0.2,
        iterative_strategy: str = "caption",
    ):
        """
        Initialize Scene Graph Detector.

        Args:
            sg_dir: Directory containing scene graph JSON files
            sg_attr_dir: Directory containing scene graph with attributes
            sg_caption_dir: Directory containing captions for objects
            ocr_train_file: Path to OCR training data
            ocr_val_file: Path to OCR validation data
            ocr_threshold: Confidence threshold for OCR
            iterative_strategy: Strategy for decoding ("sg" or "caption")
        """
        self.sg_dir = sg_dir
        self.sg_attr_dir = sg_attr_dir or sg_dir
        self.sg_caption_dir = sg_caption_dir
        self.iterative_strategy = iterative_strategy
        self.ocr_threshold = ocr_threshold

        # Load OCR data if provided
        self.train_ocr_text = {}
        self.val_ocr_text = {}

        if ocr_train_file and ocr_val_file:
            self.load_ocr(ocr_train_file, ocr_val_file)

    def load_ocr(self, train_ocr_file: str, val_ocr_file: str):
        """
        Load OCR data and match with objects.

        Args:
            train_ocr_file: Path to training OCR JSON
            val_ocr_file: Path to validation OCR JSON
        """
        train_ocr_dict = json.load(open(train_ocr_file))
        val_ocr_dict = json.load(open(val_ocr_file))

        # Process training OCR
        for key in train_ocr_dict:
            self.train_ocr_text[int(key.split("_")[-1])] = self._process_ocr_for_image(
                train_ocr_dict[key], key, is_train=True
            )

        # Process validation OCR
        for key in val_ocr_dict:
            self.val_ocr_text[int(key.split("_")[-1])] = self._process_ocr_for_image(
                val_ocr_dict[key], key, is_train=False
            )

    def _process_ocr_for_image(
        self, ocr_list: List[Dict], image_key: str, is_train: bool = False
    ) -> Dict[str, str]:
        """
        Process OCR data for a single image.

        Args:
            ocr_list: List of OCR detections
            image_key: Image identifier
            is_train: Whether this is training data

        Returns:
            Dictionary mapping object names to OCR text
        """
        ocr_text = {}

        if len(ocr_list) == 0:
            return ocr_text

        # Load corresponding scene graph
        sg_path = os.path.join(self.sg_attr_dir, f"{image_key.split('_')[-1]}.json")

        if not os.path.exists(sg_path):
            return ocr_text

        obj_list = json.load(open(sg_path))

        for ocr_item in ocr_list:
            box = ocr_item["box"]
            text = ocr_item["text"]
            conf = ocr_item["conf"]

            if conf > self.ocr_threshold:
                # Convert box format
                box_rect = [box[0][0], box[0][1], box[1][0], box[2][1]]

                # Find best matching object
                max_match_val = -1
                max_match_obj = ""

                for obj in obj_list[0]:
                    match_val = bounding_box_matching(box_rect, obj["rect"])
                    if match_val > max_match_val:
                        max_match_obj = obj["class"]
                        max_match_val = match_val

                if max_match_obj:
                    ocr_text[max_match_obj] = f"Text {text} is on the {max_match_obj}."

        return ocr_text

    def load_scene_graph(
        self,
        image_id: Union[int, str],
        include_attributes: bool = True,
        include_captions: bool = False,
    ) -> List[Dict]:
        """
        Load scene graph for an image.

        Args:
            image_id: Image ID (int or str)
            include_attributes: Whether to load attributes
            include_captions: Whether to load captions

        Returns:
            List of object detections with attributes
        """
        # Format image filename
        if isinstance(image_id, int):
            filename = f"{str(image_id).zfill(12)}.json"
        else:
            filename = f"{image_id.replace('.jpg', '')}.json"

        # Load basic scene graph
        sg_path = os.path.join(self.sg_dir, filename)

        if not os.path.exists(sg_path):
            return []

        scene_graph = json.load(open(sg_path))

        # Load attributes if requested
        if include_attributes:
            sg_attr_path = os.path.join(self.sg_attr_dir, filename)
            if os.path.exists(sg_attr_path):
                scene_graph_attr = json.load(open(sg_attr_path))
            else:
                scene_graph_attr = scene_graph
        else:
            scene_graph_attr = scene_graph

        # Load captions if requested
        scene_graph_caption = None
        if include_captions and self.sg_caption_dir:
            sg_cap_path = os.path.join(self.sg_caption_dir, filename)

            # Try backup path
            if not os.path.exists(sg_cap_path):
                sg_cap_path = os.path.join(self.sg_caption_dir + "_v2", filename)

            if os.path.exists(sg_cap_path):
                scene_graph_caption = json.load(open(sg_cap_path))

        # Process and combine data
        detections = self._process_scene_graph(scene_graph_attr, scene_graph_caption, image_id)

        return detections

    def _process_scene_graph(
        self,
        scene_graph_attr: List,
        scene_graph_caption: Optional[Union[List, Dict]],
        image_id: Union[int, str],
    ) -> List[List]:
        """
        Process scene graph into detection format.

        Args:
            scene_graph_attr: Scene graph with attributes
            scene_graph_caption: Optional captions
            image_id: Image identifier

        Returns:
            List of detections [conf, class, attrs, caption, ocr_text]
        """
        attr_list = []

        for attr_id, attr in enumerate(scene_graph_attr[0]):
            # Get caption if available
            caption = ""
            if self.iterative_strategy == "caption" and scene_graph_caption:
                if isinstance(scene_graph_caption, list):
                    caption = scene_graph_caption[attr_id]
                else:
                    rect_str = str(attr["rect"])
                    caption = scene_graph_caption.get(
                        rect_str, f"{attr['class']} is {', '.join(attr.get('attr', []))}"
                    )

            # Build detection entry
            if self.iterative_strategy == "caption":
                tmp_attr = [attr["conf"], attr["class"], attr.get("attr", []), caption]
            else:
                tmp_attr = [attr["conf"], attr["class"], attr.get("attr", [])]

            # Add OCR text if available
            img_id = (
                int(image_id) if isinstance(image_id, (int, str)) and str(image_id).isdigit() else 0
            )
            if img_id in self.val_ocr_text and attr["class"] in self.val_ocr_text[img_id]:
                tmp_attr.append(self.val_ocr_text[img_id][attr["class"]])
            else:
                tmp_attr.append("")

            attr_list.append(tmp_attr)

        # Sort by confidence
        attr_list.sort(key=lambda x: x[0], reverse=True)

        return attr_list

    def decode_scene_graph(self, attr_list: List[List]) -> str:
        """
        Decode scene graph attributes into text.

        Args:
            attr_list: List of attributes from scene graph

        Returns:
            Text description of scene
        """
        text_list = []

        for attr in attr_list:
            if self.iterative_strategy == "sg":
                # Format: "object is attr1, attr2, ..."
                if len(attr) >= 3 and attr[2]:
                    text_list.append(f"{attr[1]} is {', '.join(attr[2])}")
                else:
                    text_list.append(attr[1])

            elif self.iterative_strategy == "caption":
                # Use pre-generated caption
                if len(attr) >= 4:
                    text_list.append(attr[3])

                    # Add OCR text if available
                    if len(attr) >= 5 and attr[4]:
                        text_list.append(attr[4])

        return "\n".join(text_list)


def bounding_box_matching(box1: List[float], box2: List[float]) -> float:
    """
    Calculate IoU (Intersection over Union) between two bounding boxes.

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        IoU score (0-1)
    """
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2

    # Check if boxes don't overlap
    if ax1 >= bx2 or ax2 <= bx1 or ay1 >= by2 or ay2 <= by1:
        return 0

    # Calculate intersection
    intersection = (min(ax2, bx2) - max(ax1, bx1)) * (min(ay2, by2) - max(ay1, by1))

    # Calculate union
    area1 = (ax2 - ax1) * (ay2 - ay1)
    area2 = (bx2 - bx1) * (by2 - by1)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def detect(image_path: str, **kwargs) -> List[Dict]:
    """
    Convenience function for scene graph detection.

    Args:
        image_path: Path to image or image ID
        **kwargs: Arguments for SceneGraphDetector

    Returns:
        List of detections
    """
    detector = SceneGraphDetector(**kwargs)

    # Extract image ID from path
    image_id = Path(image_path).stem

    return detector.load_scene_graph(image_id)


# if __name__ == "__main__":
#     import pprint

#     # --- Cấu hình để test ---
#     # Vui lòng thay đổi các đường dẫn này cho phù hợp với cấu trúc project của bạn.
#     # Các đường dẫn này thường được định nghĩa trong file config (ví dụ: configs/datasets/aokvqa.yaml)
#     SG_DIR = "data/processed/input_text/scene_graph_text/scene_graph_coco17"
#     SG_ATTR_DIR = "data/processed/input_text/scene_graph_text/scene_graph_coco17_attr"
#     SG_CAPTION_DIR = "data/processed/input_text/scene_graph_text/scene_graph_coco17_caption"

#     # ID của ảnh bạn muốn test. Ví dụ: '000000000139' (cho ảnh 139.jpg)
#     TEST_IMAGE_ID = "000000000000"
#     # -------------------------

#     print("--- Bắt đầu test SceneGraphDetector ---")

#     # 1. Khởi tạo detector
#     detector = SceneGraphDetector(
#         sg_dir=SG_DIR,
#         sg_attr_dir=SG_ATTR_DIR,
#         sg_caption_dir=SG_CAPTION_DIR,
#         # Các tham số khác có thể để mặc định cho lần test này
#     )
#     print(f"Khởi tạo detector với các đường dẫn:")
#     print(f"  - sg_dir: {SG_DIR}")
#     print(f"  - sg_attr_dir: {SG_ATTR_DIR}")
#     print(f"  - sg_caption_dir: {SG_CAPTION_DIR}")

#     # 2. Gọi hàm load_scene_graph
#     print(f"\nĐang tải scene graph cho ảnh ID: {TEST_IMAGE_ID}...")
#     detections = detector.load_scene_graph(
#         image_id=TEST_IMAGE_ID,
#         include_attributes=True,
#         include_captions=True,
#     )

#     # 3. In kết quả
#     if detections:
#         print(f"\n>>> Tìm thấy {len(detections)} đối tượng. Kết quả:")
#         # Sử dụng pprint để in cho đẹp
#         pprint.pprint(detections)
#     else:
#         print("\n>>> Không tìm thấy đối tượng nào hoặc file không tồn tại.")
#         print(">>> Vui lòng kiểm tra lại TEST_IMAGE_ID và các đường dẫn.")

#     print("\n--- Test kết thúc ---")
