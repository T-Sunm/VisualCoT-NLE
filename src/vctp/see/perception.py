"""
Unified Perception Module
Combines all SEE components: detection, captioning, features, and scene graphs
Based on the Visual CoT perception pipeline from main_aokvqa.py
"""

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np

from vctp.core.interfaces import PerceptionModule
from vctp.core.registry import register_see
from vctp.core.types import DetectedObject, EvidenceBundle


@register_see("noop-see")
class NoOpPerception(PerceptionModule):
    """Minimal perception producing placeholder evidence."""

    def run(self, image_path: str, question: str, **kwargs: Dict[str, Any]) -> EvidenceBundle:
        return EvidenceBundle(
            image_id=image_path,
            global_caption="placeholder caption",
            detected_objects=[DetectedObject(name="object", bbox=[0, 0, 1, 1], score=1.0)],
            attributes={},
            relations=[],
            clip_image_embed=None,
            region_captions=None,
        )


@register_see("visualcot-perception")
class VisualCoTPerception(PerceptionModule):
    """
    Complete Visual CoT Perception Module.

    Combines:
    - Scene graph detection (VinVL/pre-computed)
    - Global image captioning (BLIP2/VinVL)
    - Local region captioning (BLIP2)
    - CLIP feature extraction
    - Scene graph processing

    Based on the perception pipeline from main_aokvqa.py
    """

    def __init__(
        self,
        # Scene Graph Config
        sg_dir: Optional[str] = None,
        sg_attr_dir: Optional[str] = None,
        sg_caption_dir: Optional[str] = None,
        ocr_train_file: Optional[str] = None,
        ocr_val_file: Optional[str] = None,
        # Strategy Config
        iterative_strategy: str = "caption",
        caption_type: str = "vinvl",
        # BLIP2 Config
        use_blip2: bool = False,
        blip2_model_type: str = "Salesforce/blip2-opt-2.7b",
        # CLIP Config
        use_clip_features: bool = True,
        clip_model_name: str = "openai/clip-vit-base-patch16",
        # General Config
        device: Optional[str] = None,
        debug: bool = False,
        **kwargs,
    ):
        """
        Initialize Visual CoT Perception Module.

        Args:
            sg_dir: Directory containing scene graph JSON files
            sg_attr_dir: Directory for scene graphs with attributes
            sg_caption_dir: Directory for object captions
            ocr_train_file: OCR training data file
            ocr_val_file: OCR validation data file
            iterative_strategy: "sg" for scene graph format, "caption" for captions
            caption_type: Type of captions to use
            use_blip2: Whether to use BLIP2 for captioning
            blip2_model_type: BLIP2 model variant
            use_clip_features: Whether to extract CLIP features
            clip_model_name: CLIP model name
            device: Device to run models on
            debug: Enable debug mode
        """
        self.iterative_strategy = iterative_strategy
        self.caption_type = caption_type
        self.use_blip2 = use_blip2
        self.use_clip_features = use_clip_features
        self.debug = debug
        self.device = device

        # Initialize Scene Graph Detector
        if sg_dir:
            from vctp.see.detectors.scene_graph_detector import SceneGraphDetector

            self.sg_detector = SceneGraphDetector(
                sg_dir=sg_dir,
                sg_attr_dir=sg_attr_dir,
                sg_caption_dir=sg_caption_dir,
                ocr_train_file=ocr_train_file,
                ocr_val_file=ocr_val_file,
                iterative_strategy=iterative_strategy,
            )
        else:
            self.sg_detector = None

        # Initialize Scene Graph Processor
        from vctp.see.graphs import SceneGraphProcessor

        self.sg_processor = SceneGraphProcessor(
            strategy=iterative_strategy, include_ocr=(caption_type == "vinvl_ocr")
        )

        # Initialize BLIP2 Captioner (if enabled) - FIXED HERE
        if use_blip2:
            from vctp.see.captions.blip_captioner import BLIP2Captioner

            self.captioner = BLIP2Captioner(
                model_type=blip2_model_type,
                device=device,
                debug=debug,
            )
        else:
            self.captioner = None

        # Initialize CLIP Feature Extractor (if enabled)
        if use_clip_features:
            from vctp.see.features.clip_extractor import CLIPFeatureExtractor

            self.clip_extractor = CLIPFeatureExtractor(model_name=clip_model_name, device=device)
        else:
            self.clip_extractor = None

    def run(
        self,
        image_path: str,
        question: str,
        max_objects: int = 50,
        generate_region_captions: bool = False,
        **kwargs: Dict[str, Any],
    ) -> EvidenceBundle:
        """Run perception pipeline on image."""
        # Extract image ID from path
        image_id = self._get_image_id(image_path)

        # Initialize debug info collector
        debug_info = {
            "scene_graph": None,
            "detected_objects": [],
            "captions": {},
            "clip_features": None,
            "processing_steps": [],
            "errors": []
        }

        # Step 1: Detect Objects (from scene graph or BLIP2)
        debug_info["processing_steps"].append("Starting object detection")
        detected_objects, scene_graph_attrs = self._detect_objects(
            image_id, image_path, max_objects, debug_info
        )

        # Step 2: Generate Global Caption
        debug_info["processing_steps"].append("Generating global caption")
        global_caption = self._generate_global_caption(
            image_path, question, scene_graph_attrs, debug_info
        )

        # Step 3: Generate Region Captions (optional)
        region_captions = None
        if generate_region_captions and detected_objects:
            debug_info["processing_steps"].append("Generating region captions")
            region_captions = self._generate_region_captions(
                image_path, question, detected_objects, debug_info
            )

        # Step 4: Extract CLIP Features
        clip_embed = None
        if self.use_clip_features:
            debug_info["processing_steps"].append("Extracting CLIP features")
            clip_embed = self._extract_clip_features(image_path, debug_info)

        # Step 5: Extract Attributes and Relations
        attributes = self._extract_attributes(scene_graph_attrs)
        relations = self._extract_relations(scene_graph_attrs)

        # Finalize debug info
        debug_info["scene_graph"] = self.sg_processor.decode(
            scene_graph_attrs, format_type="text"
        ) if scene_graph_attrs else ""
        
        debug_info["detected_objects"] = [
            {
                "name": obj.name,
                "score": obj.score,
                "attributes": obj.attributes or []
            }
            for obj in detected_objects
        ]

        # Build evidence bundle
        evidence = EvidenceBundle(
            image_id=str(image_id),
            global_caption=global_caption,
            detected_objects=detected_objects,
            attributes=attributes,
            relations=relations,
            clip_image_embed=clip_embed,
            region_captions=region_captions,
            debug_info=debug_info  # Thêm debug info
        )

        return evidence

    def _get_image_id(self, image_path: str) -> str:
        """Extract image ID from path."""
        path = Path(image_path)
        base_name = path.stem
        
        if "COCO_train2014_" in base_name:
            image_id = base_name.replace("COCO_train2014_", "")
        elif "COCO_val2014_" in base_name:
            image_id = base_name.replace("COCO_val2014_", "")
        elif "COCO_test2014_" in base_name:
            image_id = base_name.replace("COCO_test2014_", "")
        else:
            image_id = base_name
    
        return image_id

    def _detect_objects(
        self, image_id: str, image_path: str, max_objects: int, debug_info: Dict
    ) -> Tuple[List[DetectedObject], List[List]]:
        """Detect objects in image."""
        detected_objects = []
        scene_graph_attrs = []
        
        # Try to load from scene graph first
        if self.sg_detector:
            try:
                scene_graph_attrs = self.sg_detector.load_scene_graph(
                    image_id,
                    include_attributes=True,
                    include_captions=(self.iterative_strategy == "caption"),
                )
                
                debug_info["captions"]["scene_graph_source"] = "pre-computed"
                debug_info["processing_steps"].append(f"Loaded {len(scene_graph_attrs)} objects from scene graph")
                
                # Convert to DetectedObject format
                for attr in scene_graph_attrs[:max_objects]:
                    detected_objects.append(
                        DetectedObject(
                            name=attr[1],
                            bbox=[0, 0, 1, 1],
                            score=float(attr[0]),
                            attributes=attr[2] if len(attr) > 2 else [],
                        )
                    )

            except Exception as e:
                debug_info["errors"].append(f"Scene graph loading failed: {str(e)}")
                if self.debug:
                    print(f"Failed to load scene graph: {e}")

        # Fallback to BLIP2 detection
        if not detected_objects and self.use_blip2 and self.captioner:
            try:
                blip2_objects = self.captioner.detect_objects(image_path, max_objects=max_objects)
                debug_info["captions"]["blip2_detection"] = True
                debug_info["processing_steps"].append(f"BLIP2 detected {len(blip2_objects)} objects")
                
                # Convert BLIP2 format to DetectedObject
                for obj in blip2_objects:
                    detected_objects.append(
                        DetectedObject(
                            name=obj[1],
                            bbox=[0, 0, 1, 1],
                            score=float(obj[0]),
                            attributes=[],
                        )
                    )
                    scene_graph_attrs.append([obj[0], obj[1], [], ""])

            except Exception as e:
                debug_info["errors"].append(f"BLIP2 detection failed: {str(e)}")

        # Sort by confidence
        if scene_graph_attrs:
            scene_graph_attrs.sort(key=lambda x: x[0], reverse=True)

        return detected_objects, scene_graph_attrs

    def _generate_global_caption(
        self, image_path: str, question: str, scene_graph_attrs: List[List], debug_info: Dict
    ) -> str:
        """Generate global image caption."""
        global_caption = ""

        # Use BLIP2 for global captioning
        if self.use_blip2 and self.captioner:
            try:
                global_caption = self.captioner.generate_global_caption(
                    image_path, question=question
                )
                debug_info["captions"]["global_caption_source"] = "blip2"
                debug_info["captions"]["global_caption"] = global_caption
            except Exception as e:
                debug_info["errors"].append(f"BLIP2 global caption failed: {str(e)}")

        # Fallback: Use scene graph description
        if not global_caption and scene_graph_attrs:
            global_caption = self.sg_processor.decode(
                scene_graph_attrs[:10], format_type="text"
            )
            debug_info["captions"]["global_caption_source"] = "scene_graph"
            debug_info["captions"]["global_caption"] = global_caption

        return global_caption

    def _extract_clip_features(self, image_path: str, debug_info: Dict) -> Optional[np.ndarray]:
        """Extract CLIP image features."""
        if not self.clip_extractor:
            return None

        try:
            features = self.clip_extractor.extract_image_features(
                image_path, normalize=True, return_numpy=True
            )
            clip_features = features[0] if len(features.shape) == 2 else features
            
            debug_info["clip_features"] = {
                "shape": clip_features.shape,
                "norm": float(np.linalg.norm(clip_features)),
                "mean": float(np.mean(clip_features)),
                "std": float(np.std(clip_features))
            }
            
            return clip_features
        except Exception as e:
            debug_info["errors"].append(f"CLIP feature extraction failed: {str(e)}")
            return None

    def _extract_attributes(self, scene_graph_attrs: List[List]) -> Dict[str, List[str]]:
        """Extract object attributes from scene graph."""
        attributes = {}

        for attr in scene_graph_attrs:
            if len(attr) >= 3:
                obj_name = attr[1]
                obj_attrs = attr[2]

                if obj_attrs:
                    attributes[obj_name] = obj_attrs

        return attributes

    def _extract_relations(self, scene_graph_attrs: List[List]) -> List[str]:
        """Extract relationships (placeholder - can be enhanced)."""
        # This is a simplified version
        # In full implementation, would use scene graph relationships
        relations = []

        # Could be extracted from scene graph edges
        # For now, return empty list

        return relations


@register_see("clip-only-perception")
class CLIPOnlyPerception(PerceptionModule):
    """
    Lightweight perception using only CLIP.
    Useful for quick prototyping or when scene graphs are unavailable.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        device: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize CLIP-only perception.

        Args:
            model_name: CLIP model name
            device: Device to run on
        """
        from vctp.see.features.clip_extractor import CLIPFeatureExtractor

        self.clip_extractor = CLIPFeatureExtractor(model_name=model_name, device=device)

    def run(self, image_path: str, question: str, **kwargs: Dict[str, Any]) -> EvidenceBundle:
        """Run CLIP-only perception."""
        # Extract CLIP features
        clip_embed = self.clip_extractor.extract_image_features(
            image_path, normalize=True, return_numpy=True
        )

        if len(clip_embed.shape) == 2:
            clip_embed = clip_embed[0]

        # Minimal evidence bundle
        return EvidenceBundle(
            image_id=str(Path(image_path).stem),
            global_caption="Image features extracted with CLIP",
            detected_objects=[],
            attributes={},
            relations=[],
            clip_image_embed=clip_embed,
            region_captions=None,
        )


@register_see("blip2-only-perception")
class BLIP2OnlyPerception(PerceptionModule):
    """
    Perception using only BLIP2 for detection and captioning.
    Good for dynamic scenes without pre-computed scene graphs.
    """

    def __init__(
        self,
        model_type: str = "Salesforce/blip2-opt-2.7b",
        device: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize BLIP2-only perception.

        Args:
            model_type: BLIP2 model type
            device: Device to run on
        """
        from vctp.see.captions.blip_captioner import BLIP2Captioner

        self.captioner = BLIP2Captioner(
            model_type=model_type, device=device
        )

    def run(
        self, image_path: str, question: str, max_objects: int = 10, **kwargs: Dict[str, Any]
    ) -> EvidenceBundle:
        """Run BLIP2-only perception."""
        # Detect objects
        blip2_objects = self.captioner.detect_objects(image_path, max_objects)

        detected_objects = [
            DetectedObject(name=obj[1], bbox=[0, 0, 1, 1], score=float(obj[0]), attributes=[])
            for obj in blip2_objects
        ]

        # Generate global caption
        global_caption = self.captioner.generate_global_caption(image_path, question)

        # Generate region captions
        region_captions = {}
        for obj in blip2_objects[:5]:
            caption = self.captioner.generate_local_caption(obj[1], question, image_path)
            region_captions[obj[1]] = caption

        return EvidenceBundle(
            image_id=str(Path(image_path).stem),
            global_caption=global_caption,
            detected_objects=detected_objects,
            attributes={},
            relations=[],
            clip_image_embed=None,
            region_captions=region_captions,
        )


# if __name__ == "__main__":
#     import pprint
#     from pathlib import Path

#     # --- Configuration ---
#     # This assumes the script is run from the project's root directory.
#     # Adjust paths if you run it from somewhere else.
#     PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
#     SG_BASE_DIR = PROJECT_ROOT / "data/processed/input_text/scene_graph_text"
#     IMAGE_DIR = PROJECT_ROOT / "data/raw/aokvqa_images"

#     # --- Test Data ---
#     # From your expected output example.
#     TEST_IMAGE_ID = "000000000000"
#     TEST_QUESTION = "What is the man doing?"

#     test_image_path = IMAGE_DIR / f"{TEST_IMAGE_ID}.jpg"
#     if not test_image_path.exists():
#         print(f"Warning: Test image '{test_image_path}' not found.")
#         # Fallback to an image that exists in the project structure for demonstration
#         FALLBACK_IMAGE_ID = "000000000008"
#         test_image_path = IMAGE_DIR / f"{FALLBACK_IMAGE_ID}.jpg"
#         if test_image_path.exists():
#             print(f"Using fallback image: '{test_image_path}'")
#             TEST_IMAGE_ID = FALLBACK_IMAGE_ID
#         else:
#             print(f"Error: Fallback image not found either. Please check paths.")
#             exit()

#     # --- Initialize Perception Module ---
#     print("Initializing VisualCoTPerception module...")
#     perception_module = VisualCoTPerception(
#         sg_dir=str(SG_BASE_DIR / "scene_graph_coco17"),
#         sg_attr_dir=str(SG_BASE_DIR / "scene_graph_coco17_attr"),
#         sg_caption_dir=str(SG_BASE_DIR / "scene_graph_coco17_caption"),
#         use_clip_features=True,
#         use_blip2=False,  # Keep False unless you have BLIP2 set up
#         debug=True,
#     )
#     print("Initialization complete.")

#     # --- Run Perception ---
#     print(f"\nRunning perception for image ID: {TEST_IMAGE_ID}")
#     print(f"Question: {TEST_QUESTION}")

#     evidence = perception_module.run(
#         image_path=str(test_image_path),
#         question=TEST_QUESTION,
#         max_objects=10,  # Limit for cleaner output
#     )

#     # --- Display Results ---
#     print("\n" + "=" * 20 + " PERCEPTION OUTPUT " + "=" * 20)
#     pprint.pprint(evidence)
#     print("=" * 61)

#     print("\n--- Expected Output Structure ---")
#     print(
#         """
#     EvidenceBundle(
#         image_id="000000000057",
#         global_caption="A man playing tennis...",
#         detected_objects=[DetectedObject(name="man", bbox=[...], score=0.95, ...)],
#         attributes={"man": ["standing", "playing"], ...},
#         relations=[...],
#         clip_image_embed=np.array([...]), # Should not be None
#         region_captions=None or {...}
#     )
#     """
#     )
#     print("Test finished. Compare the output above with the expected structure.")
