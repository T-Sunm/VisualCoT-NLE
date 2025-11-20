"""Data models used across VCTP pipeline."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class DetectedObject:
    name: str
    bbox: List[float]  # [x1, y1, x2, y2]
    score: float
    attributes: Optional[List[str]] = None


@dataclass
class EvidenceBundle:
    image_id: str
    global_caption: Optional[str]
    detected_objects: List[DetectedObject]
    attributes: Dict[str, Any]
    relations: List[Dict[str, Any]]
    clip_image_embed: Optional[List[float]]
    region_captions: Optional[List[str]]

    # Thêm debug info cho SEE module
    debug_info: Optional[Dict[str, Any]] = None


@dataclass
class ReasoningOutput:
    candidate_answer: str
    cot_rationale: str
    used_concepts: List[str]
    confidence: Optional[float] = None 
    metadata: Optional[Dict[str, Any]] = None  

    debug_info: Optional[Dict[str, Any]] = None


@dataclass
class ConfirmationOutput:
    is_confirmed: bool
    score: float
    rationale: str
