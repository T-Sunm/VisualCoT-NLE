"""
SEE (Visual Perception) Module
Provides object detection, captioning, feature extraction, and scene graph processing
"""

from typing import Optional

# Import perception modules
try:
    from .perception import (
        NoOpPerception,
        VisualCoTPerception,
        CLIPOnlyPerception,
        BLIP2OnlyPerception,
    )
except ImportError:
    NoOpPerception = None
    VisualCoTPerception = None
    CLIPOnlyPerception = None
    BLIP2OnlyPerception = None

# Import sub-modules
try:
    from . import captions
    from . import detectors
    from . import features
    from . import graphs
except ImportError:
    pass


__all__ = [
    "NoOpPerception",
    "VisualCoTPerception",
    "CLIPOnlyPerception",
    "BLIP2OnlyPerception",
    "captions",
    "detectors",
    "features",
    "graphs",
]
