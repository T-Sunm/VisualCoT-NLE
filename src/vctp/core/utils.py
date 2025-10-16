"""Core utilities: logging and seeding helpers."""

import logging
import os
import random
from typing import Optional

import numpy as np


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Create a module-level logger with a sensible default formatter."""
    logger = logging.getLogger(name if name else __name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(os.getenv("LOGLEVEL", "INFO"))
    return logger


def set_global_seed(seed: int) -> None:
    """Set global randomness seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        # Torch not installed or unavailable; proceed with numpy/random only
        pass


def process_answer(answer: str) -> str:
    """Normalize answer text similar to main_aokvqa.process_answer."""
    cleaned = answer.replace(".", "").replace(",", "").lower()
    to_be_removed = {"a", "an", "the", "to", ""}
    tokens = [t for t in cleaned.split(" ") if t not in to_be_removed]
    return " ".join(tokens)
