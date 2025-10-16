"""Config loading and schemas (skeleton)."""

from dataclasses import dataclass
from typing import Any, Dict

import yaml


@dataclass
class ExperimentConfig:
    raw: Dict[str, Any]


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_experiment_config(path: str) -> ExperimentConfig:
    data = load_yaml(path)
    return ExperimentConfig(raw=data)
