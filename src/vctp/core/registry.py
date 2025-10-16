"""Simple registries for VCTP modules and a factory helper."""

from typing import Any, Dict, Type

from .interfaces import ConfirmationModule, PerceptionModule, ReasoningModule


SEE_REGISTRY: Dict[str, Type[PerceptionModule]] = {}
THINK_REGISTRY: Dict[str, Type[ReasoningModule]] = {}
CONFIRM_REGISTRY: Dict[str, Type[ConfirmationModule]] = {}


def register_see(name: str):
    def _decorator(cls: Type[PerceptionModule]):
        SEE_REGISTRY[name] = cls
        return cls

    return _decorator


def register_think(name: str):
    def _decorator(cls: Type[ReasoningModule]):
        THINK_REGISTRY[name] = cls
        return cls

    return _decorator


def register_confirm(name: str):
    def _decorator(cls: Type[ConfirmationModule]):
        CONFIRM_REGISTRY[name] = cls
        return cls

    return _decorator


def build_module(kind: str, name: str, *args: Any, **kwargs: Any):
    if kind == "see":
        cls = SEE_REGISTRY.get(name)
    elif kind == "think":
        cls = THINK_REGISTRY.get(name)
    elif kind == "confirm":
        cls = CONFIRM_REGISTRY.get(name)
    else:
        raise ValueError(f"Unknown module kind: {kind}")
    if cls is None:
        raise KeyError(f"No module registered for {kind}:{name}")
    return cls(*args, **kwargs)
