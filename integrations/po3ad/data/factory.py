"""Utility helpers for loading dataset-specific preprocessing modules."""

from importlib import import_module
from types import ModuleType
from typing import Tuple

_DATASET_MODULES = {
    "AnomalyShapeNet": "integrations.po3ad.data.AnomalyShapeNet.preprocessing",
    "Real3D": "integrations.po3ad.data.Real3D.preprocessing",
}


def get_dataset_module(name: str) -> ModuleType:
    try:
        module_path = _DATASET_MODULES[name]
    except KeyError as exc:  # pragma: no cover - simple guard
        raise ValueError(f"Unsupported dataset: {name}") from exc
    return import_module(module_path)


def get_dataset_class(name: str):
    module = get_dataset_module(name)
    return module.Dataset


def get_param_queues(name: str) -> Tuple[object, object, object]:
    module = get_dataset_module(name)
    standard = getattr(module, "standard_param_queue")
    rollout = getattr(module, "rollout_param_queue")
    param = getattr(module, "param_queue", standard)
    return standard, rollout, param
