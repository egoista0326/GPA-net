from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import (
        DualAttentionBaselineModel,
        OUTPUT_NAMES,
        PhaseSelfAttentionBlock,
        TaskSpecificHead,
    )


__version__ = "0.1.0"
_MODEL_EXPORTS = {
    "DualAttentionBaselineModel",
    "OUTPUT_NAMES",
    "PhaseSelfAttentionBlock",
    "TaskSpecificHead",
}

__all__ = [
    "DualAttentionBaselineModel",
    "OUTPUT_NAMES",
    "PhaseSelfAttentionBlock",
    "TaskSpecificHead",
    "__version__",
]


def __getattr__(name: str):
    if name in _MODEL_EXPORTS:
        from . import model

        value = getattr(model, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'gpanet' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
