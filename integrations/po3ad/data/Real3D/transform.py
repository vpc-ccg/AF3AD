"""Real3D dataset point cloud transforms.

This module re-exports the shared augmentation primitives used for
AnomalyShapeNet so the Real3D pipeline can depend on a local module
without duplicating implementation details.
"""

import numpy as np

from integrations.po3ad.data.AnomalyShapeNet.transform import (  # noqa: F401
    NormalizeCoord,
    CenterShift,
    RandomRotate,
    SphereCropMask,
    Compose,
    RandomPlaneCut,
    RandomEdgeSegmentCutout
)





__all__ = [
    "NormalizeCoord",
    "CenterShift",
    "RandomRotate",
    "SphereCropMask",
    "RandomPlaneCut",
    "RandomEdgeSegmentCutout",
    "Compose",
]
