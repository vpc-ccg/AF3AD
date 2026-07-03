"""AF3AD core pseudo-anomaly synthesis package.

This package contains the reusable anomaly configuration, preset factory, and
point-cloud synthesizer shared across the online and offline integrations.
"""

from .config import SmartAnomaly_Cfg
from .presets import AnomalyPreset
from .random_factory import RandomFactory, normalize_synthesis_policy
from .synthesizer import PseudoAnomalySynthesizer

__all__ = [
    "SmartAnomaly_Cfg",
    "AnomalyPreset",
    "RandomFactory",
    "PseudoAnomalySynthesizer",
    "normalize_synthesis_policy",
]
