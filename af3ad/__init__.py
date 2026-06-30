"""AF3AD core pseudo-anomaly synthesis package.

This package contains the reusable anomaly configuration, preset factory, and
point-cloud synthesizer shared across the online and offline integrations.
"""

from .config import SmartAnomaly_Cfg
from .presets import AnomalyPreset
from .synthesizer import PseudoAnomalySynthesizer

__all__ = [
    "SmartAnomaly_Cfg",
    "AnomalyPreset",
    "PseudoAnomalySynthesizer",
]
