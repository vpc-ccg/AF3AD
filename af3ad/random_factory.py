"""Randomized pseudo-anomaly configuration factory.

This module provides the lightweight ``RandomFactory`` interface used by the
PO3AD integration when sampling fully randomized AF3AD anomaly shapes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Tuple

import numpy as np

from .config import SmartAnomaly_Cfg


_SYNTHESIS_POLICY_ALIASES = {
    "af3ad_presets": "af3ad_presets",
    "presets": "af3ad_presets",
    "preset": "af3ad_presets",
    "randomfactory_raw": "randomfactory_raw",
    "random_factory_raw": "randomfactory_raw",
    "random": "randomfactory_raw",
    "randomfactory_filtered": "randomfactory_filtered",
    "random_factory_filtered": "randomfactory_filtered",
    "filtered": "randomfactory_filtered",
    "preset_randomfactory_mix": "preset_randomfactory_mix",
    "preset_random_factory_mix": "preset_randomfactory_mix",
    "mix": "preset_randomfactory_mix",
}


def normalize_synthesis_policy(policy: str) -> str:
    """Normalize synthesis-policy aliases used by detector integrations."""

    normalized = str(policy).strip().lower().replace("-", "_")
    try:
        return _SYNTHESIS_POLICY_ALIASES[normalized]
    except KeyError as exc:
        valid = "', '".join(sorted(_SYNTHESIS_POLICY_ALIASES))
        raise ValueError(
            f"Unsupported synthesis_policy '{policy}'. Expected one of '{valid}'."
        ) from exc


@dataclass
class RandomFactoryConfig:
    """Sampling and filtering controls for :class:`RandomFactory`."""

    R_low_bound: float = 0.03
    R_up_bound: float = 0.25
    R_alpha: float = 2.0
    R_beta: float = 2.0
    B_low_bound: float = 0.06
    B_up_bound: float = 0.125
    B_alpha: float = 2.0
    B_beta: float = 2.0
    eps: float = 1e-6
    max_resample_attempts: int = 10
    min_changed_points: int = 0
    min_changed_fraction: float = 0.0
    min_anomaly_l2: float = 0.0
    anomaly_strength_preset: str = "default"
    anomaly_radius_multiplier: float = 1.0
    anomaly_scale_multiplier: float = 1.0


class RandomFactory:
    """Sample randomized :class:`SmartAnomaly_Cfg` objects.

    The PO3AD integration only requires a compact interface: sample a config,
    compute displacement statistics, filter weak samples, and emit metadata.
    """

    _KERNELS = ("cosine", "gaussian", "poly", "hard")
    _DIR_MODES = ("normal_point", "normal_mean", "tangent_u", "tangent_v")
    _GATE_MODES = ("normals", "global")

    def __init__(self, args: Any = None):
        self.args = args
        self.config = RandomFactoryConfig(
            R_low_bound=float(getattr(args, "R_low_bound", 0.03)),
            R_up_bound=float(getattr(args, "R_up_bound", 0.25)),
            R_alpha=float(getattr(args, "R_alpha", 2.0)),
            R_beta=float(getattr(args, "R_beta", 2.0)),
            B_low_bound=float(getattr(args, "B_low_bound", 0.06)),
            B_up_bound=float(getattr(args, "B_up_bound", 0.125)),
            B_alpha=float(getattr(args, "B_alpha", 2.0)),
            B_beta=float(getattr(args, "B_beta", 2.0)),
            eps=float(getattr(args, "anomaly_strength_eps", 1e-6)),
            max_resample_attempts=int(getattr(args, "max_resample_attempts", 10)),
            min_changed_points=int(getattr(args, "min_changed_points", 0)),
            min_changed_fraction=float(getattr(args, "min_changed_fraction", 0.0)),
            min_anomaly_l2=float(getattr(args, "min_anomaly_l2", 0.0)),
            anomaly_strength_preset=str(
                getattr(args, "anomaly_strength_preset", "default")
            ),
            anomaly_radius_multiplier=float(
                getattr(args, "anomaly_radius_multiplier", 1.0)
            ),
            anomaly_scale_multiplier=float(
                getattr(args, "anomaly_scale_multiplier", 1.0)
            ),
        )

    def sample_config(self) -> Tuple[SmartAnomaly_Cfg, Dict[str, Any]]:
        """Return a randomized anomaly config and serializable metadata."""

        rng = np.random
        R = self._sample_beta_range(
            self.config.R_low_bound,
            self.config.R_up_bound,
            self.config.R_alpha,
            self.config.R_beta,
        )
        B = self._sample_beta_range(
            self.config.B_low_bound,
            self.config.B_up_bound,
            self.config.B_alpha,
            self.config.B_beta,
        )
        R, B = self._apply_strength_scales(R, B)

        radii = (
            float(rng.uniform(0.45, 3.0)),
            float(rng.uniform(0.45, 1.6)),
            float(rng.uniform(0.35, 1.2)),
        )
        kernel = str(rng.choice(self._KERNELS))
        dir_mode = str(rng.choice(self._DIR_MODES))
        gate_mode = str(rng.choice(self._GATE_MODES))
        one_sided = bool(rng.rand() < float(getattr(self.args, "one_sided_prob", 0.7)))
        alpha = int(1 if rng.rand() < float(getattr(self.args, "p_bulge", 0.5)) else -1)
        n_global = self._random_unit_vector()

        cfg = SmartAnomaly_Cfg(
            R=R,
            B=B,
            alpha=alpha,
            p_bulge=float(getattr(self.args, "p_bulge", 0.5)),
            kernel=kernel,
            q=float(getattr(self.args, "poly_q", rng.uniform(2.0, 6.0))),
            sigma=float(rng.uniform(0.25, 0.6)),
            radii=radii,
            dir_mode=dir_mode,
            one_sided=one_sided,
            gate_mode=gate_mode,
            n_global=tuple(float(x) for x in n_global),
            gate_soft=True,
            gate_sharpness=float(getattr(self.args, "gate_sharpness", 30.0)),
            gate_offset=float(getattr(self.args, "gate_offset", 0.0)),
        )

        metadata = asdict(cfg)
        metadata["factory"] = "random"
        metadata["radii"] = [float(x) for x in cfg.radii]
        metadata["n_global"] = [float(x) for x in cfg.n_global]
        return cfg, metadata

    def compute_stats(self, original: np.ndarray, generated: np.ndarray) -> Dict[str, float]:
        """Compute simple displacement statistics for a generated sample."""

        original = np.asarray(original, dtype=np.float32)
        generated = np.asarray(generated, dtype=np.float32)
        if original.shape != generated.shape:
            raise ValueError(
                f"Point arrays must have matching shapes, got {original.shape} and {generated.shape}."
            )

        displacement = np.linalg.norm(generated - original, axis=1)
        changed = displacement > self.config.eps
        changed_points = int(changed.sum())
        total_points = max(int(displacement.shape[0]), 1)
        anomaly_l1 = float(np.abs(generated - original).sum())
        anomaly_l2 = float(np.linalg.norm((generated - original).reshape(-1)))

        return {
            "changed_points": changed_points,
            "changed_fraction": float(changed_points) / float(total_points),
            "mean_displacement": float(displacement[changed].mean()) if changed_points else 0.0,
            "max_displacement": float(displacement.max()) if displacement.size else 0.0,
            "anomaly_l1": anomaly_l1,
            "anomaly_l2": anomaly_l2,
            "strength_score": float(anomaly_l2 / np.sqrt(float(total_points))),
        }

    def passes_filter(self, stats: Dict[str, float]) -> Tuple[bool, Tuple[str, ...]]:
        """Return whether sampled stats pass configured minimum thresholds."""

        rejected = []
        if stats.get("changed_points", 0) < self.config.min_changed_points:
            rejected.append("min_changed_points")
        if stats.get("changed_fraction", 0.0) < self.config.min_changed_fraction:
            rejected.append("min_changed_fraction")
        if stats.get("anomaly_l2", 0.0) < self.config.min_anomaly_l2:
            rejected.append("min_anomaly_l2")
        return not rejected, tuple(rejected)

    def build_metadata(
        self,
        synthesis_policy: str,
        sampled_metadata: Dict[str, Any],
        stats: Dict[str, float],
        *,
        attempt_count: int,
        accepted: bool,
        rejected_reasons: Iterable[str],
        fallback: bool,
        center: np.ndarray,
    ) -> Dict[str, Any]:
        """Combine sampling metadata and filtering stats for logging/export."""

        metadata = dict(sampled_metadata)
        metadata.update({f"rf_{key}": value for key, value in stats.items()})
        metadata.update(
            {
                "synthesis_policy": synthesis_policy,
                "rf_attempt_count": int(attempt_count),
                "rf_accepted": bool(accepted),
                "rf_fallback": bool(fallback),
                "rf_rejected_reasons": ",".join(str(x) for x in rejected_reasons),
            }
        )
        center_arr = np.asarray(center, dtype=np.float32).reshape(-1)
        if center_arr.size >= 3:
            metadata.update(
                {
                    "selected_patch_center_x": float(center_arr[0]),
                    "selected_patch_center_y": float(center_arr[1]),
                    "selected_patch_center_z": float(center_arr[2]),
                }
            )
        return metadata

    def _sample_beta_range(self, low: float, high: float, alpha: float, beta: float) -> float:
        value = low + (high - low) * np.random.beta(alpha, beta)
        return float(value)

    def _apply_strength_scales(self, radius: float, magnitude: float) -> Tuple[float, float]:
        radius_scale = 1.0
        magnitude_scale = 1.0
        preset = self.config.anomaly_strength_preset.strip().lower()
        if preset == "small":
            radius_scale *= 0.85
            magnitude_scale *= 0.75
        elif preset == "large":
            radius_scale *= 1.20
            magnitude_scale *= 1.35
        radius_scale *= self.config.anomaly_radius_multiplier
        magnitude_scale *= self.config.anomaly_scale_multiplier
        return float(radius * radius_scale), float(magnitude * magnitude_scale)

    @staticmethod
    def _random_unit_vector() -> np.ndarray:
        vec = np.random.normal(size=3)
        norm = np.linalg.norm(vec)
        if norm <= 1e-12:
            return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        return (vec / norm).astype(np.float32)
