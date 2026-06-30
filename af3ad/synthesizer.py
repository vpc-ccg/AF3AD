"""Pseudo anomaly synthesiser for 3-D point clouds.

This module provides :class:`PseudoAnomalySynthesizer`, a self-contained class
that applies preset-driven deformations to point clouds in order to create
realistic pseudo anomalies for both online and offline AF3AD workflows.
"""

import numpy as np

from .config import SmartAnomaly_Cfg
from .presets import AnomalyPreset


class PseudoAnomalySynthesizer:
    """Apply preset-based pseudo anomalies to 3-D point clouds.

    Parameters
    ----------
    args : object, optional
        Configuration object forwarded to :class:`AnomalyPreset` for
        controlling the Beta-distribution ranges of radius and magnitude.
        When *None*, sensible defaults are used.
    binary_anomaly_label : bool
        If *True*, anomaly weights are binarised (0/1) instead of soft.

    Examples
    --------
    >>> import numpy as np
    >>> from af3ad import PseudoAnomalySynthesizer
    >>> synth = PseudoAnomalySynthesizer()
    >>> rng = np.random.default_rng(42)
    >>> pts = rng.standard_normal((1024, 3)).astype(np.float32)
    >>> nrm = pts / (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-8)
    >>> center = pts[rng.integers(len(pts))]
    >>> cfg = synth.preset_factory.presets[0]()  # basic bulge
    >>> deformed = synth.generate(pts, nrm, center, cfg)
    >>> deformed.shape
    (1024, 3)
    """

    def __init__(self, args=None, binary_anomaly_label=False):
        self.preset_factory = AnomalyPreset(args)
        self.binary_anomaly_label = binary_anomaly_label

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, points, normals, center, anomaly_cfg=None):
        """Synthesise a pseudo anomaly on the given point cloud.

        Parameters
        ----------
        points : numpy.ndarray, shape (N, 3)
            Original point positions.
        normals : numpy.ndarray, shape (N, 3)
            Per-point surface normals.
        center : numpy.ndarray, shape (3,)
            Centre of the anomaly region.
        anomaly_cfg : SmartAnomaly_Cfg, optional
            Anomaly configuration.  When *None*, the default
            ``SmartAnomaly_Cfg()`` is used.

        Returns
        -------
        numpy.ndarray, shape (N, 3)
            Deformed point positions.
        """
        if anomaly_cfg is None:
            anomaly_cfg = SmartAnomaly_Cfg()

        rng = np.random.default_rng(anomaly_cfg.seed)

        P = points.astype(np.float32, copy=False)
        N = normals.astype(np.float32, copy=False)
        c = center.astype(np.float32, copy=False)

        # Normalise normals
        nrm = np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
        N = N / nrm

        # Determine radius
        diam = float(np.linalg.norm(P.max(0) - P.min(0)))
        R = anomaly_cfg.R if anomaly_cfg.R is not None else 0.2 * diam

        # Local PCA frame for anisotropy & tangents
        U = _local_frame(P, c)
        ru, rv, rn = anomaly_cfg.radii

        # Coordinates in local frame and anisotropic distance
        X = (P - c) @ U
        invQ = np.diag([
            1.0 / ((ru * R) + 1e-12) ** 2,
            1.0 / ((rv * R) + 1e-12) ** 2,
            1.0 / ((rn * R) + 1e-12) ** 2,
        ])
        t = np.sqrt(np.sum((X @ invQ) * X, axis=1))

        # Falloff weights
        w = _kernel(t, anomaly_cfg.kernel, anomaly_cfg.q, anomaly_cfg.sigma)

        # One-sided gating
        if anomaly_cfg.one_sided:
            if anomaly_cfg.gate_mode == "global":
                g = _one_side_gate(
                    P, c, anomaly_cfg.n_global,
                    offset=anomaly_cfg.gate_offset,
                    sharpness=anomaly_cfg.gate_sharpness,
                    soft=anomaly_cfg.gate_soft,
                )
            elif anomaly_cfg.gate_mode == "normals":
                nominal = U[:, 2]
                g = _normal_alignment_gate(
                    N, nominal,
                    cos_thresh=0.0,
                    sharpness=anomaly_cfg.gate_sharpness,
                    soft=anomaly_cfg.gate_soft,
                )
            else:
                raise ValueError(
                    "gate_mode must be one of {'global', 'normals'}"
                )
            w = w * g

        if self.binary_anomaly_label:
            w = (w > 1e-6).astype(np.float32)

        # Direction field
        if anomaly_cfg.dir_mode == "normal_point":
            D = N
        elif anomaly_cfg.dir_mode == "normal_mean":
            D = np.repeat(U[:, 2][None, :], len(P), axis=0)
        elif anomaly_cfg.dir_mode == "tangent_u":
            D = np.repeat(U[:, 0][None, :], len(P), axis=0)
        elif anomaly_cfg.dir_mode == "tangent_v":
            D = np.repeat(U[:, 1][None, :], len(P), axis=0)
        else:
            raise ValueError(f"Unknown dir_mode: {anomaly_cfg.dir_mode}")

        # Alpha
        alpha = anomaly_cfg.alpha
        if alpha is None:
            alpha = 1 if rng.random() < anomaly_cfg.p_bulge else -1

        # Displacement
        disp = (alpha * anomaly_cfg.B * w)[:, None] * D
        new_points = P + disp

        return new_points

    def generate_original(self, points, normals, center, distance_to_move=0.08):
        """Original (simple) pseudo anomaly generation.

        Displaces points along their normals with a cosine-cap falloff
        from *center*.

        Parameters
        ----------
        points : numpy.ndarray, shape (N, 3)
            Original point positions.
        normals : numpy.ndarray, shape (N, 3)
            Per-point surface normals.
        center : numpy.ndarray, shape (3,)
            Centre of the anomaly region.
        distance_to_move : float
            Maximum displacement magnitude.

        Returns
        -------
        numpy.ndarray, shape (N, 3)
            Deformed point positions.
        """
        distances_to_center = np.linalg.norm(points - center, axis=1)
        max_distance = np.max(distances_to_center)

        movement_ratios = 1 - (distances_to_center / max_distance)
        movement_ratios = (movement_ratios - np.min(movement_ratios)) / \
            (np.max(movement_ratios) - np.min(movement_ratios))

        directions = np.ones(points.shape[0]) * np.random.choice([-1, 1])
        movements = movement_ratios * distance_to_move * directions

        new_points = points + np.abs(normals) * movements[:, np.newaxis]
        return new_points

    def list_presets(self):
        """Return a list of ``(index, name)`` tuples for all available presets."""
        return [
            (i, fn.__doc__ or fn.__name__)
            for i, fn in enumerate(self.preset_factory.presets)
        ]


# ------------------------------------------------------------------
# Module-level helper functions (extracted from Dataset methods)
# ------------------------------------------------------------------

def _kernel(t, kind="cosine", q=4.0, sigma=0.35):
    """Falloff kernel – *t* is normalised distance; returns values in [0, 1]."""
    t = np.clip(t, 0.0, None)
    if kind == "cosine":
        x = np.clip(t, 0.0, 1.0)
        return 0.5 * (1 + np.cos(np.pi * x))
    if kind == "gaussian":
        return np.exp(-(t ** 2) / (2 * (sigma ** 2)))
    if kind == "poly":
        return np.clip(1.0 - t ** q, 0.0, 1.0)
    if kind == "hard":
        return (t < 1.0).astype(np.float32)
    raise ValueError(f"Unknown kernel: {kind}")


def _local_frame(points, center, k=64):
    """PCA frame around *center*: columns ≈ (tangent_u, tangent_v, normal)."""
    d = np.linalg.norm(points - center, axis=1)
    idx = np.argsort(d)[:k]
    Q = points[idx] - points[idx].mean(0)
    C = Q.T @ Q / max(len(idx) - 1, 1)
    w, V = np.linalg.eigh(C)
    V = V[:, np.argsort(w)[::-1]]
    return V  # shape (3, 3)


def _one_side_gate(P, center, n_hat, offset=0.0, sharpness=30.0, soft=True):
    """Global half-space gate: keep points with ``(P-center)·n_hat - offset > 0``."""
    n_hat = np.asarray(n_hat, dtype=np.float32)
    n_hat = n_hat / (np.linalg.norm(n_hat) + 1e-8)
    s = (P - center) @ n_hat - offset
    if soft:
        return 1.0 / (1.0 + np.exp(-sharpness * s))
    else:
        return (s > 0.0).astype(np.float32)


def _normal_alignment_gate(point_normals, push_dir, cos_thresh=0.0,
                           sharpness=30.0, soft=True):
    """Keep points whose normals align with *push_dir* (front-facing)."""
    push_dir = np.asarray(push_dir, dtype=np.float32)
    push_dir = push_dir / (np.linalg.norm(push_dir) + 1e-8)
    cosang = point_normals @ push_dir
    if soft:
        return 1.0 / (1.0 + np.exp(-sharpness * (cosang - cos_thresh)))
    else:
        return (cosang >= cos_thresh).astype(np.float32)
