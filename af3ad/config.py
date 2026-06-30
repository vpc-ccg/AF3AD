"""Configuration dataclass for pseudo anomaly synthesis.

This module defines ``SmartAnomaly_Cfg``, the central configuration object that
controls the shape, direction, magnitude, and gating behaviour of each
synthesised anomaly. Every preset in :mod:`af3ad.presets` ultimately returns an
instance of this class.
"""

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class SmartAnomaly_Cfg:
    """Full specification of a single pseudo-anomaly deformation.

    Parameters
    ----------
    R : float or None
        Support radius.  When *None* the synthesiser will default to
        ``0.2 * object_diameter``.
    B : float
        Displacement magnitude (the "beta" parameter).
    alpha : int or None
        +1 for a bulge, -1 for a cavity/dent.  *None* means random with
        probability ``p_bulge``.
    p_bulge : float
        Probability that *alpha* is +1 when ``alpha is None``.
    kernel : str
        Falloff kernel type.  One of ``"cosine"``, ``"gaussian"``,
        ``"poly"``, ``"hard"``.
    q : float
        Exponent for the polynomial kernel.
    sigma : float
        Width (as fraction of *R*) for the Gaussian kernel.
    radii : tuple of float
        Anisotropic ellipsoid radii ``(r_u, r_v, r_n)`` along the local
        PCA frame axes.
    dir_mode : str
        Displacement direction mode.  One of ``"normal_point"``,
        ``"normal_mean"``, ``"tangent_u"``, ``"tangent_v"``.
    one_sided : bool
        Whether to gate the deformation to one side of a plane.
    gate_mode : str
        Gating strategy: ``"global"`` or ``"normals"``.
    n_global : tuple of float
        Global outward direction used when ``gate_mode="global"``.
    gate_soft : bool
        Use a soft (logistic) gate instead of a hard step.
    gate_sharpness : float
        Steepness of the logistic gate.
    gate_offset : float
        Shift the gating plane along the normal.
    carve_strength : float
        Probability of removing centre points (holes), in ``[0, 1]``.
    smooth_steps : int
        Number of Laplacian smoothing passes inside the support region.
    smooth_lambda : float
        Laplacian smoothing weight.
    seed : int or None
        Optional random seed for reproducibility.
    """

    # size & strength
    R: Optional[float] = None
    B: float = 0.08
    alpha: Optional[int] = None
    p_bulge: float = 0.5

    # falloff kernel
    kernel: str = "cosine"
    q: float = 2.0
    sigma: float = 0.35

    # anisotropy (ellipsoid radii along local frame axes)
    radii: tuple = (1.0, 1.0, 1.0)

    # displacement direction
    dir_mode: str = "normal_point"

    # gating
    one_sided: bool = True
    gate_mode: str = "normals"
    n_global: tuple = (0.0, 0.0, 1.0)
    gate_soft: bool = True
    gate_sharpness: float = 30.0
    gate_offset: float = 0.0

    # extras
    carve_strength: float = 0.0
    smooth_steps: int = 0
    smooth_lambda: float = 0.15
    seed: Optional[int] = None

    def to_dict(self):
        """Return a plain dictionary of the configuration."""
        return asdict(self)
