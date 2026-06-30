"""Preset anomaly configurations for pseudo anomaly synthesis.

This module provides the ``AnomalyPreset`` class, which acts as a factory for
pre-designed anomaly configurations (instances of
:class:`~af3ad.config.SmartAnomaly_Cfg`).

**Contribution** – The 11 preset types defined here were hand-crafted to cover
a wide range of realistic surface defects encountered in 3-D anomaly detection
(bulges, dents, ridges, trenches, shear deformations, ripples, micro-dimple
fields, directional drag, etc.). Detector-specific integrations can select
among these presets during training or pre-generation.
"""

import numpy as np

from .config import SmartAnomaly_Cfg


def _rand_sign(p_plus=0.5):
    """+1 with probability *p_plus*, else -1."""
    return +1 if np.random.rand() < float(p_plus) else -1


class _DefaultArgs:
    """Minimal fallback args object with sensible parameter ranges."""

    R_low_bound = 0.05
    R_up_bound = 0.25
    R_alpha = 2.0
    R_beta = 5.0
    B_low_bound = 0.02
    B_up_bound = 0.15
    B_alpha = 2.0
    B_beta = 5.0


class AnomalyPreset:
    """Factory that generates :class:`SmartAnomaly_Cfg` instances from presets.

    Parameters
    ----------
    args : object, optional
        Configuration object with the following numeric attributes used for
        sampling radius (*R*) and magnitude (*B*) via a Beta distribution:

        - ``R_low_bound``, ``R_up_bound``, ``R_alpha``, ``R_beta``
        - ``B_low_bound``, ``B_up_bound``, ``B_alpha``, ``B_beta``

        Additional optional attributes (retrieved via ``getattr`` with
        defaults): ``gate_offset``, ``gate_sharpness``, ``p_bulge``,
        ``micro_beta_scale``, ``drag_beta_scale``.

        When *None*, sensible defaults are used.

    Attributes
    ----------
    presets : list of callable
        Each entry is a zero-argument method that returns a fresh
        :class:`SmartAnomaly_Cfg`.  The list order matches the preset indices
        used throughout the AF3AD integrations.

    Examples
    --------
    >>> from af3ad.presets import AnomalyPreset
    >>> factory = AnomalyPreset()          # use default parameter ranges
    >>> cfg = factory.presets[0]()          # Type 1 – basic bulge
    >>> cfg.alpha
    1
    >>> cfg = factory.presets[1]()          # Type 2 – basic dent
    >>> cfg.alpha
    -1
    """

    # Index of the micro dimple field preset in the presets list.
    MICRO_DIMPLE_PRESET_IDX = 9

    def __init__(self, args=None):
        self.args = args if args is not None else _DefaultArgs()
        self.presets = [
            self.type_1_basic_bulge,
            self.type_2_basic_dent,
            self.type_3_ridge,
            self.type_4_trench,
            self.type_5_elliptic_patch_flat_spot,
            self.type_6_skewed_impact_crater,
            self.type_7_shear_u,
            self.type_7b_shear_v,
            self.type_8_double_sided_ripple,
            self.type_9_micro_dimple_field_base,
            self.type_10_directional_drag_stretch,
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_R_B(self):
        """Sample radius *R* and magnitude *B* from Beta distributions."""
        R = self.args.R_low_bound + (self.args.R_up_bound - self.args.R_low_bound) * \
            np.random.beta(self.args.R_alpha, self.args.R_beta)
        B = self.args.B_low_bound + (self.args.B_up_bound - self.args.B_low_bound) * \
            np.random.beta(self.args.B_alpha, self.args.B_beta)
        return float(R), float(B)

    def _p(self, name, default):
        return getattr(self.args, name, default)

    # ------------------------------------------------------------------
    # Preset definitions
    # ------------------------------------------------------------------

    def type_1_basic_bulge(self):
        """Type 1 – Basic Local Bulge (isotropic, outward)."""
        R, B = self.get_R_B()
        return SmartAnomaly_Cfg(
            R=R,
            radii=(1.0, 1.0, 1.0),
            kernel="cosine",
            one_sided=True,
            gate_mode="normals",
            dir_mode="normal_mean",
            alpha=+1,
            B=B,
            sigma=0.35,
        )

    def type_2_basic_dent(self):
        """Type 2 – Basic Local Dent (isotropic, inward)."""
        R, B = self.get_R_B()
        return SmartAnomaly_Cfg(
            R=R,
            radii=(1.0, 1.0, 1.0),
            kernel="cosine",
            one_sided=True,
            gate_mode="normals",
            dir_mode="normal_mean",
            alpha=-1,
            B=B,
            sigma=0.35,
        )

    def type_3_ridge(self):
        """Type 3 – Elongated Ridge (anisotropic bulge along u)."""
        R, B = self.get_R_B()
        return SmartAnomaly_Cfg(
            R=R,
            radii=(2.5, 0.7, 0.6),
            kernel="gaussian",
            dir_mode="normal_mean",
            one_sided=True,
            gate_mode="global",
            n_global=np.array([0, 0, 1]),
            alpha=+1,
            B=B,
            sigma=0.4,
        )

    def type_4_trench(self):
        """Type 4 – Elongated Groove / Trench (anisotropic dent along u)."""
        R, B = self.get_R_B()
        return SmartAnomaly_Cfg(
            R=R,
            radii=(3.0, 0.7, 0.5),
            kernel="cosine",
            dir_mode="normal_mean",
            one_sided=True,
            gate_mode="global",
            n_global=np.array([0, 0, 1]),
            alpha=-1,
            B=B,
        )

    def type_5_elliptic_patch_flat_spot(self):
        """Type 5 – Elliptic Patch / Flat Spot (pressed region)."""
        R, B = self.get_R_B()
        return SmartAnomaly_Cfg(
            R=R,
            radii=(1.3, 1.0, 0.5),
            kernel="gaussian",
            dir_mode="normal_point",
            one_sided=True,
            gate_mode="normals",
            alpha=-1,
            B=B,
            sigma=0.5,
        )

    def type_6_skewed_impact_crater(self):
        """Type 6 – Skewed Impact Crater (oblique one-sided dent)."""
        R, B = self.get_R_B()
        gate_offset = self._p("gate_offset", 0.05)
        gate_sharpness = self._p("gate_sharpness", 8.0)
        return SmartAnomaly_Cfg(
            R=R,
            radii=(1.0, 1.0, 0.6),
            kernel="cosine",
            one_sided=True,
            gate_mode="global",
            n_global=np.array([0.2, 0.3, 0.93]),
            gate_offset=gate_offset,
            gate_sharpness=gate_sharpness,
            dir_mode="normal_point",
            alpha=-1,
            B=B,
        )

    def type_7_shear_u(self):
        """Type 7 – Shear / Slip along u (tangential)."""
        R, B = self.get_R_B()
        return SmartAnomaly_Cfg(
            R=R,
            radii=(1.2, 1.2, 0.8),
            kernel="cosine",
            dir_mode="tangent_u",
            one_sided=False,
            alpha=+1,
            B=B,
        )

    def type_7b_shear_v(self):
        """Type 7b – Shear / Slip along v (tangential, opposite)."""
        R, B = self.get_R_B()
        return SmartAnomaly_Cfg(
            R=R,
            radii=(1.2, 1.2, 0.8),
            kernel="cosine",
            dir_mode="tangent_v",
            one_sided=False,
            alpha=-1,
            B=B,
        )

    def type_8_double_sided_ripple(self):
        """Type 8 – Double-Sided Ripple (cosine, alternating sign)."""
        R, B = self.get_R_B()
        p_bulge = self._p("p_bulge", 0.5)
        alpha = _rand_sign(p_bulge)
        return SmartAnomaly_Cfg(
            R=R,
            radii=(1.0, 1.0, 0.8),
            kernel="cosine",
            one_sided=False,
            dir_mode="normal_mean",
            alpha=alpha,
            B=B,
        )

    def type_9_micro_dimple_field_base(self):
        """Type 9 – Micro Dimple Field (single tiny dimple; apply N times)."""
        _, B = self.get_R_B()
        micro_scale = self._p("micro_beta_scale", 0.25)
        B = B * micro_scale
        return SmartAnomaly_Cfg(
            radii=(0.4, 0.4, 0.4),
            kernel="cosine",
            one_sided=True,
            gate_mode="normals",
            dir_mode="normal_mean",
            alpha=-1,
            B=B,
            sigma=0.4,
        )

    def type_10_directional_drag_stretch(self):
        """Type 10 – Directional Drag / Stretch (anisotropic, gated)."""
        R, B = self.get_R_B()
        B_scaled = B * self._p("drag_beta_scale", 0.17)
        return SmartAnomaly_Cfg(
            R=R,
            radii=(2.0, 0.8, 0.6),
            kernel="gaussian",
            dir_mode="tangent_u",
            one_sided=True,
            gate_mode="global",
            n_global=np.array([0, 1, 0]),
            alpha=+1,
            B=B_scaled,
            sigma=0.5,
        )
