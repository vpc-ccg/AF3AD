# Pseudo Anomaly Synthesis Tool

> **Contribution** – This directory packages the **preset-based pseudo anomaly synthesis** method designed for the AF3AD project as a standalone, reusable tool.

## Overview

The core idea behind AF3AD's training pipeline is to deform normal 3-D point clouds with controlled, physics-inspired perturbations ("pseudo anomalies") so that a discriminator can learn to detect real defects.  Rather than applying a single random deformation, we define **11 hand-crafted preset types** that cover a wide spectrum of realistic surface defects:

| # | Preset | Key characteristics |
|---|--------|---------------------|
| 0 | **Basic Bulge** | Isotropic outward bump (`alpha=+1`). |
| 1 | **Basic Dent** | Isotropic inward cavity (`alpha=-1`). |
| 2 | **Ridge** | Anisotropic bulge elongated along u (Gaussian kernel). |
| 3 | **Trench** | Anisotropic dent elongated along u (cosine kernel). |
| 4 | **Elliptic Patch / Flat Spot** | Pressed/flattened region with per-point normals. |
| 5 | **Skewed Impact Crater** | Oblique one-sided dent with global gating. |
| 6 | **Shear U** | Tangential slip along the first tangent direction. |
| 7 | **Shear V** | Tangential slip along the second tangent direction. |
| 8 | **Double-Sided Ripple** | Cosine ripple with random ±1 sign. |
| 9 | **Micro Dimple Field** | Tiny dent (apply at many random centres for corrosion/pitting). |
| 10 | **Directional Drag / Stretch** | Anisotropic gated deformation mimicking plastic drag. |

Each preset returns a `SmartAnomaly_Cfg` dataclass that fully specifies:

* **Support radius** (`R`) and **magnitude** (`B`) sampled from configurable Beta distributions.
* **Falloff kernel** – cosine, Gaussian, polynomial, or hard.
* **Anisotropic ellipsoid radii** for non-spherical influence regions.
* **Displacement direction** – per-point normal, mean normal, or tangent.
* **One-sided gating** – global half-space or normal-alignment gate.

## Installation

No extra installation is needed beyond NumPy (which is already a dependency of AF3AD):

```bash
pip install numpy
```

## Usage

### Basic example

```python
import numpy as np
from pseudo_anomaly_synthesis import PseudoAnomalySynthesizer

# Create the synthesizer with default parameter ranges
synth = PseudoAnomalySynthesizer()

# Generate a sample point cloud (replace with your real data)
rng = np.random.default_rng(42)
points  = rng.standard_normal((2048, 3)).astype(np.float32)
normals = points / (np.linalg.norm(points, axis=1, keepdims=True) + 1e-8)
center  = points[rng.integers(len(points))]

# Pick a preset and synthesize an anomaly
cfg = synth.preset_factory.presets[0]()   # Type 0: Basic Bulge
deformed_points = synth.generate(points, normals, center, cfg)
```

### List available presets

```python
for idx, name in synth.list_presets():
    print(f"  [{idx}] {name}")
```

### Custom parameter ranges

```python
class MyArgs:
    R_low_bound = 0.03
    R_up_bound  = 0.20
    R_alpha     = 2.0
    R_beta      = 5.0
    B_low_bound = 0.01
    B_up_bound  = 0.10
    B_alpha     = 2.0
    B_beta      = 5.0

synth = PseudoAnomalySynthesizer(args=MyArgs())
```

### Using the original (simple) synthesizer

```python
deformed = synth.generate_original(points, normals, center, distance_to_move=0.08)
```

## Architecture

```
pseudo_anomaly_synthesis/
├── __init__.py      # Public API exports
├── config.py        # SmartAnomaly_Cfg dataclass
├── presets.py       # AnomalyPreset factory (11 preset types)
├── synthesizer.py   # PseudoAnomalySynthesizer + helper functions
└── README.md        # This file
```

## Integration with AF3AD

In the full AF3AD pipeline an **RL policy network** selects which preset index to apply at each training step.  This tool can be used independently of the RL loop for:

* **Data augmentation** – generate training data for any 3-D anomaly detector.
* **Ablation studies** – evaluate subsets of presets in isolation.
* **Visualisation** – inspect what each preset looks like on your point clouds.

## License

This tool is part of the AF3AD project and is distributed under the same license.