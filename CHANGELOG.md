# Changelog

## Unreleased

- Refactored the repository around a reusable top-level `af3ad` package.
- Moved detector-specific code into `integrations/po3ad` and `integrations/r3dad`.
- Added root-level scripts, example configs, and compatibility wrappers for the
  previous flat layout.


## [v0.0.1] - 2026-03-30

### Initial release
- Added the first working training pipeline for 3D anomaly detection based on the PO3AD-driven discriminator workflow with using AF3AD.
- Added dataset support structure for both AnomalyShapeNet and Real3D under a unified data factory.
- Added configurable training/evaluation settings via the `configs` package.
- Added model components for discriminator and PO3AD integration in `models`.
- Added utility modules for logging, losses, memory reporting, checkpointing, and visualization.
- Added training entrypoint and launch scripts for AnomalyShapeNet and Real3D.

### Notes
- This is the baseline public version of the codebase.

## [Unreleased]

### Planned
- Add at least one additional method (like R3DAD) as an alternative backbone option for the discriminator and/or anomaly detection module.