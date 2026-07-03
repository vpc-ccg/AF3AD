# AF3AD Integrations

This directory contains detector-specific reference integrations. The reusable
AF3AD synthesis code lives in the top-level `af3ad` package, while each
integration keeps the training code and heavier detector dependencies.

## Install The Integration Directory

The integration modules are imported from the AF3AD repository tree. There is no
separate package installer for `integrations`; make the repository root
importable before running a launcher:

```bash
cd /path/to/AF3AD
export AF3AD_ROOT="$PWD"
export PYTHONPATH="$AF3AD_ROOT:$PYTHONPATH"
```

Quick PO3AD import check after installing the PO3AD environment:

```bash
python - <<'PY'
from af3ad import PseudoAnomalySynthesizer
import integrations.po3ad

print("AF3AD PO3AD integration is importable")
PY
```

## Environment Guides

- [po3ad/](po3ad/README.md): PO3AD-style online training. This path needs
  Python 3.8, PyTorch 1.9, CUDA-compatible packages, and MinkowskiEngine built
  with OpenBLAS for legacy reproduction. It also includes a modern PyTorch 2.x
  + CUDA 12.x venv for testing checkpoints on newer GPUs.

Run the root launchers from the AF3AD repository root after activating the
matching detector environment:

```bash
python scripts/train_po3ad_integration.py --dataset Real3D --dataset-base-dir data/Real3D --category airplane
```
