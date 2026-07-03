# PO3AD AF3AD Integration

This directory contains the PO3AD-style AF3AD integration. Use the legacy
environment for reproducing the original PO3AD-based pipeline, or the modern
CUDA 12 environment for testing released AF3AD PO3AD checkpoints on newer GPUs.

The dependency stack is intentionally pinned because PO3AD uses
MinkowskiEngine and older CUDA/PyTorch APIs:

- Python 3.8
- PyTorch 1.9
- **CUDA 11.x-compatible PyTorch and CUDA Toolkit**
- MinkowskiEngine built from source with OpenBLAS
- Open3D, SciPy, scikit-learn, and the PO3AD utility dependencies

Important compatibility note: official PyTorch 1.9.0 pip wheels are available
for CUDA 11.1 and CUDA 10.2, but not CUDA 11.8. The default instructions below
therefore use `torch==1.9.0+cu111`. If you must use CUDA 11.8 with PyTorch 1.9,
use a PyTorch 1.9 build compiled for CUDA 11.8 and compile MinkowskiEngine with
the same CUDA Toolkit.

References:

- [PyTorch local install selector](https://pytorch.org/get-started/locally/)
- [PyTorch previous-version wheel matrix](https://pytorch.org/get-started/previous-versions/)
- [MinkowskiEngine install notes](https://github.com/NVIDIA/MinkowskiEngine#installation)
- [MinkowskiEngine v0.5.4 release](https://github.com/NVIDIA/MinkowskiEngine/releases/tag/v0.5.4)

## Legacy Environment: PyTorch 1.9 + CUDA 11.x

Use this path when you need the closest match to the original PO3AD-era
environment.

### 1. Install System Packages

Install Python 3.8, legacy distutils, build tools, Python headers, and OpenBLAS
development files. Package names can vary by Linux distribution.

On Ubuntu versions where Python 3.8 is not available from the default package
repositories, install it through the Deadsnakes PPA:

```bash
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.8 python3.8-dev python3.8-distutils python3.8-venv

python3.8 --version
```

Then install the build dependencies:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  git \
  libopenblas-dev \
  ninja-build
```

**Install a CUDA 11.x Toolkit that matches the PyTorch build you will use.** For
the official `torch==1.9.0+cu111` wheel, use a CUDA 11.1 Toolkit for compiling
MinkowskiEngine. If using a custom PyTorch 1.9 CUDA 11.8 build, use CUDA 11.8
for compiling MinkowskiEngine.

### 2. Create A Python 3.8 venv

Create the environment with `venv`:

```bash
python3.8 -m venv ~/.venvs/af3ad-po3ad-py38
source ~/.venvs/af3ad-po3ad-py38/bin/activate
```

Install packaging tools. Keep `setuptools` below 60 because MinkowskiEngine
v0.5.4 builds through the deprecated `numpy.distutils` path used by older
Python/CUDA stacks.

```bash
python -m pip install --upgrade pip
python -m pip install "setuptools==59.5.0" wheel
```

### 3. Install Python Dependencies

Install NumPy, SciPy, and scikit-learn inside the venv:

```bash
python -m pip install ninja
python -m pip install numpy==1.23.5 scipy==1.10.1 scikit-learn==1.2.2
```

Install the remaining PO3AD dependencies:

```bash
python -m pip install \
  matplotlib \
  open3d==0.16.0 \
  plotly \
  psutil \
  pympler \
  pyyaml \
  tensorboardX \
  tqdm
```

### 4. Install PyTorch 1.9

Default path, using the official CUDA 11.1 PyTorch 1.9 wheel:

```bash
python -m pip install \
  torch==1.9.0+cu111 \
  torchvision==0.10.0+cu111 \
  torchaudio==0.9.0 \
  -f https://download.pytorch.org/whl/torch_stable.html
```

Check the PyTorch runtime:

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

### 5. Build MinkowskiEngine With OpenBLAS

MinkowskiEngine should be compiled with a CUDA Toolkit that matches
`torch.version.cuda`.

For the official `torch==1.9.0+cu111` wheel:

```bash
export CUDA_HOME=/usr/local/cuda-11.1
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

For a custom PyTorch 1.9 CUDA 11.8 build:

```bash
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

Verify that PyTorch and `nvcc` agree before compiling:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
PY

nvcc --version
```

Build MinkowskiEngine:

```bash
mkdir -p ~/src
cd ~/src

git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
git checkout v0.5.4

export CC=gcc
export CXX=g++

python setup.py install --force_cuda --blas=openblas
```

If OpenBLAS headers are installed in a distribution-specific include directory,
pass the include and library paths explicitly:

```bash
OPENBLAS_INCLUDE_DIR="$(dirname "$(dpkg -L libopenblas-dev | grep '/cblas.h$' | head -n 1)")"
OPENBLAS_LIBRARY_DIR="$(dirname "$(dpkg -L libopenblas-dev | grep '/libopenblas.so$' | head -n 1)")"

python setup.py install \
  --force_cuda \
  --blas=openblas \
  --blas_include_dirs="$OPENBLAS_INCLUDE_DIR" \
  --blas_library_dirs="$OPENBLAS_LIBRARY_DIR"
```

If the build reports a CUDA mismatch, fix `CUDA_HOME`, `PATH`, and
`LD_LIBRARY_PATH` so `nvcc --version` matches `torch.version.cuda`. If the build
cannot find OpenBLAS, confirm that `libopenblas-dev` is installed.

If the build fails with `fatal error: Python.h: No such file or directory` or
`fatal error: cblas.h: No such file or directory`, install the missing system
headers and rerun the explicit OpenBLAS build command above:

```bash
sudo apt-get install -y python3.8-dev libopenblas-dev
```

If the build fails with `ModuleNotFoundError: No module named
'distutils.msvccompiler'`, install `python3.8-distutils` on the system and pin
setuptools inside the venv:

```bash
sudo apt-get install -y python3.8-distutils
python -m pip install --force-reinstall "setuptools==59.5.0"
```

Then rerun:

```bash
python setup.py install --force_cuda --blas=openblas
```

If the build reaches the final `pybind/minkowski.cu` step and fails at
`pybind/extern.hpp` with an error around `.def(py::self == py::self)`, patch
that binding to an explicit equality lambda:

```bash
python - <<'PY'
from pathlib import Path

path = Path("pybind/extern.hpp")
text = path.read_text()
old = "      .def(py::self == py::self);"
new = """      .def("__eq__",
           [](const minkowski::CoordinateMapKey &a,
              const minkowski::CoordinateMapKey &b) { return a == b; });"""
path.write_text(text.replace(old, new))
PY
```

Then rebuild from a clean build directory:

```bash
rm -rf build MinkowskiEngine.egg-info
MAX_JOBS=1 python setup.py install \
  --force_cuda \
  --blas=openblas \
  --blas_include_dirs="$OPENBLAS_INCLUDE_DIR" \
  --blas_library_dirs="$OPENBLAS_LIBRARY_DIR"
```

### 6. Make AF3AD Importable

From the AF3AD repository root:

```bash
cd /path/to/AF3AD
export AF3AD_ROOT="$PWD"
export PYTHONPATH="$AF3AD_ROOT:$PYTHONPATH"
```

Smoke test:

```bash
python - <<'PY'
from af3ad import PseudoAnomalySynthesizer
import MinkowskiEngine as ME
import torch

print("AF3AD import: ok")
print("MinkowskiEngine import: ok")
print("torch:", torch.__version__, "cuda runtime:", torch.version.cuda)
PY
```

## Modern Test Environment: PyTorch 2.x + CUDA 12.x

Use this path for checkpoint testing on newer GPUs that are not supported by
PyTorch 1.9, such as Blackwell/RTX 50-series GPUs. This is not the exact legacy
training environment, so validate checkpoint metrics before reporting numbers.

The commands below use Python 3.11, PyTorch CUDA 12.8 wheels, OpenBLAS, and a
CUDA-12-compatible MinkowskiEngine fork.

### 1. Install System Packages

Install Python 3.11, build tools, OpenBLAS, GCC/G++, and a CUDA Toolkit that
matches the PyTorch CUDA wheel. For RTX 50-series/Blackwell, use CUDA 12.8 or
newer.

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  gcc-12 \
  g++-12 \
  git \
  libopenblas-dev \
  ninja-build \
  python3.11 \
  python3.11-dev \
  python3.11-venv
```

### 2. Create A Python 3.11 venv

```bash
python3.11 -m venv ~/.venvs/af3ad-po3ad-py311
source ~/.venvs/af3ad-po3ad-py311/bin/activate

python -m pip install --upgrade pip setuptools wheel
```

### 3. Install PyTorch With CUDA 12.8

For RTX 50-series/Blackwell GPUs, use a CUDA 12.8 PyTorch wheel or newer:

```bash
python -m pip install \
  torch==2.9.1 \
  torchvision==0.24.1 \
  torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu128
```

Verify the runtime:

```bash
python - <<'PY'
import torch, sys

print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))
PY
```

### 4. Install Python Dependencies

```bash
python -m pip install \
  "numpy==1.26.4" \
  scipy \
  scikit-learn \
  matplotlib \
  open3d \
  plotly \
  psutil \
  pympler \
  pyyaml \
  tensorboardX \
  tqdm \
  ninja
```

### 5. Build A CUDA-12-Compatible MinkowskiEngine

Set the CUDA, compiler, BLAS, architecture, and job-count environment.

For RTX 50-series/Blackwell:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_ROOT="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export CUDAHOSTCXX="$CXX"

export TORCH_CUDA_ARCH_LIST="12.0"
export CMAKE_BUILD_PARALLEL_LEVEL=1
export MAX_JOBS=1
```

For H100/Hopper, use:

```bash
export TORCH_CUDA_ARCH_LIST="9.0"
```

Find the OpenBLAS include and library directories:

```bash
OPENBLAS_INCLUDE_DIR="$(dirname "$(dpkg -L libopenblas-dev | grep '/cblas.h$' | head -n 1)")"
OPENBLAS_LIBRARY_DIR="$(dirname "$(dpkg -L libopenblas-dev | grep '/libopenblas.so$' | head -n 1)")"
```

Clone and build the CUDA-12-compatible fork:

```bash
mkdir -p ~/src
cd ~/src

git clone https://github.com/CiSong10/MinkowskiEngine.git MinkowskiEngine-cuda12
cd MinkowskiEngine-cuda12
git checkout cuda12-installation

rm -rf build MinkowskiEngine.egg-info
python setup.py clean --all

MAX_JOBS=1 python setup.py install \
  --force_cuda \
  --blas=openblas \
  --blas_include_dirs="$OPENBLAS_INCLUDE_DIR" \
  --blas_library_dirs="$OPENBLAS_LIBRARY_DIR"
```

If CUDA 12.8 fails in `src/coordinate_map_gpu.cuh` with an ambiguous
`std::__to_address` error while assigning `map_type::create(...)` into
`m_map`, patch the `unique_ptr` to `shared_ptr` conversion explicitly:

```bash
python - <<'PY'
from pathlib import Path

path = Path("src/coordinate_map_gpu.cuh")
text = path.read_text()
old = """      m_map = map_type::create(
          compute_hash_table_size(size, m_hashtable_occupancy),
          m_unused_element, m_unused_key, m_hasher, m_equal, m_map_allocator);"""
new = """      auto new_map = map_type::create(
          compute_hash_table_size(size, m_hashtable_occupancy),
          m_unused_element, m_unused_key, m_hasher, m_equal, m_map_allocator);
      auto new_map_deleter = new_map.get_deleter();
      m_map = std::shared_ptr<map_type>(new_map.release(), new_map_deleter);"""
if old not in text:
    raise SystemExit("Expected coordinate_map_gpu.cuh block was not found")
path.write_text(text.replace(old, new))
PY

rm -rf build MinkowskiEngine.egg-info
MAX_JOBS=1 python setup.py install \
  --force_cuda \
  --blas=openblas \
  --blas_include_dirs="$OPENBLAS_INCLUDE_DIR" \
  --blas_library_dirs="$OPENBLAS_LIBRARY_DIR"
```

### 6. Verify MinkowskiEngine

```bash
python - <<'PY'
import torch
import MinkowskiEngine as ME

print("ME version:", ME.__version__)
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

features = torch.rand(4, 3, device="cuda")
coords = torch.tensor(
    [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    dtype=torch.int32,
    device="cuda",
)

st = ME.SparseTensor(features, coordinates=coords)
conv = ME.MinkowskiConvolution(3, 8, kernel_size=3, dimension=3).cuda()
out = conv(st)

print("Output:", out.F.shape, out.F.device)
PY
```

### 7. Make AF3AD Importable

From the AF3AD repository root:

```bash
cd /path/to/AF3AD
source ~/.venvs/af3ad-po3ad-py311/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
```

## Running

Activate the venv you installed and run the PO3AD launcher from the repository
root:

```bash
cd /path/to/AF3AD

# Legacy environment:
# source ~/.venvs/af3ad-po3ad-py38/bin/activate

# Modern environment:
# source ~/.venvs/af3ad-po3ad-py311/bin/activate

export PYTHONPATH="$PWD:$PYTHONPATH"

python scripts/train_po3ad_integration.py \
  --dataset Real3D \
  --dataset-base-dir data/Real3D \
  --category airplane
```

For AnomalyShapeNet:

```bash
python scripts/train_po3ad_integration.py \
  --dataset AnomalyShapeNet \
  --dataset-base-dir data/AnomalyShapeNet/dataset \
  --category ashtray0
```

When released checkpoints are available, use the checkpoint path and matching
architecture flags provided with the checkpoint. The provided launcher can load a
checkpoint with `--resume_checkpoint`:

```bash
python scripts/train_po3ad_integration.py \
  --dataset Real3D \
  --dataset-base-dir data/Real3D \
  --category airplane \
  --resume_checkpoint /path/to/checkpoint.pt
```
