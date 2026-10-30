#!/usr/bin/env bash
set -euo pipefail

# This script is for a server image that already provides PyTorch + CUDA.
# It does NOT create a new conda environment and does NOT install PyTorch.
# Run it inside the PyTorch environment provided by your server image.

PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
EXPECTED_TORCH_MAJOR_MINOR="2.1"
EXPECTED_CUDA="11.8"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_wheel() {
    local pattern="$1"
    local dir
    for dir in "$SCRIPT_DIR" "$SCRIPT_DIR/.." "$PWD"; do
        local matches=("$dir"/$pattern)
        if [ -e "${matches[0]}" ]; then
            printf '%s\n' "${matches[0]}"
            return 0
        fi
    done
    return 1
}

if ! command -v python >/dev/null 2>&1; then
    echo "ERROR: python was not found. Please run this script inside your server's PyTorch environment."
    exit 1
fi

echo "Using Python: $(command -v python)"

echo "Checking existing PyTorch environment..."
python - <<'PY'
import sys
try:
    import torch
except Exception as e:
    print('ERROR: PyTorch is not available in the current environment.')
    print('Please choose/activate a server image that already has PyTorch + CUDA installed.')
    raise SystemExit(1) from e

print('python:', sys.version.split()[0])
print('torch:', torch.__version__)
print('torch cuda:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('cuda device:', torch.cuda.get_device_name(0))
else:
    print('WARNING: torch.cuda.is_available() is False. Training main.py requires CUDA.')
PY

TORCH_VERSION="$(python - <<'PY'
import torch
print(torch.__version__)
PY
)"
TORCH_CUDA="$(python - <<'PY'
import torch
print(torch.version.cuda or '')
PY
)"

if [[ "${TORCH_VERSION}" != ${EXPECTED_TORCH_MAJOR_MINOR}* ]]; then
    echo "WARNING: local mamba_ssm/causal_conv1d wheels were downloaded for torch ${EXPECTED_TORCH_MAJOR_MINOR}.x."
    echo "Current torch version is ${TORCH_VERSION}. If wheel installation or import fails, use a PyTorch ${EXPECTED_TORCH_MAJOR_MINOR}.x + CUDA ${EXPECTED_CUDA} image."
fi

if [[ "${TORCH_CUDA}" != "${EXPECTED_CUDA}" ]]; then
    echo "WARNING: local wheels were downloaded for CUDA ${EXPECTED_CUDA}."
    echo "Current torch CUDA version is ${TORCH_CUDA:-unknown}. If wheel installation or import fails, use a CUDA ${EXPECTED_CUDA} PyTorch image."
fi

echo "Installing pip tools from Tsinghua PyPI mirror..."
python -m pip install -i "${PIP_INDEX_URL}" --upgrade pip setuptools wheel packaging ninja

echo "Installing project dependencies from Tsinghua PyPI mirror..."
python -m pip install -i "${PIP_INDEX_URL}" \
    numpy==1.26.4 \
    pillow \
    opencv-python-headless \
    scipy \
    scikit-image \
    matplotlib \
    timm==0.4.12 \
    einops

CAUSAL_WHL="$(find_wheel 'causal_conv1d-*.whl' || true)"
MAMBA_WHL="$(find_wheel 'mamba_ssm-*.whl' || true)"

if [ -z "${CAUSAL_WHL}" ]; then
    echo "ERROR: causal_conv1d wheel was not found."
    echo "Please put causal_conv1d-*.whl in either:"
    echo "  1) ${SCRIPT_DIR}"
    echo "  2) ${SCRIPT_DIR}/.."
    echo "  3) your current working directory"
    exit 1
fi

if [ -z "${MAMBA_WHL}" ]; then
    echo "ERROR: mamba_ssm wheel was not found."
    echo "Please put mamba_ssm-*.whl in either:"
    echo "  1) ${SCRIPT_DIR}"
    echo "  2) ${SCRIPT_DIR}/.."
    echo "  3) your current working directory"
    exit 1
fi

echo "Installing local wheel without pulling remote dependencies: ${CAUSAL_WHL}"
python -m pip install --no-deps "${CAUSAL_WHL}"

echo "Installing local wheel without pulling remote dependencies: ${MAMBA_WHL}"
python -m pip install --no-deps "${MAMBA_WHL}"

echo "Running final environment checks..."
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('torch cuda:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('cuda device:', torch.cuda.get_device_name(0))
from mamba_ssm import Mamba
print('mamba_ssm: ok')
import causal_conv1d
print('causal_conv1d: ok')
import cv2, numpy, PIL, scipy, skimage, matplotlib
print('basic dependencies: ok')
PY

echo "Environment is ready."
echo "To train:"
echo "  cd ${SCRIPT_DIR}"
echo "  python main.py"
