#!/usr/bin/env bash
# Fluorosis DL Project — Server Setup
# Target: RTX PRO 6000 Blackwell 96GB, CUDA 13.0, PyTorch 2.13.0
#
# Usage: bash setup.sh [--cpu]

set -euo pipefail
DEVICE="${1:---gpu}"

echo "=== Fluorosis DL Diagnosis — Environment Setup ==="
echo "Target: RTX PRO 6000 Blackwell 96GB / CUDA 13.0"
echo ""

# --- Python version check ---
PYVER=$(python3 --version 2>&1 | awk '{print $2}')
echo "[1/5] Python ${PYVER}"
REQUIRED="3.10"
if python3 -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)"; then
    echo "  OK: Python >= 3.10"
else
    echo "  ERROR: Python 3.10+ required (current: ${PYVER})"
    exit 1
fi

# --- Create virtual environment ---
echo "[2/5] Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "  Created .venv/"
else
    echo "  .venv/ already exists, skipping"
fi
source .venv/bin/activate

# --- Upgrade pip ---
pip install --upgrade pip -q

# --- PyTorch with CUDA 13.0 ---
echo "[3/5] Installing PyTorch..."
if [ "$DEVICE" = "--cpu" ]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
else
    # Nightly build for CUDA 13.0 (Blackwell architecture)
    pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130
fi

# Verify CUDA
python3 -c "
import torch
print(f'  PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.0f} GB')
"

# --- Core dependencies ---
echo "[4/5] Installing dependencies..."
pip install transformers scikit-learn scipy numpy pandas Pillow openpyxl -q
pip install matplotlib seaborn tqdm -q

# --- Verify ---
echo "[5/5] Verifying installation..."
python3 -c "
import torch, torchvision, transformers, sklearn, numpy, pandas, PIL
print('  All core packages imported successfully.')
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Quick test:"
echo "  source .venv/bin/activate"
echo "  cd 06_Implementation"
echo "  python train.py --task df --data_root .. --epochs 2 --mode edl"
