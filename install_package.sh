#!/bin/bash

# Quick fix: Install the fmri2img package in development mode

set -e

echo "=========================================="
echo "Installing fmri2img Package"
echo "=========================================="
echo ""

cd /bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/Bachelor_V2

# Activate venv (use full path since 'venv' is in different location)
echo "[1] Activating virtual environment..."
source /bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/venv/bin/activate
echo "    ✓ Virtual environment activated"
echo ""

# Install package in editable mode
echo "[2] Installing fmri2img package in development mode..."
pip install -e .
echo "    ✓ Package installed"
echo ""

# Verify installation
echo "[3] Verifying installation..."
python -c "import fmri2img; print(f'    ✓ fmri2img version: {fmri2img.__version__ if hasattr(fmri2img, \"__version__\") else \"installed\"}')"
python -c "from fmri2img.data.torch_dataset import NSDIterableDataset; print('    ✓ NSDIterableDataset imported successfully')"
echo ""

echo "=========================================="
echo "✅ Installation Complete!"
echo "=========================================="
echo ""
echo "Now you can run:"
echo "  python -m src.fmri2img.training.train_smoke --subject subj01 --session 1 --limit 8"
echo ""
