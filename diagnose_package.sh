#!/bin/bash

# Comprehensive diagnostic for package installation issue

echo "=========================================="
echo "Package Installation Diagnostic"
echo "=========================================="
echo ""

cd ~/Bachelor_V2

echo "[1] Checking git status..."
git status
echo ""

echo "[2] Checking current pyproject.toml content..."
grep -A 5 "tool.setuptools" pyproject.toml || echo "❌ [tool.setuptools] NOT FOUND - Need to pull latest code!"
echo ""

echo "[3] Checking what pip installed..."
pip show fmri2img
echo ""

echo "[4] Finding where fmri2img was installed..."
python -c "import sys; import fmri2img; print(f'fmri2img location: {fmri2img.__file__}')" 2>&1 || echo "❌ Cannot import fmri2img base module"
echo ""

echo "[5] Checking src directory structure..."
ls -la src/
echo ""
ls -la src/fmri2img/ | head -10
echo ""

echo "[6] Trying direct import with path manipulation..."
python << 'PYEOF'
import sys
sys.path.insert(0, '/bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/Bachelor_V2/src')
try:
    from fmri2img.data.torch_dataset import NSDIterableDataset
    print("✓ Direct import works! Package installation is the issue.")
except Exception as e:
    print(f"❌ Direct import also fails: {e}")
PYEOF

echo ""
echo "=========================================="
echo "Solution:"
echo "=========================================="
echo ""
echo "Run these commands:"
echo "  1. git pull origin probabilistic-distribution"
echo "  2. pip install -e . --force-reinstall --no-deps"
echo "  3. If still fails, use: python -m src.fmri2img.training.train_smoke (absolute import)"
echo ""
