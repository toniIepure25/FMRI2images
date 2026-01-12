#!/bin/bash

# Complete fix for package installation issue

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo "=========================================="
echo "Complete Package Installation Fix"
echo "=========================================="
echo ""

cd ~/Bachelor_V2

echo -e "${BLUE}[1]${NC} Pulling latest fixes..."
git pull origin probabilistic-distribution
echo -e "${GREEN}    ✓${NC} Code updated"
echo ""

echo -e "${BLUE}[2]${NC} Activating virtual environment..."
source /bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/venv/bin/activate
echo -e "${GREEN}    ✓${NC} Environment activated"
echo ""

echo -e "${BLUE}[3]${NC} Uninstalling old package..."
pip uninstall -y fmri2img 2>/dev/null || echo "    (no old installation found)"
echo -e "${GREEN}    ✓${NC} Cleaned"
echo ""

echo -e "${BLUE}[4]${NC} Installing with setup.py (more reliable)..."
pip install -e . --no-build-isolation
echo -e "${GREEN}    ✓${NC} Package installed"
echo ""

echo -e "${BLUE}[5]${NC} Verifying installation..."
echo ""

# Test 1: Check if package is findable
echo "  Test 1: Can pip find the package?"
pip show fmri2img | grep "Location:" && echo -e "${GREEN}    ✓${NC} Package found" || echo -e "${RED}    ✗${NC} Package not found"
echo ""

# Test 2: Check if base module imports
echo "  Test 2: Can import base module?"
python -c "import fmri2img; print('    ✓ Base module imports')" 2>&1 || echo -e "${RED}    ✗${NC} Failed"
echo ""

# Test 3: Check specific module
echo "  Test 3: Can import torch_dataset?"
python -c "from fmri2img.data.torch_dataset import NSDIterableDataset; print('    ✓ torch_dataset imports')" 2>&1 || echo -e "${RED}    ✗${NC} Failed"
echo ""

# Test 4: List what's actually in the package
echo "  Test 4: What modules are installed?"
python -c "import fmri2img; import os; pkg_path = os.path.dirname(fmri2img.__file__); print(f'    Package location: {pkg_path}'); import subprocess; subprocess.run(['ls', '-la', pkg_path])" 2>&1 | head -15
echo ""

echo "=========================================="
echo "Running Standalone Test"
echo "=========================================="
echo ""

python test_real_data.py

echo ""
echo "=========================================="
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo "=========================================="
