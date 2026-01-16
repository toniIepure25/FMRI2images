#!/bin/bash

# Quick fix for package installation issue

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=========================================="
echo "Fixing Package Installation"
echo "=========================================="
echo ""

cd ~/Bachelor_V2

echo -e "${BLUE}[1]${NC} Pulling latest fix from git..."
git pull origin probabilistic-distribution
echo -e "${GREEN}    ✓${NC} Code updated"
echo ""

echo -e "${BLUE}[2]${NC} Activating virtual environment..."
source /bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/venv/bin/activate
echo -e "${GREEN}    ✓${NC} Environment activated"
echo ""

echo -e "${BLUE}[3]${NC} Reinstalling package..."
pip install -e . --force-reinstall --no-deps
echo -e "${GREEN}    ✓${NC} Package reinstalled"
echo ""

echo -e "${BLUE}[4]${NC} Verifying installation..."
python -c "from fmri2img.data.torch_dataset import NSDIterableDataset; print('    ✓ Import successful!')"
python -c "from fmri2img.io.nsd_layout import NSDLayout; print('    ✓ All modules working!')"
echo ""

echo "=========================================="
echo -e "${GREEN}✅ Package Fixed!${NC}"
echo "=========================================="
echo ""
echo "Now run your smoke test:"
echo "  python -m src.fmri2img.training.train_smoke --subject subj01 --session 1 --limit 8"
echo ""
