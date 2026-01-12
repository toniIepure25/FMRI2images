#!/bin/bash

# Emergency fix: Reset server repo to match remote exactly

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "=========================================="
echo "Emergency Repo Fix"
echo "=========================================="
echo ""

cd ~/Bachelor_V2

echo -e "${YELLOW}[WARNING]${NC} Your repo is missing files!"
echo "This will reset your repo to match the remote exactly."
echo ""
read -p "Press ENTER to continue or Ctrl+C to cancel..."
echo ""

echo -e "${BLUE}[1]${NC} Fetching latest from remote..."
git fetch origin probabilistic-distribution
echo ""

echo -e "${BLUE}[2]${NC} Resetting to match remote (this will discard any local changes)..."
git reset --hard origin/probabilistic-distribution
echo ""

echo -e "${BLUE}[3]${NC} Checking file structure..."
if [ -f "src/fmri2img/data/torch_dataset.py" ]; then
    echo -e "${GREEN}    ✓${NC} torch_dataset.py found!"
else
    echo -e "${RED}    ✗${NC} torch_dataset.py STILL missing!"
    echo ""
    echo "Trying alternative: Re-clone the repository"
    cd ..
    mv Bachelor_V2 Bachelor_V2_backup_$(date +%s)
    git clone -b probabilistic-distribution https://github.com/toniIepure25/FMRI2images.git Bachelor_V2
    cd Bachelor_V2
fi
echo ""

echo -e "${BLUE}[4]${NC} Installing package..."
source /bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/venv/bin/activate
pip uninstall -y fmri2img 2>/dev/null || true
pip install -e . --no-build-isolation
echo ""

echo -e "${BLUE}[5]${NC} Testing..."
python test_real_data.py
echo ""

echo "=========================================="
echo -e "${GREEN}✅ Repository Fixed!${NC}"
echo "=========================================="
