#!/bin/bash

echo "=========================================="
echo "Server File Structure Diagnostic"
echo "=========================================="
echo ""

cd ~/Bachelor_V2

echo "[1] What branch are we on?"
git branch --show-current
echo ""

echo "[2] What files are in src/fmri2img/data/?"
ls -la src/fmri2img/data/ | head -20
echo ""

echo "[3] Does torch_dataset.py exist in the repo?"
find . -name "torch_dataset.py" -type f
echo ""

echo "[4] What's in the git index?"
git ls-files | grep "torch_dataset"
echo ""

echo "[5] Check git status"
git status --short
echo ""

echo "[6] What Python actually sees:"
python << 'PYEOF'
import sys
sys.path.insert(0, '/bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/Bachelor_V2/src')

import os
fmri_data_path = '/bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/Bachelor_V2/src/fmri2img/data'
if os.path.exists(fmri_data_path):
    files = os.listdir(fmri_data_path)
    print(f"Files in {fmri_data_path}:")
    for f in sorted(files)[:20]:
        print(f"  - {f}")
else:
    print(f"Directory doesn't exist: {fmri_data_path}")
PYEOF

echo ""
echo "[7] Maybe files weren't pulled?"
echo "Last commit:"
git log --oneline -1
echo ""
echo "Remote commits:"
git log --oneline origin/probabilistic-distribution -3

echo ""
echo "=========================================="
echo "Solution:"
echo "=========================================="
echo "If torch_dataset.py is missing, the repo might be corrupted."
echo "Try: git reset --hard origin/probabilistic-distribution"
