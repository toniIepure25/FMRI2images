#!/bin/bash

# Quick fix to download NSD stimulus images
# The HDF5 file doesn't exist in S3, so we'll download individual PNGs

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

NSD_ROOT="/bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/data/nsd"
STIM_DIR="${NSD_ROOT}/nsddata_stimuli/stimuli"
S3_BASE="s3://natural-scenes-dataset"

mkdir -p "${STIM_DIR}"

echo "================================================================================"
echo "NSD Stimulus Images Download (Alternative Method)"
echo "================================================================================"
echo ""
echo "[INFO] Target directory: ${STIM_DIR}"
echo "[INFO] Downloading stimulus PNGs from S3..."
echo ""

# The NSD dataset has stimuli organized as individual PNG files
# nsd00000.png through nsd72999.png (73,000 images)

echo "[INFO] Attempting to sync all PNG files from S3..."
echo "[INFO] This will download ~10GB of PNG images..."
echo ""

# Try syncing all PNG files
aws s3 sync \
    "${S3_BASE}/nsddata_stimuli/stimuli" \
    "${STIM_DIR}" \
    --no-sign-request \
    --exclude "*" \
    --include "nsd*.png"

if [ $? -eq 0 ]; then
    # Count downloaded files
    NUM_FILES=$(ls -1 "${STIM_DIR}"/nsd*.png 2>/dev/null | wc -l)
    TOTAL_SIZE=$(du -sh "${STIM_DIR}" | cut -f1)
    
    echo ""
    echo -e "${GREEN}[✓]${NC} Stimulus download complete!"
    echo -e "${GREEN}[✓]${NC} Downloaded ${NUM_FILES} images"
    echo -e "${GREEN}[✓]${NC} Total size: ${TOTAL_SIZE}"
    echo ""
    
    if [ ${NUM_FILES} -lt 73000 ]; then
        echo -e "${YELLOW}[WARN]${NC} Expected 73,000 images but got ${NUM_FILES}"
        echo -e "${YELLOW}[WARN]${NC} This might be sufficient for your experiments"
    fi
else
    echo ""
    echo -e "${RED}[ERROR]${NC} Failed to download stimulus images via sync"
    echo ""
    echo "Alternative approaches:"
    echo ""
    echo "1. Download a subset (faster, sufficient for testing):"
    echo "   # First 1000 images only"
    echo "   for i in \$(seq -f '%05g' 0 999); do"
    echo "     aws s3 cp ${S3_BASE}/nsddata_stimuli/stimuli/nsd\${i}.png ${STIM_DIR}/nsd\${i}.png --no-sign-request"
    echo "   done"
    echo ""
    echo "2. Check if your repository has cached stimuli:"
    echo "   ls -lh ~/Bachelor_V2/cache/stimuli/"
    echo ""
    echo "3. Use the HuggingFace mirror (if available):"
    echo "   # NSD might be hosted on HuggingFace datasets"
    echo "   python -c \"from datasets import load_dataset; ds = load_dataset('cneuro/nsd')\""
    echo ""
    exit 1
fi

echo ""
echo "================================================================================"
echo "VERIFICATION"
echo "================================================================================"
echo ""

# Verify structure
echo "[INFO] Checking downloaded data..."

if [ -d "${NSD_ROOT}/nsddata_betas/ppdata/subj01" ]; then
    BETA_COUNT=$(ls -1 "${NSD_ROOT}"/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session*.nii.gz 2>/dev/null | wc -l)
    echo -e "${GREEN}[✓]${NC} Beta files: ${BETA_COUNT}/37"
fi

if [ -d "${STIM_DIR}" ]; then
    STIM_COUNT=$(ls -1 "${STIM_DIR}"/nsd*.png 2>/dev/null | wc -l)
    echo -e "${GREEN}[✓]${NC} Stimulus images: ${STIM_COUNT}"
fi

echo ""
echo "================================================================================"
echo "NEXT STEPS"
echo "================================================================================"
echo ""
echo "1. Verify setup:"
echo "   cd ~/Bachelor_V2"
echo "   bash scripts/prepare_data.sh --verify-only"
echo ""
echo "2. Run smoke test with REAL data:"
echo "   python -m src.fmri2img.training.train_smoke --subject subj01 --session 1 --limit 8"
echo ""
echo "3. Run full experiment:"
echo "   bash scripts/run_experiment_simple.sh experiments/baseline_subj01.yaml"
echo ""
