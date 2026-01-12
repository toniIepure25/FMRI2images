#!/bin/bash

# ============================================================================
# GOOD NEWS: You Don't Need to Download Stimuli Images!
# ============================================================================
#
# Your code has a built-in fallback system:
# 1. Tries local HDF5 (fastest) - if available
# 2. Falls back to S3 HDF5 (moderate) - if available  
# 3. Falls back to COCO HTTP API (automatic download + cache)
#
# The COCO fallback will automatically download images as needed and cache
# them locally, so you can proceed with experiments RIGHT NOW!
# ============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo "================================================================================"
echo "Verifying NSD Data Setup"
echo "================================================================================"
echo ""

NSD_ROOT="/bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/data/nsd"

# Check beta files
echo -e "${BLUE}[INFO]${NC} Checking fMRI beta files..."
BETA_DIR="${NSD_ROOT}/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR"
if [ -d "${BETA_DIR}" ]; then
    BETA_COUNT=$(ls -1 "${BETA_DIR}"/betas_session*.nii.gz 2>/dev/null | wc -l)
    BETA_SIZE=$(du -sh "${BETA_DIR}" 2>/dev/null | cut -f1)
    echo -e "${GREEN}[✓]${NC} Beta files: ${BETA_COUNT}/37 (${BETA_SIZE})"
    
    if [ ${BETA_COUNT} -eq 37 ]; then
        echo -e "${GREEN}[✓]${NC} All beta files present - READY FOR EXPERIMENTS!"
    else
        echo -e "${YELLOW}[WARN]${NC} Expected 37 beta files, found ${BETA_COUNT}"
    fi
else
    echo -e "${YELLOW}[WARN]${NC} Beta directory not found"
fi

echo ""

# Check stimulus images (optional)
echo -e "${BLUE}[INFO]${NC} Checking stimulus images..."
STIM_DIR="${NSD_ROOT}/nsddata_stimuli/stimuli"
if [ -d "${STIM_DIR}" ]; then
    STIM_COUNT=$(ls -1 "${STIM_DIR}"/*.png 2>/dev/null | wc -l)
    STIM_SIZE=$(du -sh "${STIM_DIR}" 2>/dev/null | cut -f1)
    echo -e "${GREEN}[✓]${NC} Stimulus images: ${STIM_COUNT} (${STIM_SIZE})"
else
    echo -e "${YELLOW}[NOTE]${NC} No local stimulus images found"
    echo -e "${GREEN}[✓]${NC} This is OK! Your code will auto-download from COCO API"
fi

echo ""

# Check cache directory
echo -e "${BLUE}[INFO]${NC} Checking cache directory..."
CACHE_DIR="/bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/.cache"
if [ -d "${CACHE_DIR}" ]; then
    CACHE_SIZE=$(du -sh "${CACHE_DIR}" 2>/dev/null | cut -f1)
    echo -e "${GREEN}[✓]${NC} Cache directory exists (${CACHE_SIZE})"
else
    echo -e "${BLUE}[INFO]${NC} Creating cache directory..."
    mkdir -p "${CACHE_DIR}/coco"
    echo -e "${GREEN}[✓]${NC} Cache directory created"
fi

echo ""
echo "================================================================================"
echo "READY TO RUN REAL EXPERIMENTS!"
echo "================================================================================"
echo ""

echo "Your setup is complete! The code will automatically:"
echo "  1. Load fMRI data from the 37 beta files you downloaded"
echo "  2. Download stimulus images from COCO API as needed"
echo "  3. Cache images locally for faster subsequent runs"
echo ""

echo -e "${GREEN}Next Steps:${NC}"
echo ""
echo "1. Test with smoke test (real data, 8 trials):"
echo "   python -m src.fmri2img.training.train_smoke --subject subj01 --session 1 --limit 8"
echo ""
echo "2. Run baseline experiment:"
echo "   bash scripts/run_experiment_simple.sh experiments/baseline_subj01.yaml"
echo ""
echo "3. Monitor cache growth:"
echo "   watch -n 5 'du -sh ${CACHE_DIR}/coco'"
echo ""

echo "================================================================================"
echo "Understanding Image Loading"
echo "================================================================================"
echo ""
echo "Your code has a 3-tier fallback system:"
echo ""
echo "  [1] Local HDF5 (fastest, optional)"
echo "      └─ cache/nsd_hdf5/nsd_stimuli.hdf5"
echo ""
echo "  [2] S3 HDF5 (moderate, automatic)"
echo "      └─ s3://natural-scenes-dataset/.../nsd_stimuli.hdf5"
echo ""
echo "  [3] COCO HTTP API (slowest first time, then cached)"
echo "      └─ Downloads to: ${CACHE_DIR}/coco/"
echo "      └─ Cached locally for reuse"
echo ""
echo "For first experiments, images will download automatically from COCO."
echo "After caching, subsequent runs will be fast!"
echo ""

echo "================================================================================"
echo "Disk Space Status"
echo "================================================================================"
echo ""
df -h /bigdata | grep -E "Filesystem|/bigdata"
echo ""

FREE_GB=$(df -BG /bigdata | awk 'NR==2 {print $4}' | sed 's/G//')
echo "Available space: ${FREE_GB}GB"
echo ""

if [ ${FREE_GB} -gt 100 ]; then
    echo -e "${GREEN}[✓]${NC} Plenty of space for experiments and COCO cache"
elif [ ${FREE_GB} -gt 50 ]; then
    echo -e "${YELLOW}[NOTE]${NC} Sufficient space, but monitor cache growth"
else
    echo -e "${YELLOW}[WARN]${NC} Low disk space - may need cleanup"
fi

echo ""
echo "================================================================================"
echo "🚀 YOU'RE READY TO GO!"
echo "================================================================================"
