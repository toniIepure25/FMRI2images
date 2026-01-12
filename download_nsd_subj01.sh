#!/usr/bin/env bash
# =============================================================================
# download_nsd_subj01.sh - Download NSD Subject 1 Data (~40GB)
# =============================================================================
# Downloads ONLY the required data for subject 1:
# - fMRI beta files (37 sessions, ~30GB)
# - Stimulus images (73k images, ~10GB)
#
# Usage:
#   bash download_nsd_subj01.sh
#
# This will take 2-4 hours depending on your connection.
# You can run in tmux/screen to keep it running if you disconnect.
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}NSD Subject 01 Data Download${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

# Target directory
NSD_ROOT="/bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/data/nsd"
echo -e "${BLUE}[INFO]${NC} Target directory: ${NSD_ROOT}"

# Create directory structure
echo -e "${BLUE}[INFO]${NC} Creating directory structure..."
mkdir -p "${NSD_ROOT}/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR"
mkdir -p "${NSD_ROOT}/nsddata_stimuli/stimuli"
mkdir -p "${NSD_ROOT}/nsddata/ppdata/subj01/behav"

echo -e "${GREEN}[✓]${NC} Directories created"
echo ""

# Check disk space
AVAILABLE=$(df -BG "${NSD_ROOT}" | tail -1 | awk '{print $4}' | sed 's/G//')
echo -e "${BLUE}[INFO]${NC} Available disk space: ${AVAILABLE}GB"

if [ "$AVAILABLE" -lt 50 ]; then
    echo -e "${RED}[ERROR]${NC} Insufficient disk space! Need at least 50GB free."
    echo -e "${YELLOW}[WARN]${NC} You have ${AVAILABLE}GB available."
    exit 1
fi

echo -e "${GREEN}[✓]${NC} Sufficient disk space available"
echo ""

# =============================================================================
# DOWNLOAD INSTRUCTIONS
# =============================================================================

echo -e "${YELLOW}================================================================================${NC}"
echo -e "${YELLOW}DOWNLOAD INSTRUCTIONS${NC}"
echo -e "${YELLOW}================================================================================${NC}"
echo ""
echo -e "${BLUE}The NSD dataset requires registration and is hosted on AWS S3.${NC}"
echo ""
echo -e "${GREEN}Step 1: Register (if not already done)${NC}"
echo "  → Visit: https://naturalscenesdataset.org/"
echo "  → Click 'Data Access'"
echo "  → Fill in registration form"
echo "  → You'll receive download instructions via email"
echo ""
echo -e "${GREEN}Step 2: Get AWS S3 Access${NC}"
echo "  → The NSD team provides AWS CLI download commands"
echo "  → Or direct download links for each file"
echo ""
echo -e "${GREEN}Step 3: Download using this script${NC}"
echo "  → We'll use AWS CLI (public access, no credentials needed)"
echo ""

# Check if AWS CLI is available
if ! command -v aws &> /dev/null; then
    echo -e "${YELLOW}[WARN]${NC} AWS CLI not found. Installing..."
    pip install awscli
    echo -e "${GREEN}[✓]${NC} AWS CLI installed"
fi

echo ""
echo -e "${YELLOW}================================================================================${NC}"
echo -e "${YELLOW}DOWNLOADING BETA FILES (37 sessions, ~30GB)${NC}"
echo -e "${YELLOW}================================================================================${NC}"
echo ""

# NSD S3 bucket (public access)
S3_BASE="s3://natural-scenes-dataset"

# Download beta files
echo -e "${BLUE}[INFO]${NC} Downloading fMRI beta files for subj01..."
echo -e "${BLUE}[INFO]${NC} This will take 1-2 hours depending on your connection..."
echo ""

BETA_DIR="${NSD_ROOT}/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR"

# Download all 37 sessions
for session in $(seq -f "%02g" 1 37); do
    FILE="betas_session${session}.nii.gz"
    echo -e "${BLUE}[INFO]${NC} Downloading session ${session}/37: ${FILE}"
    
    aws s3 cp \
        "${S3_BASE}/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/${FILE}" \
        "${BETA_DIR}/${FILE}" \
        --no-sign-request
    
    if [ $? -eq 0 ]; then
        SIZE=$(du -h "${BETA_DIR}/${FILE}" | cut -f1)
        echo -e "${GREEN}[✓]${NC} Downloaded ${FILE} (${SIZE})"
    else
        echo -e "${RED}[ERROR]${NC} Failed to download ${FILE}"
        echo -e "${YELLOW}[INFO]${NC} You may need to download manually from:"
        echo "        https://naturalscenesdataset.org/"
        exit 1
    fi
    echo ""
done

echo -e "${GREEN}[✓]${NC} All beta files downloaded!"
echo ""

# =============================================================================
# DOWNLOAD STIMULUS IMAGES
# =============================================================================

echo ""
echo -e "${YELLOW}================================================================================${NC}"
echo -e "${YELLOW}DOWNLOADING STIMULUS IMAGES (~10GB)${NC}"
echo -e "${YELLOW}================================================================================${NC}"
echo ""

echo -e "${BLUE}[INFO]${NC} Downloading NSD stimulus images..."
echo -e "${BLUE}[INFO]${NC} This will take 30-60 minutes..."
echo ""

STIM_DIR="${NSD_ROOT}/nsddata_stimuli/stimuli"

# Download the HDF5 file (contains all 73k images)
echo -e "${BLUE}[INFO]${NC} Downloading nsd_stimuli.hdf5 (all 73,000 images in one file)..."

aws s3 cp \
    "${S3_BASE}/nsddata_stimuli/stimuli/nsd_stimuli.hdf5" \
    "${STIM_DIR}/nsd_stimuli.hdf5" \
    --no-sign-request

if [ $? -eq 0 ]; then
    SIZE=$(du -h "${STIM_DIR}/nsd_stimuli.hdf5" | cut -f1)
    echo -e "${GREEN}[✓]${NC} Downloaded nsd_stimuli.hdf5 (${SIZE})"
else
    echo -e "${RED}[ERROR]${NC} Failed to download stimulus HDF5 file"
    echo -e "${YELLOW}[INFO]${NC} Alternatively, download individual PNG files:"
    echo ""
    echo "  # Download all PNG files (slower but more reliable)"
    echo "  aws s3 sync \\"
    echo "    ${S3_BASE}/nsddata_stimuli/stimuli/ \\"
    echo "    ${STIM_DIR}/ \\"
    echo "    --no-sign-request \\"
    echo "    --exclude '*' \\"
    echo "    --include '*.png'"
    echo ""
    exit 1
fi

echo ""

# =============================================================================
# DOWNLOAD BEHAVIORAL DATA (OPTIONAL BUT RECOMMENDED)
# =============================================================================

echo ""
echo -e "${YELLOW}================================================================================${NC}"
echo -e "${YELLOW}DOWNLOADING BEHAVIORAL DATA (OPTIONAL, ~100MB)${NC}"
echo -e "${YELLOW}================================================================================${NC}"
echo ""

echo -e "${BLUE}[INFO]${NC} Downloading behavioral data (trial info, timings, etc.)..."

BEHAV_DIR="${NSD_ROOT}/nsddata/ppdata/subj01/behav"

aws s3 sync \
    "${S3_BASE}/nsddata/ppdata/subj01/behav/" \
    "${BEHAV_DIR}/" \
    --no-sign-request

if [ $? -eq 0 ]; then
    echo -e "${GREEN}[✓]${NC} Behavioral data downloaded"
else
    echo -e "${YELLOW}[WARN]${NC} Behavioral data download failed (optional)"
fi

echo ""

# =============================================================================
# VERIFY DOWNLOAD
# =============================================================================

echo ""
echo -e "${YELLOW}================================================================================${NC}"
echo -e "${YELLOW}VERIFYING DOWNLOAD${NC}"
echo -e "${YELLOW}================================================================================${NC}"
echo ""

# Count beta files
BETA_COUNT=$(ls -1 "${BETA_DIR}"/betas_session*.nii.gz 2>/dev/null | wc -l)
echo -e "${BLUE}[INFO]${NC} Beta files downloaded: ${BETA_COUNT}/37"

if [ "$BETA_COUNT" -eq 37 ]; then
    echo -e "${GREEN}[✓]${NC} All beta files present"
else
    echo -e "${RED}[ERROR]${NC} Missing beta files! Expected 37, found ${BETA_COUNT}"
fi

# Check stimulus file
if [ -f "${STIM_DIR}/nsd_stimuli.hdf5" ]; then
    STIM_SIZE=$(du -h "${STIM_DIR}/nsd_stimuli.hdf5" | cut -f1)
    echo -e "${GREEN}[✓]${NC} Stimulus HDF5 file present (${STIM_SIZE})"
else
    # Check for PNG files
    PNG_COUNT=$(ls -1 "${STIM_DIR}"/*.png 2>/dev/null | wc -l)
    if [ "$PNG_COUNT" -gt 0 ]; then
        echo -e "${GREEN}[✓]${NC} Found ${PNG_COUNT} PNG stimulus files"
    else
        echo -e "${RED}[ERROR]${NC} No stimulus files found!"
    fi
fi

# Total size
TOTAL_SIZE=$(du -sh "${NSD_ROOT}" | cut -f1)
echo ""
echo -e "${BLUE}[INFO]${NC} Total download size: ${TOTAL_SIZE}"

echo ""
echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}DOWNLOAD COMPLETE!${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo ""
echo -e "${BLUE}[INFO]${NC} Data location: ${NSD_ROOT}"
echo ""
echo -e "${BLUE}[INFO]${NC} Directory structure:"
echo "  ${NSD_ROOT}/"
echo "  ├── nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/"
echo "  │   ├── betas_session01.nii.gz"
echo "  │   ├── ..."
echo "  │   └── betas_session37.nii.gz"
echo "  ├── nsddata_stimuli/stimuli/"
echo "  │   └── nsd_stimuli.hdf5 (or *.png files)"
echo "  └── nsddata/ppdata/subj01/behav/"
echo ""
echo -e "${GREEN}[✓]${NC} Ready to run experiments!"
echo ""
echo -e "${BLUE}[INFO]${NC} Next steps:"
echo "  1. Verify data: bash scripts/prepare_data.sh --verify-only"
echo "  2. Run preflight: bash scripts/preflight.sh"
echo "  3. Run experiment: bash scripts/run_experiment_simple.sh experiments/novel_subj01.yaml"
echo ""
