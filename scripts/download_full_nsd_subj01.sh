#!/bin/bash
# Download full NSD subj01 data (all 40 sessions)

set -e

SUBJECT="subj01"
DATA_DIR="/bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/data/nsd"
BETA_DIR="${DATA_DIR}/nsddata_betas/ppdata/${SUBJECT}/func1pt8mm/betas_fithrf_GLMdenoise_RR"

echo "================================================================================"
echo "Downloading full NSD subj01 data"
echo "================================================================================"
echo "Subject: ${SUBJECT}"
echo "Target directory: ${BETA_DIR}"
echo ""
echo "This will download ~50-100GB of data (40 beta session files)"
echo "Estimated time: 2-4 hours depending on connection speed"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Create directory
mkdir -p "${BETA_DIR}"

# Download all 40 sessions
echo ""
echo "Downloading beta files from NSD S3 bucket..."
echo ""

for session in $(seq -f "%02g" 1 40); do
    BETA_FILE="betas_session${session}.nii.gz"
    TARGET="${BETA_DIR}/${BETA_FILE}"
    
    if [ -f "${TARGET}" ]; then
        echo "[${session}/40] ✓ Already exists: ${BETA_FILE}"
    else
        echo "[${session}/40] Downloading: ${BETA_FILE}"
        wget -q --show-progress \
            "https://natural-scenes-dataset.s3.amazonaws.com/nsddata_betas/ppdata/${SUBJECT}/func1pt8mm/betas_fithrf_GLMdenoise_RR/${BETA_FILE}" \
            -O "${TARGET}" || {
                echo "❌ Failed to download ${BETA_FILE}"
                rm -f "${TARGET}"
                exit 1
            }
    fi
done

echo ""
echo "================================================================================"
echo "✓ Download complete!"
echo "================================================================================"
echo ""
echo "Verifying downloaded files..."
ls -lh "${BETA_DIR}" | grep "betas_session"
echo ""
echo "Next steps:"
echo "  1. Build full index: python3 scripts/build_full_subj01_index.py"
echo "  2. Generate full CLIP embeddings: python3 scripts/cache_all_clip_embeddings.py"
echo "  3. Run experiments: ./scripts/run_all_experiments.sh"
echo ""
