#!/bin/bash
# Rebuild Target CLIP Cache with Fixed Dimensions
# ================================================
#
# This script rebuilds the target CLIP cache to fix the dimension mismatch issue.
# The old cache had mixed 1280-D and 1024-D embeddings, which causes adapter training to fail.
# The fixed script ensures all embeddings are consistently 1024-D (OpenCLIP ViT-H/14).

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Activate virtual environment
source .venv/bin/activate

# Configuration
SUBJECT="subj01"
INDEX_ROOT="data/indices/nsd_index"
MODEL_ID="stabilityai/stable-diffusion-2-1"
OUTPUT="outputs/clip_cache/target_clip_stabilityai_stable_diffusion_2_1.parquet"
BATCH_SIZE=200
INFERENCE_BATCH_SIZE=32
DEVICE="cuda"

echo "=============================================================================="
echo "REBUILDING TARGET CLIP CACHE (FIXED DIMENSIONS)"
echo "=============================================================================="
echo ""
echo "Configuration:"
echo "  Subject: ${SUBJECT}"
echo "  Model: ${MODEL_ID}"
echo "  Output: ${OUTPUT}"
echo "  Device: ${DEVICE}"
echo ""

# Check if old cache exists
if [ -f "${OUTPUT}" ]; then
    echo "⚠️  Old cache found with mixed dimensions (1280-D + 1024-D)"
    echo "   Deleting: ${OUTPUT}"
    rm -f "${OUTPUT}"
    echo "   ✓ Deleted"
    echo ""
fi

# Create output directory
mkdir -p "$(dirname "${OUTPUT}")"

# Create logs directory
LOG_DIR="logs/clip_cache"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/rebuild_target_cache_$(date +%Y%m%d_%H%M%S).log"

echo "Starting rebuild..."
echo "Log file: ${LOG_FILE}"
echo ""

# Run the fixed script
python scripts/build_target_clip_cache_robust.py \
    --subject "${SUBJECT}" \
    --index-root "${INDEX_ROOT}" \
    --model-id "${MODEL_ID}" \
    --output "${OUTPUT}" \
    --batch-size ${BATCH_SIZE} \
    --inference-batch-size ${INFERENCE_BATCH_SIZE} \
    --device "${DEVICE}" \
    2>&1 | tee "${LOG_FILE}"

EXIT_CODE=$?

echo ""
echo "=============================================================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ TARGET CACHE REBUILT SUCCESSFULLY!"
    echo ""
    echo "Verifying dimensions..."
    python scripts/check_target_cache_dimensions.py "${OUTPUT}"
    
    VERIFY_EXIT=$?
    if [ $VERIFY_EXIT -eq 0 ]; then
        echo ""
        echo "✅ All embeddings have consistent 1024-D dimension!"
        echo ""
        echo "Next steps:"
        echo "  1. Train adapter with the fixed cache:"
        echo "     python scripts/train_clip_adapter.py \\"
        echo "         --checkpoint checkpoints/mlp/subj01/mlp.pt \\"
        echo "         --clip-cache outputs/clip_cache/subj01_clip512.parquet \\"
        echo "         --target-cache ${OUTPUT} \\"
        echo "         --output checkpoints/adapter/subj01/adapter.pt \\"
        echo "         --hidden 1536 \\"
        echo "         --use-layernorm \\"
        echo "         --device cuda"
        echo ""
        echo "  2. Or rerun the full pipeline (it will skip completed steps):"
        echo "     bash scripts/run_production.sh --config configs/production_optimal.yaml"
    else
        echo ""
        echo "❌ Verification failed - cache still has dimension issues"
    fi
else
    echo "❌ REBUILD FAILED"
    echo "   Check log file: ${LOG_FILE}"
fi

echo "=============================================================================="

exit $EXIT_CODE
