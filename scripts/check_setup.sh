#!/bin/bash
# Quick Setup & Status Check for SOTA fMRI Reconstruction
# =========================================================

set -e  # Exit on error

SUBJECT=${1:-subj01}
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$BASE_DIR"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         SOTA fMRI Reconstruction - Setup Check                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to check status
check_status() {
    local item=$1
    local check_command=$2
    
    echo -n "Checking $item... "
    if eval "$check_command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ OK${NC}"
        return 0
    else
        echo -e "${RED}✗ MISSING${NC}"
        return 1
    fi
}

# Function to check file size
check_file_size() {
    local file=$1
    local min_size=$2
    local description=$3
    
    if [ -f "$file" ]; then
        local size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
        if [ "$size" -gt "$min_size" ]; then
            echo -e "${GREEN}✓${NC} $description: $(du -h "$file" | cut -f1)"
            return 0
        else
            echo -e "${YELLOW}⚠${NC} $description: Too small ($(du -h "$file" | cut -f1))"
            return 1
        fi
    else
        echo -e "${RED}✗${NC} $description: Not found"
        return 1
    fi
}

echo "=== Environment Check ==="
check_status "Python environment" "python -c 'import torch, numpy, pandas'"
check_status "CUDA availability" "python -c 'import torch; assert torch.cuda.is_available()'"
check_status "Required packages" "python -c 'import open_clip, diffusers'"

echo ""
echo "=== Data Check ==="

# Check NSD index
if [ -d "data/indices/nsd_index" ]; then
    index_file="data/indices/nsd_index/${SUBJECT}.csv"
    check_file_size "$index_file" 100000 "NSD index ($SUBJECT)"
else
    echo -e "${RED}✗${NC} NSD index directory not found"
    NEED_INDEX=1
fi

# Check CLIP cache
echo ""
echo "=== CLIP Cache Check ==="
if [ -f "outputs/clip_cache/clip.parquet" ]; then
    NUM_EMBEDDINGS=$(python -c "import pandas as pd; print(len(pd.read_parquet('outputs/clip_cache/clip.parquet')))" 2>/dev/null || echo "0")
    if [ "$NUM_EMBEDDINGS" -gt 70000 ]; then
        echo -e "${GREEN}✓${NC} CLIP cache complete: $NUM_EMBEDDINGS embeddings"
    elif [ "$NUM_EMBEDDINGS" -gt 1000 ]; then
        echo -e "${YELLOW}⚠${NC} CLIP cache partial: $NUM_EMBEDDINGS / ~73,000 embeddings"
        echo "   → Continue building: python scripts/build_clip_cache.py"
        NEED_CLIP=1
    else
        echo -e "${RED}✗${NC} CLIP cache incomplete: $NUM_EMBEDDINGS / ~73,000 embeddings"
        echo "   → Build cache: python scripts/build_clip_cache.py"
        NEED_CLIP=1
    fi
else
    echo -e "${RED}✗${NC} CLIP cache not found"
    NEED_CLIP=1
fi

# Check preprocessing
echo ""
echo "=== Preprocessing Check ==="
check_file_size "cache/preproc/${SUBJECT}_t1_scaler.pkl" 1000 "T1 scaler"
check_file_size "cache/preproc/${SUBJECT}_t2_pca_k512.npz" 100000 "T2 PCA (k=512)"

# Check trained models
echo ""
echo "=== Trained Models Check ==="
check_file_size "checkpoints/two_stage/${SUBJECT}/two_stage_best.pt" 1000000 "Two-stage encoder"
MODEL_EXISTS=$?

# Summary
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                     Setup Status Summary                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"

if [ ! -z "$NEED_CLIP" ]; then
    echo ""
    echo -e "${YELLOW}⚠ ACTION REQUIRED: Build CLIP Cache${NC}"
    echo ""
    echo "The CLIP cache is missing or incomplete. This is required for training."
    echo ""
    echo "Run this command (takes 2-3 hours):"
    echo ""
    echo "  python scripts/build_clip_cache.py \\"
    echo "    --cache-root cache \\"
    echo "    --output outputs/clip_cache/clip.parquet \\"
    echo "    --batch-size 256"
    echo ""
fi

if [ $MODEL_EXISTS -ne 0 ]; then
    echo ""
    echo -e "${YELLOW}⚠ No trained model found${NC}"
    echo ""
    echo "After building CLIP cache, train the model:"
    echo ""
    echo "  python scripts/train_two_stage.py \\"
    echo "    --config configs/sota_two_stage.yaml \\"
    echo "    --subject $SUBJECT \\"
    echo "    --output-dir checkpoints/two_stage/$SUBJECT"
    echo ""
else
    echo ""
    echo -e "${GREEN}✓ System ready for evaluation!${NC}"
    echo ""
    echo "You can now run:"
    echo ""
    echo "  # Evaluate on NSD Shared 1000"
    echo "  python scripts/eval_comprehensive.py \\"
    echo "    --subject $SUBJECT \\"
    echo "    --encoder-checkpoint checkpoints/two_stage/$SUBJECT/two_stage_best.pt \\"
    echo "    --encoder-type two_stage \\"
    echo "    --output-dir outputs/eval/$SUBJECT"
    echo ""
    echo "  # Generate comparison galleries"
    echo "  python scripts/generate_comparison_gallery.py \\"
    echo "    --subject $SUBJECT \\"
    echo "    --encoder-checkpoint checkpoints/two_stage/$SUBJECT/two_stage_best.pt \\"
    echo "    --encoder-type two_stage \\"
    echo "    --output-dir outputs/galleries/$SUBJECT \\"
    echo "    --num-samples 16"
    echo ""
fi

echo ""
echo "For complete documentation, see:"
echo "  - USAGE_EXAMPLES.md (ready-to-run commands)"
echo "  - SOTA_QUICK_START.md (detailed guide)"
echo "  - docs/EVALUATION_SUITE_GUIDE.md (evaluation docs)"
echo ""
