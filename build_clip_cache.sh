#!/bin/bash
# Correct commands for building CLIP cache
# ==========================================

echo "🚀 Building CLIP cache with correct arguments"
echo ""
echo "This will process ~73,000 NSD stimulus images."
echo "Estimated time: 2-3 hours on GPU"
echo ""

# Check if index exists
if [ ! -f "data/indices/nsd_index/subject=subj01/index.parquet" ]; then
    echo "❌ ERROR: Index file not found!"
    echo ""
    echo "First, build the index with:"
    echo "  python scripts/build_full_index.py \\"
    echo "    --subject subj01 \\"
    echo "    --output data/indices/nsd_index"
    echo ""
    exit 1
fi

echo "✓ Index file found"
echo ""
echo "Running CLIP cache builder..."
echo ""

.venv/bin/python scripts/build_clip_cache.py \
    --index-root data/indices/nsd_index \
    --subject subj01 \
    --cache outputs/clip_cache/clip.parquet \
    --batch-size 256 \
    --device cuda

echo ""
echo "✓ Done!"
echo ""
echo "Check cache status:"
echo "  python scripts/quick_status.py"
