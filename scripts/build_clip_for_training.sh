#!/bin/bash
# Quick script to build CLIP cache for NSD data
# This extracts CLIP embeddings for all NSD images used in your experiments

set -e

echo "====================================================================================================
"
echo "                           BUILDING CLIP CACHE FOR NSD DATA"
echo "===================================================================================================="
echo ""
echo "This will extract CLIP ViT-L/14 embeddings for NSD images."
echo "Expected time: ~15-30 minutes for 750 session 1 images"
echo "Output: cache/clip_embeddings/nsd_clipvitl14.parquet"
echo ""
echo "===================================================================================================="

# Run the build script (uses SD 1.x which has ViT-L/14 CLIP)
python scripts/build_target_clip_cache_robust.py \
    --subject subj01 \
    --index-root data/indices/nsd_index \
    --model-id runwayml/stable-diffusion-v1-5 \
    --batch-size 100 \
    --inference-batch-size 64 \
    --output cache/clip_embeddings/nsd_clipvitl14.parquet

echo ""
echo "===================================================================================================="
echo "✅ CLIP cache built successfully!"
echo "===================================================================================================="
echo ""
echo "You can now run training with:"
echo "  python scripts/train_ultimate_novel.py --config experiments/ultimate_novel_subj01.yaml"
echo ""
