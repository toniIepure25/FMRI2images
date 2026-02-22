#!/bin/bash
# Build embedding preprocessors for all experiments
# Run this BEFORE training any experiments

set -e

CACHE_DIR="${CACHE_DIR:-cache}"
OUTPUT_DIR="${OUTPUT_DIR:-cache}"

mkdir -p "${OUTPUT_DIR}/embedding_preproc"

echo "=" 
echo "Building Embedding Preprocessors"
echo "="
echo ""

# EXP1-EXP5: center_pcr with k=8
echo "[1/2] Building center_pcr (k=8) preprocessor..."
python3 scripts/build/build_embedding_preproc.py \
    --mode center_pcr \
    --k_components 8 \
    --cache_dir "${CACHE_DIR}" \
    --output "${OUTPUT_DIR}/embedding_preproc/center_pcr_k8.pkl"

echo ""
echo "[2/2] Building center_whiten preprocessor..."
python3 scripts/build/build_embedding_preproc.py \
    --mode center_whiten \
    --cache_dir "${CACHE_DIR}" \
    --output "${OUTPUT_DIR}/embedding_preproc/center_whiten.pkl"

echo ""
echo "✅ All preprocessors built successfully!"
echo ""
echo "Artifacts saved to:"
echo "  - ${OUTPUT_DIR}/embedding_preproc/center_pcr_k8.pkl"
echo "  - ${OUTPUT_DIR}/embedding_preproc/center_whiten.pkl"
echo ""
echo "You can now run experiments with these preprocessors."
echo ""
