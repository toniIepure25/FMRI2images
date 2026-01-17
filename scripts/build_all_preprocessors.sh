#!/bin/bash
# Build embedding preprocessors for all experiments
# Run this BEFORE training any experiments

set -e

SUBJECT="subj01"
CONFIG="configs/data.yaml"
OUTPUT_DIR="cache/embedding_preproc"
PLOT_DIR="${OUTPUT_DIR}/plots"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PLOT_DIR}"

echo "=" 
echo "Building Embedding Preprocessors"
echo "="
echo ""

# EXP1-EXP5: center_pcr with k=8
echo "[1/2] Building center_pcr (k=8) preprocessor..."
python scripts/build_embedding_preproc.py \
    --subject "${SUBJECT}" \
    --mode center_pcr \
    --k_components 8 \
    --output "${OUTPUT_DIR}/${SUBJECT}_center_pcr_k8.pkl" \
    --plot_dir "${PLOT_DIR}/${SUBJECT}_center_pcr_k8" \
    --config "${CONFIG}"

echo ""
echo "[2/2] Building center_whiten preprocessor..."
python scripts/build_embedding_preproc.py \
    --subject "${SUBJECT}" \
    --mode center_whiten \
    --output "${OUTPUT_DIR}/${SUBJECT}_center_whiten.pkl" \
    --plot_dir "${PLOT_DIR}/${SUBJECT}_center_whiten" \
    --config "${CONFIG}"

echo ""
echo "✅ All preprocessors built successfully!"
echo ""
echo "Artifacts saved to:"
echo "  - ${OUTPUT_DIR}/${SUBJECT}_center_pcr_k8.pkl"
echo "  - ${OUTPUT_DIR}/${SUBJECT}_center_whiten.pkl"
echo ""
echo "Diagnostic plots saved to:"
echo "  - ${PLOT_DIR}/${SUBJECT}_center_pcr_k8/"
echo "  - ${PLOT_DIR}/${SUBJECT}_center_whiten/"
echo ""
