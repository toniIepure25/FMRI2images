#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# V40 Full Pipeline: OOF Shortlist Reranker + Feature Ablation + Routing
#
# This script orchestrates the complete V40 pipeline including:
#   Wave A: OOF shortlist reranker (candidate MLP)
#   Wave B: VMF evidence reranker + calibration
#   Wave C: Set transformer reranker
#   Feature ablation across all waves
#   Reconstruction routing with reranker confidence
#
# Prerequisites:
#   - Fold checkpoints already trained (see prepare_oof_fold_runs.py)
#   - Fold predictions merged (see run_v40_oof_tri_gate.sh steps 1-2)
#   OR
#   - Pre-built OOF caches already exist
#
# Usage:
#   ./scripts/training/run_v40_full_pipeline.sh \
#       experimental_results/V35_legacy_teacher_distill/subj01 \
#       experimental_results/N1v28a_dual_head/subj01
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <TRI_RESULTS_DIR> <LEGACY_RESULTS_DIR> [SHORTLIST_K]"
    exit 1
fi

TRI="$1"
LEG="$2"
K="${3:-100}"
SEED="${SEED:-42}"
PYTHON="${PYTHON_BIN:-python3}"

CACHE_DIR="${TRI}/cache"
DIAGNOSTICS_DIR="${TRI}/diagnostics"
mkdir -p "${CACHE_DIR}" "${DIAGNOSTICS_DIR}"

echo "═══════════════════════════════════════════════════════════"
echo "V40 FULL PIPELINE"
echo "═══════════════════════════════════════════════════════════"
echo "TRI:    ${TRI}"
echo "LEG:    ${LEG}"
echo "K:      ${K}"
echo ""

# ─── Phase 1: Build caches (skip if they exist) ──────────────────────────

OOF_CACHE="${CACHE_DIR}/union_shortlist_train_oof_k${K}.npz"
VAL_CACHE="${CACHE_DIR}/union_shortlist_val_k${K}.npz"
S1000_CACHE="${CACHE_DIR}/union_shortlist_shared1000_k${K}.npz"

if [[ -f "${OOF_CACHE}" && -f "${VAL_CACHE}" && -f "${S1000_CACHE}" ]]; then
    echo "[Phase 1] Caches already exist, skipping build."
else
    echo "[Phase 1] Building union shortlist caches..."
    ${PYTHON} scripts/preprocessing/build_union_shortlist_cache.py \
        "${TRI}" "${LEG}" \
        --shortlist-k "${K}" \
        --splits train val shared1000 \
        --train-tri-metrics-dir "${TRI}/metrics" \
        --train-legacy-metrics-dir "${LEG}/metrics" \
        --train-split-prefix train_oof \
        --train-cache-name train_oof
fi

# ─── Phase 2: Oracle audit ───────────────────────────────────────────────

echo ""
echo "[Phase 2] Oracle audit..."
${PYTHON} scripts/evaluation/measure_union_shortlist_oracle.py \
    "${TRI}" "${LEG}" \
    --splits train val shared1000 \
    --train-tri-metrics-dir "${TRI}/metrics" \
    --train-legacy-metrics-dir "${LEG}/metrics" \
    --train-split-prefix train_oof

# ─── Phase 3: Cache inspection ───────────────────────────────────────────

echo ""
echo "[Phase 3] Inspecting caches..."
${PYTHON} scripts/evaluation/inspect_union_shortlist_cache.py \
    "${CACHE_DIR}" --all

# ─── Phase 4: Wave A — Candidate MLP reranker ────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "WAVE A: Candidate MLP Reranker"
echo "═══════════════════════════════════════════════════════════"
${PYTHON} scripts/training/train_union_shortlist_reranker.py \
    "${TRI}" "${LEG}" \
    --shortlist-k "${K}" \
    --train-cache-split train_oof \
    --val-cache-split val \
    --shared-cache-split shared1000 \
    --model-family candidate_mlp \
    --hidden-dim 128 \
    --num-layers 2 \
    --dropout 0.1 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --pairwise-margin-weight 0.1 \
    --pairwise-margin 0.2 \
    --pairwise-hard-neg-k 5 \
    --max-epochs 200 \
    --patience 30 \
    --batch-size 256 \
    --seed "${SEED}" \
    --run-tag v40_wave_a_mlp

# ─── Phase 5: Wave B — VMF Evidence reranker ─────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "WAVE B: VMF Evidence Reranker"
echo "═══════════════════════════════════════════════════════════"
${PYTHON} scripts/training/train_union_shortlist_reranker.py \
    "${TRI}" "${LEG}" \
    --shortlist-k "${K}" \
    --train-cache-split train_oof \
    --val-cache-split val \
    --shared-cache-split shared1000 \
    --model-family vmf_evidence \
    --hidden-dim 96 \
    --num-layers 2 \
    --dropout 0.1 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --pairwise-margin-weight 0.1 \
    --pairwise-margin 0.2 \
    --max-epochs 200 \
    --patience 30 \
    --batch-size 256 \
    --seed "${SEED}" \
    --run-tag v40_wave_b_vmf

# ─── Phase 6: Wave C — Set Transformer reranker ──────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "WAVE C: Set Transformer Reranker"
echo "═══════════════════════════════════════════════════════════"
${PYTHON} scripts/training/train_union_shortlist_reranker.py \
    "${TRI}" "${LEG}" \
    --shortlist-k "${K}" \
    --train-cache-split train_oof \
    --val-cache-split val \
    --shared-cache-split shared1000 \
    --model-family set_transformer \
    --hidden-dim 128 \
    --num-layers 2 \
    --dropout 0.15 \
    --lr 5e-4 \
    --weight-decay 1e-4 \
    --pairwise-margin-weight 0.1 \
    --pairwise-margin 0.2 \
    --max-epochs 200 \
    --patience 30 \
    --batch-size 128 \
    --seed "${SEED}" \
    --run-tag v40_wave_c_set

# ─── Phase 7: Feature ablation on best Wave A model ─────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "FEATURE ABLATION"
echo "═══════════════════════════════════════════════════════════"
${PYTHON} scripts/evaluation/ablate_reranker_features.py \
    "${TRI}" "${LEG}" \
    --shortlist-k "${K}" \
    --train-cache-split train_oof \
    --val-cache-split val \
    --hidden-dim 64 \
    --num-layers 2 \
    --dropout 0.1 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --max-epochs 80 \
    --patience 20 \
    --seed "${SEED}" \
    --run-tag v40_ablation

# ─── Phase 8: Reconstruction routing ────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "RECONSTRUCTION ROUTING"
echo "═══════════════════════════════════════════════════════════"

# Find the best Wave A checkpoint
WAVE_A_CKPT="${CACHE_DIR}/v40_wave_a_mlp_reranker_k${K}_best.pt"
if [[ -f "${WAVE_A_CKPT}" ]]; then
    ${PYTHON} scripts/reconstruction/route_by_reranker_confidence.py \
        --reranker-checkpoint "${WAVE_A_CKPT}" \
        --cache-dir "${CACHE_DIR}" \
        --split shared1000 \
        --shortlist-k "${K}" \
        --calibrate-on-val \
        --val-split val \
        --output-dir "${TRI}/routing/shared1000"

    ${PYTHON} scripts/reconstruction/route_by_reranker_confidence.py \
        --reranker-checkpoint "${WAVE_A_CKPT}" \
        --cache-dir "${CACHE_DIR}" \
        --split val \
        --shortlist-k "${K}" \
        --calibrate-on-val \
        --val-split val \
        --output-dir "${TRI}/routing/val"
else
    echo "WARNING: Wave A checkpoint not found at ${WAVE_A_CKPT}, skipping routing."
fi

# ─── Summary ────────────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "V40 PIPELINE COMPLETE"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Artifacts:"
echo "  Oracle audit:    ${DIAGNOSTICS_DIR}/union_shortlist_oracle.json"
echo "  Wave A summary:  ${DIAGNOSTICS_DIR}/v40_wave_a_mlp_reranker_summary.json"
echo "  Wave B summary:  ${DIAGNOSTICS_DIR}/v40_wave_b_vmf_reranker_summary.json"
echo "  Wave C summary:  ${DIAGNOSTICS_DIR}/v40_wave_c_set_reranker_summary.json"
echo "  Ablation:        ${DIAGNOSTICS_DIR}/v40_ablation_feature_ablation.json"
echo "  Routing:         ${TRI}/routing/"
echo ""
echo "Compare all waves:"
echo "  jq '.shared1000.resolver.\"R@1\"' ${DIAGNOSTICS_DIR}/v40_wave_*_reranker_summary.json"
