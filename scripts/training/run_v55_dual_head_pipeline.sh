#!/usr/bin/env bash
# V55 Multi-Subject Dual-Head Pipeline: 77.2% -> 90%+ R@1
#
# Phase A: V55a — 4-subject dual-head pretraining with uniformity
# Phase B: V55b — subj01 fine-tuning with legacy + fusion distillation
# Phase C: PPR evaluation — vMF Posterior Predictive Retrieval scoring
#
# Usage:
#   nohup bash scripts/training/run_v55_dual_head_pipeline.sh > v55_pipeline.log 2>&1 &
#
# Skip phases with:
#   SKIP_V55A=1 SKIP_V55B=1 bash scripts/training/run_v55_dual_head_pipeline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ─── Phase A: V55a Multi-Subject Dual-Head Pretraining ───────────────────
if [[ -z "${SKIP_V55A:-}" ]]; then
    log "=== Phase A: V55a multi-subject dual-head pretraining ==="

    for subj in subj01 subj02 subj05 subj07; do
        FEAT="cache/preextracted/subject=${subj}/fmri_features.npy"
        if [[ ! -f "$FEAT" ]]; then
            log "Pre-extracting fMRI for $subj..."
            make preextract SUBJECT="$subj"
        fi
    done

    log "Starting V55a training (4-subject, dual-head + uniformity)..."
    python scripts/training/train_unified.py \
        --config configs/experiments/V55a_multi_subject_dual_head.yaml

    V55A_CKPT="experimental_results/V55a_multi_subject_dual_head/subj01/checkpoint_best.pt"
    if [[ ! -f "$V55A_CKPT" ]]; then
        log "ERROR: V55a checkpoint not found at $V55A_CKPT"
        exit 1
    fi
    log "Phase A complete. Best checkpoint: $V55A_CKPT"
else
    log "Skipping Phase A (SKIP_V55A set)"
fi

# ─── Phase B: V55b Single-Subject Fine-Tuning ────────────────────────────
if [[ -z "${SKIP_V55B:-}" ]]; then
    log "=== Phase B: V55b subj01 fine-tuning ==="

    python scripts/training/train_unified.py \
        --config configs/experiments/V55b_subj01_finetune.yaml

    V55B_CKPT="experimental_results/V55b_subj01_finetune/subj01/checkpoint_best.pt"
    if [[ ! -f "$V55B_CKPT" ]]; then
        log "ERROR: V55b checkpoint not found at $V55B_CKPT"
        exit 1
    fi
    log "Phase B complete. Best checkpoint: $V55B_CKPT"
else
    log "Skipping Phase B (SKIP_V55B set)"
fi

# ─── Phase C: PPR Evaluation ─────────────────────────────────────────────
log "=== Phase C: PPR evaluation ==="

V55B_DIR="experimental_results/V55b_subj01_finetune/subj01"
LEGACY_DIR="experimental_results/N1v28a_dual_head/subj01"

log "Running PPR vs CSLS comparison..."
python scripts/evaluation/evaluate_ppr.py \
    --experiment-dir "$V55B_DIR" \
    --legacy-dir "$LEGACY_DIR" \
    --subject subj01 \
    --output-dir "$V55B_DIR/ppr_evaluation"

log "=== V55 Pipeline Complete ==="
log "Check results in: $V55B_DIR/ppr_evaluation/"
