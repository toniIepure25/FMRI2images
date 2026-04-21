#!/usr/bin/env bash
# V55 PPNR Full Pipeline: 77.2% -> 90%+ R@1
#
# Posterior Predictive Neural Retrieval (PPNR) — complete experiment pipeline:
#
#   Phase A: V55a — 4-subject dual-head pretraining + uniformity (Wave 1)
#   Phase B: V55b — subj01 fine-tuning + legacy/fusion distillation (Wave 2)
#   Phase C: V55c — fusion topology distillation refinement (Wave 2)
#   Phase D: PPR evaluation — compare PPR vs CSLS on all checkpoints
#   Phase E: Fusion sweep — find optimal scoring+weights for new student
#   Phase F: V55d+V55e — OOF fold training + PPR resolver (Wave 3)
#   Phase G: Ablation table — generate paper-ready results
#
# Usage:
#   nohup bash scripts/training/run_v55_full_pipeline.sh > v55_full.log 2>&1 &
#
# Skip phases:
#   SKIP_V55A=1 SKIP_V55B=1 SKIP_V55C=1 SKIP_OOF=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] V55-PPNR: $*"; }

log "=========================================="
log "V55 PPNR Full Pipeline"
log "Goal: 77.2% -> 90%+ R@1 on NSD SHARED1000"
log "=========================================="

# ─── Phase A: Multi-Subject Dual-Head Pretraining ────────────────────────
if [[ -z "${SKIP_V55A:-}" ]]; then
    log "=== Phase A: V55a multi-subject dual-head pretraining ==="
    log "Training on 4 subjects with dual-head + uniformity..."

    for subj in subj01 subj02 subj05 subj07; do
        FEAT="cache/preextracted/subject=${subj}/fmri_features.npy"
        if [[ ! -f "$FEAT" ]]; then
            log "Pre-extracting fMRI for $subj..."
            make preextract SUBJECT="$subj"
        fi
    done

    python3 scripts/training/train_unified.py \
        --config configs/experiments/V55a_multi_subject_dual_head.yaml

    log "Phase A complete."
else
    log "Skipping Phase A (SKIP_V55A set)"
fi

# ─── Phase B: Single-Subject Fine-Tuning ─────────────────────────────────
if [[ -z "${SKIP_V55B:-}" ]]; then
    log "=== Phase B: V55b subj01 fine-tuning ==="

    python3 scripts/training/train_unified.py \
        --config configs/experiments/V55b_subj01_finetune.yaml

    log "Phase B complete."
else
    log "Skipping Phase B (SKIP_V55B set)"
fi

# ─── Phase C: Fusion Topology Distillation ────────────────────────────────
if [[ -z "${SKIP_V55C:-}" ]]; then
    log "=== Phase C: V55c fusion topology distillation ==="

    python3 scripts/training/train_unified.py \
        --config configs/experiments/V55c_fusion_distill.yaml

    log "Phase C complete."
else
    log "Skipping Phase C (SKIP_V55C set)"
fi

# ─── Phase D: PPR Evaluation ─────────────────────────────────────────────
log "=== Phase D: PPR evaluation on all checkpoints ==="

LEGACY_DIR="experimental_results/N1v28a_dual_head/subj01"

for EXP in V55a_multi_subject_dual_head V55b_subj01_finetune V55c_fusion_distill; do
    EXP_DIR="experimental_results/${EXP}/subj01"
    if [[ -d "$EXP_DIR" ]]; then
        log "PPR eval: $EXP"
        python3 scripts/evaluation/evaluate_ppr.py \
            --experiment-dir "$EXP_DIR" \
            --subject subj01 \
            --output-dir "$EXP_DIR/ppr_evaluation" \
            || log "PPR eval failed for $EXP (non-fatal)"
    fi
done

# ─── Phase E: Fusion Weight Sweep ────────────────────────────────────────
log "=== Phase E: Fusion weight sweep with PPR ==="

BEST_STUDENT=""
for EXP in V55c_fusion_distill V55b_subj01_finetune; do
    EXP_DIR="experimental_results/${EXP}/subj01"
    if [[ -d "$EXP_DIR" ]]; then
        BEST_STUDENT="$EXP_DIR"
        break
    fi
done

if [[ -n "$BEST_STUDENT" && -d "$LEGACY_DIR" ]]; then
    log "Sweeping fusion: compact=$BEST_STUDENT legacy=$LEGACY_DIR"
    python3 scripts/evaluation/sweep_fusion_ppr.py \
        --compact-dir "$BEST_STUDENT" \
        --legacy-dir "$LEGACY_DIR" \
        --output-dir "$BEST_STUDENT/fusion_sweep" \
        || log "Fusion sweep failed (non-fatal)"
fi

# ─── Phase F: OOF Resolver (optional) ────────────────────────────────────
if [[ -z "${SKIP_OOF:-}" ]]; then
    log "=== Phase F: OOF PPR resolver ==="
    bash scripts/training/run_v55_oof_ppr_resolver.sh
    log "Phase F complete."
else
    log "Skipping Phase F (SKIP_OOF set)"
fi

# ─── Phase G: Ablation Table ─────────────────────────────────────────────
log "=== Phase G: Generating ablation table ==="

python3 scripts/evaluation/generate_ablation_table.py \
    --results-root experimental_results \
    --output-dir outputs/v55_ablation

log "=========================================="
log "V55 PPNR Pipeline Complete!"
log "Check results:"
log "  Ablation: outputs/v55_ablation/"
log "  PPR eval: experimental_results/V55*/subj01/ppr_evaluation/"
log "  Fusion sweep: experimental_results/V55*/subj01/fusion_sweep/"
log "=========================================="
