#!/usr/bin/env bash
# =============================================================================
# Roadmap Pipeline: From 77.2% to 90%+ R@1
# =============================================================================
#
# Orchestrates the complete experimental roadmap:
#   Phase A — V53: Fusion-aware student distillation (~12-24h)
#   Phase B — V54a: Multi-subject pre-training (~30-40h)
#   Phase C — V54b: Subject-01 fine-tuning from V54a (~10-20h)
#   Phase D — V52: Production OOF reranker (optional, ~60h)
#   Phase E — Ridge alignment investigation (~1h)
#
# Usage:
#   nohup bash scripts/training/run_roadmap_pipeline.sh > roadmap.log 2>&1 &
#
# Environment:
#   Requires: NSD_DATA_ROOT, DATASET_ROOT, OUTPUT_ROOT, CACHE_ROOT, HF_HOME
#   Source .env before running: set -a && source .env && set +a

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SUBJECT="subj01"
GPU="${GPU:-0}"
SUBJECTS_MULTI="${SUBJECTS_MULTI:-subj01 subj02 subj05 subj07}"
SKIP_V53="${SKIP_V53:-0}"
SKIP_V54="${SKIP_V54:-0}"
SKIP_V52="${SKIP_V52:-0}"
SKIP_RIDGE="${SKIP_RIDGE:-0}"

export CUDA_VISIBLE_DEVICES="$GPU"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
hr() { echo "========================================================================"; }

# -----------------------------------------------------------------------------
# Phase A — V53: Fusion-aware student distillation
# -----------------------------------------------------------------------------
run_v53() {
    hr
    log "PHASE A: V53 — Fusion-aware student distillation"
    hr

    local config="configs/experiments/V53_fusion_aware_student.yaml"
    local output_dir="experimental_results/V53_fusion_aware_student/${SUBJECT}"

    if [ -f "${output_dir}/checkpoint_best.pt" ] && [ "$SKIP_V53" = "1" ]; then
        log "V53 checkpoint exists and SKIP_V53=1, skipping."
        return 0
    fi

    log "Training V53 (single-subject fusion distillation)..."
    python3 scripts/training/train_unified.py \
        --config "$config" \
        --subject "$SUBJECT" \
        2>&1 | tee "${output_dir}/train.log" || true

    if [ ! -f "${output_dir}/checkpoint_best.pt" ]; then
        log "ERROR: V53 training failed — no checkpoint produced."
        return 1
    fi

    log "V53 training complete. Checking kill criteria..."
    local metrics_file="${output_dir}/metrics/summary.json"
    if [ -f "$metrics_file" ]; then
        python3 -c "
import json, sys
with open('${metrics_file}') as f:
    m = json.load(f)
csls_r1 = m.get('best_csls_r@1', m.get('val_csls_r@1', 0))
print(f'V53 compact CSLS R@1: {csls_r1:.4f}')
if csls_r1 < 0.552:
    print('KILL: compact_csls R@1 < 55.2% (V35 53.2% + 2pp). Wave V50 did not improve.')
    sys.exit(1)
print('PASS: compact_csls R@1 >= 55.2%')
" || log "WARNING: V53 kill criteria check failed (continuing anyway)"
    fi

    log "Running tri-fusion evaluation for V53..."
    python3 scripts/evaluation/sweep_tri_fusion_retrieval.py \
        --compact-dir "${output_dir}" \
        --legacy-dir "experimental_results/N1v28a_dual_head/${SUBJECT}" \
        --output-dir "${output_dir}/diagnostics" \
        --splits val shared1000 \
        2>&1 || log "WARNING: tri-fusion sweep failed"

    log "Phase A (V53) complete."
}

# -----------------------------------------------------------------------------
# Phase B — V54a: Multi-subject pre-training
# -----------------------------------------------------------------------------
run_v54a() {
    hr
    log "PHASE B: V54a — Multi-subject pre-training (4 subjects)"
    hr

    local config="configs/experiments/V54a_multi_subject_pretrain.yaml"
    local output_dir="experimental_results/V54a_multi_subject_pretrain/${SUBJECT}"

    if [ -f "${output_dir}/checkpoint_best.pt" ] && [ "$SKIP_V54" = "1" ]; then
        log "V54a checkpoint exists and SKIP_V54=1, skipping."
        return 0
    fi

    for subj in $SUBJECTS_MULTI; do
        local feat_path="cache/preextracted/subject=${subj}/fmri_features.npy"
        if [ ! -f "$feat_path" ]; then
            log "Pre-extracting features for ${subj}..."
            make preextract SUBJECT="$subj" || {
                log "ERROR: Failed to preextract features for ${subj}"
                return 1
            }
        else
            log "Features already exist for ${subj}: ${feat_path}"
        fi
    done

    log "Training V54a (4-subject pre-training)..."
    python3 scripts/training/train_unified.py \
        --config "$config" \
        2>&1 | tee "${output_dir}/train.log" || true

    if [ ! -f "${output_dir}/checkpoint_best.pt" ]; then
        log "ERROR: V54a training failed — no checkpoint produced."
        return 1
    fi

    log "Phase B (V54a) complete."
}

# -----------------------------------------------------------------------------
# Phase C — V54b: Subject-01 fine-tuning from V54a
# -----------------------------------------------------------------------------
run_v54b() {
    hr
    log "PHASE C: V54b — Subject-01 fine-tuning from V54a"
    hr

    local config="configs/experiments/V54b_subj01_finetune.yaml"
    local v54a_ckpt="experimental_results/V54a_multi_subject_pretrain/${SUBJECT}/checkpoint_best.pt"
    local output_dir="experimental_results/V54b_subj01_finetune/${SUBJECT}"

    if [ ! -f "$v54a_ckpt" ]; then
        log "ERROR: V54a checkpoint not found at ${v54a_ckpt}. Run Phase B first."
        return 1
    fi

    if [ -f "${output_dir}/checkpoint_best.pt" ] && [ "$SKIP_V54" = "1" ]; then
        log "V54b checkpoint exists and SKIP_V54=1, skipping."
        return 0
    fi

    log "Training V54b (subj01 fine-tuning with dual-head + legacy distill)..."
    python3 scripts/training/train_unified.py \
        --config "$config" \
        --subject "$SUBJECT" \
        2>&1 | tee "${output_dir}/train.log" || true

    if [ ! -f "${output_dir}/checkpoint_best.pt" ]; then
        log "ERROR: V54b training failed — no checkpoint produced."
        return 1
    fi

    log "V54b training complete. Checking kill criteria..."
    local metrics_file="${output_dir}/metrics/summary.json"
    if [ -f "$metrics_file" ]; then
        python3 -c "
import json, sys
with open('${metrics_file}') as f:
    m = json.load(f)
shared_r1 = m.get('shared1000_tri_fused_r@1', m.get('shared1000_r@1', 0))
print(f'V54b SHARED1000 R@1: {shared_r1:.4f}')
if shared_r1 < 0.79:
    print('KILL: SHARED1000 R@1 < 79%. Wave V51 did not beat V35 meaningfully.')
    sys.exit(1)
print('PASS: SHARED1000 R@1 >= 79%')
" || log "WARNING: V54b kill criteria check failed (continuing anyway)"
    fi

    log "Running tri-fusion evaluation for V54b..."
    python3 scripts/evaluation/sweep_tri_fusion_retrieval.py \
        --compact-dir "${output_dir}" \
        --legacy-dir "experimental_results/N1v28a_dual_head/${SUBJECT}" \
        --output-dir "${output_dir}/diagnostics" \
        --splits val shared1000 \
        2>&1 || log "WARNING: tri-fusion sweep failed"

    log "Phase C (V54b) complete."
}

# -----------------------------------------------------------------------------
# Phase D — V52: Production OOF reranker (optional)
# -----------------------------------------------------------------------------
run_v52() {
    hr
    log "PHASE D: V52 — Production OOF reranker"
    hr

    local base_config="configs/experiments/V52_production_oof_reranker.yaml"
    local n_folds=5
    local fold_dir="data/oof_splits/V52_production"

    log "Step 1: Creating ${n_folds}-fold splits..."
    python3 scripts/preprocessing/build_oof_split_folds.py \
        --subject "$SUBJECT" \
        --n-folds "$n_folds" \
        --output-dir "$fold_dir" \
        --seed 42 \
        2>&1 || { log "ERROR: Fold creation failed"; return 1; }

    log "Step 2: Training ${n_folds} V35-quality fold models..."
    for fold_idx in $(seq 0 $((n_folds - 1))); do
        local fold_ckpt="experimental_results/V52_production_oof_fold_${fold_idx}/${SUBJECT}/checkpoint_best.pt"
        if [ -f "$fold_ckpt" ]; then
            log "  Fold ${fold_idx} checkpoint exists, skipping."
            continue
        fi

        local fold_config="experimental_results/V52_production_oof_fold_${fold_idx}/config.yaml"
        mkdir -p "$(dirname "$fold_config")"

        python3 -c "
import yaml
with open('${base_config}') as f:
    cfg = yaml.safe_load(f)
cfg['experiment']['name'] = 'V52_production_oof_fold_${fold_idx}'
cfg['data']['split_file'] = '${fold_dir}/fold_${fold_idx}.parquet'
with open('${fold_config}', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
print('Created fold config: ${fold_config}')
"
        log "  Training fold ${fold_idx}/${n_folds}..."
        python3 scripts/training/train_unified.py \
            --config "$fold_config" \
            --subject "$SUBJECT" \
            2>&1 || log "WARNING: Fold ${fold_idx} training failed"
    done

    log "Step 3: Generating held-out predictions for each fold..."
    for fold_idx in $(seq 0 $((n_folds - 1))); do
        local fold_ckpt="experimental_results/V52_production_oof_fold_${fold_idx}/${SUBJECT}/checkpoint_best.pt"
        if [ ! -f "$fold_ckpt" ]; then
            log "  WARNING: Fold ${fold_idx} checkpoint missing, skipping."
            continue
        fi

        log "  Generating predictions for fold ${fold_idx} held-out data..."
        python3 scripts/evaluation/generate_split_predictions.py \
            --checkpoint "$fold_ckpt" \
            --config "experimental_results/V52_production_oof_fold_${fold_idx}/config.yaml" \
            --split "val" \
            --output-prefix "experimental_results/V52_production_oof_fold_${fold_idx}/${SUBJECT}/train_oof_fold${fold_idx}" \
            --subject "$SUBJECT" \
            2>&1 || log "WARNING: Prediction generation for fold ${fold_idx} failed"
    done

    log "Step 4: Merging fold predictions..."
    python3 scripts/preprocessing/merge_oof_expert_predictions.py \
        --fold-dirs $(for i in $(seq 0 $((n_folds - 1))); do echo "experimental_results/V52_production_oof_fold_${i}/${SUBJECT}"; done) \
        --output-dir "experimental_results/V52_production_oof_merged/${SUBJECT}" \
        --prefix "train_oof" \
        2>&1 || { log "ERROR: Merge failed"; return 1; }

    log "Step 5: Building union shortlist cache..."
    python3 scripts/preprocessing/build_union_shortlist_cache.py \
        --compact-dir "experimental_results/V52_production_oof_merged/${SUBJECT}" \
        --legacy-dir "experimental_results/N1v28a_dual_head/${SUBJECT}" \
        --output-dir "experimental_results/V52_production_oof_merged/${SUBJECT}/cache" \
        --split "train_oof" \
        --use-gpu \
        2>&1 || { log "ERROR: Cache build failed"; return 1; }

    log "Step 6: Training reranker on production-matched OOF cache..."
    python3 scripts/training/train_union_shortlist_reranker.py \
        --cache-dir "experimental_results/V52_production_oof_merged/${SUBJECT}/cache" \
        --val-compact-dir "experimental_results/V35_legacy_teacher_distill/${SUBJECT}" \
        --val-legacy-dir "experimental_results/N1v28a_dual_head/${SUBJECT}" \
        --output-dir "experimental_results/V52_production_reranker/${SUBJECT}" \
        --model-type "candidate_reranker" \
        --epochs 100 \
        --lr 1e-3 \
        --batch-size 256 \
        2>&1 || log "WARNING: Reranker training failed"

    log "Phase D (V52) complete."
}

# -----------------------------------------------------------------------------
# Phase E — Ridge alignment investigation
# -----------------------------------------------------------------------------
run_ridge() {
    hr
    log "PHASE E: Ridge alignment investigation"
    hr

    python3 scripts/evaluation/investigate_ridge_alignment.py \
        --subjects $SUBJECTS_MULTI \
        --target-subject "$SUBJECT" \
        --alpha-values 0.1 1.0 10.0 100.0 1000.0 10000.0 \
        --output-dir "outputs/ridge_alignment_investigation" \
        --device "cuda" \
        2>&1 || log "WARNING: Ridge investigation failed"

    log "Phase E (Ridge) complete."
}

# =============================================================================
# Main pipeline
# =============================================================================
main() {
    hr
    log "ROADMAP PIPELINE: 77.2% → 90%+ R@1"
    log "Subject: ${SUBJECT}"
    log "GPU: ${GPU}"
    log "Multi-subject: ${SUBJECTS_MULTI}"
    hr

    if [ "$SKIP_V53" != "1" ]; then
        run_v53
    else
        log "Skipping V53 (SKIP_V53=1)"
    fi

    if [ "$SKIP_V54" != "1" ]; then
        run_v54a
        run_v54b
    else
        log "Skipping V54 (SKIP_V54=1)"
    fi

    if [ "$SKIP_V52" != "1" ]; then
        run_v52
    else
        log "Skipping V52 (SKIP_V52=1)"
    fi

    if [ "$SKIP_RIDGE" != "1" ]; then
        run_ridge
    else
        log "Skipping Ridge investigation (SKIP_RIDGE=1)"
    fi

    hr
    log "ROADMAP PIPELINE COMPLETE"
    hr

    log "Results summary:"
    for exp in V53_fusion_aware_student V54a_multi_subject_pretrain V54b_subj01_finetune V52_production_reranker; do
        local mf="experimental_results/${exp}/${SUBJECT}/metrics/summary.json"
        if [ -f "$mf" ]; then
            log "  ${exp}:"
            python3 -c "
import json
with open('${mf}') as f:
    m = json.load(f)
for k in sorted(m):
    if 'r@1' in k.lower() or 'r@5' in k.lower():
        print(f'    {k}: {m[k]:.4f}')
" 2>/dev/null || true
        fi
    done

    if [ -f "outputs/ridge_alignment_investigation/ridge_alignment_results.json" ]; then
        log "  Ridge alignment: see outputs/ridge_alignment_investigation/"
    fi
}

main "$@"
