#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Full Pipeline: Road to 90%+ R@1
# =============================================================================
# Runs ALL phases sequentially on the H100 pod.
#
# Phase 2: V56a → V56c (fusion distillation)
# Phase 3a: PPR evaluation on all best checkpoints
# Phase 3b: V55d OOF 5-fold training → merge → shortlist cache
# Phase 3c: V55e PPR-aware resolver training
# Phase 4: V57a ROI Transformer
# Final: Aggregate PPR evaluation across all models
#
# Skips any phase whose checkpoint_best.pt already exists.
# Resumes from checkpoint_last.pt on crash.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/runtime_logs"
mkdir -p "$LOG_DIR"

# --- Environment: zero ephemeral writes ---
VENV_DIR="/home/jovyan/work/.venv-train"
if [[ -f "$VENV_DIR/bin/activate" ]]; then
    source "$VENV_DIR/bin/activate"
fi

if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a && source "$REPO_ROOT/.env" && set +a
fi

export PIP_CACHE_DIR="/home/jovyan/work/.cache/pip"
export TMPDIR="/home/jovyan/work/.cache/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export TORCH_HOME="/home/jovyan/work/.cache/torch"
export XDG_CACHE_HOME="/home/jovyan/work/.cache"
export MPLCONFIGDIR="/home/jovyan/work/.cache/mpl"
export HF_HOME="/home/jovyan/work/.cache/hf"
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HDF5_USE_FILE_LOCKING=FALSE
mkdir -p "$TMPDIR" "$TORCH_HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$HF_HOME" "$PIP_CACHE_DIR"

SUBJECT="subj01"
GPU=${GPU:-0}
MAX_RETRIES=3
RESULTS_ROOT="$REPO_ROOT/experimental_results"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_training() {
    local config="$1"
    local exp_name="$2"
    local extra_args="${3:-}"
    local attempt=0

    local best_ckpt="$RESULTS_ROOT/$exp_name/$SUBJECT/checkpoint_best.pt"
    if [[ -f "$best_ckpt" ]]; then
        log "SKIP $exp_name — checkpoint_best.pt already exists"
        return 0
    fi

    while [[ $attempt -lt $MAX_RETRIES ]]; do
        attempt=$((attempt + 1))
        log "=== $exp_name attempt $attempt/$MAX_RETRIES ==="

        local resume_flag=""
        local last_ckpt="$RESULTS_ROOT/$exp_name/$SUBJECT/checkpoint_last.pt"
        if [[ -f "$last_ckpt" ]]; then
            resume_flag="--resume $last_ckpt"
            log "Resuming from $last_ckpt"
        fi

        set +e
        python3 scripts/training/train_unified.py \
            --config "$config" \
            --subject "$SUBJECT" \
            --gpu "$GPU" \
            $resume_flag \
            $extra_args \
            2>&1 | tee -a "$LOG_DIR/${exp_name}.log"
        local exit_code=$?
        set -e

        if [[ $exit_code -eq 0 ]]; then
            log "$exp_name COMPLETED SUCCESSFULLY"
            return 0
        fi

        log "$exp_name attempt $attempt FAILED (exit=$exit_code). Retrying in 30s..."
        sleep 30
    done

    log "$exp_name FAILED after $MAX_RETRIES attempts. Continuing pipeline..."
    return 1
}


log "========================================================"
log "FULL PIPELINE: Road to 90%+ R@1"
log "========================================================"

# ===========================================================================
# PHASE 2: Fusion Distillation (V56a → V56c)
# ===========================================================================
log ""
log "============ PHASE 2: FUSION DISTILLATION ============"

# --- V56a ---
run_training \
    "configs/experiments/V56a_fusion_distill_fixed.yaml" \
    "V56a_fusion_distill_fixed" || true

# --- V56c (initialized from V56a best if available) ---
V56A_BEST="$RESULTS_ROOT/V56a_fusion_distill_fixed/$SUBJECT/checkpoint_best.pt"
if [[ -f "$V56A_BEST" ]]; then
    log "V56c: Updating pretrained_model_path to V56a best"
    sed -i "s|pretrained_model_path:.*|pretrained_model_path: \"$V56A_BEST\"|" \
        configs/experiments/V56c_projection_rdrop.yaml 2>/dev/null || true
fi

run_training \
    "configs/experiments/V56c_projection_rdrop.yaml" \
    "V56c_projection_rdrop" || true

# ===========================================================================
# PHASE 3a: PPR Evaluation on all available checkpoints
# ===========================================================================
log ""
log "============ PHASE 3a: PPR EVALUATION ============"
python3 scripts/evaluation/evaluate_all_checkpoints_ppr.py \
    --subject "$SUBJECT" \
    --results-root "$RESULTS_ROOT" \
    --output-dir "$RESULTS_ROOT/ppr_comparison" \
    2>&1 | tee "$LOG_DIR/phase3a_ppr_eval.log"
log "PPR evaluation complete"

# ===========================================================================
# PHASE 3b: OOF Fold Training
# ===========================================================================
log ""
log "============ PHASE 3b: OOF FOLD TRAINING ============"

# Determine best compact expert for OOF initialization
INIT_CKPT=""
for candidate in \
    "$RESULTS_ROOT/V56c_projection_rdrop/$SUBJECT/checkpoint_best.pt" \
    "$RESULTS_ROOT/V56a_fusion_distill_fixed/$SUBJECT/checkpoint_best.pt" \
    "$RESULTS_ROOT/V55b_subj01_finetune/$SUBJECT/checkpoint_best.pt"; do
    if [[ -f "$candidate" ]]; then
        INIT_CKPT="$candidate"
        break
    fi
done

if [[ -z "$INIT_CKPT" ]]; then
    log "ERROR: No compact expert checkpoint found for OOF fold initialization"
    log "Skipping Phase 3b"
else
    log "OOF init checkpoint: $INIT_CKPT"

    # Find the split.json from the init experiment
    INIT_SPLIT="$(dirname "$INIT_CKPT")/split.json"
    if [[ ! -f "$INIT_SPLIT" ]]; then
        INIT_SPLIT="$RESULTS_ROOT/V55b_subj01_finetune/$SUBJECT/split.json"
    fi

    if [[ ! -f "$INIT_SPLIT" ]]; then
        log "ERROR: No split.json found for OOF fold generation"
    else
        OOF_DIR="$RESULTS_ROOT/V55d_oof_folds"
        OOF_SPLITS_DIR="$OOF_DIR/splits"
        OOF_MANIFEST="$OOF_SPLITS_DIR/oof_fold_manifest.json"
        mkdir -p "$OOF_SPLITS_DIR"

        # Step 1: Generate fold splits (if not already done)
        if [[ ! -f "$OOF_MANIFEST" ]]; then
            log "Generating 5-fold OOF splits from $INIT_SPLIT..."
            python3 scripts/preprocessing/build_oof_split_folds.py \
                "$INIT_SPLIT" \
                --output-dir "$OOF_SPLITS_DIR" \
                --num-folds 5 \
                --seed 42 \
                2>&1 | tee "$LOG_DIR/oof_split_generation.log"
        else
            log "OOF fold splits already exist at $OOF_MANIFEST"
        fi

        # Step 2: Train each fold
        for fold in 0 1 2 3 4; do
            FOLD_NAME="V55d_oof_fold_${fold}"
            FOLD_SPLIT="$OOF_SPLITS_DIR/oof_fold_$(printf '%02d' $fold).json"
            FOLD_CONFIG="configs/experiments/V55d_oof_fold_${fold}.yaml"
            FOLD_BEST="$RESULTS_ROOT/$FOLD_NAME/$SUBJECT/checkpoint_best.pt"

            if [[ -f "$FOLD_BEST" ]]; then
                log "OOF fold $fold: checkpoint_best.pt exists. Skipping."
                continue
            fi

            if [[ ! -f "$FOLD_SPLIT" ]]; then
                log "ERROR: Fold split $FOLD_SPLIT not found. Skipping fold $fold."
                continue
            fi

            # Generate per-fold config by modifying the base V55d config
            log "Generating fold $fold config..."
            python3 -c "
import yaml
with open('configs/experiments/V55d_oof_fold.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['experiment']['name'] = '${FOLD_NAME}'
cfg['experiment']['description'] = 'V55d OOF fold ${fold}/5'
cfg['data']['split_file'] = '${FOLD_SPLIT}'
cfg['model']['pretrained_model_path'] = '${INIT_CKPT}'
with open('${FOLD_CONFIG}', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
print('Generated ${FOLD_CONFIG}')
"
            run_training "$FOLD_CONFIG" "$FOLD_NAME" || true
        done

        # Step 3: Merge OOF predictions
        log "Merging OOF predictions..."
        OOF_MERGED_DIR="$RESULTS_ROOT/V55d_oof_merged/$SUBJECT/metrics"
        mkdir -p "$OOF_MERGED_DIR"

        ALL_FOLDS_DONE=true
        for fold in 0 1 2 3 4; do
            FOLD_METRICS="$RESULTS_ROOT/V55d_oof_fold_${fold}/$SUBJECT/metrics"
            if [[ ! -d "$FOLD_METRICS" ]]; then
                log "WARNING: Fold $fold metrics missing at $FOLD_METRICS"
                ALL_FOLDS_DONE=false
            fi
        done

        if $ALL_FOLDS_DONE; then
            # Build fold results root structure for merge script
            OOF_FOLD_RESULTS="$RESULTS_ROOT/V55d_oof_fold_results"
            mkdir -p "$OOF_FOLD_RESULTS"
            for fold in 0 1 2 3 4; do
                FOLD_DIR="$OOF_FOLD_RESULTS/fold_$(printf '%02d' $fold)/metrics"
                mkdir -p "$(dirname "$FOLD_DIR")"
                if [[ ! -e "$FOLD_DIR" ]]; then
                    ln -sf "$RESULTS_ROOT/V55d_oof_fold_${fold}/$SUBJECT/metrics" "$FOLD_DIR"
                fi
            done

            python3 scripts/preprocessing/merge_oof_expert_predictions.py \
                --fold-manifest "$OOF_MANIFEST" \
                --fold-results-root "$OOF_FOLD_RESULTS" \
                --target-metrics-dir "$OOF_MERGED_DIR" \
                2>&1 | tee "$LOG_DIR/oof_merge.log"

            # Step 4: Build union shortlist cache
            log "Building union shortlist cache..."
            COMPACT_RESULTS="$RESULTS_ROOT/V55b_subj01_finetune/$SUBJECT"
            LEGACY_RESULTS="$RESULTS_ROOT/N1v28a_dual_head/$SUBJECT"

            if [[ -d "$COMPACT_RESULTS" ]] && [[ -d "$LEGACY_RESULTS" ]]; then
                python3 scripts/preprocessing/build_union_shortlist_cache.py \
                    "$COMPACT_RESULTS" \
                    "$LEGACY_RESULTS" \
                    --shortlist-k 150 \
                    --splits val shared1000 \
                    --train-tri-metrics-dir "$OOF_MERGED_DIR" \
                    --train-legacy-metrics-dir "$OOF_MERGED_DIR" \
                    --train-split-prefix "train_oof" \
                    --train-cache-name "train_oof" \
                    2>&1 | tee "$LOG_DIR/shortlist_cache.log"
            else
                log "WARNING: Compact or legacy results missing for shortlist cache"
            fi
        else
            log "WARNING: Not all OOF folds completed. Skipping merge."
        fi
    fi
fi

# ===========================================================================
# PHASE 3c: PPR-Aware Resolver
# ===========================================================================
log ""
log "============ PHASE 3c: PPR-AWARE RESOLVER ============"
RESOLVER_BEST="$RESULTS_ROOT/V55e_oof_ppr_resolver/$SUBJECT/checkpoint_best.pt"
if [[ -f "$RESOLVER_BEST" ]]; then
    log "SKIP V55e resolver — checkpoint_best.pt exists"
else
    CACHE_FILE="$RESULTS_ROOT/V55b_subj01_finetune/$SUBJECT/cache/union_shortlist_train_oof_k150.npz"
    if [[ -f "$CACHE_FILE" ]]; then
        log "Training PPR-aware resolver (V55e)..."
        run_training \
            "configs/experiments/V55e_oof_ppr_resolver.yaml" \
            "V55e_oof_ppr_resolver" || true
    else
        log "WARNING: Shortlist cache not found at $CACHE_FILE. Skipping resolver."
    fi
fi

# ===========================================================================
# PHASE 4: ROI Transformer
# ===========================================================================
log ""
log "============ PHASE 4: ROI TRANSFORMER ============"
run_training \
    "configs/experiments/V57a_roi_transformer_dual_head.yaml" \
    "V57a_roi_transformer_dual_head" || true

# ===========================================================================
# FINAL: Comprehensive PPR evaluation
# ===========================================================================
log ""
log "============ FINAL PPR EVALUATION ============"
python3 scripts/evaluation/evaluate_all_checkpoints_ppr.py \
    --subject "$SUBJECT" \
    --results-root "$RESULTS_ROOT" \
    --output-dir "$RESULTS_ROOT/ppr_comparison_final" \
    2>&1 | tee "$LOG_DIR/final_ppr_eval.log"

# Also run the original phase1 fusion eval for consistency
python3 scripts/evaluation/phase1_197k_fusion.py \
    2>&1 | tee "$LOG_DIR/final_197k_fusion.log" || true

log ""
log "========================================================"
log "FULL PIPELINE COMPLETE"
log "========================================================"
log "Results at: $RESULTS_ROOT/ppr_comparison_final/"
