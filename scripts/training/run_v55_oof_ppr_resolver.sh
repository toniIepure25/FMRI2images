#!/usr/bin/env bash
# V55 OOF PPR Resolver Pipeline
#
# Trains 5 OOF fold models, merges predictions, builds shortlist caches,
# and trains the PPR-aware set-transformer resolver.
#
# Usage:
#   nohup bash scripts/training/run_v55_oof_ppr_resolver.sh > v55_oof.log 2>&1 &
#
# Skip phases: SKIP_FOLDS=1 SKIP_MERGE=1 SKIP_CACHE=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

NUM_FOLDS="${NUM_FOLDS:-5}"
COMPACT_DIR="experimental_results/V55b_subj01_finetune/subj01"
LEGACY_DIR="experimental_results/N1v28a_dual_head/subj01"
OOF_DIR="experimental_results/V55d_oof_fold/subj01"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ─── Phase 1: Create OOF split folds ─────────────────────────────────────
if [[ -z "${SKIP_FOLDS:-}" ]]; then
    log "=== Phase 1: Building OOF split folds ==="

    python scripts/preprocessing/build_oof_split_folds.py \
        --subject subj01 \
        --num-folds "$NUM_FOLDS" \
        --seed 42 \
        --exclude-shared1000 \
        --output-dir "$OOF_DIR/splits"

    log "=== Phase 1b: Training $NUM_FOLDS fold models ==="
    for fold_idx in $(seq 0 $((NUM_FOLDS - 1))); do
        FOLD_SPLIT="$OOF_DIR/splits/fold_${fold_idx}_split.json"
        FOLD_RESULTS="$OOF_DIR/fold_${fold_idx}"

        if [[ -f "$FOLD_RESULTS/checkpoint_best.pt" ]]; then
            log "Fold $fold_idx already trained, skipping"
            continue
        fi

        log "Training fold $fold_idx / $((NUM_FOLDS - 1))..."
        python scripts/training/train_unified.py \
            --config configs/experiments/V55d_oof_fold.yaml \
            --override "data.split_file=$FOLD_SPLIT" \
            --override "experiment.name=V55d_oof_fold_${fold_idx}"
    done
    log "Phase 1 complete: $NUM_FOLDS folds trained"
else
    log "Skipping fold training (SKIP_FOLDS set)"
fi

# ─── Phase 2: Merge OOF predictions ──────────────────────────────────────
if [[ -z "${SKIP_MERGE:-}" ]]; then
    log "=== Phase 2: Merging OOF expert predictions ==="

    python scripts/preprocessing/merge_oof_expert_predictions.py \
        --fold-dir "$OOF_DIR" \
        --num-folds "$NUM_FOLDS" \
        --output-prefix "$COMPACT_DIR/train_oof"

    log "Phase 2 complete"
else
    log "Skipping OOF merge (SKIP_MERGE set)"
fi

# ─── Phase 3: Build union shortlist cache ─────────────────────────────────
if [[ -z "${SKIP_CACHE:-}" ]]; then
    log "=== Phase 3: Building union shortlist cache ==="

    python scripts/preprocessing/build_union_shortlist_cache.py \
        --compact-dir "$COMPACT_DIR" \
        --legacy-dir "$LEGACY_DIR" \
        --output-dir "$COMPACT_DIR/shortlist_cache" \
        --shortlist-k 150 \
        --use-gpu

    log "Phase 3 complete"
else
    log "Skipping shortlist cache (SKIP_CACHE set)"
fi

# ─── Phase 4: Train PPR-aware resolver ────────────────────────────────────
log "=== Phase 4: Training PPR-aware set-transformer resolver ==="

python scripts/training/train_union_shortlist_reranker.py \
    --config configs/experiments/V55e_oof_ppr_resolver.yaml \
    --compact-dir "$COMPACT_DIR" \
    --legacy-dir "$LEGACY_DIR" \
    --shortlist-cache "$COMPACT_DIR/shortlist_cache" \
    --output-dir "$COMPACT_DIR/oof_ppr_resolver"

log "=== V55 OOF PPR Resolver Pipeline Complete ==="
log "Check results in: $COMPACT_DIR/oof_ppr_resolver/"
