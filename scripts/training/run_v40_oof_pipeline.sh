#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# V40 OOF Pipeline: Generate fold predictions -> Rebuild caches -> Train reranker
#
# This script runs the complete OOF pipeline:
#   1. Create 5-fold splits from V35's split.json
#   2. Train fast vMF model on each fold (~15 min/fold)
#   3. Generate held-out predictions for each fold
#   4. Merge fold predictions into train_oof arrays
#   5. Rebuild union shortlist caches with OOF train + real val/shared1000
#   6. Train reranker on OOF cache
#
# Usage:
#   ./scripts/training/run_v40_oof_pipeline.sh
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

PYTHON="${PYTHON_BIN:-python3}"
SEED=42
N_FOLDS=5
SUBJECT=subj01

TRI=experimental_results/V35_legacy_teacher_distill/subj01
LEG=experimental_results/N1v28a_dual_head/subj01
BASE_SPLIT=${TRI}/split.json
OOF_DIR=experimental_results/V40_oof_fast_vmf
FOLD_SPLITS_DIR=${OOF_DIR}/fold_splits
MERGED_DIR=${OOF_DIR}/${SUBJECT}/metrics
CONFIG=configs/experiments/V40_oof_fast_vmf.yaml
K=100

echo "═══════════════════════════════════════════════════════════"
echo "V40 OOF PIPELINE"
echo "═══════════════════════════════════════════════════════════"
echo "Subject: ${SUBJECT}"
echo "N folds: ${N_FOLDS}"
echo "Config:  ${CONFIG}"
echo ""

# ─── Step 1: Create fold splits ─────────────────────────────────────────

if [[ -f "${FOLD_SPLITS_DIR}/oof_fold_manifest.json" ]]; then
    echo "[Step 1] Fold splits exist, skipping."
else
    echo "[Step 1] Creating ${N_FOLDS}-fold splits..."
    ${PYTHON} scripts/preprocessing/build_oof_split_folds.py \
        "${BASE_SPLIT}" \
        --output-dir "${FOLD_SPLITS_DIR}" \
        --num-folds ${N_FOLDS} \
        --seed ${SEED}
fi

# ─── Step 2: Train each fold + generate predictions ─────────────────────

for FOLD_IDX in $(seq 0 $((N_FOLDS - 1))); do
    FOLD_NAME=$(printf "fold_%02d" ${FOLD_IDX})
    FOLD_SPLIT="${FOLD_SPLITS_DIR}/oof_fold_$(printf '%02d' ${FOLD_IDX}).json"
    FOLD_OUTPUT="${OOF_DIR}/${FOLD_NAME}/${SUBJECT}"
    FOLD_CKPT="${FOLD_OUTPUT}/checkpoints/best.pt"
    FOLD_PREDS="${FOLD_OUTPUT}/metrics/val_predictions_compact.npy"

    if [[ -f "${FOLD_PREDS}" ]]; then
        echo "[Step 2] Fold ${FOLD_IDX} predictions exist, skipping."
        continue
    fi

    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "FOLD ${FOLD_IDX}/${N_FOLDS}"
    echo "═══════════════════════════════════════════════════════════"

    # Create fold-specific config
    FOLD_CONFIG="${OOF_DIR}/${FOLD_NAME}/config.yaml"
    if [[ ! -f "${FOLD_CONFIG}" ]]; then
        mkdir -p "${OOF_DIR}/${FOLD_NAME}"
        ${PYTHON} -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
cfg['data']['split_file'] = '${FOLD_SPLIT}'
cfg['experiment']['name'] = 'V40_oof_fast_vmf_${FOLD_NAME}'
cfg['paths']['output_dir'] = '${OOF_DIR}/${FOLD_NAME}'
with open('${FOLD_CONFIG}', 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f'Created fold config: ${FOLD_CONFIG}')
"
    fi

    # Train
    if [[ ! -f "${FOLD_CKPT}" ]]; then
        echo "[Step 2a] Training fold ${FOLD_IDX}..."
        ${PYTHON} scripts/training/train_unified.py \
            --config "${FOLD_CONFIG}" \
            --subject "${SUBJECT}"
    else
        echo "[Step 2a] Fold ${FOLD_IDX} checkpoint exists, skipping training."
    fi

    # Generate predictions on held-out fold (which is the "val" split in the fold config)
    echo "[Step 2b] Generating fold ${FOLD_IDX} held-out predictions..."
    ${PYTHON} scripts/evaluation/generate_split_predictions.py \
        --checkpoint "${FOLD_CKPT}" \
        --output-dir "${FOLD_OUTPUT}" \
        --split val \
        --subject "${SUBJECT}"
done

# ─── Step 3: Merge fold predictions ─────────────────────────────────────

if [[ -f "${MERGED_DIR}/train_oof_predictions_compact.npy" ]]; then
    echo ""
    echo "[Step 3] Merged OOF predictions exist, skipping."
else
    echo ""
    echo "[Step 3] Merging fold predictions into train_oof arrays..."
    ${PYTHON} -c "
import numpy as np
import json
from pathlib import Path

manifest_path = Path('${FOLD_SPLITS_DIR}/oof_fold_manifest.json')
with open(manifest_path) as f:
    manifest = json.load(f)

all_preds = []
all_gts = []
all_nsd_ids = []
all_kappas = []

for fold_info in manifest['folds']:
    fold_idx = fold_info['fold_index']
    fold_name = f'fold_{fold_idx:02d}'
    fold_dir = Path('${OOF_DIR}') / fold_name / '${SUBJECT}' / 'metrics'

    preds = np.load(fold_dir / 'val_predictions_compact.npy')
    gts = np.load(fold_dir / 'val_ground_truth_compact.npy')
    nsd_ids = np.load(fold_dir / 'val_nsd_ids.npy')
    kappas_path = fold_dir / 'val_kappas.npy'
    kappas = np.load(kappas_path) if kappas_path.exists() else None

    print(f'  Fold {fold_idx}: preds={preds.shape}, nsd_ids={nsd_ids.shape}')
    all_preds.append(preds)
    all_gts.append(gts)
    all_nsd_ids.append(nsd_ids)
    if kappas is not None:
        all_kappas.append(kappas)

preds_merged = np.concatenate(all_preds, axis=0)
gts_merged = np.concatenate(all_gts, axis=0)
nsd_ids_merged = np.concatenate(all_nsd_ids, axis=0)
kappas_merged = np.concatenate(all_kappas, axis=0) if all_kappas else None

# Sort by nsd_id for consistency
sort_idx = np.argsort(nsd_ids_merged)
preds_merged = preds_merged[sort_idx]
gts_merged = gts_merged[sort_idx]
nsd_ids_merged = nsd_ids_merged[sort_idx]
if kappas_merged is not None:
    kappas_merged = kappas_merged[sort_idx]

# Verify no duplicates and full coverage
n_unique = len(np.unique(nsd_ids_merged))
print(f'  Merged: {preds_merged.shape}, unique nsd_ids={n_unique}')
assert n_unique == len(nsd_ids_merged), 'Duplicate nsd_ids in OOF merge!'

out_dir = Path('${MERGED_DIR}')
out_dir.mkdir(parents=True, exist_ok=True)
np.save(out_dir / 'train_oof_predictions_compact.npy', preds_merged)
np.save(out_dir / 'train_oof_predictions.npy', preds_merged)
np.save(out_dir / 'train_oof_ground_truth_compact.npy', gts_merged)
np.save(out_dir / 'train_oof_ground_truth.npy', gts_merged)
np.save(out_dir / 'train_oof_nsd_ids.npy', nsd_ids_merged)
if kappas_merged is not None:
    np.save(out_dir / 'train_oof_kappas.npy', kappas_merged)

# Quick R@1 check
from numpy.linalg import norm
p = preds_merged / np.maximum(norm(preds_merged, axis=-1, keepdims=True), 1e-8)
g = gts_merged / np.maximum(norm(gts_merged, axis=-1, keepdims=True), 1e-8)
sim = p @ g.T
diag = np.diag(sim)
ranks = (sim >= diag[:, None]).sum(axis=1)
r1 = (ranks == 1).mean()
print(f'  OOF R@1: {r1:.3f} (expect ~35-55%, NOT near 100%)')
print('  Done merging.')
"
fi

# ─── Step 4: Rebuild caches with OOF train data ─────────────────────────

CACHE_DIR="${TRI}/cache"
OOF_CACHE="${CACHE_DIR}/union_shortlist_train_oof_k${K}.npz"

if [[ -f "${OOF_CACHE}" ]]; then
    echo ""
    echo "[Step 4] OOF cache exists, skipping."
else
    echo ""
    echo "[Step 4] Building union shortlist caches with OOF train data..."
    ${PYTHON} scripts/preprocessing/build_union_shortlist_cache.py \
        "${TRI}" "${LEG}" \
        --shortlist-k ${K} \
        --splits train val shared1000 \
        --train-tri-metrics-dir "${MERGED_DIR}" \
        --train-legacy-metrics-dir "${LEG}/metrics" \
        --train-split-prefix train_oof \
        --train-cache-name train_oof \
        --use-gpu
fi

# ─── Step 5: Train reranker on OOF cache ────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "OOF RERANKER TRAINING"
echo "═══════════════════════════════════════════════════════════"

echo ""
echo "=== Wave A (OOF): Candidate MLP ==="
${PYTHON} scripts/training/train_union_shortlist_reranker.py \
    "${TRI}" "${LEG}" \
    --shortlist-k ${K} \
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
    --seed ${SEED} \
    --run-tag v40_oof_wave_a_mlp

echo ""
echo "=== Wave B (OOF): VMF Evidence ==="
${PYTHON} scripts/training/train_union_shortlist_reranker.py \
    "${TRI}" "${LEG}" \
    --shortlist-k ${K} \
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
    --seed ${SEED} \
    --run-tag v40_oof_wave_b_vmf

echo ""
echo "=== Wave C (OOF): Set Transformer ==="
${PYTHON} scripts/training/train_union_shortlist_reranker.py \
    "${TRI}" "${LEG}" \
    --shortlist-k ${K} \
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
    --seed ${SEED} \
    --run-tag v40_oof_wave_c_set

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "V40 OOF PIPELINE COMPLETE"
echo "═══════════════════════════════════════════════════════════"
