#!/bin/bash
# Reproducible staged driver for V40 OOF tri-gate pipeline.
#
# This script assumes fold checkpoints/predictions are generated externally.
# It handles:
# 1) fold manifest generation
# 2) fold-heldout prediction merge -> train_oof_* metrics
# 3) union shortlist cache build (train_oof + val + shared1000)
# 4) oracle audit (train_oof + val + shared1000)
# 5) reranker training/eval on train_oof

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <TRI_RESULTS_DIR> <LEGACY_RESULTS_DIR> [NUM_FOLDS]"
  echo "Example:"
  echo "  $0 experimental_results/V35_legacy_teacher_distill/subj01 experimental_results/N1v28a_dual_head/subj01 5"
  exit 1
fi

TRI_RESULTS_DIR="$1"
LEGACY_RESULTS_DIR="$2"
NUM_FOLDS="${3:-5}"

SEED="${SEED:-42}"
SHORTLIST_K="${SHORTLIST_K:-100}"

OOF_DIR="${TRI_RESULTS_DIR}/oof"
FOLD_SPLIT_DIR="${OOF_DIR}/fold_splits"
TRI_FOLD_ROOT="${OOF_DIR}/tri_folds"
LEGACY_FOLD_ROOT="${OOF_DIR}/legacy_folds"

mkdir -p "${OOF_DIR}" "${FOLD_SPLIT_DIR}"

BASE_SPLIT="${TRI_RESULTS_DIR}/split.json"
if [[ ! -f "${BASE_SPLIT}" ]]; then
  echo "ERROR: missing base split: ${BASE_SPLIT}"
  exit 1
fi

echo "[1/5] Building OOF folds from ${BASE_SPLIT}"
python scripts/preprocessing/build_oof_split_folds.py \
  "${BASE_SPLIT}" \
  --output-dir "${FOLD_SPLIT_DIR}" \
  --num-folds "${NUM_FOLDS}" \
  --seed "${SEED}"

FOLD_MANIFEST="${FOLD_SPLIT_DIR}/oof_fold_manifest.json"

echo "[2/5] Merging TRI fold-heldout predictions -> train_oof"
python scripts/preprocessing/merge_oof_expert_predictions.py \
  --fold-manifest "${FOLD_MANIFEST}" \
  --fold-results-root "${TRI_FOLD_ROOT}" \
  --fold-dir-pattern "fold_{fold_index:02d}" \
  --metrics-subdir "metrics" \
  --split-prefix "val" \
  --target-metrics-dir "${TRI_RESULTS_DIR}/metrics" \
  --provenance-json "${TRI_RESULTS_DIR}/metrics/train_oof_merge_provenance_tri.json"

echo "[2/5] Merging LEGACY fold-heldout predictions -> train_oof"
python scripts/preprocessing/merge_oof_expert_predictions.py \
  --fold-manifest "${FOLD_MANIFEST}" \
  --fold-results-root "${LEGACY_FOLD_ROOT}" \
  --fold-dir-pattern "fold_{fold_index:02d}" \
  --metrics-subdir "metrics" \
  --split-prefix "val" \
  --target-metrics-dir "${LEGACY_RESULTS_DIR}/metrics" \
  --provenance-json "${LEGACY_RESULTS_DIR}/metrics/train_oof_merge_provenance_legacy.json"

echo "[3/5] Building union shortlist caches (train_oof + val + shared1000)"
python scripts/preprocessing/build_union_shortlist_cache.py \
  "${TRI_RESULTS_DIR}" \
  "${LEGACY_RESULTS_DIR}" \
  --shortlist-k "${SHORTLIST_K}" \
  --splits train val shared1000 \
  --train-tri-metrics-dir "${TRI_RESULTS_DIR}/metrics" \
  --train-legacy-metrics-dir "${LEGACY_RESULTS_DIR}/metrics" \
  --train-split-prefix train_oof \
  --train-cache-name train_oof \
  --fold-provenance-json "${FOLD_MANIFEST}"

echo "[4/5] Auditing oracle/disagreement on train_oof + val + shared1000"
python scripts/evaluation/measure_union_shortlist_oracle.py \
  "${TRI_RESULTS_DIR}" \
  "${LEGACY_RESULTS_DIR}" \
  --splits train val shared1000 \
  --train-tri-metrics-dir "${TRI_RESULTS_DIR}/metrics" \
  --train-legacy-metrics-dir "${LEGACY_RESULTS_DIR}/metrics" \
  --train-split-prefix train_oof

echo "[5/5] Training V40 reranker on train_oof cache"
python scripts/training/train_union_shortlist_reranker.py \
  "${TRI_RESULTS_DIR}" \
  "${LEGACY_RESULTS_DIR}" \
  --shortlist-k "${SHORTLIST_K}" \
  --train-cache-split train_oof \
  --val-cache-split val \
  --shared-cache-split shared1000 \
  --run-tag v40_oof_tri_gate

echo "Done. NOTE: fold checkpoints and fold val predictions must already exist under:"
echo "  TRI:    ${TRI_FOLD_ROOT}/fold_XX/metrics/val_*.npy"
echo "  LEGACY: ${LEGACY_FOLD_ROOT}/fold_XX/metrics/val_*.npy"
