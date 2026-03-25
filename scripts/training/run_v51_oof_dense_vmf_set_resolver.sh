#!/bin/bash
# Staged driver for V51 OOF dense-vMF set resolver.
#
# Usage:
#   bash scripts/training/run_v51_oof_dense_vmf_set_resolver.sh \
#     experimental_results/V50_all8_dense_vmf_hybrid/subj01 \
#     experimental_results/N1v28a_dual_head/subj01 5

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <DENSE_RESULTS_DIR> <LEGACY_RESULTS_DIR> [NUM_FOLDS]"
  exit 1
fi

DENSE_RESULTS_DIR="$1"
LEGACY_RESULTS_DIR="$2"
NUM_FOLDS="${3:-5}"

SEED="${SEED:-42}"
SHORTLIST_K="${SHORTLIST_K:-100}"
RUN_FOLDS="${RUN_FOLDS:-0}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
SUBJECT="${SUBJECT:-subj01}"
GPU="${GPU:-0}"
SAVE_CKPT="${SAVE_CKPT:-best}"
DENSE_BASE_CONFIG="${DENSE_BASE_CONFIG:-configs/experiments/V50_all8_dense_vmf_hybrid.yaml}"
LEGACY_BASE_CONFIG="${LEGACY_BASE_CONFIG:-configs/experiments/N1v28a_dual_head.yaml}"
PAIRWISE_MARGIN_WEIGHT="${PAIRWISE_MARGIN_WEIGHT:-0.15}"
PAIRWISE_MARGIN="${PAIRWISE_MARGIN:-0.20}"
PAIRWISE_HARD_NEG_K="${PAIRWISE_HARD_NEG_K:-5}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

OOF_DIR="${DENSE_RESULTS_DIR}/oof_v51"
FOLD_SPLIT_DIR="${OOF_DIR}/fold_splits"
DENSE_FOLD_ROOT="${OOF_DIR}/dense_folds"
LEGACY_FOLD_ROOT="${OOF_DIR}/legacy_folds"
DENSE_CONFIG_DIR="${OOF_DIR}/generated_configs/dense"
LEGACY_CONFIG_DIR="${OOF_DIR}/generated_configs/legacy"
DENSE_RUN_SCRIPT="${OOF_DIR}/run_dense_oof_folds.sh"
LEGACY_RUN_SCRIPT="${OOF_DIR}/run_legacy_oof_folds.sh"

mkdir -p "${OOF_DIR}" "${FOLD_SPLIT_DIR}" "${DENSE_FOLD_ROOT}" "${LEGACY_FOLD_ROOT}"

BASE_SPLIT="${DENSE_RESULTS_DIR}/split.json"
if [[ ! -f "${BASE_SPLIT}" ]]; then
  echo "ERROR: missing base split: ${BASE_SPLIT}"
  exit 1
fi

echo "[1/7] Building deterministic OOF folds from ${BASE_SPLIT}"
"${PYTHON_BIN}" scripts/preprocessing/build_oof_split_folds.py \
  "${BASE_SPLIT}" \
  --output-dir "${FOLD_SPLIT_DIR}" \
  --num-folds "${NUM_FOLDS}" \
  --seed "${SEED}"

FOLD_MANIFEST="${FOLD_SPLIT_DIR}/oof_fold_manifest.json"

echo "[2/7] Preparing LEGACY fold configs and export script"
"${PYTHON_BIN}" scripts/training/prepare_oof_fold_runs.py \
  --base-config "${LEGACY_BASE_CONFIG}" \
  --fold-manifest "${FOLD_MANIFEST}" \
  --config-output-dir "${LEGACY_CONFIG_DIR}" \
  --run-script-path "${LEGACY_RUN_SCRIPT}" \
  --exp-prefix V51_legacy_oof \
  --subject "${SUBJECT}" \
  --gpu "${GPU}" \
  --fold-preds-root "${LEGACY_FOLD_ROOT}" \
  --save-checkpoints "${SAVE_CKPT}" \
  --disable-shared1000-eval \
  --prediction-split val

echo "[3/7] Preparing DENSE fold configs and export script"
"${PYTHON_BIN}" scripts/training/prepare_oof_fold_runs.py \
  --base-config "${DENSE_BASE_CONFIG}" \
  --fold-manifest "${FOLD_MANIFEST}" \
  --config-output-dir "${DENSE_CONFIG_DIR}" \
  --run-script-path "${DENSE_RUN_SCRIPT}" \
  --exp-prefix V51_dense_oof \
  --subject "${SUBJECT}" \
  --gpu "${GPU}" \
  --fold-preds-root "${DENSE_FOLD_ROOT}" \
  --save-checkpoints "${SAVE_CKPT}" \
  --disable-shared1000-eval \
  --prediction-split val \
  --strip-pretrained-model \
  --fold-teacher-checkpoint-template "experimental_results/V51_legacy_oof_fold_{fold_tag}/${SUBJECT}/checkpoint_best.pt"

chmod +x "${DENSE_RUN_SCRIPT}" "${LEGACY_RUN_SCRIPT}"

if [[ "${PREPARE_ONLY}" == "1" ]]; then
  echo "Prepared OOF fold configs/scripts only."
  echo "  ${LEGACY_RUN_SCRIPT}"
  echo "  ${DENSE_RUN_SCRIPT}"
  exit 0
fi

if [[ "${RUN_FOLDS}" == "1" ]]; then
  echo "[4/7] Running LEGACY OOF fold jobs"
  GPU="${GPU}" SUBJECT="${SUBJECT}" SAVE_CKPT="${SAVE_CKPT}" bash "${LEGACY_RUN_SCRIPT}"

  echo "[4/7] Running DENSE OOF fold jobs"
  GPU="${GPU}" SUBJECT="${SUBJECT}" SAVE_CKPT="${SAVE_CKPT}" bash "${DENSE_RUN_SCRIPT}"
fi

for fold_idx in $(seq 0 $((NUM_FOLDS - 1))); do
  fold_tag=$(printf "%02d" "${fold_idx}")
  dense_metrics_dir="${DENSE_FOLD_ROOT}/fold_${fold_tag}/metrics"
  legacy_metrics_dir="${LEGACY_FOLD_ROOT}/fold_${fold_tag}/metrics"
  for required in \
    "${dense_metrics_dir}/val_predictions_compact.npy" \
    "${dense_metrics_dir}/val_ground_truth_compact.npy" \
    "${dense_metrics_dir}/val_nsd_ids.npy" \
    "${dense_metrics_dir}/val_predictions_vmf.npy" \
    "${dense_metrics_dir}/val_kappas_vmf.npy" \
    "${dense_metrics_dir}/val_predictions_rerank.npy" \
    "${dense_metrics_dir}/val_ground_truth_rerank.npy" \
    "${legacy_metrics_dir}/val_predictions_compact.npy" \
    "${legacy_metrics_dir}/val_ground_truth_compact.npy" \
    "${legacy_metrics_dir}/val_nsd_ids.npy"; do
    if [[ ! -f "${required}" ]]; then
      echo "ERROR: missing required OOF export: ${required}"
      exit 1
    fi
  done
done

echo "[5/7] Merging DENSE fold-heldout predictions -> train_oof"
"${PYTHON_BIN}" scripts/preprocessing/merge_oof_expert_predictions.py \
  --fold-manifest "${FOLD_MANIFEST}" \
  --fold-results-root "${DENSE_FOLD_ROOT}" \
  --fold-dir-pattern "fold_{fold_index:02d}" \
  --metrics-subdir "metrics" \
  --split-prefix "val" \
  --target-metrics-dir "${DENSE_RESULTS_DIR}/metrics" \
  --provenance-json "${DENSE_RESULTS_DIR}/metrics/train_oof_merge_provenance_dense_v51.json"

echo "[5/7] Merging LEGACY fold-heldout predictions -> train_oof"
"${PYTHON_BIN}" scripts/preprocessing/merge_oof_expert_predictions.py \
  --fold-manifest "${FOLD_MANIFEST}" \
  --fold-results-root "${LEGACY_FOLD_ROOT}" \
  --fold-dir-pattern "fold_{fold_index:02d}" \
  --metrics-subdir "metrics" \
  --split-prefix "val" \
  --target-metrics-dir "${LEGACY_RESULTS_DIR}/metrics" \
  --provenance-json "${LEGACY_RESULTS_DIR}/metrics/train_oof_merge_provenance_legacy_v51.json"

echo "[6/7] Building union shortlist caches (train_oof + val + shared1000)"
"${PYTHON_BIN}" scripts/preprocessing/build_union_shortlist_cache.py \
  "${DENSE_RESULTS_DIR}" \
  "${LEGACY_RESULTS_DIR}" \
  --shortlist-k "${SHORTLIST_K}" \
  --splits train val shared1000 \
  --train-tri-metrics-dir "${DENSE_RESULTS_DIR}/metrics" \
  --train-legacy-metrics-dir "${LEGACY_RESULTS_DIR}/metrics" \
  --train-split-prefix train_oof \
  --train-cache-name train_oof \
  --fold-provenance-json "${FOLD_MANIFEST}"

echo "[6/7] Auditing union oracle/disagreement on train_oof + val + shared1000"
"${PYTHON_BIN}" scripts/evaluation/measure_union_shortlist_oracle.py \
  "${DENSE_RESULTS_DIR}" \
  "${LEGACY_RESULTS_DIR}" \
  --splits train val shared1000 \
  --train-tri-metrics-dir "${DENSE_RESULTS_DIR}/metrics" \
  --train-legacy-metrics-dir "${LEGACY_RESULTS_DIR}/metrics" \
  --train-split-prefix train_oof

echo "[7/7] Training V51 dense-vMF set resolver on train_oof cache"
"${PYTHON_BIN}" scripts/training/train_union_shortlist_reranker.py \
  "${DENSE_RESULTS_DIR}" \
  "${LEGACY_RESULTS_DIR}" \
  --shortlist-k "${SHORTLIST_K}" \
  --train-cache-split train_oof \
  --val-cache-split val \
  --shared-cache-split shared1000 \
  --model-family set_transformer \
  --hidden-dim 128 \
  --num-layers 2 \
  --dropout 0.1 \
  --pairwise-margin-weight "${PAIRWISE_MARGIN_WEIGHT}" \
  --pairwise-margin "${PAIRWISE_MARGIN}" \
  --pairwise-hard-neg-k "${PAIRWISE_HARD_NEG_K}" \
  --run-tag v51_oof_dense_vmf_set_resolver

echo "Done. Review the dense+legacy fixed fusion and resolver gains on SHARED1000."
