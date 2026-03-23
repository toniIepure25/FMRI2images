#!/bin/bash
# Staged driver for V41 OOF union-shortlist vMF evidence resolver.
#
# It can both PREPARE the fold configs and COMPLETE the merge/cache/train path.
# Set RUN_FOLDS=1 to execute the generated fold scripts in this shell.
# Set PREPARE_ONLY=1 to stop after generating fold configs/scripts.

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
RUN_FOLDS="${RUN_FOLDS:-0}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
SUBJECT="${SUBJECT:-subj01}"
GPU="${GPU:-0}"
SAVE_CKPT="${SAVE_CKPT:-best}"
TRI_BASE_CONFIG="${TRI_BASE_CONFIG:-configs/experiments/V35_legacy_teacher_distill.yaml}"
LEGACY_BASE_CONFIG="${LEGACY_BASE_CONFIG:-configs/experiments/N1v28a_dual_head.yaml}"
PAIRWISE_MARGIN_WEIGHT="${PAIRWISE_MARGIN_WEIGHT:-0.10}"
PAIRWISE_MARGIN="${PAIRWISE_MARGIN:-0.20}"
PAIRWISE_HARD_NEG_K="${PAIRWISE_HARD_NEG_K:-5}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

OOF_DIR="${TRI_RESULTS_DIR}/oof_v41"
FOLD_SPLIT_DIR="${OOF_DIR}/fold_splits"
TRI_FOLD_ROOT="${OOF_DIR}/tri_folds"
LEGACY_FOLD_ROOT="${OOF_DIR}/legacy_folds"
TRI_CONFIG_DIR="${OOF_DIR}/generated_configs/tri"
LEGACY_CONFIG_DIR="${OOF_DIR}/generated_configs/legacy"
TRI_RUN_SCRIPT="${OOF_DIR}/run_tri_oof_folds.sh"
LEGACY_RUN_SCRIPT="${OOF_DIR}/run_legacy_oof_folds.sh"

mkdir -p "${OOF_DIR}" "${FOLD_SPLIT_DIR}" "${TRI_FOLD_ROOT}" "${LEGACY_FOLD_ROOT}"

BASE_SPLIT="${TRI_RESULTS_DIR}/split.json"
if [[ ! -f "${BASE_SPLIT}" ]]; then
  echo "ERROR: missing base split: ${BASE_SPLIT}"
  exit 1
fi

if [[ ! -f "${TRI_BASE_CONFIG}" ]]; then
  echo "ERROR: missing TRI base config: ${TRI_BASE_CONFIG}"
  exit 1
fi
if [[ ! -f "${LEGACY_BASE_CONFIG}" ]]; then
  echo "ERROR: missing LEGACY base config: ${LEGACY_BASE_CONFIG}"
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
  --exp-prefix V41_legacy_oof \
  --subject "${SUBJECT}" \
  --gpu "${GPU}" \
  --fold-preds-root "${LEGACY_FOLD_ROOT}" \
  --save-checkpoints "${SAVE_CKPT}" \
  --disable-shared1000-eval \
  --prediction-split val

echo "[3/7] Preparing TRI fold configs and export script"
"${PYTHON_BIN}" scripts/training/prepare_oof_fold_runs.py \
  --base-config "${TRI_BASE_CONFIG}" \
  --fold-manifest "${FOLD_MANIFEST}" \
  --config-output-dir "${TRI_CONFIG_DIR}" \
  --run-script-path "${TRI_RUN_SCRIPT}" \
  --exp-prefix V41_tri_oof \
  --subject "${SUBJECT}" \
  --gpu "${GPU}" \
  --fold-preds-root "${TRI_FOLD_ROOT}" \
  --save-checkpoints "${SAVE_CKPT}" \
  --disable-shared1000-eval \
  --prediction-split val \
  --strip-pretrained-model \
  --fold-teacher-checkpoint-template "experimental_results/V41_legacy_oof_fold_{fold_tag}/${SUBJECT}/checkpoint_best.pt"

chmod +x "${TRI_RUN_SCRIPT}" "${LEGACY_RUN_SCRIPT}"

if grep -R "pretrained_model_path" "${TRI_CONFIG_DIR}"/*.yaml >/dev/null 2>&1; then
  echo "ERROR: generated TRI OOF configs still contain pretrained_model_path; this would leak held-out fold images."
  exit 1
fi
if ! grep -R "V41_legacy_oof_fold_" "${TRI_CONFIG_DIR}"/*.yaml >/dev/null 2>&1; then
  echo "ERROR: generated TRI OOF configs are not wired to fold-specific legacy teacher checkpoints."
  exit 1
fi

if [[ "${PREPARE_ONLY}" == "1" ]]; then
  echo "Prepared OOF fold configs/scripts only. Next run the generated scripts:"
  echo "  ${TRI_RUN_SCRIPT}"
  echo "  ${LEGACY_RUN_SCRIPT}"
  exit 0
fi

if [[ "${RUN_FOLDS}" == "1" ]]; then
  echo "[4/7] Running LEGACY OOF fold jobs"
  GPU="${GPU}" SUBJECT="${SUBJECT}" SAVE_CKPT="${SAVE_CKPT}" bash "${LEGACY_RUN_SCRIPT}"

  for fold_idx in $(seq 0 $((NUM_FOLDS - 1))); do
    fold_tag=$(printf "%02d" "${fold_idx}")
    legacy_ckpt="experimental_results/V41_legacy_oof_fold_${fold_tag}/${SUBJECT}/checkpoint_best.pt"
    if [[ ! -f "${legacy_ckpt}" ]]; then
      echo "ERROR: missing fold-specific legacy teacher checkpoint: ${legacy_ckpt}"
      exit 1
    fi
  done

  echo "[4/7] Running TRI OOF fold jobs"
  GPU="${GPU}" SUBJECT="${SUBJECT}" SAVE_CKPT="${SAVE_CKPT}" bash "${TRI_RUN_SCRIPT}"
fi

for fold_idx in $(seq 0 $((NUM_FOLDS - 1))); do
  fold_tag=$(printf "%02d" "${fold_idx}")
  tri_metrics_dir="${TRI_FOLD_ROOT}/fold_${fold_tag}/metrics"
  legacy_metrics_dir="${LEGACY_FOLD_ROOT}/fold_${fold_tag}/metrics"
  for required in \
    "${tri_metrics_dir}/val_predictions_compact.npy" \
    "${tri_metrics_dir}/val_ground_truth_compact.npy" \
    "${tri_metrics_dir}/val_nsd_ids.npy" \
    "${tri_metrics_dir}/val_predictions_compact_component_mu.npy" \
    "${tri_metrics_dir}/val_predictions_compact_component_kappa.npy" \
    "${legacy_metrics_dir}/val_predictions_compact.npy" \
    "${legacy_metrics_dir}/val_ground_truth_compact.npy" \
    "${legacy_metrics_dir}/val_nsd_ids.npy"; do
    if [[ ! -f "${required}" ]]; then
      echo "ERROR: missing required OOF export: ${required}"
      echo "If folds have not been run yet, execute:"
      echo "  RUN_FOLDS=1 bash $0 ${TRI_RESULTS_DIR} ${LEGACY_RESULTS_DIR} ${NUM_FOLDS}"
      exit 1
    fi
  done
done

echo "[5/7] Merging TRI fold-heldout predictions -> train_oof"
"${PYTHON_BIN}" scripts/preprocessing/merge_oof_expert_predictions.py \
  --fold-manifest "${FOLD_MANIFEST}" \
  --fold-results-root "${TRI_FOLD_ROOT}" \
  --fold-dir-pattern "fold_{fold_index:02d}" \
  --metrics-subdir "metrics" \
  --split-prefix "val" \
  --target-metrics-dir "${TRI_RESULTS_DIR}/metrics" \
  --provenance-json "${TRI_RESULTS_DIR}/metrics/train_oof_merge_provenance_tri_v41.json"

echo "[5/7] Merging LEGACY fold-heldout predictions -> train_oof"
"${PYTHON_BIN}" scripts/preprocessing/merge_oof_expert_predictions.py \
  --fold-manifest "${FOLD_MANIFEST}" \
  --fold-results-root "${LEGACY_FOLD_ROOT}" \
  --fold-dir-pattern "fold_{fold_index:02d}" \
  --metrics-subdir "metrics" \
  --split-prefix "val" \
  --target-metrics-dir "${LEGACY_RESULTS_DIR}/metrics" \
  --provenance-json "${LEGACY_RESULTS_DIR}/metrics/train_oof_merge_provenance_legacy_v41.json"

echo "[6/7] Building union shortlist caches (train_oof + val + shared1000)"
"${PYTHON_BIN}" scripts/preprocessing/build_union_shortlist_cache.py \
  "${TRI_RESULTS_DIR}" \
  "${LEGACY_RESULTS_DIR}" \
  --shortlist-k "${SHORTLIST_K}" \
  --splits train val shared1000 \
  --train-tri-metrics-dir "${TRI_RESULTS_DIR}/metrics" \
  --train-legacy-metrics-dir "${LEGACY_RESULTS_DIR}/metrics" \
  --train-split-prefix train_oof \
  --train-cache-name train_oof \
  --fold-provenance-json "${FOLD_MANIFEST}"

echo "[6/7] Auditing union oracle/disagreement on train_oof + val + shared1000"
"${PYTHON_BIN}" scripts/evaluation/measure_union_shortlist_oracle.py \
  "${TRI_RESULTS_DIR}" \
  "${LEGACY_RESULTS_DIR}" \
  --splits train val shared1000 \
  --train-tri-metrics-dir "${TRI_RESULTS_DIR}/metrics" \
  --train-legacy-metrics-dir "${LEGACY_RESULTS_DIR}/metrics" \
  --train-split-prefix train_oof

echo "[7/7] Training V41 vMF-evidence resolver on train_oof cache"
"${PYTHON_BIN}" scripts/training/train_union_shortlist_reranker.py \
  "${TRI_RESULTS_DIR}" \
  "${LEGACY_RESULTS_DIR}" \
  --shortlist-k "${SHORTLIST_K}" \
  --train-cache-split train_oof \
  --val-cache-split val \
  --shared-cache-split shared1000 \
  --model-family vmf_evidence \
  --hidden-dim 96 \
  --num-layers 2 \
  --dropout 0.1 \
  --pairwise-margin-weight "${PAIRWISE_MARGIN_WEIGHT}" \
  --pairwise-margin "${PAIRWISE_MARGIN}" \
  --pairwise-hard-neg-k "${PAIRWISE_HARD_NEG_K}" \
  --run-tag v41_oof_union_vmf_resolver

echo "Done. Review the shared1000 resolver metrics against the fixed tri baseline (77.2%)."
