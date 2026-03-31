#!/bin/bash
set -euo pipefail

SUBJECT="${SUBJECT:-subj01}"
GPU="${GPU:-0}"
SAVE_CKPT="${SAVE_CKPT:-best}"
SUBJECTS_CSV="${SUBJECTS_CSV:-subj01,subj02,subj03,subj04,subj05,subj06,subj07,subj08}"
TOKEN_CACHE="${TOKEN_CACHE:-outputs/clip_cache/tokens_ViT-L-14_projected_all8.h5}"
RERANK_CACHE="${RERANK_CACHE:-outputs/rerank_cache/pca_trainonly_dim2048_seed42_all8.npz}"
LOG_ROOT="${LOG_ROOT:-/home/jovyan/work/preserved_localdata/experiment_archive/all8_v35_v42_v43b_chain_logs}"

mkdir -p "${LOG_ROOT}"
touch "${LOG_ROOT}/chain_master.log"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${PYTHONPATH:-src}"

link_result_dir() {
  local name="$1"
  local dst="/home/jovyan/work/preserved_localdata/experiment_archive/${name}"
  local src="experimental_results/${name}"
  mkdir -p "${dst}"
  rm -rf "${src}"
  ln -s "${dst}" "${src}"
}

run_step() {
  local name="$1"
  shift
  echo "[$(date -Iseconds)] START ${name}" | tee -a "${LOG_ROOT}/chain_master.log"
  "$@" 2>&1 | tee "${LOG_ROOT}/${name}.log"
  echo "[$(date -Iseconds)] DONE ${name}" | tee -a "${LOG_ROOT}/chain_master.log"
}

for exp in \
  V35_all8_legacy_teacher_distill \
  V42_all8_multi_hypothesis_vmf_retrieval \
  V43b_all8_stable_anti_collapse_multi_hypothesis_vmf
do
  link_result_dir "${exp}"
done

run_step token_cache_all8 \
  python3 scripts/build/build_token_clip_cache.py \
    --mode projected \
    --subjects "${SUBJECTS_CSV}" \
    --output "${TOKEN_CACHE}" \
    --batch-size 32 \
    --device cuda

run_step rerank_cache_all8 \
  python3 scripts/preprocessing/build_rerank_cache.py \
    --config configs/experiments/V35_all8_legacy_teacher_distill.yaml \
    --method pca \
    --subjects "${SUBJECTS_CSV}" \
    --token-cache "${TOKEN_CACHE}" \
    --output-path "${RERANK_CACHE}" \
    --device cuda \
    --batch-size 256

run_step v35_all8 \
  python3 scripts/training/train_unified.py \
    --config configs/experiments/V35_all8_legacy_teacher_distill.yaml \
    --subject "${SUBJECT}" \
    --gpu 0 \
    --save-checkpoints "${SAVE_CKPT}"

run_step v42_all8 \
  python3 scripts/training/train_unified.py \
    --config configs/experiments/V42_all8_multi_hypothesis_vmf_retrieval.yaml \
    --subject "${SUBJECT}" \
    --gpu 0 \
    --save-checkpoints "${SAVE_CKPT}"

run_step v43b_all8 \
  python3 scripts/training/train_unified.py \
    --config configs/experiments/V43b_all8_stable_anti_collapse_multi_hypothesis_vmf.yaml \
    --subject "${SUBJECT}" \
    --gpu 0 \
    --save-checkpoints "${SAVE_CKPT}"
