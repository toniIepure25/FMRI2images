#!/bin/bash
set -euo pipefail

SUBJECT="${SUBJECT:-subj01}"
GPU="${GPU:-0}"
SAVE_CKPT="${SAVE_CKPT:-best}"
START_AT="${START_AT:-token_cache_all8}"
SUBJECTS_CSV="${SUBJECTS_CSV:-subj01,subj02,subj03,subj04,subj05,subj06,subj07,subj08}"
TOKEN_CACHE="${TOKEN_CACHE:-outputs/clip_cache/tokens_ViT-L-14_projected_all8.h5}"
RERANK_CACHE="${RERANK_CACHE:-outputs/rerank_cache/pca_trainonly_dim2048_seed42_all8.npz}"
LOG_ROOT="${LOG_ROOT:-/home/jovyan/work/preserved_localdata/experiment_archive/all8_v35_v42_v43b_chain_logs}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/home/jovyan/work/preserved_runtime}"
RUNTIME_PYTHON="${RUNTIME_PYTHON:-${RUNTIME_ROOT}/python_runtime}"
NSD_HDF5_DEFAULT="/home/jovyan/work/data/nsd/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"

mkdir -p "${LOG_ROOT}"
touch "${LOG_ROOT}/chain_master.log"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
if [[ -d "${RUNTIME_PYTHON}" ]]; then
  export PYTHONPATH="${RUNTIME_PYTHON}:src:${PYTHONPATH:-}"
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${RUNTIME_ROOT}/pip_cache}"
  export HF_HOME="${HF_HOME:-${RUNTIME_ROOT}/hf_cache}"
  export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${RUNTIME_ROOT}/hf_cache}"
  export TORCH_HOME="${TORCH_HOME:-${RUNTIME_ROOT}/torch_cache}"
else
  export PYTHONPATH="${PYTHONPATH:-src}"
fi

if [[ -z "${NSD_HDF5:-}" && -f "${NSD_HDF5_DEFAULT}" ]]; then
  export NSD_HDF5="${NSD_HDF5_DEFAULT}"
fi

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

should_run_stage() {
  local stage="$1"
  local stages=(
    token_cache_all8
    rerank_cache_all8
    v35_all8
    v42_all8
    v43b_all8
  )
  local start_seen=0
  for s in "${stages[@]}"; do
    if [[ "${s}" == "${START_AT}" ]]; then
      start_seen=1
    fi
    if [[ "${s}" == "${stage}" ]]; then
      [[ "${start_seen}" -eq 1 ]]
      return
    fi
  done
  echo "Unknown START_AT stage: ${START_AT}" >&2
  exit 1
}

for exp in \
  V35_all8_legacy_teacher_distill \
  V42_all8_multi_hypothesis_vmf_retrieval \
  V43b_all8_stable_anti_collapse_multi_hypothesis_vmf
do
  link_result_dir "${exp}"
done

if should_run_stage token_cache_all8; then
  run_step token_cache_all8 \
    python3 scripts/build/build_token_clip_cache.py \
      --mode projected \
      --subjects "${SUBJECTS_CSV}" \
      --output "${TOKEN_CACHE}" \
      --batch-size 32 \
      --device cuda
fi

if should_run_stage rerank_cache_all8; then
  run_step rerank_cache_all8 \
    python3 scripts/preprocessing/build_rerank_cache.py \
      --config configs/experiments/V35_all8_legacy_teacher_distill.yaml \
      --method pca \
      --subjects "${SUBJECTS_CSV}" \
      --token-cache "${TOKEN_CACHE}" \
      --output-path "${RERANK_CACHE}" \
      --device cuda \
      --batch-size 256
fi

if should_run_stage v35_all8; then
  run_step v35_all8 \
    python3 scripts/training/train_unified.py \
      --config configs/experiments/V35_all8_legacy_teacher_distill.yaml \
      --subject "${SUBJECT}" \
      --gpu 0 \
      --save-checkpoints "${SAVE_CKPT}"
fi

if should_run_stage v42_all8; then
  run_step v42_all8 \
    python3 scripts/training/train_unified.py \
      --config configs/experiments/V42_all8_multi_hypothesis_vmf_retrieval.yaml \
      --subject "${SUBJECT}" \
      --gpu 0 \
      --save-checkpoints "${SAVE_CKPT}"
fi

if should_run_stage v43b_all8; then
  run_step v43b_all8 \
    python3 scripts/training/train_unified.py \
      --config configs/experiments/V43b_all8_stable_anti_collapse_multi_hypothesis_vmf.yaml \
      --subject "${SUBJECT}" \
      --gpu 0 \
      --save-checkpoints "${SAVE_CKPT}"
fi
