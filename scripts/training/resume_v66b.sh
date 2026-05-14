#!/usr/bin/env bash
# Resume V66b after V66a completes: patch checkpoint path and train.
# Requires: V66a checkpoint at experimental_results/V66a_roi_pretrain/subj01/checkpoint_best.pt
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a && source .env && set +a
export HDF5_USE_FILE_LOCKING=FALSE
export TMPDIR="${TMPDIR:-/home/jovyan/work/tmp}"
mkdir -p "$TMPDIR"

CFG="configs/experiments/V66b_roi_finetune_197k.yaml"
V66A_BEST=""
for c in \
  "experimental_results/V66a_roi_pretrain/subj01/checkpoint_best.pt" \
  "experimental_results/V66a_roi_pretrain/subj01/checkpoints/best.pt" \
  "experimental_results/V66a_roi_pretrain/subj01/checkpoints/checkpoint_best.pt"; do
  if [[ -f "$c" ]]; then
    V66A_BEST="$c"
    break
  fi
done
if [[ -z "$V66A_BEST" ]]; then
  V66A_BEST="$(find experimental_results/V66a_roi_pretrain -name '*.pt' -type f 2>/dev/null | head -1 || true)"
fi
if [[ -z "$V66A_BEST" || ! -f "$V66A_BEST" ]]; then
  echo "ERROR: V66a checkpoint not found. Train V66a first."
  exit 1
fi

echo "Using V66a checkpoint: $V66A_BEST"
# Reset config line if re-running (idempotent)
git checkout -- "$CFG" 2>/dev/null || true
sed -i "s|PLACEHOLDER_V66A_BEST_CHECKPOINT|${V66A_BEST}|g" "$CFG"
grep "pretrained_encoder_path:" "$CFG" | head -1

LOG="logs/v65_v66/v66b_train_resume.log"
mkdir -p "$(dirname "$LOG")"
python3 scripts/training/train_unified.py --config "$CFG" 2>&1 | tee "$LOG"
