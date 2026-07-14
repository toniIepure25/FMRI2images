#!/bin/bash
# PCD Phase 2: Per-Subject Fine-Tuning
#
# Run after PCD_v1_8subject pretraining completes.
# Fine-tunes the pretrained model for each subject individually.
#
# Usage:
#   bash scripts/orchestration/run_pcd_finetune.sh [GPU_ID]

set -euo pipefail

GPU=${1:-0}
PRETRAIN_DIR="experimental_results/PCD_v1_8subject"
PRETRAIN_CKPT="${PRETRAIN_DIR}/subj01/checkpoints/best.pt"

if [ ! -f "$PRETRAIN_CKPT" ]; then
    echo "ERROR: Pretrained checkpoint not found at $PRETRAIN_CKPT"
    echo "Run PCD_v1_8subject pretraining first."
    exit 1
fi

SUBJECTS=(subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08)

for subj in "${SUBJECTS[@]}"; do
    echo "============================================"
    echo " Fine-tuning PCD for ${subj}"
    echo "============================================"
    
    OUT_DIR="experimental_results/PCD_v1_finetune/${subj}"
    
    if [ -f "${OUT_DIR}/checkpoints/best.pt" ]; then
        echo "Skipping ${subj} — checkpoint already exists"
        continue
    fi
    
    python3 scripts/training/train_unified.py \
        --config configs/experiments/PCD_v1_finetune.yaml \
        --subject "$subj" \
        --gpu "$GPU" \
        --resume "$PRETRAIN_CKPT" \
        2>&1 | tee "${OUT_DIR}/finetune.log"
    
    echo "Completed fine-tuning for ${subj}"
done

echo "============================================"
echo " All fine-tuning complete!"
echo "============================================"
