#!/bin/bash
#
# Phase 3: Train Probabilistic Multi-Layer Encoder with Uncertainty Modeling
# ==========================================================================
#
# Features:
# - ProbabilisticMultiLayerTwoStageEncoder (mu/logvar outputs)
# - KL divergence loss with annealing schedule
# - Multi-layer CLIP supervision (layer_4, layer_8, layer_12, final)
# - PCA-512 preprocessing for memory efficiency
# - 4,323 samples (filtered for multi-layer + text-CLIP)
#
# Expected:
# - Training time: ~30 minutes (50 epochs)
# - Memory usage: ~5.5 GB
# - Model size: ~4.5M parameters (slightly larger than Phase 2 due to mu/logvar heads)
#

set -e  # Exit on error

SUBJECT="subj01"
SAVE_NAME="phase3_probabilistic"
LOG_FILE="logs/clip_adapter/phase3_probabilistic_training.log"

# Create log directory
mkdir -p logs/clip_adapter

echo "================================================================================"
echo "Phase 3: Probabilistic Multi-Layer Encoder Training"
echo "================================================================================"
echo ""
echo "Subject: ${SUBJECT}"
echo "Save name: ${SAVE_NAME}"
echo "Log file: ${LOG_FILE}"
echo ""
echo "Starting training..."
echo ""

# Run training with probabilistic mode enabled
"/home/tonystark/Desktop/Bachelor V2/.venv/bin/python" scripts/train_two_stage.py \
  --subject "${SUBJECT}" \
  --use-preproc \
  --preproc-dir outputs/preproc \
  --pca-k 512 \
  --multi-layer \
  --multilayer-cache cache/clip_embeddings/nsd_clipcache_multilayer.parquet \
  --predict-text-clip \
  --text-clip-cache cache/clip_embeddings/text_clip.parquet \
  --text-clip-weight 0.3 \
  --probabilistic \
  --kl-weight 1e-4 \
  --kl-anneal-epochs 10 \
  --latent-dim 512 \
  --n-blocks 4 \
  --dropout 0.1 \
  --head-type linear \
  --batch-size 128 \
  --epochs 50 \
  --lr 1e-3 \
  --patience 10 \
  --checkpoint-dir checkpoints/clip_adapter \
  --save-name "${SAVE_NAME}" \
  2>&1 | tee "${LOG_FILE}"

echo ""
echo "================================================================================"
echo "Training complete!"
echo "================================================================================"
echo ""
echo "Checkpoint: checkpoints/clip_adapter/${SUBJECT}/${SAVE_NAME}"
echo "Log: ${LOG_FILE}"
echo ""
echo "Next steps:"
echo "  1. Evaluate probabilistic model with uncertainty metrics"
echo "  2. Compare Phase 2 (deterministic) vs Phase 3 (probabilistic)"
echo "  3. Analyze prediction uncertainty calibration"
echo ""
