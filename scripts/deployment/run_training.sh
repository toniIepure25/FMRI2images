#!/bin/bash
# Complete Training Pipeline for subj01
# ======================================
# After successful CLIP cache build (10,004 embeddings)

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         SOTA fMRI Training Pipeline - subj01                   ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

SUBJECT="subj01"

# Step 1: Check CLIP cache
echo "Step 1/4: Verifying CLIP cache..."
CLIP_COUNT=$(python -c "import pandas as pd; print(len(pd.read_parquet('outputs/clip_cache/clip.parquet')))")
echo "  ✓ CLIP cache has $CLIP_COUNT embeddings"
echo ""

# Step 2: Preprocess - T1 Scaler
echo "Step 2/4: Running T1 preprocessing (2 minutes)..."
if [ ! -f "cache/preproc/${SUBJECT}_t1_scaler.pkl" ]; then
    python scripts/preprocess_fmri.py \
        --subject $SUBJECT \
        --method t1 \
        --output cache/preproc/${SUBJECT}_t1_scaler.pkl
    echo "  ✓ T1 scaler created"
else
    echo "  ✓ T1 scaler already exists"
fi
echo ""

# Step 3: Preprocess - T2 PCA
echo "Step 3/4: Running T2 PCA preprocessing (10 minutes)..."
if [ ! -f "cache/preproc/${SUBJECT}_t2_pca_k512.npz" ]; then
    python scripts/preprocess_fmri.py \
        --subject $SUBJECT \
        --method t2 \
        --pca-dim 512 \
        --output cache/preproc/${SUBJECT}_t2_pca_k512.npz
    echo "  ✓ T2 PCA components created"
else
    echo "  ✓ T2 PCA components already exist"
fi
echo ""

# Step 4: Train Two-Stage Encoder
echo "Step 4/4: Training two-stage encoder (6 hours)..."
echo ""
echo "⏱️  This will take ~6 hours. Consider using tmux:"
echo "   tmux new -s training"
echo "   bash run_training.sh"
echo "   # Detach: Ctrl+B, then D"
echo ""
read -p "Continue with training? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python scripts/train_two_stage.py \
        --config configs/sota_two_stage.yaml \
        --subject $SUBJECT \
        --output-dir checkpoints/two_stage/$SUBJECT
    
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                     Training Complete! 🎉                      ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Evaluate on NSD Shared 1000:"
    echo "   python scripts/eval_comprehensive.py \\"
    echo "     --subject $SUBJECT \\"
    echo "     --encoder-checkpoint checkpoints/two_stage/$SUBJECT/two_stage_best.pt \\"
    echo "     --encoder-type two_stage \\"
    echo "     --output-dir outputs/eval/$SUBJECT"
    echo ""
    echo "2. Generate comparison galleries:"
    echo "   python scripts/generate_comparison_gallery.py \\"
    echo "     --subject $SUBJECT \\"
    echo "     --encoder-checkpoint checkpoints/two_stage/$SUBJECT/two_stage_best.pt \\"
    echo "     --encoder-type two_stage \\"
    echo "     --output-dir outputs/galleries/$SUBJECT \\"
    echo "     --num-samples 16"
    echo ""
else
    echo ""
    echo "Training skipped. Run manually when ready:"
    echo "  python scripts/train_two_stage.py \\"
    echo "    --config configs/sota_two_stage.yaml \\"
    echo "    --subject $SUBJECT \\"
    echo "    --output-dir checkpoints/two_stage/$SUBJECT"
    echo ""
fi
