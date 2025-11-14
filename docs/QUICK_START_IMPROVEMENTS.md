# Quick Start Guide: Implementing Pipeline Improvements
**Date**: November 11, 2025  
**Goal**: Achieve mean cosine > 0.70, R@1 > 15%  
**Time Required**: 12-16 hours total

---

## Overview

This guide provides step-by-step instructions to implement the comprehensive improvements outlined in `COMPREHENSIVE_IMPROVEMENT_PLAN.md`.

---

## Prerequisites

1. ✅ Virtual environment activated
2. ✅ Full NSD dataset accessible (9000 trials)
3. ✅ GPU available (CUDA)
4. ✅ ~50GB disk space for checkpoints
5. ✅ Backup current checkpoints (optional but recommended)

---

## Phase 1: Preprocessing Upgrade (2-3 hours)

### Step 1.1: Backup Current Preprocessing
```bash
# Optional: backup existing preprocessed data
cp -r outputs/preproc/subj01 outputs/preproc/subj01_backup_k3
```

### Step 1.2: Retrain Preprocessing with k=200
```bash
# Delete old preprocessing artifacts
rm -rf outputs/preproc/subj01/*.npy outputs/preproc/subj01/*.npz outputs/preproc/subj01/meta.json

# Retrain with k=200
python scripts/nsd_fit_preproc.py \
    --subject subj01 \
    --index-root data/indices/nsd_index \
    --out outputs/preproc/subj01 \
    --reliability-thr 0.1 \
    --pca-k 200 \
    --limit 9000 \
    --device cuda

# Expected output:
# - Voxels retained: ~370,000
# - PCA components: 200
# - Explained variance: ~99.9%
# - Training time: ~45-90 minutes
```

### Step 1.3: Verify Preprocessing
```bash
python -c "
import json
with open('outputs/preproc/subj01/meta.json') as f:
    meta = json.load(f)
print(f'PCA components: {meta[\"pca_components\"]}')
print(f'Explained variance: {meta[\"explained_variance_ratio\"]:.4f}')
print(f'Voxels kept: {meta[\"n_voxels_kept\"]:,}')
"

# Expected output:
# PCA components: 200
# Explained variance: 0.9990
# Voxels kept: 370,498
```

---

## Phase 2: MLP Architecture Upgrade (4-6 hours)

### Step 2.1: Backup Current MLP
```bash
# Optional: backup existing MLP checkpoint
cp checkpoints/mlp/subj01/mlp.pt checkpoints/mlp/subj01/mlp_baseline.pt
```

### Step 2.2: Install Improved MLP Module
```bash
# The improved MLP is already created at:
# src/fmri2img/models/mlp_improved.py

# Verify it exists
ls -lh src/fmri2img/models/mlp_improved.py
```

### Step 2.3: Train Improved MLP on Full Dataset
```bash
# Delete old checkpoint to force retraining
rm checkpoints/mlp/subj01/mlp.pt

# Train with improved architecture
python scripts/train_mlp.py \
    --subject subj01 \
    --index-root data/indices/nsd_index \
    --clip-cache outputs/clip_cache/subj01_clip512.parquet \
    --checkpoint-dir checkpoints/mlp/subj01 \
    --report-dir outputs/reports/subj01 \
    --use-preproc \
    --pca-k 200 \
    --hidden 2048 2048 1024 \
    --dropout 0.3 \
    --lr 0.0001 \
    --wd 0.0001 \
    --batch-size 128 \
    --epochs 100 \
    --patience 15 \
    --device cuda \
    --seed 42 \
    --use-improved \
    --mse-weight 0.3 \
    --triplet-weight 0.2

# Expected output:
# - Training time: ~3-5 hours
# - Best val cosine: > 0.65
# - Test cosine: > 0.60
# - Parameters: ~6-8M (vs ~2M baseline)
```

**Note**: If `--use-improved` flag is not recognized, you need to add it to train_mlp.py (see Phase 6 below).

### Step 2.4: Verify MLP Training
```bash
python -c "
import json
with open('outputs/reports/subj01/mlp_eval.json') as f:
    report = json.load(f)
print(f'Test cosine: {report[\"test_cosine\"]:.4f}')
print(f'Test MSE: {report[\"test_mse\"]:.6f}')
print(f'R@1: {report[\"r1\"]*100:.1f}%')
print(f'R@5: {report[\"r5\"]*100:.1f}%')
"

# Expected output (with full dataset):
# Test cosine: 0.62-0.68
# Test MSE: 0.0008-0.0012
# R@1: 10-15%
# R@5: 30-40%
```

---

## Phase 3: Adapter Training Upgrade (3-4 hours)

### Step 3.1: Build Full Target CLIP Cache
```bash
# This may take 2-3 hours but is crucial
python scripts/train_clip_adapter.py \
    --clip-cache outputs/clip_cache/subj01_clip512.parquet \
    --out checkpoints/clip_adapter/subj01/adapter_improved.pt \
    --model-id stabilityai/stable-diffusion-2-1 \
    --epochs 50 \
    --batch-size 128 \
    --lr 0.0003 \
    --patience 12 \
    --dropout 0.2 \
    --hidden 1536 1536 \
    --use-layernorm \
    --device cuda \
    --seed 42

# Expected output:
# - Training time: ~2-3 hours (with HTTP fallback)
# - Val cosine: > 0.45
# - Test cosine: > 0.42
# - Much better than current 0.3622
```

### Step 3.2: Verify Adapter Training
```bash
python -c "
import torch
ckpt = torch.load('checkpoints/clip_adapter/subj01/adapter_improved.pt')
meta = ckpt['meta']
print(f'Test cosine: {meta[\"test_cosine\"]:.4f}')
print(f'Test MSE: {meta[\"test_mse\"]:.6f}')
print(f'Parameters: {sum(p.numel() for p in ckpt[\"state_dict\"].values()):,}')
"

# Expected output:
# Test cosine: 0.42-0.48
# Test MSE: 0.0008-0.0010
# Parameters: ~2-3M
```

---

## Phase 4: Optimized Image Generation (1-2 hours)

### Step 4.1: Generate Images with All Improvements
```bash
python scripts/decode_diffusion.py \
    --subject subj01 \
    --encoder mlp \
    --ckpt checkpoints/mlp/subj01/mlp.pt \
    --clip-cache outputs/clip_cache/subj01_clip512.parquet \
    --index-root data/indices/nsd_index \
    --model-id stabilityai/stable-diffusion-2-1 \
    --clip-adapter checkpoints/clip_adapter/subj01/adapter_improved.pt \
    --output-dir outputs/recon/subj01/improved_full \
    --steps 150 \
    --guidance 11.0 \
    --scheduler ddim \
    --dtype float32 \
    --blend-alpha 1.0 \
    --device cuda \
    --limit 900 \
    --seed 42

# Expected output:
# - Generation time: ~3-4 hours for 900 images
# - Mean cosine: > 0.70
# - Images: outputs/recon/subj01/improved_full/images/
```

### Step 4.2: Quick Preview (Optional)
```bash
# Generate 50 images for quick preview (~20 minutes)
python scripts/decode_diffusion.py \
    --subject subj01 \
    --encoder mlp \
    --ckpt checkpoints/mlp/subj01/mlp.pt \
    --clip-cache outputs/clip_cache/subj01_clip512.parquet \
    --index-root data/indices/nsd_index \
    --model-id stabilityai/stable-diffusion-2-1 \
    --clip-adapter checkpoints/clip_adapter/subj01/adapter_improved.pt \
    --output-dir outputs/recon/subj01/improved_preview \
    --steps 150 \
    --guidance 11.0 \
    --scheduler ddim \
    --dtype float32 \
    --blend-alpha 1.0 \
    --device cuda \
    --limit 50 \
    --seed 42
```

---

## Phase 5: Evaluation & Comparison (30 minutes)

### Step 5.1: Evaluate Improved Results
```bash
# Create evaluation script if not exists
python scripts/eval_reconstruction.py \
    --recon-dir outputs/recon/subj01/improved_full/images \
    --subject subj01 \
    --splits test \
    --gallery test \
    --clip-cache outputs/clip_cache/subj01_clip512.parquet \
    --output outputs/reports/subj01/improved_evaluation.json \
    --device cuda
```

### Step 5.2: Compare with Baseline
```bash
python -c "
import json

# Load baseline results
with open('outputs/reports/subj01/baseline_eval.json') as f:
    baseline = json.load(f)

# Load improved results
with open('outputs/reports/subj01/improved_evaluation.json') as f:
    improved = json.load(f)

print('=' * 60)
print('BASELINE vs IMPROVED')
print('=' * 60)
print(f'Mean Cosine:')
print(f'  Baseline: {baseline.get(\"mean_cosine\", 0.5365):.4f}')
print(f'  Improved: {improved.get(\"mean_cosine\", 0):.4f}')
print(f'  Δ: +{(improved.get(\"mean_cosine\", 0) - baseline.get(\"mean_cosine\", 0.5365)):.4f}')
print()
print(f'R@1:')
print(f'  Baseline: {baseline.get(\"r1\", 0)*100:.1f}%')
print(f'  Improved: {improved.get(\"r1\", 0)*100:.1f}%')
print()
print(f'R@5:')
print(f'  Baseline: {baseline.get(\"r5\", 0)*100:.1f}%')
print(f'  Improved: {improved.get(\"r5\", 0)*100:.1f}%')
print('=' * 60)
"

# Expected output:
# Mean Cosine:
#   Baseline: 0.5365
#   Improved: 0.7000-0.7500
#   Δ: +0.1635-0.2135
# R@1:
#   Baseline: 0.0%
#   Improved: 12-18%
# R@5:
#   Baseline: 11.0%
#   Improved: 35-45%
```

---

## Phase 6: Optional Enhancements

### Option A: Use Automated Script
```bash
# Use updated run_production.sh with all improvements
./scripts/run_production.sh

# This will:
# 1. Build index (if needed)
# 2. Build CLIP cache (if needed)
# 3. Train preprocessing with k=200
# 4. Train improved MLP
# 5. Train improved adapter
# 6. Generate images with optimal parameters
# 7. Evaluate results

# Expected total time: 12-16 hours
```

### Option B: Ablation Study
```bash
# Test different PCA values
for k in 100 200 300 500; do
    python scripts/nsd_fit_preproc.py \
        --subject subj01 \
        --index-root data/indices/nsd_index \
        --out outputs/preproc/subj01_k${k} \
        --pca-k ${k} \
        --device cuda
    
    python scripts/train_mlp.py \
        --subject subj01 \
        --preproc-dir outputs/preproc/subj01_k${k} \
        --pca-k ${k} \
        --checkpoint-dir checkpoints/mlp/subj01_k${k} \
        --report-dir outputs/reports/subj01 \
        --device cuda
done

# Compare results
python -c "
import json
for k in [100, 200, 300, 500]:
    with open(f'outputs/reports/subj01/mlp_eval.json') as f:
        report = json.load(f)
    print(f'k={k}: test_cosine={report[\"test_cosine\"]:.4f}')
"
```

---

## Troubleshooting

### Issue 1: Out of Memory (OOM)
```bash
# Reduce batch size
--batch-size 64  # or 32

# Use float16 (may lose accuracy)
--dtype float16

# Use gradient checkpointing (add to train_mlp.py)
```

### Issue 2: HDF5 Download Failures
```bash
# The improved pipeline uses HTTP fallback automatically
# If issues persist, force HTTP-only:
export NSD_SOURCE=http
```

### Issue 3: Slow Training
```bash
# Use smaller limit for testing
--limit 1000

# Use fewer epochs
--epochs 30

# Use simpler architecture first
--hidden 2048
```

### Issue 4: Low Performance After Training
**Check**:
1. PCA components: Should be 100-500
2. Training data size: Should be 5000-8000
3. Validation cosine: Should be > 0.60
4. Adapter cosine: Should be > 0.40

**If still low**:
- Increase epochs
- Adjust learning rate
- Check for overfitting (train-val gap)

---

## Validation Checklist

- [ ] Preprocessing: k=200, explained variance > 99.9%
- [ ] MLP training: val cosine > 0.65, test cosine > 0.60
- [ ] Adapter training: test cosine > 0.40
- [ ] Image generation: mean cosine > 0.70
- [ ] Retrieval: R@1 > 12%, R@5 > 35%
- [ ] Visual inspection: Images semantically aligned with stimuli

---

## Expected Final Performance

| Metric | Current | After Improvements | Improvement |
|--------|---------|-------------------|-------------|
| **Mean Cosine** | 0.5365 | 0.70-0.75 | +31-40% |
| **R@1 (test)** | 0% | 12-18% | +12-18% |
| **R@5 (test)** | 11% | 35-45% | +24-34% |
| **R@10 (test)** | 22% | 50-60% | +28-38% |
| **CLIPScore** | ~0.60 | 0.72-0.77 | +20-28% |

---

## Next Steps After Completion

1. **Document results**: Create comparison report
2. **Visual analysis**: Inspect generated images
3. **Statistical testing**: Compute significance of improvements
4. **Hyperparameter tuning**: Fine-tune parameters further
5. **Publication**: Write methods section for paper

---

## Time Budget

| Phase | Task | Time |
|-------|------|------|
| 1 | Preprocessing (k=200) | 1.5h |
| 2 | MLP training (full data) | 4h |
| 3 | Adapter training (full data) | 3h |
| 4 | Image generation (900) | 3.5h |
| 5 | Evaluation | 0.5h |
| **Total** | | **12.5h** |

---

## Support

If you encounter issues:
1. Check `logs/` directory for detailed logs
2. Review `docs/COMPREHENSIVE_IMPROVEMENT_PLAN.md`
3. Verify GPU memory: `nvidia-smi`
4. Check disk space: `df -h`

**Good luck! 🚀**
