# � Comprehensive Usage Examples

> **Complete command reference for the Brain-to-Image reconstruction system**

This guide provides production-ready commands for all components: data preparation, training, evaluation, reconstruction, ablation studies, and reporting. All examples are tested and ready to run.

---

## 📋 Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Data Preparation](#2-data-preparation)
3. [Model Training](#3-model-training)
4. [Evaluation](#4-evaluation)
5. [Image Reconstruction](#5-image-reconstruction)
6. [Ablation Studies](#6-ablation-studies)
7. [Reporting & Visualization](#7-reporting--visualization)
8. [Advanced Workflows](#8-advanced-workflows)
9. [Troubleshooting](#9-troubleshooting)
10. [Quick Reference](#10-quick-reference)

---

## 1. Environment Setup

### Verify Installation

```bash
# Navigate to project root
cd /path/to/fmri2img

# Activate conda environment
conda activate fmri2img

# Verify Python version
python --version  # Should be 3.10+

# Check CUDA availability
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Run quick tests
python scripts/test_full_workflow.py
```

### GPU Configuration

```bash
# Check GPU info
nvidia-smi

# Set specific GPU (if multiple)
export CUDA_VISIBLE_DEVICES=0

# Force CPU mode (not recommended)
export CUDA_VISIBLE_DEVICES=""
```

---

## 2. Data Preparation

### 2.1 Download NSD Dataset

```bash
# Automated download (recommended)
python scripts/download_nsd_data.py \
  --output cache/ \
  --components stimuli betas

# Manual download alternative
# Visit: http://naturalscenesdataset.org/
# Required files:
#   - nsd_stimuli.hdf5 (39GB) → cache/nsd_hdf5/
#   - ppdata/subj0X/betas/ → cache/betas/

# Verify download
ls -lh cache/nsd_hdf5/nsd_stimuli.hdf5
ls -lh cache/betas/subj01/
```

### 2.2 Build Subject Indices

```bash
# Build complete index for subject 01
python scripts/build_full_index.py \
  --cache-root cache \
  --subject subj01 \
  --output data/indices/nsd_index/

# Build for all subjects
for subj in subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08; do
  python scripts/build_full_index.py \
    --cache-root cache \
    --subject $subj \
    --output data/indices/nsd_index/
done

# Build limited index for testing (100 samples)
python scripts/build_full_index.py \
  --cache-root cache \
  --subject subj01 \
  --limit 100 \
  --output data/indices/nsd_index/test/

# Verify output
ls -lh data/indices/nsd_index/subject=subj01/
# Should contain: trial_train.parquet, trial_val.parquet, trial_test.parquet
```

**Output:** Parquet files with columns: `[nsdId, subject, session, trial, split]`

### 2.3 Build CLIP Cache

#### Full Cache (Production)

```bash
# Extract CLIP embeddings for all 73,000 NSD images
python scripts/build_clip_cache.py \
  --cache-root cache \
  --output outputs/clip_cache/clip.parquet \
  --batch-size 256 \
  --device cuda

# Estimated time: 2-3 hours on RTX 3090
# Output size: ~500MB (73K × 512D float32)
```

#### Resume Interrupted Build

```bash
# Automatically resumes from last checkpoint
python scripts/build_clip_cache.py \
  --output outputs/clip_cache/clip.parquet \
  --batch-size 256

# Check progress
python -c "
import pandas as pd
df = pd.read_parquet('outputs/clip_cache/clip.parquet')
print(f'Progress: {len(df):,} / 73,000 embeddings ({100*len(df)/73000:.1f}%)')
"
```

#### Test Cache (Quick)

```bash
# Build mini cache for testing (100 images, 1 minute)
python scripts/build_clip_cache.py \
  --cache-root cache \
  --output outputs/clip_cache/clip_test.parquet \
  --limit 100 \
  --batch-size 32
```

#### Monitor Progress

```bash
# Watch in real-time
watch -n 5 "python -c 'import pandas as pd; df = pd.read_parquet(\"outputs/clip_cache/clip.parquet\"); print(f\"{len(df):,} embeddings\")'"

# Check GPU usage
nvidia-smi -l 1

# Use tmux for long-running processes
tmux new -s clipcache
python scripts/build_clip_cache.py --output outputs/clip_cache/clip.parquet
# Detach: Ctrl+B, then D
# Reattach: tmux attach -t clipcache
```

### 2.4 Preprocess fMRI Data (Optional)

```bash
# Z-score normalization
python scripts/preprocess_fmri.py \
  --subject subj01 \
  --method zscore \
  --output cache/preproc/subj01_zscore_scaler.pkl

# PCA dimensionality reduction
python scripts/preprocess_fmri.py \
  --subject subj01 \
  --method pca \
  --pca-dim 512 \
  --output cache/preproc/subj01_pca_k512.npz

# Combined preprocessing
python scripts/preprocess_fmri.py \
  --subject subj01 \
  --method combined \
  --pca-dim 512 \
  --output cache/preproc/subj01_combined.pkl
```

**Note:** Preprocessing is now integrated into training scripts and can be done on-the-fly.

---

## 3. Model Training

### 3.1 Ridge Regression Baseline

**Use case:** Fast baseline, interpretability, linear relationships

```bash
# Basic training
python scripts/train_ridge.py \
  --subject subj01 \
  --config configs/ridge_baseline.yaml \
  --output-dir checkpoints/ridge/subj01

# With custom hyperparameters
python scripts/train_ridge.py \
  --subject subj01 \
  --alpha 1.0 \
  --use-preproc \
  --pca-k 100 \
  --output-dir checkpoints/ridge/subj01

# Quick test (100 samples, 30 seconds)
python scripts/train_ridge.py \
  --subject subj01 \
  --limit 100 \
  --output-dir checkpoints/ridge/test
```

**Training time:** ~5 minutes (closed-form solution)  
**Parameters:** None (linear weights only)  
**Expected performance:** R@1 ≈ 12%, CLIP-I ≈ 0.52

### 3.2 MLP Encoder

**Use case:** Strong baseline, good speed/performance trade-off

```bash
# Standard MLP (single hidden layer)
python scripts/train_mlp.py \
  --subject subj01 \
  --config configs/mlp_standard.yaml \
  --output-dir checkpoints/mlp/subj01

# Custom architecture
python scripts/train_mlp.py \
  --subject subj01 \
  --use-preproc \
  --pca-k 512 \
  --hidden-dim 256 \
  --dropout 0.2 \
  --batch-size 128 \
  --epochs 50 \
  --learning-rate 1e-3 \
  --output-dir checkpoints/mlp/custom

# Fast training (reduced epochs)
python scripts/train_mlp.py \
  --subject subj01 \
  --config configs/mlp_standard.yaml \
  --epochs 20 \
  --output-dir checkpoints/mlp/fast

# Multi-subject training
for subj in subj01 subj02 subj03; do
  python scripts/train_mlp.py \
    --subject $subj \
    --config configs/mlp_standard.yaml \
    --output-dir checkpoints/mlp/${subj}
done
```

**Training time:** ~2 hours  
**Parameters:** ~148K  
**Expected performance:** R@1 ≈ 19%, CLIP-I ≈ 0.61

### 3.3 Two-Stage Encoder (SOTA)

**Use case:** Best performance, recommended for research

#### Standard Training

```bash
# SOTA configuration
python scripts/train_two_stage.py \
  --subject subj01 \
  --config configs/two_stage_sota.yaml \
  --output-dir checkpoints/two_stage/subj01

# Monitor training
tail -f logs/train_two_stage_subj01_*.log
```

#### Custom Hyperparameters

```bash
# Architecture variations
python scripts/train_two_stage.py \
  --subject subj01 \
  --latent-dim 768 \
  --n-blocks 4 \
  --dropout 0.3 \
  --head-type mlp \
  --head-hidden 512 \
  --output-dir checkpoints/two_stage/arch_768_4b

# Loss function weighting
python scripts/train_two_stage.py \
  --subject subj01 \
  --config configs/two_stage_sota.yaml \
  --mse-weight 0.3 \
  --cosine-weight 0.3 \
  --info-nce-weight 0.4 \
  --temperature 0.05 \
  --output-dir checkpoints/two_stage/infonce_0.4

# Training hyperparameters
python scripts/train_two_stage.py \
  --subject subj01 \
  --config configs/two_stage_sota.yaml \
  --batch-size 64 \
  --learning-rate 5e-5 \
  --epochs 100 \
  --weight-decay 1e-4 \
  --output-dir checkpoints/two_stage/custom_hp
```

#### Self-Supervised Pretraining

```bash
# Masked pretraining followed by supervised fine-tuning
python scripts/train_two_stage.py \
  --subject subj01 \
  --config configs/two_stage_sota.yaml \
  --self-supervised \
  --ssl-objective masked \
  --ssl-epochs 20 \
  --output-dir checkpoints/two_stage/ssl_masked

# Denoising pretraining
python scripts/train_two_stage.py \
  --subject subj01 \
  --config configs/two_stage_sota.yaml \
  --self-supervised \
  --ssl-objective denoising \
  --ssl-epochs 20 \
  --output-dir checkpoints/two_stage/ssl_denoising
```

#### Runtime Overrides

```bash
# Override specific config parameters
python scripts/train_two_stage.py \
  --config configs/two_stage_sota.yaml \
  --override "encoder.n_blocks=6" \
  --override "training.learning_rate=5e-5" \
  --override "preprocessing.pca_k=768"
```

**Training time:** ~6-8 hours  
**Parameters:** ~413K  
**Expected performance:** R@1 ≈ 24%, CLIP-I ≈ 0.66

### 3.4 CLIP Adapter

**Use case:** Dimension adaptation (512→768/1024), cross-model alignment

```bash
# Train adapter for SD 2.1 (1024-D)
python scripts/train_clip_adapter.py \
  --subject subj01 \
  --target-dim 1024 \
  --config configs/adapter_vitl14.yaml \
  --output-dir checkpoints/clip_adapter/sd21

# Train adapter for SD 1.5 (768-D)
python scripts/train_clip_adapter.py \
  --subject subj01 \
  --target-dim 768 \
  --output-dir checkpoints/clip_adapter/sd15

# Quick test
python scripts/train_clip_adapter.py \
  --subject subj01 \
  --target-dim 1024 \
  --limit 1000 \
  --epochs 10 \
  --output-dir checkpoints/clip_adapter/test
```

**Training time:** ~4 hours  
**Parameters:** ~395K (768-D) or ~527K (1024-D)

### 3.5 Training Best Practices

#### Monitor GPU Usage

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Log GPU usage to file
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.free --format=csv -l 10 > gpu_log.csv
```

#### Checkpoint Management

```bash
# Training automatically saves:
# - two_stage_best.pt (best validation performance)
# - two_stage_last.pt (last epoch)
# - training_log.json (per-epoch metrics)
# - config.yaml (saved configuration)

# Resume from checkpoint
python scripts/train_two_stage.py \
  --subject subj01 \
  --config configs/two_stage_sota.yaml \
  --resume checkpoints/two_stage/subj01/two_stage_last.pt \
  --output-dir checkpoints/two_stage/subj01
```

#### Distributed Training (Multi-GPU)

```bash
# Coming soon: DDP support
# python -m torch.distributed.launch --nproc_per_node=2 \
#   scripts/train_two_stage.py --config configs/two_stage_sota.yaml
```

---

## 4. Evaluation

### 4.1 Comprehensive Evaluation

```bash
# Full evaluation with all metrics
python scripts/eval_comprehensive.py \
  --subject subj01 \
  --encoder-checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
  --encoder-type two_stage \
  --output-dir outputs/eval/subj01 \
  --clip-cache outputs/clip_cache/clip.parquet

# Quick test (100 samples)
python scripts/eval_comprehensive.py \
  --subject subj01 \
  --encoder-checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
  --encoder-type two_stage \
  --max-samples 100 \
  --output-dir outputs/eval/test

# NSD Shared1000 evaluation (average 3 repetitions)
python scripts/eval_comprehensive.py \
  --subject subj01 \
  --encoder-checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
  --encoder-type two_stage \
  --split shared1000 \
  --average-reps \
  --output-dir outputs/eval/subj01_shared1000
```

**Output files:**
- `eval_comprehensive.log` - Detailed log
- `retrieval_metrics.json` - R@K, ranking metrics
- `predictions.npy` - Predicted CLIP embeddings
- `targets.npy` - Ground-truth CLIP embeddings

**Metrics computed:**
- Retrieval: R@1, R@5, R@10, R@20, R@50, R@100
- Ranking: Mean rank, median rank, MRR
- Similarity: CLIP-I score (cosine similarity)

### 4.2 Retrieval-Only Evaluation

```bash
# Evaluate on test set with test gallery
python scripts/eval_retrieval.py \
  --subject subj01 \
  --encoder-type two_stage \
  --checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
  --split test \
  --gallery test \
  --clip-cache outputs/clip_cache/clip.parquet \
  --output-json outputs/eval/retrieval_test.json

# Harder: test set vs full gallery (73K images)
python scripts/eval_retrieval.py \
  --subject subj01 \
  --encoder-type two_stage \
  --checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
  --split test \
  --gallery full \
  --output-json outputs/eval/retrieval_full.json

# Shared1000 evaluation
python scripts/eval_retrieval.py \
  --subject subj01 \
  --encoder-type two_stage \
  --checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
  --split shared1000 \
  --gallery shared1000 \
  --average-reps \
  --output-json outputs/eval/retrieval_shared1000.json
```

### 4.3 Multi-Gallery Evaluation

```bash
# Evaluate with multiple gallery types in one run
python scripts/run_reconstruct_and_eval.py \
  --encoder checkpoints/two_stage/subj01/two_stage_best.pt \
  --encoder-type two_stage \
  --subject subj01 \
  --galleries test shared1000 full \
  --output-dir outputs/eval/multi_gallery \
  --skip-reconstruction  # Only eval, no image generation

# Results will be in:
# - outputs/eval/multi_gallery/test/metrics.json
# - outputs/eval/multi_gallery/shared1000/metrics.json
# - outputs/eval/multi_gallery/full/metrics.json
```

### 4.4 Model Comparison

```bash
# Create comparison directory
mkdir -p outputs/eval/comparison

# Evaluate all models
for model in ridge mlp two_stage; do
  python scripts/eval_comprehensive.py \
    --subject subj01 \
    --encoder-checkpoint checkpoints/${model}/subj01/${model}_best.pt \
    --encoder-type $model \
    --output-dir outputs/eval/comparison/${model}
done

# Generate comparison report
python scripts/compare_models.py \
  --eval-dirs outputs/eval/comparison/{ridge,mlp,two_stage} \
  --output outputs/eval/comparison/report.md
```

### 4.5 Cross-Subject Analysis

```bash
# Evaluate trained model on all subjects
for subj in subj01 subj02 subj03; do
  python scripts/eval_comprehensive.py \
    --subject $subj \
    --encoder-checkpoint checkpoints/two_stage/${subj}/two_stage_best.pt \
    --encoder-type two_stage \
    --output-dir outputs/eval/cross_subject/${subj}
done

# Aggregate results
python scripts/aggregate_results.py \
  --results-dir outputs/eval/cross_subject \
  --output outputs/eval/cross_subject_summary.csv
```

---

## 5. Image Reconstruction

### 5.1 Basic Reconstruction

```bash
# Generate reconstructions with Stable Diffusion
python scripts/reconstruct.py \
  --subject subj01 \
  --encoder-checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
  --encoder-type two_stage \
  --num-images 50 \
  --output-dir outputs/reconstructions/subj01 \
  --diffusion-model stabilityai/stable-diffusion-2-1 \
  --num-inference-steps 50

# Quick test (10 images, 25 steps)
python scripts/reconstruct.py \
  --subject subj01 \
  --encoder-checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
  --encoder-type two_stage \
  --num-images 10 \
  --num-inference-steps 25 \
  --output-dir outputs/reconstructions/test
```

**First run:** Downloads Stable Diffusion model (~5GB)  
**Time:** ~3-6 seconds per image (50 steps), ~1.5 seconds (25 steps)

### 5.2 Comparison Galleries

```bash
# Generate side-by-side comparisons with multiple strategies
python scripts/generate_comparison_gallery.py \
  --subject subj01 \
  --encoder-checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
  --encoder-type two_stage \
  --output-dir outputs/galleries/subj01 \
  --num-samples 16 \
  --strategies single best_of_8 \
  --grid-cols 4 \
  --num-inference-steps 50 \
  --guidance-scale 7.5 \
  --seed 42

# Quick preview (4 samples, single strategy)
python scripts/generate_comparison_gallery.py \
  --subject subj01 \
  --encoder-checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
  --encoder-type two_stage \
  --output-dir outputs/galleries/preview \
  --num-samples 4 \
  --strategies single \
  --num-inference-steps 25
```

**Output:**
- `single/` - Individual reconstructions (single sample)
- `best_of_8/` - Best of 8 candidates (quality)
- `comparison_grid.png` - Combined visualization

**Time estimates:**
- Single (16 samples): ~5 minutes
- Best-of-8 (16 samples): ~40 minutes
- Best-of-16 (16 samples): ~80 minutes

### 5.3 Advanced Sampling Strategies

#### Best-of-N Sampling

```bash
# Generate N candidates, keep best by CLIP score
python scripts/generate_comparison_gallery.py \
  --subject subj01 \
  --encoder-checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
  --encoder-type two_stage \
  --num-samples 16 \
  --strategies best_of_4 best_of_8 best_of_16 \
  --output-dir outputs/galleries/best_of_n

# Compare quality vs time trade-off
```

#### BOI-Lite Refinement

```bash
# Bayesian Optimization + InfoNCE (requires encoding model)
python scripts/generate_comparison_gallery.py \
  --subject subj01 \
  --encoder-checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
  --encoder-type two_stage \
  --encoding-model-checkpoint checkpoints/encoding_model/subj01/encoding_model.pt \
  --num-samples 16 \
  --strategies single boi_lite \
  --output-dir outputs/galleries/boi_lite

# BOI-Lite parameters
#   --boi-n-init 8     # Initial candidates
#   --boi-n-refine 4   # Refinement iterations
```

### 5.4 Diffusion Model Selection

```bash
# Stable Diffusion 1.5 (faster, 768-D CLIP)
python scripts/reconstruct.py \
  --subject subj01 \
  --encoder-checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
  --encoder-type two_stage \
  --diffusion-model runwayml/stable-diffusion-v1-5 \
  --clip-adapter checkpoints/clip_adapter/sd15/adapter.pt \
  --output-dir outputs/reconstructions/sd15

# Stable Diffusion 2.1 (better quality, 1024-D CLIP)
python scripts/reconstruct.py \
  --subject subj01 \
  --encoder-checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
  --encoder-type two_stage \
  --diffusion-model stabilityai/stable-diffusion-2-1 \
  --clip-adapter checkpoints/clip_adapter/sd21/adapter.pt \
  --output-dir outputs/reconstructions/sd21

# Stable Diffusion XL (highest quality, slower)
python scripts/reconstruct.py \
  --subject subj01 \
  --encoder-checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
  --encoder-type two_stage \
  --diffusion-model stabilityai/stable-diffusion-xl-base-1.0 \
  --output-dir outputs/reconstructions/sdxl
```

### 5.5 Batch Reconstruction

```bash
# Reconstruct all test samples
python scripts/batch_reconstruct.py \
  --subject subj01 \
  --encoder-checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
  --encoder-type two_stage \
  --split test \
  --output-dir outputs/reconstructions/batch_test \
  --batch-size 8 \
  --num-inference-steps 50

# Reconstruct specific trial indices
python scripts/batch_reconstruct.py \
  --subject subj01 \
  --encoder-checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
  --encoder-type two_stage \
  --trial-indices 0,100,500,1000 \
  --output-dir outputs/reconstructions/specific
```

### 5.6 Perceptual Metrics (After Reconstruction)

```bash
# Compute LPIPS, SSIM, PixCorr on reconstructed images
python scripts/eval_perceptual.py \
  --reconstructions-dir outputs/reconstructions/subj01 \
  --ground-truth-dir cache/nsd_png/ \
  --subject subj01 \
  --output-json outputs/eval/perceptual_metrics.json

# Generate comparison plots
python scripts/plot_perceptual.py \
  --metrics-json outputs/eval/perceptual_metrics.json \
  --output-dir outputs/plots/perceptual
```

---

## 6. Ablation Studies

### 6.1 PCA Dimensionality Ablation

```bash
# Test different PCA dimensions
python scripts/ablation_pca_dims.py \
  --subject subj01 \
  --base-config configs/two_stage_sota.yaml \
  --pca-values 100 256 512 768 1024 \
  --output-dir outputs/ablations/pca_dims

# Results: outputs/ablations/pca_dims/results.csv
# Columns: pca_k, r1, r5, r10, median_rank, clip_i, train_time
```

**Expected findings:**
- k=100: Fast, lower performance
- k=512: Best speed/performance trade-off
- k=1024: Marginal gains, longer training

### 6.2 InfoNCE Weight Ablation

```bash
# Sweep InfoNCE loss weight
python scripts/ablation_infonce.py \
  --subject subj01 \
  --base-config configs/two_stage_sota.yaml \
  --infonce-weights 0.0 0.1 0.2 0.3 0.4 0.5 0.6 \
  --output-dir outputs/ablations/infonce

# Quick version (fewer epochs)
python scripts/ablation_infonce.py \
  --subject subj01 \
  --base-config configs/two_stage_sota.yaml \
  --infonce-weights 0.0 0.2 0.4 0.6 \
  --epochs 30 \
  --output-dir outputs/ablations/infonce_quick
```

**Expected findings:**
- 0.0: Baseline (MSE + cosine only)
- 0.2-0.4: Best performance
- 0.6+: Diminishing returns

### 6.3 Architecture Depth Ablation

```bash
# Test different numbers of residual blocks
python scripts/ablation_depth.py \
  --subject subj01 \
  --base-config configs/two_stage_sota.yaml \
  --n-blocks 2 3 4 6 8 \
  --output-dir outputs/ablations/depth

# Analyze parameter count vs performance
python scripts/plot_ablation.py \
  --results-csv outputs/ablations/depth/results.csv \
  --x-axis n_params \
  --y-axis r1 \
  --output outputs/ablations/depth/params_vs_r1.png
```

### 6.4 Best-of-N Ablation (Generation Only)

```bash
# Test different N values (no training required)
python scripts/ablation_best_of_n.py \
  --subject subj01 \
  --encoder-checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
  --encoder-type two_stage \
  --n-values 1 2 4 8 16 32 \
  --num-samples 50 \
  --output-dir outputs/ablations/best_of_n

# Results show quality vs compute trade-off
```

### 6.5 Learning Rate Sweep

```bash
# Grid search over learning rates
for lr in 1e-5 5e-5 1e-4 5e-4 1e-3; do
  python scripts/train_two_stage.py \
    --subject subj01 \
    --config configs/two_stage_sota.yaml \
    --learning-rate $lr \
    --output-dir outputs/ablations/lr/lr_${lr}
done

# Aggregate results
python scripts/aggregate_results.py \
  --results-dir outputs/ablations/lr \
  --output outputs/ablations/lr/summary.csv
```

### 6.6 Preprocessing Comparison

```bash
# Compare normalization methods
for norm in zscore minmax robust none; do
  python scripts/train_two_stage.py \
    --subject subj01 \
    --config configs/two_stage_sota.yaml \
    --normalization $norm \
    --output-dir outputs/ablations/norm/${norm}
done

# Compare PCA vs no PCA
python scripts/train_two_stage.py \
  --subject subj01 \
  --config configs/two_stage_sota.yaml \
  --no-pca \
  --output-dir outputs/ablations/no_pca
```

### 6.7 Automated Ablation Driver

```bash
# Run comprehensive ablation study
python scripts/ablation_driver.py \
  --subject subj01 \
  --ablation-types pca_dims infonce_weight arch_depth \
  --base-config configs/two_stage_sota.yaml \
  --output-dir outputs/ablations/comprehensive

# Dry run (see what would be executed)
python scripts/ablation_driver.py \
  --subject subj01 \
  --ablation-types pca_dims \
  --base-config configs/two_stage_sota.yaml \
  --output-dir outputs/ablations/test \
  --dry-run

# Generate consolidated report
python scripts/generate_ablation_report.py \
  --ablations-dir outputs/ablations/comprehensive \
  --output outputs/ablations/comprehensive/report.pdf
```

**Time estimates:**
- PCA dims (5 values): ~10-15 hours
- InfoNCE weight (7 values): ~14-21 hours
- Architecture depth (5 values): ~10-15 hours
- Best-of-N (generation only): ~2-4 hours

---

## 7. Reporting & Visualization

### 7.1 Generate Evaluation Reports

```bash
# Comprehensive report with LaTeX tables
python scripts/generate_report.py \
  --results-dir outputs/eval/subj01 \
  --output-dir outputs/reports/subj01 \
  --report-type full

# Output files:
#   - summary.md (Markdown overview)
#   - retrieval_table.tex (LaTeX table for papers)
#   - metrics_plot.png (Performance visualization)
```

### 7.2 Ablation Study Reports

```bash
# Create report for ablation study
python scripts/generate_report.py \
  --results-dir outputs/ablations/infonce \
  --output-dir outputs/reports/ablation_infonce \
  --report-type ablation

# Output files:
#   - ablation_table.tex (LaTeX comparison table)
#   - ablation_plot.png (Parameter vs performance)
#   - summary.md (Text summary)
```

### 7.3 Multi-Subject Comparison

```bash
# Compare performance across subjects
python scripts/compare_subjects.py \
  --checkpoints-dir checkpoints/two_stage \
  --subjects subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08 \
  --output-dir outputs/reports/cross_subject

# Generate publication-ready figures
python scripts/plot_cross_subject.py \
  --summary-csv outputs/reports/cross_subject/summary.csv \
  --output-dir outputs/reports/cross_subject/figures
```

### 7.4 Visualization Tools

#### Retrieval Results Visualization

```bash
# Plot retrieval metrics over training
python scripts/plot_training.py \
  --log-file checkpoints/two_stage/subj01/training_log.json \
  --metrics r1 r5 r10 clip_i \
  --output outputs/plots/training_curves.png

# Visualize confusion matrix
python scripts/plot_retrieval_confusion.py \
  --predictions outputs/eval/subj01/predictions.npy \
  --targets outputs/eval/subj01/targets.npy \
  --output outputs/plots/confusion.png
```

#### Embedding Space Visualization

```bash
# t-SNE visualization of CLIP embeddings
python scripts/visualize_embeddings.py \
  --embeddings-file outputs/eval/subj01/predictions.npy \
  --labels-file data/indices/nsd_index/subject=subj01/categories.csv \
  --method tsne \
  --output outputs/plots/tsne_embeddings.png

# UMAP alternative (faster)
python scripts/visualize_embeddings.py \
  --embeddings-file outputs/eval/subj01/predictions.npy \
  --method umap \
  --output outputs/plots/umap_embeddings.png
```

#### Reconstruction Quality Gallery

```bash
# Create HTML gallery with best/worst reconstructions
python scripts/create_gallery.py \
  --reconstructions-dir outputs/reconstructions/subj01 \
  --metrics-json outputs/eval/perceptual_metrics.json \
  --num-best 20 \
  --num-worst 20 \
  --output outputs/gallery/index.html

# View in browser
python -m http.server 8000 --directory outputs/gallery
# Open: http://localhost:8000
```

### 7.5 LaTeX Tables for Publications

```bash
# Generate publication-ready tables
python scripts/generate_latex_tables.py \
  --results-dir outputs/eval \
  --subjects subj01 subj02 subj03 \
  --models ridge mlp two_stage \
  --output outputs/latex/

# Output files:
#   - main_results.tex (retrieval metrics)
#   - ablation_pca.tex (PCA ablation)
#   - ablation_infonce.tex (InfoNCE ablation)
#   - cross_subject.tex (subject comparison)
```

### 7.6 Summary Statistics

```bash
# Compute summary statistics across runs
python scripts/summarize_results.py \
  --results-pattern "outputs/eval/subj*/retrieval_metrics.json" \
  --output outputs/summary_stats.csv

# Format: mean ± std for all metrics
# Example output:
# R@1: 23.7 ± 2.1
# R@5: 54.8 ± 3.4
# CLIP-I: 0.658 ± 0.012
```

---

## 8. Advanced Workflows

### 8.1 Complete Automated Pipeline

```bash
# Run everything end-to-end using Makefile
make pipeline

# This executes:
# 1. Build CLIP cache
# 2. Train encoders for all subjects
# 3. Evaluate on multiple galleries
# 4. Generate reconstructions
# 5. Create summary reports

# Estimated time: 15-20 hours
```

### 8.2 Custom Pipeline Script

```bash
#!/bin/bash
# custom_pipeline.sh - Customizable end-to-end workflow

set -e  # Exit on error

SUBJECT="subj01"
BASE_DIR="/path/to/fmri2img"
CONFIG="configs/two_stage_sota.yaml"

cd "$BASE_DIR"

echo "=== Step 1: Build CLIP Cache ==="
python scripts/build_clip_cache.py \
  --output outputs/clip_cache/clip.parquet \
  --batch-size 256

echo "=== Step 2: Train Encoder ==="
python scripts/train_two_stage.py \
  --config $CONFIG \
  --subject $SUBJECT \
  --output-dir checkpoints/two_stage/$SUBJECT

echo "=== Step 3: Evaluate ==="
python scripts/eval_comprehensive.py \
  --subject $SUBJECT \
  --encoder-checkpoint checkpoints/two_stage/$SUBJECT/two_stage_best.pt \
  --encoder-type two_stage \
  --output-dir outputs/eval/$SUBJECT

echo "=== Step 4: Generate Reconstructions ==="
python scripts/generate_comparison_gallery.py \
  --subject $SUBJECT \
  --encoder-checkpoint checkpoints/two_stage/$SUBJECT/two_stage_best.pt \
  --encoder-type two_stage \
  --num-samples 50 \
  --strategies single best_of_8 \
  --output-dir outputs/galleries/$SUBJECT

echo "=== Step 5: Create Report ==="
python scripts/generate_report.py \
  --results-dir outputs/eval/$SUBJECT \
  --output-dir outputs/reports/$SUBJECT \
  --report-type full

echo "=== Pipeline Complete! ==="
echo "Results: outputs/reports/$SUBJECT/"
```

Save and run:
```bash
chmod +x custom_pipeline.sh
./custom_pipeline.sh
```

### 8.3 Parallel Multi-Subject Training

```bash
# Train multiple subjects in parallel (requires multiple GPUs)
parallel --jobs 4 \
  python scripts/train_two_stage.py \
    --config configs/two_stage_sota.yaml \
    --subject {} \
    --device cuda:{%} \
    --output-dir checkpoints/two_stage/{} \
  ::: subj01 subj02 subj03 subj04

# Or use GNU parallel with specific GPU assignment
parallel --jobs 2 \
  CUDA_VISIBLE_DEVICES={%} python scripts/train_two_stage.py \
    --config configs/two_stage_sota.yaml \
    --subject {1} \
    --output-dir checkpoints/two_stage/{1} \
  ::: subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08
```

### 8.4 Hyperparameter Optimization

```bash
# Grid search with Optuna
python scripts/optimize_hyperparameters.py \
  --subject subj01 \
  --n-trials 50 \
  --study-name two_stage_opt \
  --output-dir outputs/optuna/two_stage

# Search space includes:
#   - learning_rate: [1e-5, 1e-3]
#   - n_blocks: [2, 8]
#   - latent_dim: [256, 1024]
#   - dropout: [0.1, 0.5]
#   - infonce_weight: [0.0, 0.6]

# View results
python scripts/plot_optuna_results.py \
  --study-dir outputs/optuna/two_stage \
  --output outputs/optuna/two_stage/plots/
```

### 8.5 Transfer Learning

```bash
# Pre-train on multiple subjects
python scripts/train_multisub ject.py \
  --subjects subj01 subj02 subj03 \
  --config configs/two_stage_sota.yaml \
  --output-dir checkpoints/multisubject/pretrained

# Fine-tune on target subject
python scripts/train_two_stage.py \
  --subject subj04 \
  --config configs/two_stage_sota.yaml \
  --pretrained-checkpoint checkpoints/multisubject/pretrained/encoder.pt \
  --freeze-layers encoder.stage1 \
  --output-dir checkpoints/two_stage/subj04_transfer
```

### 8.6 Ensemble Methods

```bash
# Train ensemble of models with different seeds
for seed in 42 123 456 789 101112; do
  python scripts/train_two_stage.py \
    --subject subj01 \
    --config configs/two_stage_sota.yaml \
    --seed $seed \
    --output-dir checkpoints/ensemble/seed_${seed}
done

# Ensemble prediction (average embeddings)
python scripts/eval_ensemble.py \
  --checkpoints checkpoints/ensemble/seed_*/two_stage_best.pt \
  --subject subj01 \
  --output-dir outputs/eval/ensemble
```

### 8.7 Continuous Monitoring

```bash
# Launch TensorBoard
tensorboard --logdir checkpoints/ --port 6006

# Launch MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# Custom logging webhook
python scripts/train_two_stage.py \
  --config configs/two_stage_sota.yaml \
  --subject subj01 \
  --webhook-url http://your-server.com/webhook \
  --output-dir checkpoints/two_stage/subj01
```

---

## 9. Troubleshooting

### 9.1 Common Errors

#### "CLIP embedding missing for nsdId=XXX"

**Cause:** CLIP cache not built or incomplete

**Solution:**
```bash
# Build complete cache
python scripts/build_clip_cache.py --output outputs/clip_cache/clip.parquet

# Verify cache
python -c "
import pandas as pd
df = pd.read_parquet('outputs/clip_cache/clip.parquet')
print(f'Cache contains {len(df):,} embeddings')
print(f'nsdId range: {df.nsdId.min()} to {df.nsdId.max()}')
"
```

#### "RuntimeError: CUDA out of memory"

**Cause:** Batch size too large for available GPU memory

**Solution:**
```bash
# Option 1: Reduce batch size
python scripts/train_two_stage.py \
  --config configs/two_stage_sota.yaml \
  --override "training.batch_size=64"  # Default is 128

# Option 2: Use gradient accumulation
python scripts/train_two_stage.py \
  --config configs/two_stage_sota.yaml \
  --batch-size 64 \
  --accumulation-steps 2  # Effective batch size: 128

# Option 3: Use mixed precision training
python scripts/train_two_stage.py \
  --config configs/two_stage_sota.yaml \
  --mixed-precision fp16

# Option 4: Use CPU (slow)
python scripts/train_two_stage.py \
  --config configs/two_stage_sota.yaml \
  --device cpu
```

#### "FileNotFoundError: nsd_stimuli.hdf5"

**Cause:** NSD dataset not downloaded

**Solution:**
```bash
# Automated download
python scripts/download_nsd_data.py --output cache/

# Or manual download from:
# http://naturalscenesdataset.org/
```

#### "ValueError: PCA components not compatible"

**Cause:** Trying to load preprocessing from different configuration

**Solution:**
```bash
# Always use same preprocessing config during training and evaluation
# Or retrain with correct preprocessing

# Clear preprocessing cache
rm -rf cache/preproc/*

# Retrain
python scripts/train_two_stage.py --config configs/two_stage_sota.yaml --subject subj01
```

#### "ImportError: No module named 'diffusers'"

**Cause:** Missing dependencies

**Solution:**
```bash
# Install diffusion dependencies
pip install diffusers transformers accelerate

# Or reinstall environment
conda env update -f environment.yml
```

### 9.2 Performance Issues

#### Slow Training

```bash
# Check data loading bottleneck
python scripts/profile_dataloading.py --subject subj01

# Solutions:
# 1. Increase num_workers
#    --override "data.num_workers=8"
#
# 2. Use SSD for cache
#    --cache-dir /path/to/ssd/cache
#
# 3. Preload data to RAM
#    --preload-data
```

#### Slow CLIP Cache Building

```bash
# Increase batch size (requires more VRAM)
python scripts/build_clip_cache.py \
  --batch-size 512  # Default: 256

# Use faster CLIP model (lower quality)
python scripts/build_clip_cache.py \
  --clip-model ViT-B/32  # Default: ViT-L/14
```

### 9.3 Debugging Tools

```bash
# Enable debug logging
python scripts/train_two_stage.py \
  --config configs/two_stage_sota.yaml \
  --log-level DEBUG

# Profile memory usage
python -m memory_profiler scripts/train_two_stage.py \
  --config configs/two_stage_sota.yaml \
  --limit 100

# Check for NaN/Inf in training
python scripts/train_two_stage.py \
  --config configs/two_stage_sota.yaml \
  --detect-anomaly

# Visualize gradient flow
python scripts/train_two_stage.py \
  --config configs/two_stage_sota.yaml \
  --plot-gradients \
  --output-dir checkpoints/debug
```

### 9.4 Verification Checks

```bash
# Run all tests
python scripts/test_full_workflow.py && \
python scripts/test_e2e_integration.py && \
python scripts/test_extended_components.py --test-real-data

# Verify data integrity
python scripts/verify_data.py \
  --cache-root cache \
  --clip-cache outputs/clip_cache/clip.parquet

# Check model outputs
python scripts/check_model_outputs.py \
  --checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
  --subject subj01 \
  --num-samples 10
```

---

## 10. Quick Reference

### 10.1 Essential Commands

| Task | Command | Time |
|------|---------|------|
| **Installation check** | `python scripts/test_full_workflow.py` | 2s |
| **Build CLIP cache** | `python scripts/build_clip_cache.py --output outputs/clip_cache/clip.parquet` | 2-3h |
| **Train Ridge baseline** | `python scripts/train_ridge.py --subject subj01 --config configs/ridge_baseline.yaml` | 5min |
| **Train MLP** | `python scripts/train_mlp.py --subject subj01 --config configs/mlp_standard.yaml` | 2h |
| **Train Two-Stage (SOTA)** | `python scripts/train_two_stage.py --subject subj01 --config configs/two_stage_sota.yaml` | 6-8h |
| **Evaluate** | `python scripts/eval_comprehensive.py --subject subj01 --encoder-checkpoint <path> --encoder-type two_stage` | 10min |
| **Generate images** | `python scripts/generate_comparison_gallery.py --subject subj01 --encoder-checkpoint <path> --encoder-type two_stage --num-samples 16` | 5-40min |
| **Full pipeline** | `make pipeline` | 15-20h |

### 10.2 Config File Cheat Sheet

```yaml
# Quick reference for two_stage_sota.yaml parameters

dataset:
  subject: subj01           # Subject ID (subj01-08)
  train_ratio: 0.80         # Training split (0.0-1.0)

preprocessing:
  pca_k: 512                # PCA dimensions (100-1024)
  reliability_threshold: 0.1 # Voxel filtering (0.0-1.0)

encoder:
  latent_dim: 768           # Latent dimensions (256-1024)
  n_blocks: 4               # Residual blocks (2-8)
  dropout: 0.3              # Dropout rate (0.0-0.5)

training:
  batch_size: 128           # Batch size (32-256)
  learning_rate: 1e-4       # Learning rate (1e-5 to 1e-3)
  epochs: 100               # Max epochs (50-200)
  early_stopping_patience: 10 # Early stopping (5-20)
```

### 10.3 File Structure Reference

```
fmri2img/
├── cache/                          # Downloaded NSD data
│   ├── nsd_hdf5/nsd_stimuli.hdf5  # 73K images (39GB)
│   └── betas/subj01/               # fMRI data
├── data/indices/nsd_index/         # Train/val/test splits
│   └── subject=subj01/*.parquet
├── outputs/
│   ├── clip_cache/clip.parquet     # CLIP embeddings (500MB)
│   ├── eval/subj01/*.json          # Evaluation metrics
│   ├── reconstructions/subj01/*.png # Generated images
│   └── galleries/subj01/*.png      # Comparison grids
├── checkpoints/
│   ├── ridge/subj01/               # Ridge checkpoints
│   ├── mlp/subj01/                 # MLP checkpoints
│   └── two_stage/subj01/           # Two-Stage checkpoints
│       ├── two_stage_best.pt       # Best model
│       ├── two_stage_last.pt       # Latest model
│       ├── training_log.json       # Training history
│       └── config.yaml             # Saved config
└── logs/                           # Training logs
```

### 10.4 Expected Outputs

#### After Training
```json
{
  "epoch": 100,
  "train_loss": 0.234,
  "val_loss": 0.312,
  "train_clip_i": 0.721,
  "val_clip_i": 0.658,
  "train_r1": 0.312,
  "val_r1": 0.237
}
```

#### After Evaluation
```json
{
  "r1": 0.237,
  "r5": 0.548,
  "r10": 0.714,
  "r20": 0.823,
  "r50": 0.912,
  "r100": 0.956,
  "mean_rank": 47.3,
  "median_rank": 12,
  "mrr": 0.412,
  "clip_i": 0.658
}
```

### 10.5 Performance Targets

| Model | R@1 | R@5 | R@10 | Median Rank | CLIP-I | Training Time |
|-------|-----|-----|------|-------------|--------|---------------|
| Ridge | 12% | 39% | 56% | 187 | 0.52 | 5 min |
| MLP | 19% | 47% | 64% | 92 | 0.61 | 2 hours |
| Two-Stage | 24% | 55% | 71% | 47 | 0.66 | 6-8 hours |

*Test set (3K gallery), subj01*

### 10.6 Resource Requirements

| Task | GPU VRAM | RAM | Storage | Time |
|------|----------|-----|---------|------|
| **CLIP cache build** | 6GB | 8GB | 1GB | 2-3h |
| **Ridge training** | - | 16GB | 500MB | 5min |
| **MLP training** | 6GB | 16GB | 1GB | 2h |
| **Two-Stage training** | 8-12GB | 16GB | 2GB | 6-8h |
| **Evaluation** | 2GB | 8GB | 500MB | 10min |
| **Image generation** | 6-8GB | 16GB | 2GB | 3-6s/image |

### 10.7 Useful Shell Aliases

```bash
# Add to ~/.bashrc or ~/.zshrc

alias fmri-train="python scripts/train_two_stage.py --config configs/two_stage_sota.yaml"
alias fmri-eval="python scripts/eval_comprehensive.py"
alias fmri-test="python scripts/test_full_workflow.py && python scripts/test_e2e_integration.py"
alias fmri-cache="python scripts/build_clip_cache.py --output outputs/clip_cache/clip.parquet"
alias fmri-gpu="watch -n 1 nvidia-smi"

# Usage:
# fmri-train --subject subj01 --output-dir checkpoints/two_stage/subj01
# fmri-eval --subject subj01 --encoder-checkpoint checkpoints/two_stage/subj01/two_stage_best.pt --encoder-type two_stage
```

### 10.8 Environment Variables

```bash
# Set default paths
export FMRI_CACHE_ROOT="cache"
export FMRI_CLIP_CACHE="outputs/clip_cache/clip.parquet"
export FMRI_INDEX_ROOT="data/indices/nsd_index"
export FMRI_CHECKPOINT_DIR="checkpoints"

# Set GPU preferences
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Set parallel processing
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

---

## 📚 Additional Resources

### Documentation
- [Main README](README.md) - Overview and installation
- [START_HERE.md](START_HERE.md) - Quick start guide
- [docs/QUICK_START.md](docs/QUICK_START.md) - Detailed tutorial
- [docs/COMPLETE_TEST_SUITE.md](docs/COMPLETE_TEST_SUITE.md) - Testing guide

### Papers & References
- [Natural Scenes Dataset](http://naturalscenesdataset.org/)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [Stable Diffusion Paper](https://arxiv.org/abs/2112.10752)
- [Two-Stage Architecture Guide](docs/ARCHITECTURE.md)

### Community
- [GitHub Issues](https://github.com/yourusername/fmri2img/issues)
- [Discussions](https://github.com/yourusername/fmri2img/discussions)

---

## ⏱️ Time Budget Planning

### Quick Test (1 hour)
```bash
# Verify everything works
python scripts/test_full_workflow.py  # 2s
python scripts/train_mlp.py --limit 100 --epochs 10  # 5min
python scripts/eval_comprehensive.py --max-samples 50  # 2min
```

### Half-Day Experiment (4 hours)
```bash
# Build mini cache + train MLP
python scripts/build_clip_cache.py --limit 5000  # 30min
python scripts/train_mlp.py --epochs 50  # 2h
python scripts/eval_comprehensive.py  # 10min
python scripts/generate_comparison_gallery.py --num-samples 16 --strategies single  # 5min
```

### Full Day (8-10 hours)
```bash
# Train SOTA + evaluate + generate
python scripts/build_clip_cache.py  # 2-3h
python scripts/train_two_stage.py --config configs/two_stage_sota.yaml  # 6-8h
python scripts/eval_comprehensive.py  # 10min
python scripts/generate_comparison_gallery.py --strategies single best_of_8  # 40min
```

### Week-Long Study (40 hours)
```bash
# Multi-subject + ablations + comprehensive evaluation
make pipeline  # 15-20h for 3 subjects
# + Ablation studies: 20h
# + Analysis and reporting: 5h
```

---

<div align="center">

**🎓 Ready to decode the visual brain!**

For questions, issues, or contributions, visit our [GitHub repository](https://github.com/yourusername/fmri2img)

*Last updated: December 7, 2025*

</div>
