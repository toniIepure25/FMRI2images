# 🚀 Quick Start Guide

> **Get up and running with the Brain-to-Image reconstruction system in minutes**

This guide provides a streamlined path to training and evaluating neural decoders for visual reconstruction from fMRI data. For detailed documentation, see the [main README](README.md).

---

## 📋 Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.10+ | 3.11+ |
| **GPU** | 6GB VRAM (GTX 1060) | 24GB VRAM (RTX 3090/4090) |
| **RAM** | 16GB | 32GB+ |
| **Storage** | 50GB free | 200GB+ free |
| **OS** | Linux/macOS | Linux (Ubuntu 20.04+) |

### Environment Setup

```bash
# Clone repository
git clone https://github.com/yourusername/fmri2img.git
cd fmri2img

# Create and activate conda environment
conda env create -f environment.yml
conda activate fmri2img

# Verify installation
python scripts/test_full_workflow.py  # Should complete in ~2 seconds
```

---

## 🎯 Choose Your Path

### Path 1: Quick Test (15 minutes)
**Recommended for first-time users**

Test the complete pipeline with a small dataset before committing to full training:

```bash
# 1. Run automated tests
python scripts/test_full_workflow.py && \
python scripts/test_e2e_integration.py && \
python scripts/test_extended_components.py --test-real-data

# 2. Train on 100 samples (5 minutes)
python scripts/train_mlp.py \
  --subject subj01 \
  --limit 100 \
  --epochs 10 \
  --output-dir checkpoints/test

# 3. Verify outputs
ls -lh checkpoints/test/
```

### Path 2: Full Pipeline (10-15 hours)
**For production training and evaluation**

Complete end-to-end workflow with full NSD dataset:

```bash
# Use the automated Makefile
make pipeline

# Or run step-by-step (see Section below)
```

### Path 3: Use Pre-trained Models
**Coming soon - skip training entirely**

```bash
# Download pre-trained checkpoints
python scripts/download_pretrained.py --subject subj01 --model two_stage

# Run evaluation immediately
python scripts/eval_comprehensive.py \
  --subject subj01 \
  --encoder-checkpoint checkpoints/pretrained/subj01_two_stage.pt \
  --encoder-type two_stage
```

---

## 📦 Step-by-Step Workflow

### Step 1: Data Preparation (2-3 hours, one-time)

#### Download NSD Data

```bash
# Option A: Automated download (recommended)
python scripts/download_nsd_data.py --output cache/

# Option B: Manual download from NSD website
# Visit: http://naturalscenesdataset.org/
# Download: nsd_stimuli.hdf5 (39GB) → cache/nsd_hdf5/
```

#### Build Subject Index

```bash
# Create trial indices with train/val/test splits
python scripts/build_full_index.py \
  --cache-root cache \
  --subject subj01 \
  --output data/indices/nsd_index/

# Verify output
ls -lh data/indices/nsd_index/
# Should contain: subject=subj01/trial_*.parquet
```

#### Build CLIP Cache

```bash
# Extract CLIP embeddings for all 73K NSD images
python scripts/build_clip_cache.py \
  --cache-root cache \
  --output outputs/clip_cache/clip.parquet \
  --batch-size 256

# Time: ~2-3 hours on GPU
# Output: ~500MB parquet file with 73,000 embeddings
```

**Resumable:** If interrupted, simply rerun the same command - it will skip already-processed images.

**Progress monitoring:**
```bash
# Watch cache grow
watch -n 10 "python -c 'import pandas as pd; print(len(pd.read_parquet(\"outputs/clip_cache/clip.parquet\")))"
```

### Step 2: Training (2-8 hours depending on model)

#### Train Ridge Baseline (Fast)

```bash
python scripts/train_ridge.py \
  --subject subj01 \
  --config configs/ridge_baseline.yaml \
  --output-dir checkpoints/ridge/subj01

# Time: ~5 minutes
# Use: Quick baseline for comparison
```

#### Train MLP Encoder (Standard)

```bash
python scripts/train_mlp.py \
  --subject subj01 \
  --config configs/mlp_standard.yaml \
  --output-dir checkpoints/mlp/subj01

# Time: ~2 hours
# Use: Strong baseline, good speed/performance trade-off
```

#### Train Two-Stage Encoder (SOTA)

```bash
python scripts/train_two_stage.py \
  --subject subj01 \
  --config configs/two_stage_sota.yaml \
  --output-dir checkpoints/two_stage/subj01

# Time: ~6-8 hours
# Use: Best performance, recommended for research
```

**Monitor training:**
```bash
# Watch logs in real-time
tail -f logs/train_two_stage_subj01.log

# Check GPU usage
nvidia-smi -l 1
```

### Step 3: Evaluation (10-30 minutes)

```bash
# Comprehensive evaluation with multiple galleries
python scripts/eval_comprehensive.py \
  --subject subj01 \
  --encoder-checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
  --encoder-type two_stage \
  --output-dir outputs/eval/subj01

# Quick evaluation (test set only)
python scripts/eval_comprehensive.py \
  --subject subj01 \
  --encoder-checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
  --encoder-type two_stage \
  --max-samples 100 \
  --output-dir outputs/eval/test
```

**Output metrics:**
- Retrieval: R@1, R@5, R@10, R@20, R@50, R@100
- Ranking: Mean/median rank, MRR
- Similarity: CLIP-I score (cosine similarity)

### Step 4: Image Reconstruction (Optional, 5-60 minutes)

```bash
# Generate reconstructions with Stable Diffusion
python scripts/generate_comparison_gallery.py \
  --subject subj01 \
  --encoder-checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
  --encoder-type two_stage \
  --output-dir outputs/galleries/subj01 \
  --num-samples 16 \
  --strategies single best_of_8 \
  --num-inference-steps 50

# Time: ~5 min (single) or ~40 min (best-of-8)
```

**Note:** Requires Stable Diffusion model (~5GB). First run will download automatically.

---

## 🔧 Configuration

### Using Config Files (Recommended)

```bash
# Use predefined configurations
python scripts/train_two_stage.py \
  --config configs/two_stage_sota.yaml \
  --subject subj01
```

### Runtime Overrides

```bash
# Override specific parameters
python scripts/train_two_stage.py \
  --config configs/two_stage_sota.yaml \
  --override "training.learning_rate=5e-5" \
  --override "encoder.n_blocks=6"
```

### Key Configuration Parameters

```yaml
# configs/two_stage_sota.yaml
dataset:
  subject: subj01
  train_ratio: 0.80    # 80% train, 10% val, 10% test

preprocessing:
  pca_k: 512           # PCA dimensions (100-1024)
  reliability_threshold: 0.1

encoder:
  latent_dim: 768      # Latent representation size
  n_blocks: 4          # Residual blocks (2-8)
  dropout: 0.3         # Regularization

training:
  batch_size: 128      # Adjust based on GPU memory
  learning_rate: 1e-4
  epochs: 100
  early_stopping_patience: 10
```

---

## 📊 Expected Performance

### Retrieval Metrics (subj01, Test Set)

| Encoder | R@1 | R@5 | R@10 | Median Rank | CLIP-I |
|---------|-----|-----|------|-------------|--------|
| Ridge | 12.3% | 38.7% | 56.2% | 187 | 0.524 |
| MLP | 18.9% | 47.3% | 64.1% | 92 | 0.612 |
| Two-Stage | 23.7% | 54.8% | 71.4% | 47 | 0.658 |

*Gallery: 3,000 test images*

### Training Convergence

- **Ridge**: Instant (closed-form solution)
- **MLP**: 30-50 epochs (~2 hours)
- **Two-Stage**: 50-80 epochs (~6-8 hours)

---

## 🐛 Troubleshooting

### Common Issues

#### 1. "CLIP embedding missing for nsdId=XXX"
**Cause:** CLIP cache not built or incomplete  
**Solution:**
```bash
python scripts/build_clip_cache.py --output outputs/clip_cache/clip.parquet
```

#### 2. "CUDA out of memory"
**Cause:** Batch size too large for GPU  
**Solution:**
```bash
# Reduce batch size in config or via override
--override "training.batch_size=64"
```

#### 3. "FileNotFoundError: nsd_stimuli.hdf5"
**Cause:** NSD data not downloaded  
**Solution:**
```bash
python scripts/download_nsd_data.py --output cache/
```

#### 4. Slow training on CPU
**Cause:** No GPU detected  
**Solution:**
```bash
# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Force GPU device
--device cuda
```

### Getting Help

- **Documentation**: See [docs/](docs/) for detailed guides
- **Issues**: [GitHub Issues](https://github.com/yourusername/fmri2img/issues)
- **Tests**: Run `python scripts/test_full_workflow.py` to verify installation

---

## 📚 Next Steps

### For Researchers
1. **Run ablations**: Test different hyperparameters (see [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md))
2. **Multi-subject analysis**: Train on all 8 NSD subjects
3. **Custom architectures**: Extend `src/fmri2img/models/`

### For Developers
1. **Read architecture docs**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
2. **Review test suite**: [docs/COMPLETE_TEST_SUITE.md](docs/COMPLETE_TEST_SUITE.md)
3. **Contributing guide**: [CONTRIBUTING.md](CONTRIBUTING.md)

### For Quick Results
1. **Use Makefile**: `make pipeline` for full automation
2. **Download pretrained**: Skip training entirely (coming soon)
3. **Try examples**: [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) for ready-to-run commands

---

## 🎓 Learning Resources

- **NSD Dataset**: [Official Documentation](http://naturalscenesdataset.org/)
- **CLIP Paper**: [Radford et al., 2021](https://arxiv.org/abs/2103.00020)
- **Stable Diffusion**: [Rombach et al., 2022](https://arxiv.org/abs/2112.10752)
- **Our Technical Docs**: [docs/](docs/)

---

## ⚡ Quick Commands Reference

```bash
# Installation check
python scripts/test_full_workflow.py

# Build CLIP cache (one-time, 2-3 hours)
python scripts/build_clip_cache.py --output outputs/clip_cache/clip.parquet

# Train SOTA model (6-8 hours)
python scripts/train_two_stage.py --config configs/two_stage_sota.yaml --subject subj01

# Evaluate (10 minutes)
python scripts/eval_comprehensive.py --subject subj01 \
  --encoder-checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
  --encoder-type two_stage

# Generate images (5-40 minutes)
python scripts/generate_comparison_gallery.py --subject subj01 \
  --encoder-checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \
  --encoder-type two_stage --num-samples 16

# Full automated pipeline
make pipeline
```

---

<div align="center">

**Ready to reconstruct the brain? Let's go! 🧠→🖼️**

[Full Documentation](README.md) • [Usage Examples](USAGE_EXAMPLES.md) • [API Reference](docs/API.md)

</div>
