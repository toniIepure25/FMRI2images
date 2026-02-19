# Running Experiments Guide

> **Complete guide for training, evaluating, and analyzing experiments**

---

## 🚀 Quick Start

### Single Experiment

```bash
# Activate environment
source activate_env.sh  # or: conda activate fmri2img

# Train baseline model
python scripts/train.py --config configs/experiments/exp0_baseline.yaml

# Monitor training
tensorboard --logdir outputs/
```

### Full Ablation Study

```bash
# Build preprocessors (required once, 1-2 hours)
bash scripts/build_all_preprocessors.sh

# Run all experiments (3-7 days on A100)
bash scripts/run_all_experiments.sh 0

# Results saved to: experimental_results/
```

---

## 📋 Experiment Configurations

### Overview of Experiments

| Experiment | Description | Novel Feature | Expected Result |
|------------|-------------|---------------|-----------------|
| **EXP0** | Baseline | None | ~0.1% Top-1 accuracy |
| **EXP1** | + Preprocessing | Center + PCR (k=8) | 50-100x improvement |
| **EXP2** | + Memory queue | MoCo-style (Q=8192) | Moderate improvement |
| **EXP3** | + Gaussian NLL | Heteroscedastic regression | Similar retrieval, poor calibration |
| **EXP4** | + Gaussian-NCE | Distribution-aware loss ⭐ | Best retrieval + calibration |
| **EXP5** | + KL annealing | Free-bits regularization | Similar to EXP4 |
| **EXP6** | Ablation | Whitening vs PCR | Compare preprocessing |

⭐ = Novel contribution

### Config Files

All configs located in `configs/experiments/`:
- `exp0_baseline.yaml`
- `exp1_preproc.yaml`
- `exp2_queue.yaml`
- `exp3_gaussian_nll.yaml`
- `exp4_gaussian_nce.yaml` ⭐
- `exp5_kl_anneal.yaml`
- `exp6_whiten.yaml`

---

## 🎯 Training Workflow

### Step 1: Prepare Environment

```bash
# Check system health
bash scripts/preflight.sh

# Expected output:
# ✅ Python environment
# ✅ PyTorch + CUDA
# ✅ GPU available (20GB VRAM)
# ✅ Disk space (150GB free)
# ✅ Data files present
```

### Step 2: Build Preprocessors (First Time Only)

```bash
# Build all preprocessing caches
bash scripts/build_all_preprocessors.sh

# This creates:
# - cache/preproc_center_pcr_k8/*.pt    (~500MB)
# - cache/preproc_center_whiten/*.pt    (~500MB)
```

**Time:** 1-2 hours  
**Required:** Only once per dataset

### Step 3: Run Training

#### Option A: Single Experiment

```bash
python scripts/train.py \
  --config configs/experiments/exp0_baseline.yaml \
  --output_dir outputs/exp0_baseline \
  --device cuda \
  --num_workers 4
```

#### Option B: All Experiments (Batch)

```bash
# Run experiments sequentially
bash scripts/run_all_experiments.sh 0

# Logs saved to:
# experimental_results/logs/run_all_experiments_YYYYMMDD_HHMMSS.log
```

#### Option C: Specific Experiments

```bash
# Run only EXP1, EXP2, EXP4
for exp in exp1_preproc exp2_queue exp4_gaussian_nce; do
  python scripts/train.py --config configs/experiments/${exp}.yaml
done
```

### Step 4: Monitor Training

#### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir outputs/ --port 6006

# Access at: http://localhost:6006
```

#### Check Logs

```bash
# Follow live training log
tail -f outputs/exp0_baseline/train.log

# View specific metrics
grep "Top-1" outputs/exp0_baseline/train.log
```

---

## 📊 Training Parameters

### Key Hyperparameters

```yaml
# Model architecture
encoder:
  hidden_dims: [2048, 1024, 512]  # MLP layers
  dropout: 0.1
  use_layer_norm: true

# Training
batch_size: 32                     # Adjust based on GPU
learning_rate: 0.0001
num_epochs: 50
warmup_epochs: 5

# Optimization
optimizer: adamw
weight_decay: 0.01
gradient_clip: 1.0

# Loss function
loss_type: "infonce"               # or "gaussian_nll", "gaussian_nce"
temperature: 0.07                  # For contrastive losses
```

### GPU Memory Management

| GPU VRAM | Batch Size | Training Time (50 epochs) |
|----------|------------|---------------------------|
| 8GB | 8 | ~24 hours |
| 16GB | 16 | ~16 hours |
| 20GB (A100) | 32 | ~12 hours |
| 24GB (RTX 3090) | 64 | ~8 hours |

**Out of memory?** Reduce batch size in config:
```bash
nano configs/experiments/exp0_baseline.yaml
# Change: batch_size: 32 → batch_size: 8
```

---

## 🧪 Evaluation

### During Training (Automatic)

Validation runs every epoch:
- Retrieval metrics (Top-1, Top-5, Top-10)
- Mean/Median rank
- Two-way identification
- Loss values

Results saved to: `outputs/{experiment}/metrics.json`

### After Training

#### Standard Evaluation

```bash
python scripts/evaluate.py \
  --checkpoint outputs/exp0_baseline/best.pt \
  --test_split test \
  --output_dir experimental_results/exp0_baseline
```

#### Probabilistic Evaluation (for EXP3-6)

```bash
python scripts/evaluate_probabilistic.py \
  --checkpoint outputs/exp4_gaussian_nce/best.pt \
  --test_split test \
  --num_samples 100 \
  --output_dir experimental_results/exp4_gaussian_nce
```

### Evaluation Metrics

#### Embedding Metrics (All experiments)
- **Retrieval@K:** Top-1, Top-5, Top-10 accuracy
- **Ranking:** Mean rank, Median rank, MRR
- **2AFC:** Two-way identification with bootstrap CI
- **Separability:** ROC AUC, Cohen's d
- **RSA:** Representational Similarity Analysis
- **Collapse:** Per-dim std, pairwise similarity

#### Bayesian Metrics (EXP3-6 only)
- **Proper scoring:** Gaussian NLL, Energy Score
- **Bayesian retrieval:** Distribution-aware ranking
- **Probabilistic 2AFC:** Uncertainty propagation
- **Calibration:** Reliability diagrams, χ² test
- **Risk-coverage:** Selective prediction, AURC
- **Conformal:** Distribution-free UQ

---

## 📈 Result Analysis

### Compare Experiments

```bash
# Generate comparison table
python scripts/compare_experiments.py \
  --experiments exp0,exp1,exp2,exp3,exp4,exp5,exp6 \
  --metrics top1,top5,mrr,auc \
  --output experiment_comparison/

# Output:
# - comparison_table.csv
# - comparison_table.tex
# - statistical_summary.json
```

### Visualizations

```bash
# Generate all plots
python scripts/visualize_results.py \
  --experiments exp0,exp1,exp2,exp3,exp4,exp5,exp6 \
  --output_dir figures/

# Creates:
# - retrieval_curves.pdf
# - calibration_plots.pdf
# - risk_coverage.pdf
# - ablation_comparison.pdf
```

### Statistical Tests

```bash
# Run significance tests
python scripts/statistical_tests.py \
  --baseline exp0_baseline \
  --experiments exp1,exp2,exp3,exp4,exp5,exp6 \
  --alpha 0.05

# Output:
# - p-values for each comparison
# - Effect sizes (Cohen's d)
# - Confidence intervals
```

---

## 🔧 Advanced Usage

### Resume Training

```bash
# Automatically resumes from latest checkpoint
python scripts/train.py \
  --config configs/experiments/exp0_baseline.yaml \
  --resume

# Or specify checkpoint explicitly
python scripts/train.py \
  --config configs/experiments/exp0_baseline.yaml \
  --resume_from outputs/exp0_baseline/epoch_25.pt
```

### Debug Mode

```bash
# Run with minimal data for quick debugging
python scripts/train.py \
  --config configs/experiments/exp0_baseline.yaml \
  --debug \
  --max_steps 10
```

### Distributed Training

```bash
# Multi-GPU training
torchrun --nproc_per_node=4 scripts/train.py \
  --config configs/experiments/exp0_baseline.yaml \
  --distributed
```

### Custom Configuration

```bash
# Override config parameters from command line
python scripts/train.py \
  --config configs/experiments/exp0_baseline.yaml \
  --batch_size 64 \
  --learning_rate 0.0002 \
  --num_epochs 100
```

---

## 📝 Output Structure

### Training Outputs

```
outputs/exp0_baseline/
├── config.yaml              # Saved configuration
├── train.log                # Training logs
├── metrics.json             # Validation metrics
├── checkpoints/
│   ├── epoch_01.pt
│   ├── epoch_02.pt
│   └── ...
├── best.pt                  # Best model (highest Top-1)
└── tensorboard/
    └── events.out.tfevents.*
```

### Evaluation Outputs

```
experimental_results/exp0_baseline/
├── embedding_metrics.json    # Retrieval, 2AFC, RSA, etc.
├── probabilistic_metrics.json # Bayesian metrics (if applicable)
├── predictions.npz           # Saved predictions
├── figures/
│   ├── retrieval_curve.pdf
│   ├── calibration.pdf
│   └── risk_coverage.pdf
└── summary_report.md         # Human-readable summary
```

---

## ⏱️ Time Estimates

### On JupyterHub Cluster (A100-20GB)

| Task | Time | GPU Usage |
|------|------|-----------|
| Build preprocessors | 1-2 hours | Minimal |
| Single experiment (50 epochs) | 12-16 hours | 100% |
| Full ablation (7 experiments) | 3-5 days | 100% |
| Evaluation (per experiment) | 15-30 minutes | 50% |

### On Local Workstation (RTX 3090)

| Task | Time | GPU Usage |
|------|------|-----------|
| Build preprocessors | 2-3 hours | Minimal |
| Single experiment (50 epochs) | 16-24 hours | 100% |
| Full ablation (7 experiments) | 5-7 days | 100% |
| Evaluation (per experiment) | 20-40 minutes | 50% |

---

## 🐛 Troubleshooting

### Training Issues

**Issue: Out of GPU memory**
```bash
# Solution: Reduce batch size
nano configs/experiments/exp0_baseline.yaml
# Change: batch_size: 32 → batch_size: 8
```

**Issue: Training stuck / not converging**
```bash
# Check learning rate
grep "learning_rate" outputs/*/train.log

# Try lower learning rate
python scripts/train.py --config ... --learning_rate 0.00005
```

**Issue: NaN losses**
```bash
# Enable gradient clipping (should be on by default)
# Check config:
nano configs/experiments/exp0_baseline.yaml
# Ensure: gradient_clip: 1.0
```

### Data Issues

**Issue: Data files not found**
```bash
# Verify data presence
bash scripts/preflight.sh

# Re-download if needed
bash download_nsd_subj01.sh
```

**Issue: Preprocessor not found**
```bash
# Rebuild preprocessors
bash scripts/build_all_preprocessors.sh
```

### Checkpoint Issues

**Issue: Can't load checkpoint**
```bash
# Inspect checkpoint
python inspect_checkpoint.py outputs/exp0_baseline/best.pt

# If corrupted, use earlier epoch
python scripts/train.py --config ... --resume_from outputs/exp0_baseline/epoch_48.pt
```

---

## 📖 Additional Resources

- **Setup guide:** [docs/guides/SETUP.md](SETUP.md)
- **Evaluation details:** [docs/guides/EVALUATION_SUITE_GUIDE.md](EVALUATION_SUITE_GUIDE.md)
- **Novel contributions:** [docs/guides/NOVEL_CONTRIBUTIONS_PIPELINE.md](NOVEL_CONTRIBUTIONS_PIPELINE.md)
- **Architecture:** [docs/architecture/PIPELINE_ARCHITECTURE.md](../architecture/PIPELINE_ARCHITECTURE.md)
- **Troubleshooting:** [SETUP_TROUBLESHOOTING.md](../../SETUP_TROUBLESHOOTING.md)
