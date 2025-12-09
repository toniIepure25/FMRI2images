# Training Configurations

**Model Training Settings for fMRI-to-Image Reconstruction**

---

## 📚 Available Configurations

### **Baseline Models**

#### `ridge_baseline.yaml` - Ridge Regression Baseline
```yaml
Purpose: Fast baseline for comparison
Architecture: Linear regression with PCA preprocessing
Training Time: ~5 minutes
GPU Memory: N/A (CPU only)
Performance: Cosine Similarity 0.25-0.35
```

**When to use**:
- Quick experiments and prototyping
- Baseline comparison
- Debugging data pipeline
- Limited computational resources

**Command**:
```bash
python scripts/train_ridge.py \
    --config configs/training/ridge_baseline.yaml \
    --subject subj01
```

---

### **Standard Models**

#### `mlp_standard.yaml` - Multi-Layer Perceptron (Standard)
```yaml
Purpose: Standard neural encoder
Architecture: Single hidden layer (256-D)
Training Time: ~2 hours
GPU Memory: 8GB VRAM
Performance: Cosine Similarity 0.35-0.45
```

**Features**:
- Dropout regularization (0.1)
- Cosine learning rate schedule
- Early stopping (patience=10)
- Batch size: 32

**When to use**:
- Standard training workflows
- Good performance/speed tradeoff
- Limited GPU memory
- Initial experiments before scaling

**Command**:
```bash
python scripts/train_mlp.py \
    --config configs/training/mlp_standard.yaml \
    --subject subj01
```

**Tuning tips**:
```bash
# Increase capacity
--override "model.hidden_dim=512"

# Adjust learning rate
--override "training.learning_rate=0.0005"

# Longer training
--override "training.epochs=200"
```

---

### **State-of-the-Art Models**

#### `two_stage_sota.yaml` ⭐ - Two-Stage Encoder (SOTA)
```yaml
Purpose: Best performance, state-of-the-art
Architecture: 4 residual blocks, 768-D latent
Training Time: ~4 hours
GPU Memory: 12GB VRAM
Performance: Cosine Similarity 0.45-0.55
Retrieval R@1: ~24%
```

**Features**:
- Multi-layer supervision
- InfoNCE contrastive loss (weight 0.4)
- Residual connections
- Layer normalization
- Advanced data augmentation

**When to use**:
- Final results and publication
- Maximum performance required
- Sufficient GPU resources available
- After hyperparameter tuning

**Command**:
```bash
python scripts/train_two_stage.py \
    --config configs/training/two_stage_sota.yaml \
    --subject subj01
```

**Advanced usage**:
```bash
# Higher quality (longer training)
--override "training.epochs=300" \
--override "training.patience=20"

# Stronger regularization
--override "model.dropout=0.2" \
--override "training.weight_decay=0.0001"

# Multi-GPU training
--override "training.distributed=true"
```

#### `sota_two_stage.yaml` (DEPRECATED)
Legacy naming. Use `two_stage_sota.yaml` instead.

---

### **Specialized Models**

#### `adapter_vitl14.yaml` - CLIP Adapter
```yaml
Purpose: Adapt fMRI→CLIP for diffusion models
Architecture: 512-D → 768-D adaptation layer
Training Time: ~30 minutes
GPU Memory: 6GB VRAM
Performance: N/A (task-specific)
```

**Features**:
- Dimension adaptation (512→768 or 512→1024)
- Compatible with Stable Diffusion 1.5/2.1
- Fast convergence
- Lightweight architecture

**When to use**:
- After training main encoder
- For diffusion-based image generation
- When using CLIP-based metrics

**Command**:
```bash
python scripts/train_clip_adapter.py \
    --config configs/training/adapter_vitl14.yaml \
    --pretrained checkpoints/two_stage/best.pt
```

#### `clip2fmri.yaml` - Inverse Mapping (Research)
```yaml
Purpose: Research tool for brain-consistency loss
Architecture: CLIP → fMRI (inverse direction)
Training Time: ~1 hour
GPU Memory: 8GB VRAM
Use Case: Ablation studies, research
```

**When to use**:
- Brain-consistency loss experiments
- Ablation studies
- Understanding CLIP→brain mapping

---

### **Development**

#### `dev_fast.yaml` - Fast Development/Debugging
```yaml
Purpose: Quick iteration and debugging
Training Time: 2-5 minutes
Samples: 1,000 (reduced)
Epochs: 10 (reduced)
```

**Features**:
- Reduced dataset (1K samples)
- Short epochs (10)
- Fast validation
- Minimal logging

**When to use**:
- Debugging code
- Testing new features
- Rapid iteration
- Sanity checks

**Command**:
```bash
python scripts/train_mlp.py \
    --config configs/training/dev_fast.yaml \
    --subject subj01
```

**⚠️ Note**: Not for final results. Performance will be poor due to limited data/epochs.

---

## 📊 Performance Comparison

| Config | R@1 | R@5 | R@10 | Cosine Sim | Time | Memory |
|--------|-----|-----|------|------------|------|--------|
| `ridge_baseline` | 8.2% | 18.5% | 27.3% | 0.28 | 5 min | 4GB RAM |
| `mlp_standard` | 15.7% | 32.4% | 44.1% | 0.41 | 2 hrs | 8GB VRAM |
| `two_stage_sota` ⭐ | 24.3% | 45.8% | 58.2% | 0.52 | 4 hrs | 12GB VRAM |

*Benchmarks on full NSD dataset (30K samples), NVIDIA A100 40GB*

---

## 🎯 Recommended Workflow

### **1. Start with Ridge Baseline** (5 min)
```bash
python scripts/train_ridge.py \
    --config configs/training/ridge_baseline.yaml \
    --subject subj01
```
Establishes baseline performance quickly.

### **2. Train MLP Standard** (2 hrs)
```bash
python scripts/train_mlp.py \
    --config configs/training/mlp_standard.yaml \
    --subject subj01
```
Good performance with reasonable training time.

### **3. Scale to Two-Stage SOTA** (4 hrs)
```bash
python scripts/train_two_stage.py \
    --config configs/training/two_stage_sota.yaml \
    --subject subj01
```
Best performance for final results.

### **4. Add Adapter** (30 min)
```bash
python scripts/train_clip_adapter.py \
    --config configs/training/adapter_vitl14.yaml \
    --pretrained checkpoints/two_stage/best.pt
```
Enables diffusion-based image generation.

---

## 🔧 Common Customizations

### **Adjust Learning Rate**
```bash
--override "training.learning_rate=0.0005"
```

### **Increase Model Capacity**
```bash
--override "model.hidden_dim=512"  # For MLP
--override "model.latent_dim=1024"  # For Two-Stage
```

### **Longer Training**
```bash
--override "training.epochs=200" \
--override "training.patience=20"
```

### **Larger Batch Size**
```bash
--override "training.batch_size=64"  # Requires more GPU memory
```

### **Data Augmentation**
```bash
--override "data.augmentation=true" \
--override "data.noise_std=0.1"
```

---

## 📝 Creating Custom Configs

### **1. Copy Template**
```bash
cp configs/training/mlp_standard.yaml configs/training/my_experiment.yaml
```

### **2. Edit Parameters**
```yaml
_base_: ../base.yaml

model:
  architecture: mlp
  hidden_dim: 512        # Increased capacity
  dropout: 0.2           # Stronger regularization

training:
  epochs: 150            # Longer training
  learning_rate: 0.0005  # Reduced LR
  batch_size: 64         # Larger batches
```

### **3. Document Purpose**
Add comments explaining your changes:
```yaml
# Experiment: Higher capacity MLP with stronger regularization
# Hypothesis: Larger hidden dim + higher dropout → better generalization
# Expected: Improved validation performance, slower training
```

### **4. Test Configuration**
```bash
python scripts/train_mlp.py \
    --config configs/training/my_experiment.yaml \
    --dry-run  # Validates config without training
```

---

## 🎓 Best Practices

### **Reproducibility**
```yaml
# Always set seed
training:
  seed: 42
  deterministic: true
```

### **Checkpointing**
```yaml
checkpoints:
  save_best: true
  save_last: true
  save_every_n_epochs: 10
```

### **Logging**
```yaml
logging:
  log_every_n_steps: 100
  wandb_project: fmri2img
  tensorboard: true
```

### **Early Stopping**
```yaml
training:
  early_stopping: true
  patience: 10
  monitor: val_cosine_similarity
```

---

## 🐛 Troubleshooting

### **Out of Memory**
```bash
# Reduce batch size
--override "training.batch_size=16"

# Enable gradient accumulation
--override "training.accumulate_grad_batches=2"
```

### **Poor Performance**
```bash
# Increase model capacity
--override "model.hidden_dim=512"

# Train longer
--override "training.epochs=200"

# Reduce learning rate
--override "training.learning_rate=0.00005"
```

### **Training Too Slow**
```bash
# Use fast config for development
--config configs/training/dev_fast.yaml

# Reduce dataset size (testing only)
--override "data.limit_samples=5000"

# Mixed precision training
--override "training.precision=16"
```

---

## 📚 Related Documentation

- **[Main Config README](../README.md)** - Overview
- **[Inference Configs](../inference/README.md)** - Generation settings
- **[Usage Examples](../../USAGE_EXAMPLES.md)** - Complete commands
- **[Training Guides](../../docs/guides/)** - Model-specific guides

---

**Last Updated**: December 7, 2025  
**Status**: Production-Ready  
**Recommended**: `two_stage_sota.yaml` for best performance
