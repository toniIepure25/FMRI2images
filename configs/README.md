# Configuration Management System# =============================================================================

# Configuration Files Guide

**fMRI-to-Image Reconstruction**  # =============================================================================

*Research-Level YAML-Based Configuration*

This directory contains professional configuration files for different use cases

---and experiment types. All configs are in YAML format with comprehensive

documentation and expected results.

## 📁 Professional Organization

## Configuration Hierarchy

The configuration system follows a **hierarchical, modular architecture** for maximum flexibility and maintainability.

### Base Configuration

```- **base.yaml** - Default settings shared across all experiments

configs/  - Contains sensible defaults

├── base.yaml                    # Base configuration (inheritance root)  - Override in specific configs

│  - ~150 parameters documented

├── 📚 training/                 # Model training configurations

│   ├── README.md## Training Configurations

│   ├── ridge_baseline.yaml      # Ridge regression (baseline)

│   ├── mlp_standard.yaml        # MLP encoder (standard)### Baseline & Standard

│   ├── two_stage_sota.yaml      # Two-Stage encoder (SOTA) ⭐- **ridge_baseline.yaml** - Ridge regression baseline

│   ├── sota_two_stage.yaml      # Legacy naming (deprecated)  - Fast training (~5 min)

│   ├── adapter_vitl14.yaml      # CLIP adapter  - Cosine similarity: 0.25-0.35

│   ├── clip2fmri.yaml           # Inverse mapping (research)  - Good for quick experiments

│   └── dev_fast.yaml            # Fast development/debugging  

│- **mlp_standard.yaml** - Standard MLP encoder

├── 🚀 inference/                # Image generation configurations  - Moderate training (~1-2 hours)

│   ├── README.md  - Cosine similarity: 0.35-0.45

│   ├── production.yaml          # Production deployment ⭐  - Good performance/speed tradeoff

│   ├── production_improved.yaml # Enhanced production

│   ├── fast_inference.yaml      # Speed-optimized### Advanced

│   └── highres_quality.yaml     # Quality-optimized- **two_stage_sota.yaml** - State-of-the-art two-stage encoder

│  - Best performance (~3-6 hours)

├── ⚙️ system/                   # System & infrastructure  - Cosine similarity: 0.45-0.55

│   ├── README.md  - Multi-layer supervision, InfoNCE loss

│   ├── data.yaml                # Dataset configuration (NSD)  - Recommended for final results

│   ├── clip.yaml                # CLIP model settings

│   └── logging.yaml             # Logging configuration- **adapter_vitl14.yaml** - CLIP adapter training

│  - Maps 512D → 768D CLIP space

├── 🧪 experiments/              # Research & ablation studies  - Quick training (~30 min)

│   ├── README.md  - Improves generation quality

│   └── ablation.yaml            # Ablation study template  - Use after training main encoder

│

└── README.md                    # This file### Specialized Training

```- **clip2fmri.yaml** - Inverse mapping (CLIP → fMRI)

  - For brain-consistency loss

---  - Research/ablation use

  

## 🎯 Quick Start## Inference Configurations



### **Training a Model**### Production

```bash- **production.yaml** - Production inference settings

# Ridge baseline (5 minutes)  - Balanced quality/speed

python scripts/train_ridge.py \  - 50 diffusion steps

    --config configs/training/ridge_baseline.yaml \  - ~10s per image

    --subject subj01  - Recommended for deployment



# MLP encoder (2 hours)### Quality-Focused

python scripts/train_mlp.py \- **highres_quality.yaml** - Maximum quality generation

    --config configs/training/mlp_standard.yaml \  - 1024px resolution

    --subject subj01  - 200 diffusion steps

  - ~30-60s per image

# Two-Stage SOTA (4 hours, best performance)  - Best for publications/demos

python scripts/train_two_stage.py \

    --config configs/training/two_stage_sota.yaml \### Speed-Focused

    --subject subj01- **fast_inference.yaml** - Rapid generation

```  - 512px resolution

  - 25 diffusion steps

### **Generating Images**  - ~3-5s per image

```bash  - Good for batch processing

# Production (balanced quality/speed)

python scripts/decode_diffusion.py \## Development & Testing

    --config configs/inference/production.yaml \

    --checkpoint checkpoints/two_stage/best.pt- **dev_fast.yaml** - Quick development testing

  - 1000 samples, 10 epochs

# High quality (slow, best for demos)  - ~2-5 min training

python scripts/decode_diffusion.py \  - For debugging & rapid iteration

    --config configs/inference/highres_quality.yaml \  - Not for final results

    --checkpoint checkpoints/two_stage/best.pt

```## Analysis



---- **ablation.yaml** - Systematic ablation studies

  - Template for component analysis

## 📖 Configuration Hierarchy  - Statistical testing built-in

  - Comprehensive result tracking

### **Inheritance System**

## Data & Model Configs

All configurations inherit from `base.yaml`:

- **data.yaml** - NSD dataset configuration

```yaml  - S3 access settings

# configs/training/mlp_standard.yaml  - File paths & structure

_base_: ../base.yaml  # Inherit defaults  - Preprocessing pipelines



model:- **clip.yaml** - CLIP model settings

  hidden_dim: 256     # Override specific parameters  - Model selection

```  - Embedding dimensions

  - Feature extraction

### **Runtime Overrides**

- **logging.yaml** - Logging configuration

```bash  - Log levels & formats

python scripts/train_mlp.py \  - Output destinations

    --config configs/training/mlp_standard.yaml \

    --override "training.learning_rate=0.001" \## Deprecated Configs

    --override "training.batch_size=64"

```- **sota_two_stage.yaml** → Use **two_stage_sota.yaml**

- **production_improved.yaml** → Use **production.yaml**

---

These are kept for backward compatibility but redirect to new configs.

## 📚 Configuration Categories

## Usage Examples

### **1. Training** (`training/`) - See [training/README.md](training/README.md)

### Training

| Config | Performance | Time | Use Case |```bash

|--------|-------------|------|----------|# Quick baseline

| `ridge_baseline.yaml` | 0.25-0.35 | 5 min | Baseline, quick experiments |python -m fmri2img.training.train_ridge \

| `mlp_standard.yaml` | 0.35-0.45 | 2 hours | Standard training |    --config configs/ridge_baseline.yaml

| `two_stage_sota.yaml` ⭐ | 0.45-0.55 | 4 hours | Best performance (SOTA) |

| `adapter_vitl14.yaml` | N/A | 30 min | Diffusion integration |# Standard MLP

| `dev_fast.yaml` | N/A | 2-5 min | Development/debugging |python -m fmri2img.training.train_mlp \

    --config configs/mlp_standard.yaml

### **2. Inference** (`inference/`) - See [inference/README.md](inference/README.md)

# SOTA two-stage

| Config | Resolution | Speed | Quality | Use Case |python -m fmri2img.training.train_two_stage \

|--------|-----------|-------|---------|----------|    --config configs/two_stage_sota.yaml \

| `production.yaml` ⭐ | 512px | ~10s | Good | Deployment |    --subject subj01

| `fast_inference.yaml` | 512px | ~5s | Fair | Batch processing |

| `highres_quality.yaml` | 1024px | ~60s | Excellent | Publications |# Train adapter

python -m fmri2img.training.train_clip_adapter \

### **3. System** (`system/`) - See [system/README.md](system/README.md)    --config configs/adapter_vitl14.yaml

```

Infrastructure configurations: data paths, CLIP models, logging.

### Inference

### **4. Experiments** (`experiments/`) - See [experiments/README.md](experiments/README.md)```bash

# Production inference

Research and ablation study templates.python -m fmri2img.generation.decode_diffusion \

    --config configs/production.yaml \

---    --checkpoint checkpoints/two_stage/subj01/best.pt



## 📊 Performance Benchmarks# High-quality generation

python -m fmri2img.generation.decode_diffusion \

### **Training Time** (NVIDIA A100 40GB)    --config configs/highres_quality.yaml \

    --checkpoint checkpoints/two_stage/subj01/best.pt

| Configuration | Time | Peak Memory | Performance (Cosine Sim) |

|--------------|------|-------------|-------------------------|# Fast batch processing

| Ridge Baseline | 5 min | 4GB RAM | 0.28 |python -m fmri2img.generation.decode_diffusion \

| MLP Standard | 2 hours | 8GB VRAM | 0.41 |    --config configs/fast_inference.yaml \

| Two-Stage SOTA | 4 hours | 12GB VRAM | 0.52 ⭐ |    --checkpoint checkpoints/two_stage/subj01/best.pt \

    --batch-size 8

### **Inference Speed** (per image)```



| Configuration | Resolution | Time | Quality |### Development

|--------------|-----------|------|---------|```bash

| Fast | 512px | ~5s | Fair |# Quick test

| Production | 512px | ~10s | Good ⭐ |python -m fmri2img.training.train_mlp \

| High Quality | 1024px | ~60s | Excellent |    --config configs/dev_fast.yaml



---# Ablation study

python -m fmri2img.eval.ablation_driver \

## 🔧 Common Workflows    --base-config configs/ablation.yaml \

    --components infonce_loss,multi_layer,dropout

### **Quick Experiment**```

```bash

python scripts/train_ridge.py \## Configuration Override

    --config configs/training/dev_fast.yaml

```All configs support command-line overrides:



### **Production Training**```bash

```bash# Override specific parameters

python scripts/train_two_stage.py \python -m fmri2img.training.train_mlp \

    --config configs/training/two_stage_sota.yaml \    --config configs/mlp_standard.yaml \

    --subject subj01    --training.learning_rate 5e-5 \

```    --training.batch_size 128 \

    --preprocessing.pca_k 1024

### **High-Quality Generation**```

```bash

python scripts/decode_diffusion.py \## Adding New Configurations

    --config configs/inference/highres_quality.yaml \

    --checkpoint checkpoints/two_stage/best.pt1. Copy an existing config as template

```2. Update experiment metadata

3. Modify parameters as needed

---4. Document expected results

5. Test thoroughly before committing

## 📝 Configuration Format

## Best Practices

All configs use YAML with inheritance:

### For Research

```yaml- Use **two_stage_sota.yaml** for best results

_base_: ../base.yaml  # Inherit from base- Run **ablation.yaml** for component analysis

- Document all parameter changes

model:- Include expected results in config

  architecture: mlp

  hidden_dim: 256### For Production

- Use **production.yaml** as starting point

training:- Adjust based on quality/speed requirements

  epochs: 100- Monitor memory usage and generation time

  learning_rate: 0.0001- Use **fast_inference.yaml** for real-time needs

```

### For Development

**Override at runtime**:- Use **dev_fast.yaml** for quick iteration

```bash- Test with full config before final run

--override "training.learning_rate=0.001"- Keep configs in version control

```- Document any custom modifications



---## Configuration Validation



## 🎓 Best PracticesValidate configs before use:

```bash

1. **Use appropriate config for task**python -m fmri2img.utils.validate_config \

   - Development → `training/dev_fast.yaml`    --config configs/your_config.yaml

   - Production → `training/two_stage_sota.yaml````



2. **Override at runtime** (don't modify files)## File Organization

   ```bash

   --override "param=value"```

   ```configs/

├── README.md                    # This file

3. **Document custom configs** (purpose, performance, requirements)├── base.yaml                    # Base configuration

├── Training/

4. **Version control** (commit templates, ignore personal configs)│   ├── ridge_baseline.yaml

│   ├── mlp_standard.yaml

---│   ├── two_stage_sota.yaml

│   └── adapter_vitl14.yaml

## 🆕 Creating New Configurations├── Inference/

│   ├── production.yaml

```bash│   ├── highres_quality.yaml

# 1. Copy template│   └── fast_inference.yaml

cp configs/training/mlp_standard.yaml configs/training/my_experiment.yaml├── Development/

│   ├── dev_fast.yaml

# 2. Modify parameters│   └── ablation.yaml

vim configs/training/my_experiment.yaml└── Data/

    ├── data.yaml

# 3. Test with dry run    ├── clip.yaml

python scripts/train_mlp.py \    └── logging.yaml

    --config configs/training/my_experiment.yaml \```

    --dry-run

```## Support



---For issues or questions about configurations:

1. Check this README

## 📚 Related Documentation2. Review config comments

3. See docs/USAGE_EXAMPLES.md

- **Category READMEs**: See subdirectory READMEs for detailed info4. Check START_HERE.md

- **[Usage Examples](../USAGE_EXAMPLES.md)** - Command-line usage

- **[Training Guides](../docs/guides/)** - Model-specific guides## Version History

- **[Quick Start](../START_HERE.md)** - Getting started

- v3.0 (Dec 2025): Complete reorganization, professional configs

---- v2.0 (Nov 2025): Added multi-layer and ablation configs

- v1.0 (Oct 2025): Initial configs

## 🐛 Troubleshooting

---

**Config not found?****Last Updated**: December 6, 2025

```bash**Version**: 3.0

python script.py --config configs/training/mlp_standard.yaml
```

**Override not working?**
```bash
--override "training.learning_rate=0.001"  # Use quotes!
```

**Inheritance failing?**
```yaml
_base_: ../base.yaml  # Correct relative path
```

---

## 📞 Support

1. Check category-specific README (`training/`, `inference/`, etc.)
2. Review [Usage Examples](../USAGE_EXAMPLES.md)
3. Open GitHub issue with config file

---

**Last Updated**: December 7, 2025  
**Version**: 2.0 (Professionally Organized)  
**Status**: Production-Ready
