# =============================================================================
# Configuration Files Guide
# =============================================================================

This directory contains professional configuration files for different use cases
and experiment types. All configs are in YAML format with comprehensive
documentation and expected results.

## Configuration Hierarchy

### Base Configuration
- **base.yaml** - Default settings shared across all experiments
  - Contains sensible defaults
  - Override in specific configs
  - ~150 parameters documented

## Training Configurations

### Baseline & Standard
- **ridge_baseline.yaml** - Ridge regression baseline
  - Fast training (~5 min)
  - Cosine similarity: 0.25-0.35
  - Good for quick experiments
  
- **mlp_standard.yaml** - Standard MLP encoder
  - Moderate training (~1-2 hours)
  - Cosine similarity: 0.35-0.45
  - Good performance/speed tradeoff

### Advanced
- **two_stage_sota.yaml** - State-of-the-art two-stage encoder
  - Best performance (~3-6 hours)
  - Cosine similarity: 0.45-0.55
  - Multi-layer supervision, InfoNCE loss
  - Recommended for final results

- **adapter_vitl14.yaml** - CLIP adapter training
  - Maps 512D → 768D CLIP space
  - Quick training (~30 min)
  - Improves generation quality
  - Use after training main encoder

### Specialized Training
- **clip2fmri.yaml** - Inverse mapping (CLIP → fMRI)
  - For brain-consistency loss
  - Research/ablation use
  
## Inference Configurations

### Production
- **production.yaml** - Production inference settings
  - Balanced quality/speed
  - 50 diffusion steps
  - ~10s per image
  - Recommended for deployment

### Quality-Focused
- **highres_quality.yaml** - Maximum quality generation
  - 1024px resolution
  - 200 diffusion steps
  - ~30-60s per image
  - Best for publications/demos

### Speed-Focused
- **fast_inference.yaml** - Rapid generation
  - 512px resolution
  - 25 diffusion steps
  - ~3-5s per image
  - Good for batch processing

## Development & Testing

- **dev_fast.yaml** - Quick development testing
  - 1000 samples, 10 epochs
  - ~2-5 min training
  - For debugging & rapid iteration
  - Not for final results

## Analysis

- **ablation.yaml** - Systematic ablation studies
  - Template for component analysis
  - Statistical testing built-in
  - Comprehensive result tracking

## Data & Model Configs

- **data.yaml** - NSD dataset configuration
  - S3 access settings
  - File paths & structure
  - Preprocessing pipelines

- **clip.yaml** - CLIP model settings
  - Model selection
  - Embedding dimensions
  - Feature extraction

- **logging.yaml** - Logging configuration
  - Log levels & formats
  - Output destinations

## Deprecated Configs

- **sota_two_stage.yaml** → Use **two_stage_sota.yaml**
- **production_improved.yaml** → Use **production.yaml**

These are kept for backward compatibility but redirect to new configs.

## Usage Examples

### Training
```bash
# Quick baseline
python -m fmri2img.training.train_ridge \
    --config configs/ridge_baseline.yaml

# Standard MLP
python -m fmri2img.training.train_mlp \
    --config configs/mlp_standard.yaml

# SOTA two-stage
python -m fmri2img.training.train_two_stage \
    --config configs/two_stage_sota.yaml \
    --subject subj01

# Train adapter
python -m fmri2img.training.train_clip_adapter \
    --config configs/adapter_vitl14.yaml
```

### Inference
```bash
# Production inference
python -m fmri2img.generation.decode_diffusion \
    --config configs/production.yaml \
    --checkpoint checkpoints/two_stage/subj01/best.pt

# High-quality generation
python -m fmri2img.generation.decode_diffusion \
    --config configs/highres_quality.yaml \
    --checkpoint checkpoints/two_stage/subj01/best.pt

# Fast batch processing
python -m fmri2img.generation.decode_diffusion \
    --config configs/fast_inference.yaml \
    --checkpoint checkpoints/two_stage/subj01/best.pt \
    --batch-size 8
```

### Development
```bash
# Quick test
python -m fmri2img.training.train_mlp \
    --config configs/dev_fast.yaml

# Ablation study
python -m fmri2img.eval.ablation_driver \
    --base-config configs/ablation.yaml \
    --components infonce_loss,multi_layer,dropout
```

## Configuration Override

All configs support command-line overrides:

```bash
# Override specific parameters
python -m fmri2img.training.train_mlp \
    --config configs/mlp_standard.yaml \
    --training.learning_rate 5e-5 \
    --training.batch_size 128 \
    --preprocessing.pca_k 1024
```

## Adding New Configurations

1. Copy an existing config as template
2. Update experiment metadata
3. Modify parameters as needed
4. Document expected results
5. Test thoroughly before committing

## Best Practices

### For Research
- Use **two_stage_sota.yaml** for best results
- Run **ablation.yaml** for component analysis
- Document all parameter changes
- Include expected results in config

### For Production
- Use **production.yaml** as starting point
- Adjust based on quality/speed requirements
- Monitor memory usage and generation time
- Use **fast_inference.yaml** for real-time needs

### For Development
- Use **dev_fast.yaml** for quick iteration
- Test with full config before final run
- Keep configs in version control
- Document any custom modifications

## Configuration Validation

Validate configs before use:
```bash
python -m fmri2img.utils.validate_config \
    --config configs/your_config.yaml
```

## File Organization

```
configs/
├── README.md                    # This file
├── base.yaml                    # Base configuration
├── Training/
│   ├── ridge_baseline.yaml
│   ├── mlp_standard.yaml
│   ├── two_stage_sota.yaml
│   └── adapter_vitl14.yaml
├── Inference/
│   ├── production.yaml
│   ├── highres_quality.yaml
│   └── fast_inference.yaml
├── Development/
│   ├── dev_fast.yaml
│   └── ablation.yaml
└── Data/
    ├── data.yaml
    ├── clip.yaml
    └── logging.yaml
```

## Support

For issues or questions about configurations:
1. Check this README
2. Review config comments
3. See docs/USAGE_EXAMPLES.md
4. Check START_HERE.md

## Version History

- v3.0 (Dec 2025): Complete reorganization, professional configs
- v2.0 (Nov 2025): Added multi-layer and ablation configs
- v1.0 (Oct 2025): Initial configs

---
**Last Updated**: December 6, 2025
**Version**: 3.0
