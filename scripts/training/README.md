# Training Scripts

Model training implementations for fMRI-to-CLIP embedding mapping.

**11 training scripts**

## Entry Point

- **`train.py`** - Main wrapper forwarding to `train_unified.py`

## Implementations

- **`train_unified.py`** ⭐ - Unified training supporting all architectures and losses
  - Ridge, MLP, Unified, Gaussian architectures
  - MSE, InfoNCE, Gaussian NLL, Gaussian-NCE, KL losses
  - Preprocessing: PCA/PCR, soft reliability
  - Memory queue for contrastive learning
  
- `train_ridge.py` - Fast ridge regression baseline
- `train_mlp.py` - MLP encoder baseline
- `train_clip_adapter.py` - CLIP dimension adapter (512D→768/1024D)
- `train_two_stage.py` - Two-stage pipeline (fMRI→latent→CLIP)
- `train_real_nsd.py` - Training on real NSD data
- `train_clip_to_fmri.py` - Reverse direction (CLIP→fMRI encoding)

## Usage

```bash
# Using wrapper (recommended)
python scripts/training/train.py --config configs/experiments/exp0_baseline.yaml

# Direct invocation
python scripts/training/train_unified.py --config configs/experiments/exp0_baseline.yaml
```
