# Training Configurations

Training recipes for models outside the main ablation ladder.

## Files

| Config | Purpose | Script | Time |
|--------|---------|--------|------|
| `dev_fast.yaml` | Quick sanity checks (1K trials, 10 epochs) | `train_unified.py` | ~2 min |
| `ridge_baseline.yaml` | Linear ridge regression baseline | `train_ridge.py` | ~5 min |
| `adapter_vitl14.yaml` | CLIP adapter: ViT-B/32 (512) -> ViT-L/14 (768) | `train_clip_adapter.py` | ~30 min |

## Usage

```bash
# Quick development loop
python3 scripts/training/train_unified.py \
    --config configs/training/dev_fast.yaml --gpu 0

# Ridge baseline
python3 scripts/training/train_ridge.py \
    --config configs/training/ridge_baseline.yaml --subject subj01

# CLIP adapter (only needed for Phase 1 encoder reuse)
python3 scripts/training/train_clip_adapter.py \
    --config configs/training/adapter_vitl14.yaml --subject subj01
```

## Note

For the main ablation experiments (B0-N4), use `configs/experiments/{B0,B1,N1,N2,N3,N4}_*.yaml`
with `train_unified.py`. These training configs serve supplementary roles.
