# Inference Configurations

Diffusion model generation configs with different quality/speed tradeoffs.

## Files

| Config | Steps | Resolution | Speed | Use Case |
|--------|-------|-----------|-------|----------|
| `production.yaml` | 50 | 768px | ~10s | Standard evaluation |
| `fast_inference.yaml` | 25 | 512px | ~3s | Batch processing |
| `highres_quality.yaml` | 200 | 1024px | ~60s | Paper figures |

All configs use Stable Diffusion 2.1 (`stabilityai/stable-diffusion-2-1`).

## Usage

```bash
# Production inference
python3 scripts/reconstruction/decode_diffusion.py \
    --config configs/inference/production.yaml \
    --checkpoint experimental_results/N4_full_system/best_model.pt \
    --subject subj01

# Fast batch evaluation
python3 scripts/reconstruction/decode_diffusion.py \
    --config configs/inference/fast_inference.yaml \
    --checkpoint path/to/model.pt --subject subj01

# High-res for publication figures
python3 scripts/reconstruction/decode_diffusion.py \
    --config configs/inference/highres_quality.yaml \
    --checkpoint path/to/model.pt --subject subj01
```

## Uncertainty-Aware CFG (Phase 2)

When using a vMF model with ROI-DCF (N3+), `production.yaml` supports
decomposed uncertainty-aware CFG. Enable it in the config or at runtime:

```bash
python3 scripts/reconstruction/decode_diffusion.py \
    --config configs/inference/production.yaml \
    --checkpoint path/to/N4_model.pt \
    --override "ua_cfg.enabled=true"
```

This maps model uncertainty to diffusion parameters:
- kappa (concentration) -> guidance scale w
- delta (ROI disagreement) -> ensemble size K and diffusion steps
