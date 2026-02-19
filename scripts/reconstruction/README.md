# Reconstruction Scripts

Image reconstruction and generation from fMRI embeddings.

**5 reconstruction scripts**

## Entry Point

- **`reconstruct.py`** - Main wrapper forwarding to `generate_images.py`

## Implementations

- **`generate_images.py`** ⭐ - Generate reconstructed images from fMRI
  - Takes trained model checkpoint
  - Generates images using diffusion models
  - Saves results and visualizations
  
- `decode_diffusion.py` - Diffusion model decoding utilities
- `decode_two_stage.py` - Two-stage reconstruction pipeline
- `reconstruct_nn.py` - Nearest neighbor reconstruction

## Usage

```bash
# Using wrapper (recommended)
python scripts/reconstruction/reconstruct.py \
    --checkpoint outputs/exp0/best.pt \
    --n_images 100

# Direct invocation
python scripts/reconstruction/generate_images.py \
    --checkpoint outputs/exp0/best.pt \
    --n_images 100
```
