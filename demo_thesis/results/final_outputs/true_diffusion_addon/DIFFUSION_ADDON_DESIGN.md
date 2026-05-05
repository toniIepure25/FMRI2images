# DIFFUSION ADDON DESIGN

## Audit of previous code

### `scripts/reconstruction/decode_two_stage.py`

This path is **not thesis-grade** for the frozen final system.
It creates a fresh random `Linear(512 -> 1024)` projection at inference time and injects that into Stable Diffusion prompt embeddings.
That projection is never trained, never justified, and is not aligned with the validated retrieval architecture.
It is scientifically unsuitable for final reporting.

### `scripts/reconstruction/generate_images.py`

This is largely legacy scaffolding for embedding export and visualization.
It is not an end-to-end retrieval-aware reconstruction system.
It does not use the frozen tri-fusion benchmark outputs.

### `scripts/reconstruction/decode_diffusion.py`

This path is stronger than the legacy scripts, but still not sufficient as the final thesis addon.
It performs generic CLIP-to-diffusion conditioning and optional pooling tricks, but it does **not** make principled use of the frozen tri-expert retrieval system.
It ignores the validated tri-fusion shortlist/top-k structure and therefore is only loosely aligned with the reportable architecture.

### Reusable components

The following pieces are reused:

- frozen best-system loader from `scripts/evaluation/final_best_system.py`
- NSD stimulus loading from `scripts/evaluation/export_final_qualitatives.py`
- image metrics from `src/fmri2img/eval/image_metrics.py`
- reconstruction metrics from `src/fmri2img/eval/recon_eval.py`

## Final algorithm

The final addon is architecture-aligned because it uses the actual frozen retrieval outputs as the semantic and visual prior:

1. compute the frozen fixed tri-fusion shortlist using the approved production recipe
2. take fused top-k retrieved images as the visual prior
3. turn fused local scores into score weights
4. construct a score-weighted latent prior in the diffusion VAE space
5. run true Stable Diffusion img2img from that prior
6. generate multiple seeded candidates deterministically
7. select the final sample using retrieval-prior consistency only, never ground truth

## Why this is more professional

- it keeps the frozen 77.2% retrieval benchmark untouched
- it visibly exploits the real compact/legacy/fused architecture instead of bypassing it
- it uses retrieval top-k as the reconstruction prior rather than arbitrary text prompting
- it avoids random untrained projections at inference time
- it forbids heuristic fallback under the label of diffusion
- it records dependency availability and the exact diffusion path used

## Runtime result for this export

- true diffusion run: `True`
- diffusion model: `sd2-community/stable-diffusion-2-1`
- diffusion model path: `sd2-community/stable-diffusion-2-1`
- dependency versions: `{'diffusers': '0.37.0', 'transformers': '5.3.0', 'accelerate': '1.13.0', 'safetensors': '0.7.0', 'open_clip': '3.3.0', 'lpips': 'unknown', 'skimage': '0.26.0'}`
