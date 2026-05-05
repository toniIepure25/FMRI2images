# True Diffusion Add-on

This directory contains the final thesis-grade qualitative reconstruction addon built on top of the frozen best retrieval system.

## Frozen Retrieval Backbone

- tri/main results dir: `experimental_results/V35_legacy_teacher_distill/subj01`
- legacy results dir: `experimental_results/N1v28a_dual_head/subj01`
- frozen tri-fusion config: `{'compact_score': 'csls', 'legacy_score': 'csls', 'family': 'normalized_weighted', 'normalization': 'zscore', 'shortlist_k': 150, 'alpha': 0.3, 'beta': 0.0, 'gamma': 0.7}`
- frozen SHARED1000 R@1: `77.2%`

## Method

This addon performs:

1. frozen tri-fusion shortlist retrieval
2. score-weighted top-k prior construction
3. score-weighted latent prior formation in the diffusion VAE space
4. true Stable Diffusion img2img generation
5. multi-sample candidate selection using retrieval-prior consistency only

## Integrity constraints

- no retrieval retraining
- no fusion changes
- no GT during generation or candidate selection
- no heuristic fallback in this script

## Runtime status

- true diffusion run: `True`
- diffusion model: `sd2-community/stable-diffusion-2-1`
- diffusion path: `sd2-community/stable-diffusion-2-1`
- dependency versions: `{'diffusers': '0.37.0', 'transformers': '5.3.0', 'accelerate': '1.13.0', 'safetensors': '0.7.0', 'open_clip': '3.3.0', 'lpips': 'unknown', 'skimage': '0.26.0'}`

## Outputs

- `DIFFUSION_ADDON_DESIGN.md`
- `selected_examples.json`
- `selected_examples.csv`
- `metrics/`
- `figures/`
- `reconstructions/`
