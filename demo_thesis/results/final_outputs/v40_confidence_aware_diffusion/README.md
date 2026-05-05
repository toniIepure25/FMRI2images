# V40 Confidence-Aware Retrieval-Guided Diffusion

This directory contains the qualitative V40 diffusion addon built on top of the frozen final retrieval system.

## Frozen Retrieval Backbone

- tri/main results dir: `experimental_results/V35_legacy_teacher_distill/subj01`
- legacy results dir: `experimental_results/N1v28a_dual_head/subj01`
- frozen tri-fusion config: `{'compact_score': 'csls', 'legacy_score': 'csls', 'family': 'normalized_weighted', 'normalization': 'zscore', 'shortlist_k': 150, 'alpha': 0.3, 'beta': 0.0, 'gamma': 0.7}`
- frozen SHARED1000 R@1: `77.2%`

## What V40 Changes

- keeps the retrieval benchmark frozen
- routes each query through a confidence-aware controller
- preserves exact / near-exact retrieval cases with `identity_pass`
- uses an anchor-preserving latent prior instead of unconditional top-k latent averaging
- selects the final output using anchor similarity, shortlist consistency, and layout preservation only
- never uses ground truth for generation or candidate selection

## True Diffusion

- true diffusion run: `True`
- method: `score_weighted_anchor_residual_img2img`
- model: `sd2-community/stable-diffusion-2-1`

## Main Outputs

- `selected_examples.json`, `selected_examples.csv`
- `metrics/per_example_metrics.csv`
- `metrics/summary_metrics.json`
- `metrics/summary_metrics.csv`
- `metrics/bucket_summary.csv`
- `figures/examples/`
- `figures/contact_sheets/`
- `reconstructions/`
