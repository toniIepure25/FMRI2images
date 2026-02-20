# Experiment Notes: exp001_baseline_ultimate

**Phase 1 — Probabilistic Two-Stage Encoder (All 7 Novel Contributions)**
**Date**: 2026-01-17
**Status**: Training stopped at epoch 28/50 (early stopping did not trigger; manual stop)

## Overview

Phase 1 baseline combining all 7 original contributions: soft reliability weighting,
multi-layer CLIP targets (layers 4/8/12), InfoNCE contrastive loss, KL-regularized
variational head, brain-consistency loss, adaptive-K sampling, and MC Dropout
uncertainty. CLIP ViT-B/32 (512-D) target space.

## Hypothesis

All 7 contributions combined should produce a strong baseline for embedding quality
(cosine > 0.5) and provide a foundation for Phase 2 improvements (vMF, ROI-DCF).

## Results

| Metric | Value | Assessment |
|--------|-------|------------|
| Cosine Similarity (mean) | 0.683 +/- 0.055 | Strong embedding alignment |
| Cosine Similarity (median) | 0.687 | Consistent with mean |
| R@1 (gallery=75) | 1.33% | Poor — likely hubness-dominated |
| R@5 (gallery=75) | 6.67% | Below expectations |
| R@10 (gallery=75) | 13.33% | Moderate |
| Mean Rank (gallery=75) | 38.0 | Near median — no better than random |
| KL Divergence | 0.167 | Good balance (posterior not collapsed) |
| MSE | 0.000825 | Low absolute error |
| L2 Distance | 0.793 +/- 0.066 | — |

**Key finding**: High cosine similarity (0.68) but near-chance retrieval (R@1 = 1.3%)
indicates **severe hubness** in the 512-D embedding space. The model produces
embeddings that are globally similar to ground truth but not discriminative enough
for identification. This motivates Phase 2: embedding preprocessing (center_pcr),
proper contrastive objectives (vMF-NCE with queue), and CSLS anti-hubness correction.

## Training Observations

- **Stability**: Stable convergence, no loss spikes observed
- **Convergence**: Total loss reached 0.437 at epoch 28; still decreasing slowly
- **Training time**: Not recorded (cluster run)
- **Issues**: Small evaluation set (N=75) limits retrieval metric reliability

## Insights

1. Cosine similarity alone is a misleading metric — high cos-sim does not imply
   good retrieval when the embedding space is anisotropic (hubness problem).
2. Batch-level R@1 (27.5%) vs global R@1 (1.3%) gap confirms hubness: embeddings
   are discriminative within a batch but collapse when evaluated against the full gallery.
3. KL = 0.167 suggests the variational head is healthy but may benefit from
   tighter posterior (lower KL -> more concentrated predictions -> Phase 2 vMF).

## Phase 2 Follow-Up

These Phase 1 results motivated the Phase 2 research plan:
- EXP0-EXP6: Systematic ablation of preprocessing and loss functions
- EXP7: vMF-NCE to match the spherical geometry of normalized CLIP embeddings
- EXP8-EXP9: ROI Transformer + ROI-DCF for brain-topology-aware encoding
- EXP10-EXP14: Full innovation stack (mixture sampling, dual UA-CFG, etc.)
