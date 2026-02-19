# Evaluation Protocol

## Overview
This document specifies the exact evaluation protocol for all experiments to ensure reproducibility and fair comparison.

## Dataset Splits

### NSD Subject 01
- **Total unique images**: ~10,000 images (check exact number)
- **Train**: 70% of unique images
- **Validation**: 15% of unique images
- **Test**: 15% of unique images

### Split Generation
- **Method**: Stratified random split by image ID
- **Seed**: 42 (fixed across all experiments)
- **File**: Save split indices to `data/indices/subj01_split.json`

**Important**: Split by unique image IDs, not trials. Each image may have 3 repetitions in NSD; all repetitions of an image must be in the same split.

## Embedding Preprocessing

### Fitting Procedure (CRITICAL)
1. **Fit on train split only**:
   ```python
   preprocessor = EmbeddingPreprocessor(mode="center_pcr", k_components=8, seed=42)
   preprocessor.fit(train_embeddings)  # Only train!
   preprocessor.save("cache/embedding_preproc/subj01_center_pcr_k8.pkl")
   ```

2. **Apply to all splits**:
   - Train: Apply fitted preprocessor
   - Validation: Apply same fitted preprocessor
   - Test: Apply same fitted preprocessor

3. **Apply to all computations**:
   - Ground truth CLIP embeddings (for gallery/targets)
   - Model predictions (mu and sampled z)
   - All similarity/retrieval computations

### Preprocessing Modes
- **center_pcr**: Center, remove top-k PCs, L2-normalize
  - Default: k=8 (ablate: k ∈ {2, 4, 8, 16})
- **center_whiten**: Center, PCA-whiten, L2-normalize

### Artifacts to Save
- Mean vector: (D,)
- PCA components: (D, k) for PCR or (D, D) for whitening
- Explained variance: (k,) or (D,)
- Whitening matrix: (D, D) for center_whiten
- Diagnostics: anisotropy scores, AUC, Cohen's d

## Gallery Construction

### Deterministic Subsampling
- **Seed**: 42 (same across all experiments)
- **Method**: `np.random.seed(42); np.random.choice(N, size=gallery_size, replace=False)`
- **Constraint**: Ensure all test query IDs are present in gallery
  - If missing, replace random items with missing targets

### Gallery Sizes for Curves
- Standard: [2, 10, 50, 100, 500, 1000]
- Extended (optional): [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]

### Full Gallery
- Use all available test samples for final metrics (N_test)

## Metrics Definitions

### Retrieval Metrics

**Recall@K (R@K)**:
- Proportion of queries where correct item is in top-K retrieved
- Compute for K ∈ {1, 5, 10}
- **Chance baseline**: K / N_gallery

**Mean Rank (MeanR)**:
- Average rank of correct item (1-indexed)
- **Chance baseline**: (N_gallery + 1) / 2

**Median Rank (MedR)**:
- Median rank of correct item
- **Chance baseline**: (N_gallery + 1) / 2

**Mean Reciprocal Rank (MRR)**:
- Average of 1/rank for correct items
- **Chance baseline**: $\sum_{r=1}^{N} \frac{1}{r \cdot N} = \frac{H_N}{N}$ (harmonic)

**Normalized Discounted Cumulative Gain (nDCG@K)**:
- Normalized DCG at position K
- Relevance: Binary (1 for correct, 0 otherwise)

### Identification Metrics

**2-Alternative Forced Choice (2AFC)**:
- Sample N_pairs pairs of (query, target) from test set
- For each pair: Compute similarity to both targets
- Correct if positive pair has higher similarity
- **Default N_pairs**: 5000
- **Report**: Accuracy with 95% bootstrap CI (1000 iterations)
- **Chance baseline**: 0.5

**AUC (Positive vs Negative)**:
- Sample N_pairs positive and negative pairs
- Compute ROC AUC for discriminating positive vs negative
- **Chance baseline**: 0.5

**Cohen's d**:
- Effect size: (mean_pos - mean_neg) / pooled_std
- Standardized measure of separation

**d-prime (d')**:
- Signal detection theory: (mean_pos - mean_neg) / sqrt(0.5 * (var_pos + var_neg))

### Structural Metrics

**RSA (Representational Similarity Analysis)**:
- Compute pairwise distance matrices (RDMs) for predicted and GT embeddings
- Correlate flattened RDMs using Spearman rank correlation
- Metric: ρ (Spearman's rho)

**Linear CKA (Centered Kernel Alignment)**:
- Compute centered Gram matrices for predicted and GT
- CKA = ||K_1 K_2||_F^2 / (||K_1||_F ||K_2||_F)
- Range: [0, 1], higher is better

### Probabilistic Metrics

**NLL per-dimension**:
- Compute Gaussian NLL: 0.5 * (logvar + err²/exp(logvar))
- Average over all samples and dimensions
- **Critical**: Report per-dimension (divide by D=768) for interpretability

**ΔNLL vs Baseline**:
- Baseline: Constant prediction with empirical variance
- ΔNLL = NLL_model - NLL_baseline
- Negative values indicate better than baseline

**Energy Score**:
- Proper scoring rule for multivariate distributions
- ES = E[||Y - X||] - 0.5 * E[||Y - Y'||]
- Sample S=64 times from predicted distribution
- Lower is better

**Coverage (Calibration)**:
- Use chi-square thresholds: χ²(D, α) for α ∈ {0.80, 0.95}
- Mahalanobis distance: D² = Σ(x-μ)²/σ²
- Empirical coverage: Proportion of samples with D² ≤ threshold
- **Ideal**: Coverage@80 ≈ 0.80, Coverage@95 ≈ 0.95

**Expected Calibration Error (ECE)**:
- Bin samples by confidence (1/variance)
- Compute |confidence - accuracy| in each bin
- Weight by bin size
- N_bins = 10

**Area Under Risk-Coverage (AURC)**:
- Sort samples by uncertainty (ascending)
- Compute cumulative risk (error) at different coverage levels
- AURC = ∫ risk(coverage) d(coverage)
- Lower is better

**Probabilistic 2AFC**:
- Use likelihood ratios instead of cosine similarity
- Choose prediction with higher likelihood under predicted distribution
- Measures if uncertainty is decision-relevant

## Oracle Sanity Checks

**Purpose**: Ensure evaluation code is correct before trusting results.

**Test**: Ground truth embeddings should perfectly retrieve themselves.

**Procedure**:
1. Use GT embeddings as both queries and gallery
2. Compute retrieval metrics
3. **Pass criteria**:
   - R@1 ≥ 0.95 (tolerate small numerical errors)
   - MeanR ≤ 1.1
   - MRR ≥ 0.95

**If Failed**:
- Check for ID mismatches between queries and gallery
- Check for normalization issues
- Check for accidental shuffling
- **DO NOT PROCEED** with experiment evaluation until fixed

## Sampling Strategy

### For Probabilistic Metrics
- **Monte Carlo samples**: S=64 per prediction
- **Sampling method**: Reparameterization trick
  ```python
  z_sample = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
  ```
- **Seed**: Set per-sample deterministically (sample_id * 1000 + s)

### For Pair Sampling (2AFC, AUC)
- **N_pairs**: 5000 (default)
- **Seed**: 42 (fixed)
- **Method**: Random sampling without replacement
- **Ensure**: Equal numbers of positive and negative pairs (balance)

## Inference Modes

### Mode 1: Deterministic (for standard metrics)
- **Use**: μ(f) only (no sampling)
- **Metrics**: All retrieval, identification, RSA, CKA

### Mode 2: Probabilistic (for uncertainty metrics)
- **Use**: Sample S times from N(μ, Σ)
- **Metrics**: NLL, Energy Score, Coverage, ECE, AURC, Prob-2AFC

**Config**:
```yaml
inference:
  use_mean_for_retrieval: true
  mc_samples_prob_metrics: 64
  mc_samples_prob_2afc: 64
  seed: 42
```

## Reporting Guidelines

### Always Report
1. **Chance baselines** for each metric and gallery size
2. **Gallery size** or N for all metrics
3. **95% CI** for 2AFC (bootstrap)
4. **ΔNLL** instead of raw NLL when comparing methods

### Formatting
- **Percentages**: Report as 0-1 scale with 4 decimals (e.g., 0.8523)
- **Ranks**: Integer or 2 decimals (e.g., 12.45)
- **Correlation**: 4 decimals (e.g., 0.7821)
- **NLL**: Per-dimension, 4 decimals (e.g., 2.3456)

### Tables
- One row per experiment
- Columns: Exp name, R@1, R@5, R@10, MeanR, MRR, 2AFC, AUC, RSA, [Prob metrics if available]
- Include chance baseline row
- **Bold** best result per column

### Figures
1. **Retrieval curves**: R@K vs gallery size (log scale x-axis)
2. **Calibration plot**: Nominal vs empirical coverage
3. **Risk-coverage curves**: Error vs coverage
4. **Ablation bar chart**: Side-by-side comparison
5. **Separation histograms**: Pos vs neg similarity before/after preprocessing

## Reproducibility Checklist

- [ ] Fixed seed for data splits (42)
- [ ] Fixed seed for gallery subsampling (42)
- [ ] Fixed seed for pair sampling (42)
- [ ] Preprocessor fit only on train split
- [ ] Same preprocessor applied to all splits and metrics
- [ ] Oracle check passed
- [ ] Git commit hash saved with results
- [ ] Config file saved with results
- [ ] Random seeds saved with results
- [ ] Preprocessor artifact path + checksum saved

## Output Structure

Each experiment should produce:
```
experimental_results/<exp_name>/
├── config.yaml                      # Experiment config
├── metadata.json                    # Git hash, seeds, timestamps
├── checkpoints/
│   └── best.ckpt                    # Best model checkpoint
├── evaluation/
│   ├── metrics.json                 # All metrics as JSON
│   ├── summary_report.md            # Human-readable report
│   └── plots/
│       ├── retrieval_curves.png
│       ├── identification_summary.png
│       ├── structure_metrics.png
│       ├── calibration_curve.png
│       ├── risk_coverage.png
│       └── separation_histograms.png
└── logs/
    └── training.log
```

## Version Control

Document version: 1.0
Date: 2026-01-17
Author: Research Engineering Team
Changes from v0.9:
- Added oracle sanity checks
- Specified ΔNLL vs raw NLL reporting
- Added inference mode config
- Clarified preprocessing fit procedure
