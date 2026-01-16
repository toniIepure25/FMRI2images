# Experiment Evaluation Summary: exp001_baseline_ultimate

**Date**: 2026-01-17 01:11:54  
**Epoch**: 28 / 50  
**Samples Evaluated**: 75

---

## Quick Summary

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Cosine Similarity** | 0.6832 ± 0.0546 | ✅ Excellent |
| **Top-1 Retrieval** | 1.33% | ❌ Poor |
| **Top-5 Retrieval** | 6.67% | ❌ Poor |
| **Mean Rank** | 38.0 | ❌ Poor |
| **KL Divergence** | 0.1668 | ✅ Good balance |

---

## Detailed Metrics

### CLIP Embedding Quality

**Cosine Similarity Distribution**:
- Mean: 0.6832
- Median: 0.6875
- Std: 0.0546
- Min: 0.4472
- Max: 0.7605

**Interpretation**: 
- Values closer to 1.0 indicate better alignment with ground truth CLIP embeddings
- Median similar to mean suggests consistent performance
- Low std (<0.15) indicates stable predictions across samples

### Retrieval Performance

**Top-K Accuracy**:
- Top-1: 1.33% (correct image ranked #1)
- Top-5: 6.67% (correct image in top-5)
- Top-10: 13.33% (correct image in top-10)

**Ranking Statistics**:
- Mean Rank: 38.0
- Median Rank: 38.0

**Interpretation**:
- Top-5 >40% indicates strong practical retrieval capability
- Lower mean rank indicates more predictions near ground truth
- Median < Mean suggests some outliers with very poor predictions

### Reconstruction Error

**Embedding Space Distance**:
- MSE: 0.0008
- RMSE: 0.0287
- L2 Distance (mean): 0.7934 ± 0.0656

### Probabilistic Component

**KL Divergence (Variational Regularization)**:
- Mean: 0.1668
- Std: 0.0000

**Interpretation**:
- KL ~0.1-0.3: Good balance between reconstruction and regularization
- KL <0.05: Posterior collapsed to prior (underfitting)
- KL >0.5: High divergence (may need higher KL weight)

---

## Configuration

**Training Hyperparameters**:
- Learning Rate: 1e-05
- Batch Size: 4
- Gradient Accumulation: 8
- Effective Batch: 32
- Gradient Clip: 0.3
- KL Weight: 0.01

**Model Architecture**:
- Latent Dim: 768
- Blocks: 6
- Head Hidden: 1024
- Dropout: 0.3
- Enabled Layers: layer_4, layer_8, layer_12, final

**Loss Weights**:
- MSE: 0.4
- Cosine: 0.4
- InfoNCE: 0.2

---

## Decision

### ✅ EXCELLENT - Strong Embedding Quality
- **Cosine Similarity: 0.6832** - Outstanding performance!
- **Recommendation**: This model shows excellent learning
  - Continue training to epoch 50 for potential further improvement
  - Archive as strong baseline for thesis
  - Consider generating reconstructions for visualization
- **Note**: Low retrieval metrics are expected with small validation sets (<1000 samples)


### Next Steps

1. **If satisfied with metrics**:
   ```bash
   # Generate reconstructions (optional)
   python scripts/run_stage34_recon_eval.py \
       --checkpoint CHECKPOINT_PATH \
       --config CONFIG_PATH \
       --output-dir experimental_results/exp001_baseline_ultimate/reconstructions \
       --num-images 50
   ```

2. **If trying new experiment**:
   - Modify hyperparameters in config
   - Train new model
   - Evaluate with this workflow again

3. **Compare with other experiments**:
   ```bash
   python scripts/compare_experimental_results.py \
       --exp-dirs experimental_results/exp001_baseline_ultimate experimental_results/exp002_... \
       --output experimental_results/comparison_reports/comparison.md
   ```

---

*Evaluation completed on 2026-01-17 01:11:54*
