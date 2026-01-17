# Research-Grade Evaluation Suite

**Complete evaluation framework for probabilistic two-stage fMRI-to-image pipeline**

This evaluation suite provides field-standard and novel metrics for assessing:
1. **Stage 1**: fMRI → CLIP embedding predictions (deterministic + probabilistic)
2. **Stage 2**: CLIP → Image reconstructions
3. **End-to-end**: Sampling and uncertainty-aware decoding

---

## 📋 Overview

### Implemented Modules

#### Core Evaluation (`src/fmri2img/eval/`)

1. **`embedding_eval.py`** - Standard embedding metrics
   - Retrieval@K (Top-1, Top-5, Top-10)
   - Mean/Median rank, MRR
   - Two-way identification (2AFC)
   - Matched vs mismatched separability (AUC)
   - Representational Similarity Analysis (RSA)
   - Collapse diagnostics
   - Gallery size scaling analysis

2. **`probabilistic_eval.py`** - Novel Bayesian metrics
   - Proper scoring rules (NLL, Energy Score)
   - Distribution-aware ("Bayesian") retrieval
   - Probabilistic 2AFC with uncertainty propagation
   - Calibration analysis (reliability diagrams)
   - Risk-coverage curves (selective prediction)
   - Conformal prediction sets (distribution-free)

3. **`recon_eval.py`** - Reconstruction metrics
   - Low-level: PixCorr, SSIM, PSNR
   - Perceptual: LPIPS
   - Semantic: CLIP-image similarity, 2AFC identification

4. **`sampling_eval.py`** - Sampling evaluation
   - Expected-of-N vs Best-of-N analysis
   - Diversity metrics
   - Oracle upper bounds

5. **`report_generation.py`** - Visualization and reporting
   - Calibration curves
   - Risk-coverage plots
   - Retrieval vs gallery size
   - Expected/best-of-N curves
   - Comprehensive markdown reports

#### CLI Scripts (`scripts/`)

- **`eval_stage1_embeddings.py`** - Standard embedding evaluation
- **`eval_stage1_probabilistic.py`** - Probabilistic evaluation

#### Tests (`tests/`)

- **`test_evaluation_sanity.py`** - Critical sanity tests
  - Oracle retrieval (must be 100%)
  - Random baseline (must be near chance)
  - Calibration correctness
  - NLL computation verification
  - Reproducibility checks

---

## 🚀 Quick Start

### 1. Run Sanity Tests (Important!)

Before using the evaluation suite, verify correctness:

```bash
# Direct execution
python tests/test_evaluation_sanity.py

# Or with pytest
pytest tests/test_evaluation_sanity.py -v
```

**Expected output**: All tests pass with ✅ 

If any test fails, there's a bug in the evaluation code or data alignment.

### 2. Evaluate Stage 1 Embeddings

Standard metrics (retrieval, 2AFC, RSA, collapse):

```bash
python scripts/eval_stage1_embeddings.py \
    --checkpoint runs/20260115_172845_ultimate_novel_subj01/checkpoints/best_model.pt \
    --config experiments/ultimate_novel_subj01.yaml \
    --output-dir experimental_results/exp001/evaluation \
    --num-samples 1000 \
    --gallery-sizes 2 10 50 100 1000 \
    --seed 42
```

**Outputs**:
- `experimental_results/exp001/evaluation/embedding_metrics_val.json`
- `experimental_results/exp001/evaluation/embedding_summary_val.md`

**Key metrics to check**:
- Top-1 retrieval > 10% = strong
- 2AFC accuracy > 60% = well above chance
- Collapse ratio ≈ 1.0 = no collapse

### 3. Evaluate Probabilistic (Novel)

Bayesian metrics for uncertainty quantification:

```bash
python scripts/eval_stage1_probabilistic.py \
    --checkpoint runs/20260115_172845_ultimate_novel_subj01/checkpoints/best_model.pt \
    --config experiments/ultimate_novel_subj01.yaml \
    --output-dir experimental_results/exp001/evaluation \
    --mc-samples 64 \
    --run-conformal \
    --seed 42
```

**Outputs**:
- `experimental_results/exp001/evaluation/probabilistic_metrics_val.json`
- `experimental_results/exp001/evaluation/probabilistic_summary_val.md`
- `experimental_results/exp001/evaluation/plots/calibration_curve.png`
- `experimental_results/exp001/evaluation/plots/risk_coverage_curve.png`

**Key metrics to check**:
- Bayesian retrieval > Cosine baseline = uncertainty helps
- Calibration error < 5% = well-calibrated
- AURC: lower is better (selective prediction works)

### 4. Evaluate Reconstructions (Stage 2)

```python
# Example usage (implement CLI script similarly)
from fmri2img.eval.recon_eval import evaluate_reconstructions, LPIPSEvaluator, CLIPImageEvaluator

# Load images
imgs_recon = [...]  # List of numpy arrays (H, W, 3) in [0, 1]
imgs_gt = [...]

# Evaluate
lpips_eval = LPIPSEvaluator(device='cuda')
clip_eval = CLIPImageEvaluator(clip_model, preprocess, device='cuda')

results = evaluate_reconstructions(
    imgs_recon, imgs_gt,
    lpips_evaluator=lpips_eval,
    clip_evaluator=clip_eval
)

print(results.to_dict())
```

### 5. Evaluate Sampling (End-to-End)

```python
from fmri2img.eval.sampling_eval import evaluate_sampling_from_disk

results = evaluate_sampling_from_disk(
    imgs_gt_dir='outputs/gt_images/',
    recon_dirs=[
        'outputs/recons/sample_0/',
        'outputs/recons/sample_1/',
        'outputs/recons/sample_2/',
        'outputs/recons/sample_3/',
    ],
    n_values=[1, 2, 4, 8]
)

print(results.to_dict())
```

---

## 📊 Output Structure

```
experimental_results/exp001/evaluation/
├── embedding_metrics_val.json          # Standard metrics (JSON)
├── embedding_summary_val.md            # Human-readable summary
├── probabilistic_metrics_val.json      # Bayesian metrics (JSON)
├── probabilistic_summary_val.md        # Probabilistic summary
├── plots/
│   ├── calibration_curve.png           # Reliability diagram
│   ├── risk_coverage_curve.png         # Selective prediction
│   ├── retrieval_vs_gallery_size.png   # Scaling analysis
│   └── sampling_*.png                  # Sampling curves
└── full_evaluation_report.md           # Combined report
```

---

## 🎯 Interpretation Guidelines

### Stage 1: Embedding Metrics

| Metric | Good | Excellent | Interpretation |
|--------|------|-----------|----------------|
| Top-1 Retrieval | >10% | >15% | Correct image in #1 position |
| Top-5 Retrieval | >40% | >50% | Correct image in top-5 |
| 2AFC Accuracy | >60% | >70% | Forced-choice identification |
| Mean Rank | <100 | <50 | Average position (lower better) |
| RSA Spearman | >0.3 | >0.5 | Structure preservation |
| Collapse Ratio | 0.8-1.2 | 0.9-1.1 | No mode collapse |
| ROC AUC | >0.7 | >0.85 | Match/mismatch separability |

**Chance baselines**:
- Top-1 = 1/N (e.g., 0.1% for N=1000)
- 2AFC = 50%
- Mean rank = N/2

### Stage 1: Probabilistic Metrics (Novel)

| Metric | Good | Excellent | Interpretation |
|--------|------|-----------|----------------|
| Bayesian improvement | >0% | >5% | vs cosine baseline |
| Calibration error | <10% | <5% | Reliability of uncertainties |
| AURC | <0.2 | <0.15 | Selective prediction quality |
| Prob 2AFC | >55% | >65% | Uncertainty-aware identification |

**Key questions**:
1. Does Bayesian retrieval beat cosine? → Uncertainty is useful
2. Is calibration error <5%? → Model is well-calibrated
3. Does risk decrease with coverage? → Can abstain on hard examples

### Stage 2: Reconstruction Metrics

| Metric | Good | Excellent | Range |
|--------|------|-----------|-------|
| PixCorr | >0.25 | >0.35 | [-1, 1] |
| SSIM | >0.30 | >0.45 | [0, 1] |
| PSNR | >18 dB | >22 dB | dB scale |
| LPIPS | <0.50 | <0.35 | [0, ∞), lower better |
| CLIP-2AFC | >60% | >70% | [0, 100%] |

### Sampling: Expected vs Best-of-N

- **Expected-of-N**: Average quality across N samples (should be stable)
- **Best-of-N**: Best quality among N samples (should improve with N)
- **Gap**: Best - Expected (larger gap = higher diversity)

**Interpretation**:
- If Best-of-N >> Expected-of-N: Sampling helps! Use best-of-N decoding.
- If Best-of-N ≈ Expected-of-N: Low diversity, deterministic model is fine.

---

## 🔬 Scientific Context

### Standard Metrics (Literature Precedent)

1. **Retrieval@K**: Ozcelik & VanRullen (2023), Scotti et al. (2023) - MindEye, Brain Diffuser
2. **2AFC**: Green & Swets (1966) - Signal detection theory from cognitive neuroscience
3. **RSA**: Kriegeskorte et al. (2008) - Representational similarity analysis
4. **PixCorr, SSIM, PSNR**: Naselaris et al. (2009), Wang et al. (2004)
5. **LPIPS**: Zhang et al. (2018) - Perceptual similarity

### Novel Contributions (This Work)

1. **Bayesian Retrieval**: Uses full posterior q(z|x), not just mean μ
   - Score candidates by log-likelihood under predicted distribution
   - Principled way to incorporate uncertainty in retrieval

2. **Proper Scoring Rules**: NLL and Energy Score (Gneiting & Raftery 2007)
   - Incentivize well-calibrated probabilistic predictions
   - Energy Score: multivariate, uses samples directly

3. **Calibration Analysis**: Reliability diagrams for neural decoding
   - Are 90% credible regions correct 90% of the time?
   - Uses Mahalanobis distance + Chi-square quantiles

4. **Risk-Coverage Curves**: Selective prediction (Geifman & El-Yaniv 2017)
   - Can model abstain on uncertain predictions?
   - Plots error vs coverage fraction

5. **Conformal Prediction**: Distribution-free uncertainty (Vovk et al. 2005)
   - Provable coverage guarantees under exchangeability
   - Complements Bayesian approach

6. **Probabilistic 2AFC**: Uncertainty propagation through identification
   - Estimate P(correct) via Monte Carlo sampling
   - More principled than thresholding on mean

7. **Sampling Evaluation**: Expected-of-N vs Best-of-N
   - Quantifies benefit of generating multiple samples
   - Inspired by nucleus sampling (Holtzman et al. 2020)

---

## 🛠️ Implementation Details

### Dependencies

Core:
- `numpy`, `scipy`, `scikit-learn`
- `torch`, `torchvision`
- `matplotlib`, `seaborn`

Optional:
- `lpips` (for LPIPS metric): `pip install lpips`
- `scikit-image` (for SSIM/PSNR): `pip install scikit-image`
- `open_clip` or `clip` (for CLIP-image eval)

### Performance Tips

1. **Cache embeddings**: Save predictions/GT to disk (`.npz`) to avoid recomputation
2. **Use GPU**: LPIPS and CLIP encoding benefit from GPU
3. **Batch processing**: Use provided batch functions for LPIPS/CLIP
4. **Gallery size**: For large N (>10k), use faiss for fast retrieval (not implemented yet, uses numpy)
5. **MC samples**: 64 samples usually sufficient for Energy Score

### Extensibility

To add new metrics:

1. **Add function to appropriate module** (`embedding_eval.py`, `probabilistic_eval.py`, etc.)
2. **Update dataclass** (`EmbeddingEvalResults`, `ProbabilisticEvalResults`)
3. **Update CLI script** to call new function
4. **Add sanity test** to `test_evaluation_sanity.py`
5. **Update report generation** to include new metric

Example:
```python
# In embedding_eval.py
def compute_new_metric(predictions, ground_truth):
    # ...
    return {'new_metric': value}

# In dataclass
@dataclass
class EmbeddingEvalResults:
    # ... existing fields
    new_metric: float
    
# In evaluate_embeddings()
new_results = compute_new_metric(predictions, ground_truth)
return EmbeddingEvalResults(
    # ... existing
    new_metric=new_results['new_metric']
)
```

---

## 📝 Citation

If you use this evaluation suite, please cite:

```bibtex
@software{fmri_eval_suite_2026,
  title={Research-Grade Evaluation Suite for Probabilistic fMRI-to-Image Reconstruction},
  author={[Your Name]},
  year={2026},
  url={https://github.com/toniIepure25/FMRI2images}
}
```

And relevant papers:
- Gneiting & Raftery (2007) - Proper scoring rules
- Kriegeskorte et al. (2008) - RSA
- Vovk et al. (2005) - Conformal prediction
- Ozcelik & VanRullen (2023) - Neural decoding benchmarks

---

## 🐛 Troubleshooting

### "Oracle retrieval is not 100%"

**Cause**: ID misalignment between predictions and ground truth.

**Fix**:
1. Check dataset returns samples in consistent order
2. Verify predictions[i] corresponds to ground_truth[i]
3. Run sanity test: `python tests/test_evaluation_sanity.py`

### "Calibration error is very high (>20%)"

**Causes**:
1. Model variances are poorly estimated
2. Training didn't converge
3. KL weight too low/high

**Fixes**:
1. Check training loss curves (NLL, KL divergence)
2. Tune KL annealing schedule
3. Verify logvar is not clamped too aggressively

### "Bayesian retrieval worse than cosine"

**Causes**:
1. Predicted variances are not informative
2. All variances are similar (collapse)
3. Implementation bug

**Fixes**:
1. Check variance statistics: `np.exp(logvar).mean()`
2. Visualize uncertainty vs error correlation
3. Run sanity tests

### "LPIPS/CLIP evaluator fails"

**Cause**: Missing dependencies or CUDA out of memory.

**Fixes**:
```bash
pip install lpips scikit-image
# If OOM, reduce batch size or use CPU
lpips_eval = LPIPSEvaluator(device='cpu')
```

---

## ✅ Checklist for Paper-Ready Evaluation

Before submitting/publishing results:

- [ ] Run sanity tests (`test_evaluation_sanity.py`) - all pass
- [ ] Evaluate on held-out test set (not validation)
- [ ] Report sample sizes (N) and confidence intervals where applicable
- [ ] Include chance baselines in tables/plots
- [ ] Generate calibration curves for probabilistic model
- [ ] Report both standard (cosine retrieval) and novel (Bayesian) metrics
- [ ] Compare multiple experiments side-by-side
- [ ] Document hyperparameters (MC samples, gallery sizes, seeds)
- [ ] Save all metrics as JSON for reproducibility
- [ ] Include representative reconstructions (not just metrics)

---

## 📞 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check sanity tests first: `python tests/test_evaluation_sanity.py`
- Provide: checkpoint path, config, error message, sanity test results

---

**Happy Evaluating! 🎉**
