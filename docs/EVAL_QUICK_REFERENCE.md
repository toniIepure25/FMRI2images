# Evaluation Suite Quick Reference

## 🎯 Quick Start (3 Commands)

```bash
# 1. Verify correctness
python3 tests/test_evaluation_sanity.py

# 2. Run standard metrics
python3 scripts/eval_stage1_embeddings.py \
    --checkpoint YOUR_CHECKPOINT.pt \
    --config YOUR_CONFIG.yaml \
    --output-dir results/

# 3. Run novel Bayesian metrics
python3 scripts/eval_stage1_probabilistic.py \
    --checkpoint YOUR_CHECKPOINT.pt \
    --config YOUR_CONFIG.yaml \
    --output-dir results/
```

## 📊 Key Metrics At-A-Glance

### Standard (compare to literature)

| Metric              | Good | Chance | Interpretation         |
| ------------------- | ---- | ------ | ---------------------- |
| **Top-1 Retrieval** | >10% | 1/N    | Correct in #1 position |
| **Top-5 Retrieval** | >40% | 5/N    | Correct in top-5       |
| **2AFC Accuracy**   | >60% | 50%    | Forced-choice ID       |
| **Mean Rank**       | <100 | N/2    | Average position       |

### Novel (this work - for your thesis)

| Metric                 | Good | Interpretation               |
| ---------------------- | ---- | ---------------------------- |
| **Bayesian vs Cosine** | >0%  | Does uncertainty help?       |
| **Calibration Error**  | <5%  | Are uncertainties reliable?  |
| **AURC**               | <0.2 | Selective prediction quality |

## 📁 What Gets Evaluated

```
Stage 1: fMRI → CLIP Embedding
├─ Standard metrics (compare to MindEye, Brain Diffuser)
│  ├─ Retrieval@K (Top-1, Top-5, Top-10)
│  ├─ 2AFC identification
│  ├─ RSA (structure preservation)
│  └─ Collapse detection
└─ Novel Bayesian metrics (your contribution!)
   ├─ Bayesian retrieval (uses full posterior)
   ├─ Calibration (reliability diagrams)
   ├─ Risk-coverage (selective prediction)
   └─ Conformal sets (distribution-free)

Stage 2: CLIP → Image Reconstruction
├─ Low-level: PixCorr, SSIM, PSNR
├─ Perceptual: LPIPS
└─ Semantic: CLIP-image similarity, 2AFC

End-to-End: Sampling
├─ Expected-of-N (average quality)
├─ Best-of-N (max quality)
└─ Diversity (variance across samples)
```

## 🔬 Interpretation Guide

### If Top-1 retrieval is...

- **<5%**: Model barely works, check training
- **5-10%**: Moderate, baseline-level performance
- **10-15%**: Good, competitive with literature
- **>15%**: Excellent, state-of-the-art level

### If Bayesian retrieval improvement is...

- **<0%**: Uncertainty not helping (check calibration)
- **0-3%**: Modest improvement
- **>3%**: Strong evidence uncertainty is useful

### If calibration error is...

- **<5%**: Well-calibrated, uncertainties are reliable
- **5-10%**: Moderately calibrated
- **>10%**: Poorly calibrated, uncertainty unreliable

## 🐛 Common Issues

### "Oracle retrieval is not 100%"

→ **Bug**: ID misalignment. Predictions[i] doesn't match GT[i].
→ **Fix**: Check dataset ordering, run sanity tests.

### "Random baseline is not 50%"

→ **Expected**: Statistical variation with small N.
→ **Fix**: Increase trials or tolerance in tests.

### "Bayesian worse than cosine"

→ **Cause**: Poorly estimated variances.
→ **Fix**: Check KL weight, variance statistics, calibration.

## 📖 Files You Need

### Core Modules (in `src/fmri2img/eval/`)

- `embedding_eval.py` - Standard metrics
- `probabilistic_eval.py` - Novel Bayesian metrics
- `recon_eval.py` - Image reconstruction metrics
- `sampling_eval.py` - End-to-end sampling
- `report_generation.py` - Plots and reports

### CLI Scripts (in `scripts/`)

- `eval_stage1_embeddings.py` - Run standard eval
- `eval_stage1_probabilistic.py` - Run Bayesian eval
- `eval_comprehensive.py` - Run all stages

### Tests (in `tests/`)

- `test_evaluation_sanity.py` - Critical correctness checks

## 🎓 For Your Thesis

### Standard Section

"We evaluate using standard retrieval metrics (Top-K, 2AFC) following Ozcelik & VanRullen (2023)."

### Novel Contributions Section

"We extend evaluation to probabilistic predictions using:

1. **Bayesian Retrieval**: Log-likelihood scoring (Gneiting & Raftery 2007)
2. **Calibration Analysis**: Reliability diagrams for neural decoding
3. **Risk-Coverage**: Selective prediction (Geifman & El-Yaniv 2017)
4. **Conformal Sets**: Distribution-free uncertainty (Vovk et al. 2005)"

### Tables to Include

1. Retrieval comparison (your model vs baselines)
2. Bayesian vs Cosine (show improvement from uncertainty)
3. Calibration table (nominal vs empirical coverage)
4. Sampling analysis (Expected vs Best-of-N)

### Figures to Include

1. Calibration curve (shows reliability)
2. Risk-coverage plot (shows selective prediction)
3. Retrieval vs gallery size (shows scaling)
4. Sampling curves (shows benefit of multiple samples)

## 💡 Pro Tips

1. **Always run sanity tests first** - catches bugs early
2. **Use validation set for development** - save test set for final eval
3. **Report chance baselines** - readers need context
4. **Include confidence intervals** - for 2AFC, bootstrap estimates
5. **Compare to literature** - use same metrics as MindEye, etc.
6. **Document everything** - save configs, seeds, sample sizes
7. **Generate plots** - visualizations are more convincing than tables

## 🚀 Reproducibility Checklist

- [ ] Fixed random seed (--seed 42)
- [ ] Same data split (validation vs test)
- [ ] Documented sample size (N=?)
- [ ] Saved full metrics (JSON files)
- [ ] Recorded checkpoint path
- [ ] Noted hyperparameters (MC samples, gallery sizes)
- [ ] Ran sanity tests (all pass)
- [ ] Generated plots (saved PNGs)

---

**Need more details?** → See `docs/EVALUATION_SUITE_README.md`
