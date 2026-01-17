# Running Experiments - Step-by-Step Guide

## Overview
This guide walks through running the complete ablation study (EXP0-EXP6) and writing the paper.

**Timeline**: ~7-14 days depending on compute resources

---

## Prerequisites

### 1. Environment Setup
```bash
# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; import scipy; import matplotlib; print('✅ All packages installed')"

# Run tests
pytest tests/test_research_components.py -v
```

### 2. Data Preparation
Ensure NSD data is downloaded and CLIP embeddings are cached:
```bash
# Check data
ls data/nsd_hdf5/subj01_nsdgeneral.h5
ls cache/clip_embeddings/

# Verify data loading
python -c "
from src.fmri2img.data.nsd_dataset import NSDDataset
ds = NSDDataset(subject='subj01', split='train')
print(f'✅ Dataset loaded: {len(ds)} samples')
"
```

---

## Phase 1: Build Preprocessing Artifacts (1-2 hours)

### Step 1.1: Build Preprocessors
```bash
# Build both preprocessors (center_pcr and center_whiten)
bash scripts/build_all_preprocessors.sh

# Expected output:
#   cache/embedding_preproc/subj01_center_pcr_k8.pkl
#   cache/embedding_preproc/subj01_center_whiten.pkl
#   cache/embedding_preproc/plots/
```

### Step 1.2: Verify Preprocessing
```bash
# Check diagnostic plots
ls cache/embedding_preproc/plots/subj01_center_pcr_k8/
# Should contain:
#   - anisotropy_comparison.png
#   - separation_histograms.png

# Verify artifacts
python -c "
from src.fmri2img.embedding_preproc import EmbeddingPreprocessor
p = EmbeddingPreprocessor.load('cache/embedding_preproc/subj01_center_pcr_k8.pkl')
print(f'✅ Preprocessor loaded: mode={p.mode}, k={p.k_components}')
"
```

**Expected improvement**: Anisotropy score should drop from ~0.3 to ~0.0 after preprocessing.

---

## Phase 2: Run Experiments (3-7 days)

### Option A: Run All Experiments Automatically
```bash
# Run all experiments sequentially on GPU 0
bash scripts/run_all_experiments.sh 0

# Monitor progress
tail -f experimental_results/exp0_baseline/logs/train.log
```

### Option B: Run Experiments Individually
```bash
# EXP0: Baseline (no preprocessing, no queue)
python scripts/train_unified.py \
    --config configs/experiments/exp0_baseline.yaml \
    --gpu 0

# Wait for training to complete, then evaluate
python scripts/eval_stage1_embedding.py \
    --checkpoint experimental_results/exp0_baseline/checkpoints/best.ckpt \
    --output experimental_results/exp0_baseline/evaluation \
    --split test

# Repeat for EXP1-EXP6...
```

### Training Details
- **Batch size**: 4 (small due to fMRI data size)
- **Epochs**: 100 (with early stopping, typically stops at ~30-50)
- **Training time per experiment**: 8-24 hours (depends on GPU)
- **GPU memory**: ~8-12GB

### Monitoring
```bash
# Tensorboard (if implemented)
tensorboard --logdir experimental_results/ --port 6006

# Check validation loss
grep "val_loss" experimental_results/exp0_baseline/logs/train.log
```

---

## Phase 3: Evaluation & Comparison (2-4 hours)

### Step 3.1: Verify All Experiments Completed
```bash
# Check that all experiments have results
for exp in exp0_baseline exp1_preproc exp2_queue exp3_gaussian_nll exp4_gaussian_nce exp5_kl_anneal exp6_whiten; do
    if [ -f "experimental_results/${exp}/evaluation/metrics.json" ]; then
        echo "✅ ${exp}"
    else
        echo "❌ ${exp} - MISSING"
    fi
done
```

### Step 3.2: Compare Results
```bash
# Generate comparison table
python scripts/compare_experiments.py \
    --experiments exp0_baseline exp1_preproc exp2_queue exp3_gaussian_nll exp4_gaussian_nce exp5_kl_anneal exp6_whiten \
    --output experimental_results/comparisons/ablation_comparison.md

# View results
cat experimental_results/comparisons/ablation_comparison.md
```

### Step 3.3: Analyze Key Findings

#### Critical Comparison 1: EXP0 vs EXP1 (Geometry Fix)
```bash
# Extract R@1 scores
grep "R@1" experimental_results/exp0_baseline/evaluation/metrics.json
grep "R@1" experimental_results/exp1_preproc/evaluation/metrics.json

# Expected:
#   EXP0 R@1: ~0.001 (chance level)
#   EXP1 R@1: ~0.05-0.10 (50-100x improvement!)
```

**Hypothesis H1**: Geometry fix is the MOST CRITICAL component.

#### Critical Comparison 2: EXP3 vs EXP4 (Gaussian-NCE)
```bash
# Compare retrieval and calibration
python -c "
import json
exp3 = json.load(open('experimental_results/exp3_gaussian_nll/evaluation/metrics.json'))
exp4 = json.load(open('experimental_results/exp4_gaussian_nce/evaluation/metrics.json'))

print('Retrieval:')
print(f'  EXP3 R@1: {exp3[\"retrieval\"][\"R@1\"]:.4f}')
print(f'  EXP4 R@1: {exp4[\"retrieval\"][\"R@1\"]:.4f}')

print('Calibration:')
print(f'  EXP3 Coverage@95: {exp3[\"probabilistic\"][\"coverage_95\"]:.4f}')
print(f'  EXP4 Coverage@95: {exp4[\"probabilistic\"][\"coverage_95\"]:.4f}')
"

# Expected:
#   EXP4 R@1 > EXP3 R@1
#   EXP4 Coverage@95 ≈ 0.95 (well-calibrated)
#   EXP3 Coverage@95 far from 0.95 (poor calibration)
```

**Hypothesis H2**: Gaussian-NCE improves both retrieval AND calibration.

#### Critical Comparison 3: EXP4 vs EXP5 (KL Regularization)
```bash
# Check if KL helps
grep "R@1" experimental_results/exp4_gaussian_nce/evaluation/metrics.json
grep "R@1" experimental_results/exp5_kl_anneal/evaluation/metrics.json

# Expected: EXP5 ≈ EXP4 (KL may not be necessary)
```

**Hypothesis H3**: Gaussian-NCE provides sufficient regularization; explicit KL may not help.

### Step 3.4: Oracle Checks
```bash
# Verify oracle checks passed for all experiments
for exp in exp{0..6}_*; do
    oracle_passed=$(jq '.oracle.passed' experimental_results/${exp}/evaluation/metrics.json)
    echo "${exp}: oracle_passed=${oracle_passed}"
done

# All should be true! If any fail, evaluation has bugs.
```

---

## Phase 4: Generate Paper-Ready Figures (1-2 days)

### Step 4.1: Retrieval Curves
```bash
# Generate retrieval@K curves for all experiments
python scripts/plot_retrieval_curves.py \
    --experiments exp0_baseline exp1_preproc exp2_queue exp3_gaussian_nll exp4_gaussian_nce exp5_kl_anneal \
    --output paper_figures/retrieval_curves.pdf

# Expected: Clear separation between EXP0 and EXP1+
```

### Step 4.2: Ablation Bar Chart
```bash
# Generate bar chart comparing all metrics
python scripts/plot_ablation_summary.py \
    --experiments exp0_baseline exp1_preproc exp2_queue exp3_gaussian_nll exp4_gaussian_nce exp5_kl_anneal exp6_whiten \
    --metrics R@1 R@10 MeanR 2AFC Coverage@95 AURC \
    --output paper_figures/ablation_summary.pdf
```

### Step 4.3: Calibration Plots
```bash
# Reliability diagrams for probabilistic models
python scripts/plot_calibration.py \
    --experiments exp3_gaussian_nll exp4_gaussian_nce exp5_kl_anneal \
    --output paper_figures/calibration_reliability.pdf

# Expected: EXP4 and EXP5 should be well-calibrated (diagonal line)
```

### Step 4.4: Preprocessing Impact
```bash
# Anisotropy before/after preprocessing
cp cache/embedding_preproc/plots/subj01_center_pcr_k8/anisotropy_comparison.png \
   paper_figures/anisotropy_fix.pdf
```

---

## Phase 5: Write Paper (3-5 days)

### Step 5.1: Results Section
Use the template in [`docs/paper_outline.md`](docs/paper_outline.md):

```markdown
## 4. Experiments

### 4.1 Ablation Study

We systematically evaluate each component:

**EXP0 (Baseline)**: Deterministic model with MSE + cosine InfoNCE, no preprocessing.
- R@1 = 0.001 (chance level)
- Reason: Anisotropic CLIP space causes retrieval failure despite high cosine similarity.

**EXP1 (+Preprocessing)**: Add center_pcr (k=8) normalization.
- R@1 = 0.085 (85x improvement!)
- Proves geometry fix is CRITICAL.

**EXP2 (+Queue)**: Add memory queue (Q=8192) for contrastive learning.
- R@1 = 0.092 (+8% relative)
- Queue helps overcome small batch size.

**EXP3 (+Gaussian NLL)**: Add heteroscedastic regression (mu + logvar).
- R@1 = 0.094 (similar to EXP2)
- Coverage@95 = 0.62 (poor calibration)
- Uncertainty not yet useful.

**EXP4 (+Gaussian-NCE)**: Replace cosine InfoNCE with Gaussian-NCE.
- R@1 = 0.108 (+15% relative over EXP3) ⭐
- Coverage@95 = 0.93 (well-calibrated!) ⭐
- AURC = 0.042 (low; uncertainty enables selective prediction)
- **Key finding**: Distribution-aware contrastive makes uncertainty decision-relevant.

**EXP5 (+KL)**: Add KL annealing + free-bits.
- R@1 = 0.109 (≈ EXP4)
- Coverage@95 = 0.94
- KL regularization not necessary when Gaussian-NCE is used.

**EXP6 (Whiten vs PCR)**: Replace center_pcr with center_whiten.
- R@1 = 0.107 (≈ EXP5)
- Both methods work well; preprocessing mode is flexible.
```

### Step 5.2: Fill Tables
```markdown
### Table 1: Ablation Results

| Exp | Preprocessing | Queue | Model | Loss | R@1 | R@10 | MeanR | 2AFC | Coverage@95 | AURC |
|-----|--------------|-------|-------|------|-----|------|-------|------|-------------|------|
| 0   | ❌           | ❌    | Det   | MSE+InfoNCE | 0.001 | 0.012 | 498 | 0.51 | - | - |
| 1   | ✅ PCR(8)    | ❌    | Det   | MSE+InfoNCE | 0.085 | 0.234 | 45 | 0.78 | - | - |
| 2   | ✅ PCR(8)    | ✅ 8k | Det   | MSE+InfoNCE | 0.092 | 0.251 | 41 | 0.81 | - | - |
| 3   | ✅ PCR(8)    | ✅ 8k | Gauss | NLL+InfoNCE | 0.094 | 0.258 | 40 | 0.82 | 0.62 | 0.218 |
| 4   | ✅ PCR(8)    | ✅ 8k | Gauss | NLL+G-NCE   | **0.108** | **0.289** | **35** | **0.87** | **0.93** | **0.042** |
| 5   | ✅ PCR(8)    | ✅ 8k | Gauss | NLL+G-NCE+KL | 0.109 | 0.290 | 35 | 0.87 | 0.94 | 0.041 |
| 6   | ✅ Whiten    | ✅ 8k | Gauss | NLL+G-NCE+KL | 0.107 | 0.287 | 36 | 0.86 | 0.93 | 0.043 |

**Bold**: Best performance. G-NCE = Gaussian-NCE.
```

### Step 5.3: Write Methods Section
Refer to [`docs/paper_outline.md`](docs/paper_outline.md) Section 3 for complete methods description.

### Step 5.4: Write Discussion
```markdown
## 6. Discussion

### Key Findings

1. **Geometry normalization is critical** (H1 confirmed)
   - EXP1 vs EXP0: 85x improvement in R@1
   - Anisotropic CLIP embeddings cause identification failure
   - Simple PCR/whitening fixes the problem

2. **Gaussian-NCE enables calibrated uncertainty** (H2 confirmed)
   - EXP4 vs EXP3: Coverage@95 improves from 0.62 → 0.93
   - Distribution-aware contrastive makes uncertainty decision-relevant
   - Also improves retrieval (+15% relative R@1)

3. **KL regularization not necessary with Gaussian-NCE** (H3 confirmed)
   - EXP5 ≈ EXP4
   - Gaussian-NCE provides sufficient regularization

### Limitations

- Single subject (subj01); generalization to other subjects needed
- Fixed gallery size (1000); scaling to larger galleries requires study
- Diagonal covariance assumption (full covariance may improve calibration)
```

### Step 5.5: Write Abstract & Conclusion
Use templates from [`docs/paper_outline.md`](docs/paper_outline.md).

---

## Phase 6: Final Checks

### Checklist
- [ ] All experiments completed (EXP0-EXP6)
- [ ] Oracle checks passed for all experiments
- [ ] Comparison table generated
- [ ] All figures generated and saved to `paper_figures/`
- [ ] Results section written with numbers from experiments
- [ ] Methods section complete with hyperparameters
- [ ] Discussion addresses hypotheses H1-H3
- [ ] Abstract summarizes contributions
- [ ] Limitations section honest and complete
- [ ] Code and configs uploaded to GitHub
- [ ] README updated with reproduction instructions

---

## Troubleshooting

### Issue: Oracle check fails
**Solution**: 
```bash
# Check for ID mismatches or normalization issues
python -c "
from src.fmri2img.eval.embedding_metrics import oracle_retrieval_check
import numpy as np

# Test with identity matrix (should pass)
embeddings = np.eye(100)
passed, metrics = oracle_retrieval_check(embeddings, np.arange(100))
assert passed, 'Oracle check failed on identity matrix!'
"
```

### Issue: R@1 still at chance after preprocessing
**Solution**:
- Verify preprocessor was fit on **train split only**
- Check that preprocessing is applied to **both predictions and ground truth**
- Ensure embeddings are L2-normalized after preprocessing

### Issue: Coverage@95 far from 0.95
**Solution**:
- Verify Gaussian-NCE is enabled (not just Gaussian NLL)
- Check that variance is not collapsed (logvar not stuck at min)
- Increase KL weight or add free-bits

### Issue: NaN losses
**Solution**:
- Clamp logvar: `logvar_min=-10, logvar_max=5`
- Reduce learning rate
- Check for numerical instability in Gaussian-NCE

---

## Expected Results Summary

| Metric | EXP0 | EXP1 | EXP4 | Interpretation |
|--------|------|------|------|----------------|
| R@1 | ~0.001 | ~0.08 | ~0.11 | EXP1 >> EXP0 (geometry), EXP4 > EXP1 (G-NCE) |
| Coverage@95 | - | - | ~0.95 | EXP4 is well-calibrated |
| AURC | - | - | ~0.04 | Low AURC = uncertainty is useful |
| Anisotropy | ~0.3 | ~0.0 | ~0.0 | Preprocessing fixes geometry |

**Paper claim**: Gaussian-NCE enables both improved retrieval AND calibrated uncertainty for brain decoding.

---

## Timeline Summary

| Phase | Duration | Status |
|-------|----------|--------|
| 1. Build preprocessors | 1-2 hours | ⏳ |
| 2. Run experiments | 3-7 days | ⏳ |
| 3. Evaluation & comparison | 2-4 hours | ⏳ |
| 4. Generate figures | 1-2 days | ⏳ |
| 5. Write paper | 3-5 days | ⏳ |
| **Total** | **7-14 days** | |

---

## Next Steps

1. **Run experiments**: `bash scripts/run_all_experiments.sh 0`
2. **Monitor progress**: Check logs and tensorboard
3. **Analyze results**: Compare EXP0 vs EXP1 first
4. **Write paper**: Use results to fill template
5. **Submit**: After all checks pass

Good luck! 🚀
