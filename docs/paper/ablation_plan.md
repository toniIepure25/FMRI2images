# Ablation Plan

## Overview
Systematic ablation study to isolate the contribution of each component. All experiments use the same dataset, split, seeds, and evaluation protocol.

## Experimental Conditions

### EXP0: Baseline (Deterministic)
**Purpose**: Establish baseline with standard approach.

**Configuration**:
- Model: Deterministic (single output, no variance)
- Loss: MSE + Cosine similarity (standard InfoNCE)
- Preprocessing: ❌ None
- Queue: ❌ No memory queue
- Batch size: 4 (same as all experiments)

**Expected**:
- High cosine similarity (~0.6-0.8)
- **Chance-level retrieval** due to anisotropy (R@1 ≈ 1/N)
- No uncertainty estimates

**Proves**: Without geometry fix, identification fails despite high similarity.

---

### EXP1: + Embedding Preprocessing
**Purpose**: Isolate effect of geometry normalization.

**Changes from EXP0**:
- ✅ Preprocessing: center_pcr with k=8 components removed
- Everything else same as EXP0

**Expected**:
- **Dramatic improvement** in R@1 (from ~0.001 to ~0.05-0.10)
- Mean rank decreases substantially
- AUC(pos vs neg) increases significantly
- Anisotropy score drops from ~0.3 to ~0.0

**Proves**: Geometry normalization is the **single most critical component**.

**Hypothesis**: EXP1 >> EXP0 by X% absolute R@1

---

### EXP2: + Memory Queue for InfoNCE
**Purpose**: Test if queue-augmented contrastive learning helps.

**Changes from EXP1**:
- ✅ Queue: MoCo-style memory queue (size Q=8192)
- ✅ InfoNCE with queue negatives
- ✅ Learnable temperature (logit_scale)

**Expected**:
- Moderate improvement in retrieval (more negatives → better contrastive signal)
- Improved 2AFC accuracy
- Better separation of positive/negative pairs

**Proves**: Queue helps overcome small batch size limitation.

**Hypothesis**: EXP2 > EXP1 by Y% absolute R@1

---

### EXP3: + Gaussian NLL (Heteroscedastic Regression)
**Purpose**: Introduce probabilistic predictions without distribution-aware contrastive.

**Changes from EXP2**:
- ✅ Model outputs: mu + logvar
- ✅ Loss: Gaussian NLL (replaces MSE)
- ✅ Inference: use_mean_for_retrieval=true (deterministic metrics use mu)
- ❌ Contrastive: Still uses cosine InfoNCE (not Gaussian-NCE yet)

**Expected**:
- Similar retrieval to EXP2 (still using mu for retrieval)
- **Uncalibrated uncertainty**: NLL high, Coverage@95 ≠ 0.95
- AURC poor (uncertainty not useful)

**Proves**: Heteroscedastic regression alone doesn't calibrate uncertainty.

**Hypothesis**: EXP3 ≈ EXP2 for retrieval, but poor calibration

---

### EXP4: + Gaussian-NCE (Novel Contribution)
**Purpose**: Test distribution-aware contrastive objective.

**Changes from EXP3**:
- ✅ Loss: Gaussian-NCE (uses likelihood as similarity score)
- ✅ Contrastive learning now optimizes for Bayesian retrieval

**Expected**:
- **Best retrieval performance**: R@1 highest, MeanR lowest
- **Calibrated uncertainty**: Coverage@95 ≈ 0.95
- **Low AURC**: Uncertainty enables effective selective prediction
- Probabilistic 2AFC works (likelihood ratios are meaningful)

**Proves**: Gaussian-NCE makes uncertainty decision-relevant and improves retrieval.

**Hypothesis**: EXP4 > EXP3 for both retrieval AND calibration

---

### EXP5: + KL Annealing + Free-Bits
**Purpose**: Test if explicit KL regularization is necessary.

**Changes from EXP4**:
- ✅ KL divergence term with annealing schedule
  - Start weight: 0.0
  - End weight: 0.001 (small, just regularization)
  - Annealing: Linear over 10,000 steps
- ✅ Free-bits: 0.5 per dimension (prevent collapse)

**Expected**:
- Similar or slightly better calibration
- Possible reduction in posterior collapse (if it exists)
- May not improve retrieval if Gaussian-NCE already provides sufficient regularization

**Proves**: Whether explicit KL is needed given Gaussian-NCE.

**Hypothesis**: EXP5 ≈ EXP4 (Gaussian-NCE may be sufficient)

---

### EXP6: Ablation - Whitening vs PCR
**Purpose**: Compare preprocessing methods.

**Changes from EXP5**:
- ✅ Preprocessing: center_whiten (instead of center_pcr)
- Everything else same as EXP5

**Expected**:
- Similar performance to EXP5
- Slightly different anisotropy scores
- Both methods should work well

**Proves**: Preprocessing mode is flexible; both center_pcr and center_whiten work.

**Hypothesis**: EXP6 ≈ EXP5

---

## Ablation Matrix

| Component | EXP0 | EXP1 | EXP2 | EXP3 | EXP4 | EXP5 | EXP6 |
|-----------|------|------|------|------|------|------|------|
| **Preprocessing** | ❌ | center_pcr | center_pcr | center_pcr | center_pcr | center_pcr | center_whiten |
| **Queue** | ❌ | ❌ | 8192 | 8192 | 8192 | 8192 | 8192 |
| **Model** | Deterministic | Deterministic | Deterministic | Gaussian | Gaussian | Gaussian | Gaussian |
| **Loss: NLL** | MSE | MSE | MSE | ✅ | ✅ | ✅ | ✅ |
| **Loss: InfoNCE** | Cosine | Cosine | Cosine+Queue | Cosine+Queue | ❌ | ❌ | ❌ |
| **Loss: Gaussian-NCE** | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Loss: KL** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| **Inference Mode** | mu | mu | mu | mu | mu | mu | mu |

## Evaluation Metrics per Experiment

### Core Metrics (All Experiments)
- Retrieval: R@1, R@5, R@10, MeanR, MRR, nDCG@10
- Identification: 2AFC, AUC, Cohen's d, d'
- Structure: RSA, CKA
- Oracle check: Pass/Fail

### Probabilistic Metrics (EXP3-EXP6 only)
- NLL per-dim, ΔNLL
- Energy Score (with sampling)
- Coverage@80, Coverage@95
- ECE, AURC
- Probabilistic 2AFC

## Key Comparisons

### Critical Test 1: Geometry Matters
**Compare**: EXP0 vs EXP1
**Metric**: R@1, Mean Rank, AUC
**Expected**: Massive improvement (10-100x R@1)
**Significance**: This is the biggest contribution

### Critical Test 2: Queue Helps Small Batches
**Compare**: EXP1 vs EXP2
**Metric**: R@1, 2AFC
**Expected**: Moderate improvement (1.5-2x R@1)
**Significance**: Validates queue design

### Critical Test 3: Gaussian-NCE Enables Calibration
**Compare**: EXP3 vs EXP4
**Metric**: Coverage@95, AURC, Prob-2AFC
**Expected**: EXP4 has Coverage@95 ≈ 0.95, low AURC; EXP3 does not
**Significance**: Proves Gaussian-NCE makes uncertainty useful

### Critical Test 4: Gaussian-NCE Also Improves Retrieval
**Compare**: EXP3 vs EXP4
**Metric**: R@1, MeanR
**Expected**: EXP4 > EXP3 for retrieval too
**Significance**: Dual benefit (retrieval + calibration)

### Critical Test 5: KL May Not Be Necessary
**Compare**: EXP4 vs EXP5
**Metric**: All metrics
**Expected**: EXP5 ≈ EXP4 (small or no improvement)
**Significance**: If true, simplifies training

## Hyperparameters (Shared Across Experiments)

### Model Architecture
- fMRI encoder: MLP [V → 4096 → 2048 → 1024]
- Embedding decoder: MLP [1024 → 1536 → 768]
- For Gaussian models (EXP3-6): Dual head for mu and logvar

### Training
- Optimizer: AdamW
- Learning rate: 1e-4 with cosine decay
- Batch size: 4 (all experiments)
- Epochs: 100 (early stopping on val loss)
- Gradient clipping: 1.0

### Loss Weights (EXP5-6)
- NLL: 1.0
- KL: Annealed 0.0 → 0.001
- Gaussian-NCE: 1.0

### Queue (EXP2-6)
- Size: 8192
- Update: Every batch (enqueue GT embeddings)
- Min size before use: 256

### Preprocessing (EXP1-6)
- center_pcr: k=8
- center_whiten: epsilon=1e-5

### Inference
- Deterministic metrics: Use mu only
- Probabilistic metrics: Sample S=64 times

## Statistical Testing

For each critical comparison, perform:
1. **Bootstrap test** (1000 iterations) for significance of R@1 difference
2. **Paired t-test** on per-sample metrics (2AFC, RSA)
3. Report **effect sizes** (Cohen's d) for all comparisons

**Significance threshold**: p < 0.05 (after Bonferroni correction for 5 tests)

## Expected Outcomes

### Success Criteria
1. EXP1 >> EXP0: R@1 improves by ≥5x (e.g., 0.001 → 0.005+)
2. EXP4 > EXP3: Coverage@95 within [0.90, 1.00] for EXP4, not for EXP3
3. EXP4: AURC < 0.5 (usable selective prediction)
4. Any experiment with oracle_passed=false is INVALID

### Failure Modes
- If EXP1 ≈ EXP0: Preprocessing not working → check implementation
- If EXP4 ≈ EXP3: Gaussian-NCE not helping → check loss weight, temperature
- If oracle check fails: BUG in eval code → fix before continuing

## Visualization Plan

### Figure 1: Ablation Bar Chart
- X-axis: EXP0, EXP1, EXP2, EXP3, EXP4, EXP5, EXP6
- Y-axis: R@1 (left), 2AFC (middle), Coverage@95 (right)
- 3 subplots side-by-side
- Color code by component added

### Figure 2: Retrieval Curves
- X-axis: Gallery size (log scale)
- Y-axis: R@1
- 7 lines (one per experiment)
- Dashed line for chance baseline

### Figure 3: Calibration
- X-axis: Nominal coverage
- Y-axis: Empirical coverage
- One line per probabilistic experiment (EXP3-6)
- Diagonal line for perfect calibration

### Figure 4: Risk-Coverage
- X-axis: Coverage (0 to 1)
- Y-axis: Risk (error)
- Lines for EXP3, EXP4, EXP5, EXP6
- Annotate with AURC values

## Timeline and Resources

### Estimated Time per Experiment
- EXP0: 2 hours (training) + 30 min (eval) = 2.5 hours
- EXP1-6: Similar (~2-3 hours each)
- **Total**: ~18-21 hours for all experiments

### Compute Requirements
- GPU: 1x A100 or V100 (min 16GB VRAM)
- Storage: ~50GB (checkpoints + cached data)

### Parallelization
- Can run experiments in parallel if multiple GPUs available
- Queue experiments: EXP0, EXP1 → EXP2 → EXP3 → EXP4 → EXP5, EXP6

## Deliverables

1. **Metrics JSON** for each experiment
2. **Plots** for each experiment (in evaluation/plots/)
3. **Comparison report** (compare_experiments.py output)
4. **LaTeX table** for paper (auto-generated from metrics)
5. **Statistical test results** (p-values, effect sizes)

## Notes

- If any experiment crashes or produces NaN: Debug before continuing
- If oracle check fails: Fix evaluation code immediately
- Save all configs and seeds for exact reproducibility
- Document any deviations from plan in experiment logs
