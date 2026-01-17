# Paper Outline: Probabilistic fMRI-to-CLIP Embedding Decoding

## Title
**Geometry-Aware Probabilistic Decoding of fMRI Activity to Foundation Model Embeddings with Calibrated Uncertainty**

## Abstract (Target: 150-200 words)

We address the problem of decoding fMRI brain activity into foundation model (CLIP) embedding spaces for image reconstruction and retrieval. Current approaches suffer from two critical failures: (1) anisotropic embedding geometry causing chance-level retrieval despite high cosine similarity, and (2) uncalibrated uncertainty estimates that provide no decision value. We introduce three contributions: **First**, we demonstrate that embedding preprocessing (centering + principal component removal/whitening + L2 normalization) is essential to fix geometry-induced identification collapse, improving retrieval from chance to >X% R@1. **Second**, we propose Gaussian-NCE, a novel distribution-aware contrastive objective that uses likelihood under predicted Gaussian distributions as similarity scores, making uncertainty directly relevant for Bayesian retrieval. **Third**, we establish a rigorous probabilistic evaluation protocol using proper scoring rules (NLL, Energy Score), calibration tests (coverage, ECE), and risk-coverage curves (AURC) for selective prediction. Across systematic ablations on Natural Scenes Dataset, we show that geometry normalization + Gaussian-NCE achieve state-of-the-art retrieval (R@1: X%, 2AFC: Y%) while producing well-calibrated uncertainty (Coverage@95: ~0.95, AURC: Z), enabling trustworthy brain decoding for downstream applications.

**Keywords**: fMRI decoding, CLIP embeddings, uncertainty quantification, contrastive learning, anisotropy correction

---

## 1. Introduction

### 1.1 Motivation
- Brain decoding to foundation model embeddings enables zero-shot image reconstruction/retrieval
- Current challenge: models achieve high cosine similarity but **fail at identification** (Retrieval@K ≈ chance)
- Hypothesis: Anisotropic embedding spaces + uncalibrated uncertainty

### 1.2 Problem Statement
Given fMRI activity $f \in \mathbb{R}^V$ (V voxels), decode to CLIP embedding $z \in \mathbb{R}^D$ (D=768) such that:
1. **Retrieval**: Query embedding $\hat{z}$ retrieves correct image from gallery
2. **Uncertainty**: Predicted distribution $q(\hat{z}|f)$ is well-calibrated for selective prediction
3. **Geometry**: Normalized embeddings respect semantic structure

### 1.3 Key Contributions
1. **Embedding Geometry Fix**: Demonstrate that anisotropy causes identification collapse; propose center+PCR/whiten+normalize preprocessing that is fit on train split and applied consistently to all embeddings and metrics. Show X% absolute improvement in R@1.

2. **Gaussian-NCE**: Introduce distribution-aware contrastive loss that optimizes likelihood $p(z_{\text{key}} | \mu_q, \Sigma_q)$ instead of cosine similarity, directly training for Bayesian retrieval with meaningful uncertainty.

3. **Probabilistic Evaluation Protocol**: Establish rigorous uncertainty evaluation using NLL (vs baseline), Energy Score, calibration (chi-square thresholds), ECE, and AURC for risk-coverage curves. Show that Gaussian-NCE produces calibrated posteriors (Coverage@95 ≈ 0.95) unlike baseline methods.

---

## 2. Related Work

### 2.1 fMRI-to-Image Decoding
- Early work: Voxel-wise encoding models, reconstruction from V1
- Modern: Foundation model embeddings (CLIP, DALL-E, Stable Diffusion)
- Gap: Most report cosine similarity, not retrieval/identification

### 2.2 Anisotropic Embeddings
- CLIP embeddings exhibit cone effect / hubness (Gao et al., Liang et al.)
- Solutions: Temperature scaling, whitening, contrastive fine-tuning
- Gap: Not addressed in brain decoding literature

### 2.3 Uncertainty in Neural Decoding
- Monte Carlo Dropout, ensembles, heteroscedastic regression
- Gap: Uncertainty rarely evaluated beyond visualization; not decision-relevant

### 2.4 Contrastive Learning with Distributions
- MoCo/SimCLR for deterministic embeddings
- Probabilistic contrastive: SupCon variants, few works
- Gap: No Gaussian-NCE using likelihoods for contrastive retrieval

---

## 3. Method

### 3.1 Problem Formulation
- Input: fMRI activity $f \in \mathbb{R}^V$ (flattened ROI)
- Target: CLIP embedding $z \in \mathbb{R}^D$ (ViT-L/14, D=768)
- Model: $p_\theta(\hat{z} | f) = \mathcal{N}(\mu_\theta(f), \text{diag}(\exp(\log\sigma^2_\theta(f))))$

### 3.2 Embedding Preprocessing (Geometry Fix)
**Algorithm**:
1. Fit on training split:
   - Compute mean $\bar{z}_{\text{train}}$
   - PCA on centered embeddings
   - For **center_pcr**: Store top-$k$ components $U_k$
   - For **center_whiten**: Store whitening matrix $W = U \Lambda^{-1/2} U^T$
2. Apply to all splits:
   - Center: $z' = z - \bar{z}_{\text{train}}$
   - **PCR**: $z'' = z' - U_k (U_k^T z')$ (remove top-k PCs)
   - **Whiten**: $z'' = W z'$
   - Normalize: $\tilde{z} = z'' / \|z''\|_2$

**Diagnostics**:
- Anisotropy score: Mean cosine similarity of random pairs (should decrease to ~0)
- Pos/neg separation: AUC, Cohen's $d$

### 3.3 Contrastive Learning with Memory Queue
- Small batch size (B=4) insufficient for InfoNCE
- Maintain MoCo-style queue of GT embeddings (size Q=8k-32k)
- Learnable temperature (CLIP-style $\exp(\text{logit\_scale})$)

**InfoNCE Loss**:
$$
\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(s(\hat{z}_i, z_i)/\tau)}{\sum_{j \in B \cup Q} \exp(s(\hat{z}_i, z_j)/\tau)}
$$

where $s(\cdot, \cdot)$ is cosine similarity.

### 3.4 Gaussian-NCE (Novel Contribution)
Replace cosine similarity with Gaussian log-likelihood:

$$
s(\hat{z}_q, z_k) = \log \mathcal{N}(z_k; \mu_q, \text{diag}(\exp(\log\sigma^2_q)))
$$

**Gaussian-NCE Loss**:
$$
\mathcal{L}_{\text{GaussianNCE}} = -\log \frac{\exp(s(\hat{z}_i, z_i)/\tau)}{\sum_{j \in B \cup Q} \exp(s(\hat{z}_i, z_j)/\tau)}
$$

**Key insight**: Uncertainty is now decision-relevant — high variance reduces likelihood for all keys, enabling selective prediction.

### 3.5 Probabilistic Training Objective
**Total Loss**:
$$
\mathcal{L} = \mathcal{L}_{\text{NLL}} + \beta(t) \mathcal{L}_{\text{KL}} + \lambda \mathcal{L}_{\text{GaussianNCE}}
$$

- **NLL**: Gaussian NLL with variance clamping $[\log\sigma^2_{\min}, \log\sigma^2_{\max}]$
- **KL**: Annealed $\beta(t)$ from 0 to $\beta_{\max}$ over $T$ steps, with free-bits $\lambda_{\text{fb}}$
- **Gaussian-NCE**: Distribution-aware contrastive

### 3.6 Inference Modes
- **Deterministic retrieval**: Use $\mu(f)$ only
- **Probabilistic evaluation**: Sample $S$ times from $\mathcal{N}(\mu, \Sigma)$ for NLL/Energy/Calibration

---

## 4. Evaluation Protocol

### 4.1 Deterministic Metrics (Embedding Quality)
- **Retrieval**: R@1, R@5, R@10, MeanR, MedR, MRR, nDCG@10
  - Curves vs gallery size: [2, 10, 50, 100, 500, 1000]
  - Chance baselines: $1/N_{\text{gallery}}$
- **Identification**: 2AFC with bootstrap CI (95%)
- **Discriminability**: AUC(pos vs neg), Cohen's $d$, $d'$
- **Structure**: RSA (Spearman), Linear CKA

### 4.2 Probabilistic Metrics (Uncertainty Quality)
- **NLL per-dim**: Report $\Delta\text{NLL}$ vs baseline (empirical variance)
- **Energy Score**: $\mathbb{E}[\|Y - X\|] - 0.5 \mathbb{E}[\|Y - Y'\|]$ (proper scoring)
- **Calibration**: Coverage@80, Coverage@95 using $\chi^2(D)$ thresholds
- **ECE**: Expected calibration error (binned confidence vs accuracy)
- **AURC**: Area under risk-coverage curve for selective prediction
- **Prob-2AFC**: Likelihood-ratio based identification

### 4.3 Oracle Sanity Check
- GT embeddings must retrieve themselves: R@1 ≥ 0.95, MeanR ≈ 1
- Fail loudly if oracle check fails (ID mismatch / normalization bug)

---

## 5. Experiments

### 5.1 Dataset and Setup
- **Dataset**: Natural Scenes Dataset (NSD), Subject 01
- **ROI**: nsdgeneral (whole-brain mask)
- **Split**: Train/Val/Test (sizes TBD)
- **CLIP Model**: ViT-L/14 (D=768)
- **Gallery Sizes**: Evaluate on [2, 10, 50, 100, 500, 1000]
- **Seeds**: All experiments use same seeds for reproducibility

### 5.2 Ablation Study (EXP0-EXP6)

| Exp | Description | Preproc | Queue | NLL | Gaussian-NCE | KL |
|-----|-------------|---------|-------|-----|--------------|-----|
| EXP0 | Baseline (MSE+cosine) | ❌ | ❌ | ❌ | ❌ | ❌ |
| EXP1 | + Embedding preprocessing | center_pcr k=8 | ❌ | ❌ | ❌ | ❌ |
| EXP2 | + Queue InfoNCE | center_pcr k=8 | ✅ Q=8k | ❌ | ❌ | ❌ |
| EXP3 | + Gaussian NLL | center_pcr k=8 | ✅ | ✅ | ❌ | ❌ |
| EXP4 | + Gaussian-NCE | center_pcr k=8 | ✅ | ✅ | ✅ | ❌ |
| EXP5 | + KL anneal+free-bits | center_pcr k=8 | ✅ | ✅ | ✅ | ✅ |
| EXP6 | Ablation: center_whiten | center_whiten | ✅ | ✅ | ✅ | ✅ |

### 5.3 Hypotheses
- H1: EXP1 >> EXP0 (geometry is critical)
- H2: EXP2 > EXP1 (queue helps contrastive)
- H3: EXP4 > EXP3 (Gaussian-NCE improves retrieval + calibration)
- H4: EXP5 ≈ EXP4 (KL may not be necessary if Gaussian-NCE is strong)
- H5: EXP6 ≈ EXP5 (both preprocessing modes work)

---

## 6. Results

### 6.1 Main Results (Table 1)
Report for each EXP:
- R@1, R@5, R@10, MeanR, MRR, nDCG@10
- 2AFC (with CI), AUC, Cohen's $d$
- RSA, CKA
- NLL/dim, $\Delta$NLL, ECE, AURC, Coverage@95

### 6.2 Retrieval Curves (Figure 1)
- R@K vs gallery size for all experiments
- Show exponential decay and separation between methods

### 6.3 Calibration Analysis (Figure 2)
- **Top**: Pos vs neg similarity histograms (before/after preprocessing)
- **Bottom**: Calibration curve (nominal vs empirical coverage)

### 6.4 Risk-Coverage Curves (Figure 3)
- Error vs coverage for different experiments
- Show AURC values
- Demonstrate that Gaussian-NCE enables effective selective prediction

### 6.5 Ablation Bar Chart (Figure 4)
- Side-by-side bars for R@1, 2AFC, $\Delta$NLL, ECE across EXP0-EXP6

### 6.6 Key Findings
1. **Geometry fix is critical**: EXP1 improves R@1 from X% to Y% (Z% absolute gain)
2. **Gaussian-NCE works**: EXP4 achieves best retrieval + calibration
3. **Uncertainty is calibrated**: Coverage@95 ≈ 0.95 for EXP4/EXP5
4. **Queue helps**: Small batch problem solved with Q=8k negatives

---

## 7. Analysis and Discussion

### 7.1 Why Anisotropy Matters
- High mean cosine similarity (e.g., 0.3) between unrelated images
- Creates ambiguous retrieval even with "good" predictions
- Preprocessing reduces anisotropy score from X to Y

### 7.2 Gaussian-NCE vs Cosine InfoNCE
- Cosine: Ignores uncertainty, all predictions treated equally
- Gaussian-NCE: Low-confidence predictions have flat likelihood → safe rejection
- Enables selective prediction with risk-coverage trade-off

### 7.3 Calibration Without KL?
- Gaussian-NCE alone may provide sufficient regularization
- KL with free-bits prevents collapse but may not improve metrics
- Empirical: EXP4 ≈ EXP5 suggests Gaussian-NCE is sufficient

### 7.4 Failure Analysis
- Show examples where uncertainty is high and prediction is wrong
- Demonstrate that risk-coverage allows filtering bad predictions

---

## 8. Limitations

1. **Single subject**: Results on subj01 only; generalization TBD
2. **Computational cost**: Gaussian-NCE adds overhead vs cosine
3. **Embedding choice**: CLIP may not be optimal target; other foundations (DINOv2, Imagebind) unexplored
4. **Small gallery**: Real-world retrieval would have much larger galleries
5. **Preprocessing overhead**: Requires fitting on train split; not online

---

## 9. Conclusion and Future Work

We demonstrated that **geometry normalization** and **Gaussian-NCE** are critical for reliable fMRI-to-CLIP decoding. Our contributions enable:
- **Trustworthy retrieval**: R@1 from chance to X% with proper evaluation
- **Calibrated uncertainty**: Coverage@95 ≈ 0.95, AURC = Y
- **Selective prediction**: Risk-coverage curves for decision-making

**Future directions**:
- Multi-subject evaluation and transfer
- Hierarchical/mixture models for multimodal uncertainty
- Application to closed-loop neurofeedback
- Extension to video/dynamic stimuli

---

## Appendix

### A.1 Hyperparameters
- Model architecture: [Specify layers, dims]
- Learning rate schedule: [Specify]
- Queue size: 8192
- Variance clamps: $[-10, 5]$
- KL annealing: Linear from 0 to $10^{-3}$ over 10k steps
- Free-bits: 0.5 per dimension

### A.2 Computational Resources
- GPU: [Specify]
- Training time per experiment: [Specify]

### A.3 Additional Ablations
- k_components: [2, 4, 8, 16] for center_pcr
- Queue size: [4k, 8k, 16k, 32k]
- Temperature initialization

### A.4 Code and Data Availability
- Code: [GitHub link]
- Preprocessed embeddings: [Data link if allowed by NSD terms]
