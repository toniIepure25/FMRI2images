# Paper Outline: Hyperspherical Probabilistic Brain Decoding

## Title Options

### NeurIPS (ML focus)
**"Probabilistic Contrastive Decoding on the Hypersphere: Calibrated Uncertainty for Brain-to-Image Retrieval"**

### MICCAI (Medical imaging focus)
**"Brain-Topology-Aware Probabilistic Decoding of Visual Cortex Activity with Calibrated Uncertainty"**

## Abstract (Target: 150-200 words)

We address the problem of decoding fMRI brain activity into foundation model (CLIP) embedding spaces for image reconstruction and retrieval. Current approaches suffer from three critical failures: (1) anisotropic embedding geometry causing chance-level retrieval despite high cosine similarity, (2) a distribution mismatch — L2-normalised embeddings live on the hypersphere $S^{d-1}$ but are modelled with Euclidean Gaussians, and (3) lack of interpretable, brain-topology-aware architectures. We introduce three contributions: **First**, we propose **vMF-NCE**, a novel contrastive objective that uses von Mises-Fisher log-density as similarity scores, providing a principled probabilistic model on the correct manifold where the concentration parameter $\kappa$ directly encodes confidence. **Second**, we introduce an **ROI-tokenised Transformer encoder** that groups voxels by anatomical brain region, enabling interpretability through attention weights. **Third**, we establish a rigorous **probabilistic evaluation protocol** with proper scoring rules (NLL, Energy Score), calibration tests, risk-coverage curves, and noise-ceiling normalisation. Across systematic ablations on the Natural Scenes Dataset (4 subjects), we show that vMF-NCE with ROI-aware encoding achieves state-of-the-art retrieval while producing well-calibrated uncertainty and revealing which cortical regions drive each prediction.

**Keywords**: fMRI decoding, CLIP embeddings, von Mises-Fisher, uncertainty quantification, contrastive learning, brain-computer interfaces

---

## 1. Introduction

### 1.1 Motivation
- Brain decoding to foundation model embeddings enables zero-shot image reconstruction/retrieval
- Current challenge: models achieve high cosine similarity but **fail at identification** (Retrieval@K ≈ chance)
- Three overlooked problems: (a) anisotropy, (b) wrong distributional assumption, (c) no topology

### 1.2 Problem Statement
Given fMRI activity $f \in \mathbb{R}^V$ (V voxels), decode to CLIP embedding $z \in S^{d-1}$ such that:
1. **Retrieval**: Query embedding $\hat{z}$ retrieves correct image from gallery
2. **Uncertainty**: Predicted concentration $\kappa(f)$ is well-calibrated for selective prediction
3. **Interpretability**: Architecture reveals which brain regions contribute to each prediction

### 1.3 Key Contributions
1. **vMF-NCE**: Distribution-aware contrastive objective on the hypersphere using $\log p(z_k | \mu_q, \kappa_q)$ as similarity score. Theoretically motivated by the mismatch between Gaussian models and L2-normalised embeddings.

2. **ROI-Tokenised Transformer**: Brain-region-aware encoder that groups voxels by anatomical ROI (V1-V4, FFA, PPA, EBA, etc.), providing interpretability through attention weights.

3. **Comprehensive Evaluation Protocol**: Proper scoring rules, calibration tests, risk-coverage curves, noise ceiling normalisation, SOTA comparison, and image reconstruction metrics.

---

## 2. Related Work

### 2.1 fMRI-to-Image Decoding
- Early work: Voxel-wise encoding models
- Modern: MindEye (Scotti et al., NeurIPS 2023), MindEye2 (Scotti et al., 2024), Brain Diffuser (Ozcelik & VanRullen, 2023)
- Gap: Most use MSE/cosine loss without principled uncertainty

### 2.2 Anisotropic Embeddings and Hubness
- CLIP embeddings exhibit cone effect / hubness (Radovanovic et al. 2010, Liang et al. 2022)
- Solutions: Temperature scaling, whitening, contrastive fine-tuning
- Gap: Not addressed in brain decoding literature

### 2.3 Directional Statistics and Hyperspherical VAEs
- Von Mises-Fisher distribution on $S^{d-1}$ (Banerjee et al. 2005)
- Hyperspherical VAEs (Davidson et al. 2018)
- Gap: Never applied to brain decoding or contrastive learning

### 2.4 Brain-Topology-Aware Architectures
- MindEye2 subspace tokenisation
- BrainSCUBA: cross-attention
- Gap: No systematic ROI tokenisation with interpretable attention

---

## 3. Method

### 3.1 Problem Formulation
- Input: fMRI $f \in \mathbb{R}^V$
- Target: CLIP embedding $z \in S^{d-1}$ (unit hypersphere, $d=768$)
- Model: $p_\theta(\hat{z} | f) = \text{vMF}(\hat{z}; \mu_\theta(f), \kappa_\theta(f))$

### 3.2 Embedding Preprocessing (Geometry Fix)
- Center + PCR/whiten + L2-normalise
- Reduces anisotropy and hubness
- Fit on train split; apply to all

### 3.3 ROI-Tokenised Transformer Encoder
$$
f \in \mathbb{R}^V \xrightarrow{\text{ROI split}} \{f_{r_1}, \ldots, f_{r_R}\} \xrightarrow{\text{per-ROI MLP}} \{t_1, \ldots, t_R\} \xrightarrow{\text{Transformer}} h_{\text{CLS}}
$$
- $R \approx 17$ ROIs from NSD atlas
- Learnable [CLS] token; anatomical positional encoding
- 4 layers, 8 heads, $d_\text{model} = 512$

### 3.4 Von Mises-Fisher Decoder
$$
p(z | \mu, \kappa) = C_d(\kappa) \cdot \exp(\kappa \cdot \mu^\top z), \quad z \in S^{d-1}
$$

Decoder outputs $\mu = \text{normalize}(\text{MLP}(h_{\text{CLS}}))$ and $\log \kappa = \text{MLP}(h_{\text{CLS}})$.

### 3.5 vMF-NCE Loss (Novel)
Replace cosine similarity with vMF log-density:
$$
s(\hat{z}_q, z_k) = \log C_d(\kappa_q) + \kappa_q \cdot \mu_q^\top z_k
$$

**vMF-NCE Loss**:
$$
\mathcal{L}_{\text{vMF-NCE}} = -\log \frac{\exp(s(\hat{z}_i, z_i))}{\sum_{j \in B \cup Q} \exp(s(\hat{z}_i, z_j))}
$$

**Key insight**: $\kappa$ encodes confidence. Low $\kappa$ → flat density → all keys score similarly → model abstains. High $\kappa$ → peaked density → only correct key scores high.

### 3.6 Training Objective
$$
\mathcal{L} = \mathcal{L}_{\text{vMF-NLL}} + \lambda \mathcal{L}_{\text{vMF-NCE}}
$$

### 3.7 Comparison: Why Not Gaussian?
- L2-normalised $\mu$ + diagonal Gaussian concentrates mass **off** $S^{d-1}$
- vMF is the natural exponential family distribution on $S^{d-1}$
- We keep Gaussian-NCE as ablation (EXP4) to demonstrate vMF advantage (EXP7)

---

## 4. Evaluation Protocol

### 4.1 Embedding Metrics
- R@1, R@5, R@10, MeanR, MedR, MRR, nDCG@10
- 2AFC with bootstrap CI
- RSA (Spearman), Linear CKA
- Hubness: skewness of $N_k$, hub/antihub fraction

### 4.2 Probabilistic Metrics
- NLL per-dim, ΔNLL vs baseline
- Energy Score (proper scoring rule)
- Coverage@80, Coverage@95
- ECE, AURC for selective prediction
- Prob-2AFC (likelihood ratio)

### 4.3 Image Reconstruction Metrics
- Low-level: PixCorr, SSIM
- Mid-level: AlexNet(2), AlexNet(5), LPIPS
- High-level: CLIP-I (image cosine)

### 4.4 Noise Ceiling Normalisation
- Spearman-Brown prophecy on 3 NSD repeats
- Report % of explainable variance captured

### 4.5 Information-Theoretic Analysis
- MI lower bound via InfoNCE: $I \geq \log N - \mathcal{L}_\text{NCE}$
- Compare MI bounds across experiments

---

## 5. Experiments

### 5.1 Dataset and Setup
- NSD, Subjects 01/02/05/07
- ROI: nsdgeneral mask
- CLIP: ViT-L/14 (D=768)
- Train/Val/Test: 70/15/15

### 5.2 Extended Ablation Study (EXP0-EXP8)

| Exp | Architecture | Distribution | Loss | Preprocessing |
|-----|-------------|-------------|------|--------------|
| EXP0 | MLP | - | MSE+cos | None |
| EXP1 | MLP | - | MSE+cos | center_pcr |
| EXP2 | MLP | - | InfoNCE+Q | center_pcr |
| EXP3 | MLP | Gaussian | NLL+InfoNCE | center_pcr |
| EXP4 | MLP | Gaussian | NLL+G-NCE | center_pcr |
| EXP5 | MLP | Gaussian | NLL+G-NCE+KL | center_pcr |
| EXP6 | MLP | Gaussian | NLL+G-NCE+KL | whiten |
| **EXP7** | **MLP** | **vMF** | **vMF-NLL+vMF-NCE** | **center_pcr** |
| **EXP8** | **ROI-Trans** | **vMF** | **vMF-NLL+vMF-NCE** | **center_pcr** |

### 5.3 Hypotheses
- H1: EXP1 >> EXP0 (geometry critical)
- H2: EXP4 > EXP3 (Gaussian-NCE improves calibration)
- **H7: EXP7 > EXP4** (vMF is principled; better calibration + retrieval)
- **H8: EXP8 > EXP7** (ROI-Transformer improves retrieval through inductive bias)
- Cross-subject: EXP7/EXP8 generalise across subjects

---

## 6. Results

### 6.1 Main Results Table
SOTA comparison: our method vs MindEye, MindEye2, Brain Diffuser

### 6.2 Ablation Table (EXP0-EXP8)
Full metrics across all experiments

### 6.3 Retrieval Curves (Figure 1)
R@1 vs gallery size for all methods

### 6.4 Calibration Analysis (Figure 2)
- Coverage curve: nominal vs empirical
- Risk-coverage curve with AURC values
- vMF vs Gaussian calibration comparison

### 6.5 ROI Contribution Analysis (Figure 3)
- Cortical flatmap of attention weights
- ROI ablation importance bars
- Category-ROI interaction heatmap

### 6.6 Uncertainty-Reliability Correlation (Figure 4)
- Scatter: model uncertainty vs neural reliability
- Shows that uncertainty tracks measurement noise

---

## 7. Analysis and Discussion

### 7.1 Why vMF > Gaussian on the Hypersphere
- Geometric argument: Gaussian on $S^{d-1}$ wastes probability mass off-manifold
- Empirical: vMF achieves better calibration (Coverage@95 closer to 0.95)
- Information-theoretic: tighter MI bound with vMF-NCE

### 7.2 Brain Regions that Drive Decoding
- V1-V4: contribute to low-level features (PixCorr, SSIM)
- FFA/PPA/EBA: contribute to high-level features (CLIP-I)
- RSC: spatial layout information

### 7.3 Selective Prediction in Practice
- Risk-coverage tradeoff enables deployment in BCIs
- Abstain on uncertain predictions improves effective accuracy

---

## 8. Limitations

1. Single dataset (NSD); needs validation on other fMRI datasets
2. Computational cost: Transformer encoder increases training time ~2x vs MLP
3. vMF normalising constant approximation in high dimensions
4. ROI tokenisation depends on atlas quality

---

## 9. Conclusion and Future Work

**Contributions**: vMF-NCE, ROI-tokenised Transformer, comprehensive evaluation protocol.

**Future**:
- Multi-subject alignment and transfer learning
- Video/temporal decoding
- Closed-loop neurofeedback with calibrated uncertainty
- Normalising flows on the hypersphere for richer posteriors

---

## Appendix

### A.1 Hyperparameters
Full table of all hyperparameters per experiment

### A.2 vMF Normalising Constant
Derivation of $C_d(\kappa)$ and approximation in high $d$

### A.3 Additional Ablations
- $k$ in center_pcr: [2, 4, 8, 16]
- Queue size: [4k, 8k, 16k, 32k]
- Number of ROI Transformer layers: [2, 4, 6, 8]
- $\kappa$ clamp range

### A.4 Multi-Subject Results
Per-subject tables and cross-subject aggregation

### A.5 Noise Ceiling Analysis
Spearman-Brown results per subject and ROI

### A.6 Additional Qualitative Results
Image reconstruction grids with uncertainty annotations
