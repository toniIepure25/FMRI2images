# Brain Regions as Directional Experts: Calibrated Uncertainty Decomposition for Neural Image Decoding

## Title Options

### NeurIPS / ICML (ML focus)
**"Brain Regions as Directional Experts: Calibrated Uncertainty Decomposition for Neural Image Decoding"**

### MICCAI (Medical imaging focus)
**"Directional Expert Consensus with Dual Uncertainty for Brain-to-Image Reconstruction"**

---

## Abstract (Target: 200 words)

We address fMRI-to-image decoding by treating brain regions as directional experts that vote on perceived content via probability distributions on the hypersphere. Current methods produce point estimates without principled uncertainty, ignore the mismatch between Euclidean loss functions and L2-normalized CLIP embeddings on $S^{d-1}$, and lack brain-topology-aware architectures. We introduce four contributions: **First**, ROI Directional Consensus Fusion (ROI-DCF), where each brain region predicts an independent von Mises-Fisher distribution and consensus is formed via spherical weighted averaging, naturally yielding a disagreement score $\delta$. **Second**, a dual uncertainty decomposition for generation: the per-region concentration $\kappa$ (aleatoric) controls classifier-free guidance scale, while inter-region disagreement $\delta$ (epistemic-like) controls ensemble diversity. **Third**, vMF-NCE, a Bessel-free contrastive objective on $S^{d-1}$ where $\kappa$ encodes confidence, combined with a concentration-aware self-paced curriculum (kappa-SPCL). **Fourth**, the first comprehensive probabilistic evaluation protocol for brain decoding: risk-coverage curves with AURC for selective prediction, noise-ceiling normalization, and hierarchical selective classification. Across systematic ablations on 4 NSD subjects, we demonstrate state-of-the-art retrieval with calibrated, interpretable uncertainty.

**Keywords**: fMRI decoding, von Mises-Fisher, directional statistics, uncertainty decomposition, contrastive learning, brain-computer interfaces

---

## 1. Introduction

### 1.1 Motivation

- Brain decoding to foundation-model embeddings enables zero-shot image retrieval and reconstruction
- Current methods achieve high cosine similarity but fail at identification (R@K near chance) due to anisotropy and hubness
- Three critical overlooked problems:
  - (a) **Distribution mismatch**: L2-normalized CLIP embeddings live on $S^{d-1}$ but are modeled with Euclidean Gaussians
  - (b) **No topology**: Flat MLP encoders ignore brain region structure
  - (c) **No principled uncertainty**: Predictions lack calibrated confidence for safe deployment

### 1.2 Key Insight

Brain regions are *directional experts*: each ROI (V1, FFA, PPA, etc.) has a specialized view of the perceived stimulus. Their predictions should be directional distributions, not point estimates. When experts agree (low $\delta$), we can be confident; when they disagree (high $\delta$), we need caution — and this is a *different* uncertainty from measurement noise (captured by $\kappa$).

### 1.3 Problem Statement

Given fMRI $\mathbf{f} \in \mathbb{R}^V$, decode to CLIP embedding $\mathbf{z} \in S^{d-1}$ with:
1. **Retrieval**: $\hat{\mathbf{z}}$ retrieves correct image from gallery
2. **Dual uncertainty**: $\kappa(\mathbf{f})$ for within-region noise, $\delta(\mathbf{f})$ for between-region disagreement
3. **Interpretability**: Per-ROI contributions reveal which brain regions drive each prediction

### 1.4 Contributions

1. **ROI-DCF**: Per-ROI vMF directional experts with spherical consensus fusion — each brain region predicts a full distribution, not a point estimate. The disagreement $\delta$ is a natural byproduct.

2. **Dual Uncertainty Decomposition for Generation**: $\kappa$ (aleatoric) $\to$ CFG guidance scale; $\delta$ (epistemic-like) $\to$ ensemble size and diffusion steps. Different uncertainty sources control different generation aspects.

3. **vMF-NCE with Brain-Specific Innovations**: Bessel-free contrastive loss on the correct manifold, kappa-SPCL self-paced curriculum, noise-ceiling-informed temperature.

4. **Neuroscience-Grounded Evaluation Protocol**: Risk-coverage curves (AURC), noise-ceiling normalization, hierarchical selective prediction, ROI importance with bootstrap CIs.

---

## 2. Related Work

### 2.1 fMRI-to-Image Decoding
- MindEye (Scotti et al., NeurIPS 2023): MLP + contrastive + diffusion prior
- MindEye2 (Scotti et al., ICML 2024): Cross-subject ridge + SDXL unCLIP
- Brain Diffuser (Ozcelik & VanRullen, 2023): Dual-stream latent + semantic
- Brain-IT (2025): Brain Interaction Transformer with cross-subject clusters
- MoRE-Brain (2025): Hierarchical MoE with dual routing
- SRST + SG-MoE (2025): Selective ROI Spherical Tokenizer + Structure-Guided MoE
- **Gap**: All output point estimates. None model per-ROI distributions or decompose uncertainty.

### 2.2 Directional Statistics and Hyperspherical Models
- vMF distribution (Banerjee et al., 2005)
- Hyperspherical VAEs (Davidson et al., 2018)
- Probabilistic CL with vMF (2025, withdrawn ICLR): General CV application
- **Gap**: Never applied to brain decoding; no brain-topology integration

### 2.3 Uncertainty in Generative Models
- Classifier-Free Guidance (Ho & Salimans, 2022)
- Adaptive CFG via SOC (Karras et al., 2025)
- $\beta$-CFG (2025): Gradient-based adaptive normalization
- **Gap**: No brain-signal-derived uncertainty decomposition for generation control

### 2.4 Selective Prediction
- Risk-coverage curves (Geifman & El-Yaniv, 2017)
- Hierarchical selective classification (NeurIPS 2024)
- **Gap**: Never applied to neural decoding; critical for BCI safety

---

## 3. Method

### 3.1 Problem Formulation

Input: $\mathbf{f} \in \mathbb{R}^V$, Target: $\mathbf{z} \in S^{d-1}$ (CLIP ViT-L/14, $d=768$)

Per-ROI model: $p(\mathbf{z} | \mathbf{f}) = \sum_{r=1}^{R} \alpha_r \cdot \text{vMF}(\mathbf{z}; \boldsymbol{\mu}_r(\mathbf{f}), \kappa_r(\mathbf{f}))$

### 3.2 ROI-Tokenized Transformer Encoder

$$\mathbf{f} \xrightarrow{\text{ROI split}} \{f_{r_1}, \ldots, f_{r_R}\} \xrightarrow{\text{per-ROI MLP}} \{t_1, \ldots, t_R\} \xrightarrow{\text{Transformer}} [\text{CLS}], \{h_1, \ldots, h_R\}$$

- $R \approx 17$ NSD ROIs, [CLS] aggregation, 4 layers, 8 heads
- Attention weights $\alpha_r$ from [CLS]-to-ROI in final layer

### 3.3 ROI Directional Consensus Fusion (ROI-DCF)

Each post-Transformer ROI token gets its own vMF head:
$$(\boldsymbol{\mu}_r, \kappa_r) = \text{vMFHead}(h_r) \quad \forall r \in \{1, \ldots, R\}$$

Consensus via spherical weighted mean (Mardia & Jupp, 2000):
$$\mathbf{R} = \sum_r \alpha_r \kappa_r \boldsymbol{\mu}_r, \quad \boldsymbol{\mu}_{\text{fused}} = \frac{\mathbf{R}}{\|\mathbf{R}\|}, \quad \kappa_{\text{consensus}} = \|\mathbf{R}\|$$

Disagreement score:
$$\delta = 1 - \sum_{r,s} \alpha_r \alpha_s \cos(\boldsymbol{\mu}_r, \boldsymbol{\mu}_s) \in [0, 1]$$

**Properties**: When ROIs agree, $\kappa_{\text{consensus}} \approx \sum_r \alpha_r \kappa_r$. When they disagree, the resultant shortens $\to$ lower $\kappa_{\text{consensus}}$.

### 3.4 vMF-NCE Loss (Bessel-Free)

$$s(\hat{\mathbf{z}}_q, \mathbf{z}_k) = \kappa_q \cdot \boldsymbol{\mu}_q^\top \mathbf{z}_k / \tau$$

$$\mathcal{L}_{\text{vMF-NCE}} = -\log \frac{\exp(s(\hat{\mathbf{z}}_i, \mathbf{z}_i))}{\sum_{j \in B \cup Q} \exp(s(\hat{\mathbf{z}}_i, \mathbf{z}_j))}$$

**Theorem**: $\log C_d(\kappa_q)$ cancels in the softmax ratio (constant across keys for fixed query).

**Multi-task**: $\mathcal{L} = \mathcal{L}_{\text{vMF-NCE}}^{\text{fused}} + \lambda_{\text{aux}} \frac{1}{R} \sum_r \mathcal{L}_{\text{vMF-NCE}}^{(r)}$

### 3.5 Kappa-SPCL Curriculum

Self-paced weighting: $w_i = \text{softmax}(\kappa_i / T_c)$, annealing $T_c$ from high (uniform) to low (focus on confident samples).

### 3.6 Dual Uncertainty-Aware Generation

| $\kappa$ | $\delta$ | Interpretation | Strategy |
|----------|----------|----------------|----------|
| High | Low | Confident + agree | Strong CFG, 1 sample, few steps |
| Low | Low | Noisy + consistent | Moderate CFG |
| High | High | Confident + conflicting | Multiple mixture samples |
| Low | High | Fully uncertain | Abstain or conservative |

$$w = w_{\min} + f(\kappa) \cdot g(\delta) \cdot (w_{\max} - w_{\min})$$
$$K = K_{\min} + h(\delta) \cdot (K_{\max} - K_{\min})$$

### 3.7 vMF Mixture Sampling for Generation

Sample from the full mixture $p(\mathbf{z}|\mathbf{f}) = \sum_r \alpha_r \cdot \text{vMF}(\boldsymbol{\mu}_r, \kappa_r)$ using:
1. Component selection: $r \sim \text{Categorical}(\boldsymbol{\alpha})$
2. Direction sampling: $\mathbf{z} \sim \text{vMF}(\boldsymbol{\mu}_r, \kappa_r)$ via Wood (1994) rejection

---

## 4. Evaluation Protocol

### 4.1 Embedding Metrics
R@1, R@5, R@10, MeanR, MedR, MRR, nDCG@10, 2AFC, hubness metrics, CSLS

### 4.2 Probabilistic Metrics
- **Energy Score**: Proper scoring rule for the vMF mixture
- **Coverage**: Nominal vs empirical (vMF-based, not Gaussian)
- **ECE**: Expected calibration error
- **AURC**: Area Under Risk-Coverage curve (first in brain decoding)

### 4.3 Risk-Coverage for Selective Prediction (Novel)
- Order samples by $\kappa$ (or combined $\kappa \cdot (1-\beta\delta)$)
- Compute risk at each coverage level
- Report AURC, E-AURC, R@80%, R@90%, R@95%
- Compare: kappa-only, delta-only, combined, random ordering
- **Hierarchical selective**: Fall back to category-level when instance uncertain

### 4.4 Image Reconstruction
PixCorr, SSIM, AlexNet(2), AlexNet(5), LPIPS, CLIPScore

### 4.5 Noise Ceiling Normalization (Novel for brain decoding)
- Spearman-Brown from 3 NSD repeats
- Report all metrics as % of theoretical maximum
- Enables fair cross-subject and cross-study comparison

### 4.6 Neuroscience Analysis
- Per-ROI attention importance with bootstrap CIs
- Category-ROI interaction heatmaps
- Uncertainty-reliability correlation

---

## 5. Experiments

### 5.1 Dataset and Setup
- NSD, Subjects 01/02/05/07, CLIP ViT-L/14 ($d=768$)
- ROI: nsdgeneral mask with 17 sub-regions
- Train/Val/Test: 70/15/15

### 5.2 Ablation Ladder (EXP0–EXP14)

| Exp | Architecture | Distribution | Loss | Key Innovation |
|-----|-------------|-------------|------|---------------|
| EXP0 | MLP | — | MSE+cos | Baseline |
| EXP1 | MLP | — | MSE+cos | +center_pcr |
| EXP2 | MLP | — | InfoNCE+Q | +contrastive |
| EXP3 | MLP | Gaussian | NLL+InfoNCE | +probabilistic |
| EXP4 | MLP | Gaussian | G-NCE | +Gaussian-NCE |
| EXP5 | MLP | Gaussian | G-NCE+KL | +KL annealing |
| EXP6 | MLP | Gaussian | G-NCE+KL | +whitening |
| **EXP7** | **MLP** | **vMF** | **vMF-NCE** | **+spherical model** |
| **EXP8** | **ROI-Trans** | **vMF** | **vMF-NCE** | **+ROI Transformer** |
| **EXP9** | **ROI-Trans** | **vMF-DCF** | **Multi-vMF-NCE** | **+ROI-DCF consensus** |
| **EXP10** | ROI-Trans | vMF-DCF | Multi-vMF-NCE | +mixture sampling |
| **EXP11** | ROI-Trans | vMF-DCF | Multi-vMF-NCE | +decomposed UA-CFG |
| **EXP12** | ROI-Trans | vMF-DCF | Multi-vMF-NCE | +ceiling temperature |
| **EXP13** | ROI-Trans | vMF-DCF | kappa-SPCL | +curriculum learning |
| **EXP14** | ROI-Trans | vMF-DCF | kappa-SPCL | **Full system** |

### 5.3 Hypotheses
- **H7**: vMF > Gaussian on $S^{d-1}$ (distribution matches geometry)
- **H8**: ROI-Transformer > MLP (inductive bias helps)
- **H9**: ROI-DCF > single-head (per-ROI distributions are richer)
- **H10**: Mixture sampling > consensus point (more diversity for generation)
- **H11**: Decomposed UA-CFG > heuristic UA-CFG (principled uncertainty use)
- **H12**: Ceiling temperature improves calibration without hurting retrieval
- **H13**: kappa-SPCL curriculum improves convergence
- **H14**: Full system > any ablation

---

## 6. Results

### 6.1 Main Results Table
SOTA comparison: our EXP14 vs MindEye, MindEye2, Brain Diffuser, Brain-IT

### 6.2 Ablation Table (EXP0–EXP14)
Full metrics across all experiments, all subjects, with significance tests

### 6.3 Risk-Coverage Curves (Figure 1) — KEY FIGURE
- kappa-only, delta-only, combined, random orderings
- With hierarchical selective prediction overlay
- AURC comparison table

### 6.4 Dual Uncertainty Quadrant Analysis (Figure 2) — KEY FIGURE
- 2D scatter: kappa vs delta for all test samples
- Color-coded by error / success
- Shows the four quadrants (Table from Section 3.6) empirically

### 6.5 ROI Contribution Analysis (Figure 3)
- Cortical flatmap of attention weights
- Per-ROI kappa distributions (which ROIs are most/least certain)
- Category-ROI interaction heatmap (faces engage FFA, places engage PPA)

### 6.6 Noise-Ceiling Normalized Results (Table)
- All metrics expressed as % of theoretical maximum
- Per-subject comparison showing how close we get to the limit

### 6.7 Image Reconstructions (Figure 4)
- Grid organized by kappa-delta quadrant
- Shows how generation quality varies with uncertainty
- Highlights abstention cases

---

## 7. Analysis and Discussion

### 7.1 Why Per-ROI Distributions Beat Point Estimates
- Geometric argument: each ROI sees different visual aspects
- The consensus naturally captures multi-view agreement
- Ablation: EXP9 vs EXP8 shows per-ROI vMF > single vMF

### 7.2 Dual Uncertainty Is Not Redundant
- $\kappa$ and $\delta$ are weakly correlated (show scatter)
- Each captures distinct failure modes
- Ablation: decomposed UA-CFG > single-scalar UA-CFG

### 7.3 Risk-Coverage for Clinical Deployment
- At 80% coverage, error drops by X%
- At 95% coverage, we maintain Y% of accuracy
- Hierarchical fallback recovers Z% of abstained predictions
- Implication: principled "I don't know" for BCIs

### 7.4 Frequency-ROI Interactions (if included)
- V1/V2 weigh high-frequency bands most heavily
- PPA/RSC weigh low-frequency (spatial layout) bands
- Consistent with known neuroscience (De Valois et al., 1982)

---

## 8. Limitations

1. **Single dataset**: NSD only; needs validation on BOLD5000, GOD, etc.
2. **Subject-specific**: Cross-subject alignment implemented but not fully validated
3. **vMF in 768-D**: Rejection sampling can be slow for very low $\kappa$
4. **ROI atlas dependence**: Performance depends on atlas quality
5. **Computational cost**: ROI-DCF adds parameters vs. single-head vMF

---

## 9. Broader Impact

- Enables safer BCI deployment through principled selective prediction
- Risk-coverage framework applicable to any neural decoding task
- Neuroscience interpretability: reveals which regions drive specific predictions
- Dual uncertainty decomposition generalizable beyond brain decoding

---

## 10. Conclusion

We introduced brain regions as directional experts — a principled framework where each ROI predicts a vMF distribution on the CLIP hypersphere. The resulting dual uncertainty (concentration $\kappa$ and disagreement $\delta$) provides distinct signals for both selective prediction and generation control. Combined with a comprehensive evaluation protocol including risk-coverage curves and noise-ceiling normalization, this work establishes a new standard for uncertainty-aware neural image decoding.

---

## Appendix

### A.1 vMF-NCE Bessel Cancellation Proof
Full derivation showing $\log C_d(\kappa)$ cancels in the cross-entropy softmax.

### A.2 Hyperparameters
Full table per experiment, all subjects.

### A.3 Additional Ablations
- PCR $k \in \{2, 4, 8, 16\}$
- Queue size $\in \{4K, 8K, 16K, 32K\}$
- ROI Transformer layers $\in \{2, 4, 6, 8\}$
- $\kappa$ bound range, SPCL temperature schedule
- Shared vs independent per-ROI heads

### A.4 Multi-Subject Results
Per-subject tables and cross-subject aggregation with paired tests.

### A.5 Noise Ceiling Analysis
Spearman-Brown results per subject and ROI.

### A.6 Frequency-ROI Band Weights
Learned band importance per ROI (interpretable neuroscience result).

### A.7 Qualitative Reconstructions
Extended image grids organized by uncertainty quadrant.

---

## References

- Allen, E.J. et al. (2022). A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence. *Nature Neuroscience*.
- Banerjee, A. et al. (2005). Clustering on the unit hypersphere using von Mises-Fisher distributions. *JMLR*.
- Davidson, T.R. et al. (2018). Hyperspherical variational auto-encoders. *UAI*.
- Geifman, Y. & El-Yaniv, R. (2017). Selective classification for deep neural networks. *NeurIPS*.
- Ho, J. & Salimans, T. (2022). Classifier-Free Diffusion Guidance. *arXiv*.
- Mardia, K.V. & Jupp, P.E. (2000). *Directional Statistics*. Wiley.
- Ozcelik, F. & VanRullen, R. (2023). Natural scene reconstruction from fMRI signals using generative latent diffusion. *Scientific Reports*.
- Schoppe, O. et al. (2016). Measuring the performance of neural models. *Frontiers in Computational Neuroscience*.
- Scotti, P. et al. (2024). MindEye2: Shared-Subject Models Enable fMRI-To-Image With 1 Hour of Data. *ICML*.
- Wood, A.T.A. (1994). Simulation of the von Mises Fisher distribution. *Communications in Statistics - Simulation and Computation*.
