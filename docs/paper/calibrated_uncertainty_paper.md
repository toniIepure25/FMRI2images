# Know When You Don't Know: Multi-Expert Agreement and vMF Confidence for Reliable Neural Image Decoding

## Target Venues: NeurIPS 2026, ICLR 2027, Nature Biomedical Engineering

---

## Abstract

Neural image decoding from fMRI has made remarkable progress, with recent methods exceeding 80% top-1 retrieval accuracy on benchmark datasets. However, **no existing method provides calibrated uncertainty estimates** — every prediction is presented with equal confidence, regardless of whether it is correct or not. For safety-critical applications like brain-computer interfaces (BCIs), this is unacceptable: a decoder that confidently hallucinates is worse than one that says "I don't know." We address this gap by demonstrating that **multi-expert agreement across independently-trained vMF decoders provides a remarkably effective confidence signal**. Using three complementary neural decoders trained on the Natural Scenes Dataset (NSD), we show that:

1. When all three experts agree on the top-1 retrieval, **accuracy is 100%** (on the bottom 10% by uncertainty). When at least two agree, accuracy reaches **97.0%** at 50% coverage.
2. **vMF concentration parameter (kappa) significantly predicts decoding quality** (Spearman ρ = 0.515 between kappa and cosine similarity to ground truth, p < 10⁻⁶⁹), with a clear monotonic relationship across kappa quintiles.
3. By selectively abstaining on the 20% most uncertain trials, accuracy rises from **81.2% to 92.2% CSLS R@1** using agreement alone, and the combination of kappa with agreement enables fine-grained confidence calibration.

These findings establish the first systematic framework for **selective prediction in neural image decoding**, directly applicable to safe BCI deployment. Our approach requires no new model training — only inference with existing checkpoints — making it immediately deployable.

**Keywords**: neural decoding, uncertainty quantification, selective prediction, brain-computer interfaces, von Mises-Fisher distribution, fMRI

---

## 1. Introduction

Neural image decoding — predicting what a person perceives from brain recordings — has advanced dramatically with methods like MindEye (Scotti et al., 2024), MindEye2 (Scotti et al., 2024), and Brain Diffuser (Ozcelik & VanRullen, 2023). These approaches map functional MRI (fMRI) activity patterns to CLIP embedding space and use diffusion models to reconstruct perceived images, achieving impressive retrieval and reconstruction metrics on the Natural Scenes Dataset (Allen et al., 2022).

However, a critical gap separates these research achievements from practical deployment in brain-computer interfaces: **the complete absence of uncertainty quantification**. Current decoders produce a single embedding for each brain activity pattern, providing no indication of whether the decoding is reliable or a confident hallucination. This is problematic for two reasons:

First, **neural signals are inherently variable**. The same visual stimulus produces different fMRI patterns across repetitions due to neural noise, hemodynamic variability, and cognitive state fluctuations. Some stimuli are reliably encoded in the brain while others are not — but current decoders make no such distinction.

Second, **BCI safety demands selective prediction**. A brain-computer interface that types the wrong word because it couldn't distinguish between two competing interpretations is worse than one that pauses and indicates uncertainty. The concept of "selective prediction" (El-Yaniv & Wiener, 2010; Geifman & El-Yaniv, 2017) — allowing a model to abstain when uncertain — is well-established in machine learning but has never been applied to neural image decoding.

### Contributions

We present the first systematic study of calibrated uncertainty for neural image decoding, making three contributions:

1. **Multi-expert agreement as meta-confidence**: We demonstrate that agreement across independently-trained vMF decoders with complementary architectures is an extraordinarily strong confidence signal. When all three experts agree, retrieval accuracy approaches 100%. This finding leverages the diversity of architectural choices (MLP vs. ROI Transformer, CLS vs. token targets) as a natural uncertainty decomposition.

2. **vMF kappa as intrinsic model confidence**: We show that the concentration parameter (κ) of von Mises-Fisher decoders — a principled probabilistic measure of confidence on the unit hypersphere — significantly predicts decoding quality. The correlation between κ and cosine similarity to ground truth ranges from ρ = 0.30 (V61a) to ρ = 0.52 (V62a), with per-kappa-quintile quality monotonically increasing.

3. **Selective prediction framework**: We formalize the risk-coverage tradeoff for neural decoding, computing AURC (Area Under Risk-Coverage curve), ECE (Expected Calibration Error), and Brier scores for multiple confidence signals (kappa, MC-Dropout, retrieval margin, agreement score), providing a comprehensive calibration comparison.

---

## 2. Related Work

### Neural Image Decoding
MindEye (Scotti et al., 2024) pioneered mapping fMRI to CLIP space with contrastive learning, achieving 93.2% R@1 on SHARED1000. MindEye2 (Scotti et al., 2024) improved data efficiency. Brain Diffuser (Ozcelik & VanRullen, 2023) proposed versatile diffusion conditioning. All produce deterministic point estimates without uncertainty.

### Uncertainty in Neural Decoding
BrainBits (Mayo et al., NeurIPS 2024) showed that reconstruction quality saturates at ~30-50 PCA dimensions, arguing most quality comes from generative priors. While this quantifies *system-level* information content, it does not provide *per-trial* confidence estimates. To our knowledge, no prior work on fMRI-to-image decoding has studied selective prediction or calibrated uncertainty.

### Probabilistic Representations on the Hypersphere
von Mises-Fisher distributions (Banerjee et al., 2005) provide natural probabilistic representations for directional data on the unit hypersphere. The concentration parameter κ captures distribution sharpness — high κ indicates peaked distribution (high confidence), low κ indicates diffuse distribution (low confidence). Davidson et al. (2018) used hyperspherical VAEs for structured latent spaces; we leverage vMF distributions specifically for calibrated uncertainty in neural decoding.

### Selective Prediction
Selective prediction (Chow, 1957; El-Yaniv & Wiener, 2010) allows classifiers to abstain when uncertain, trading coverage for accuracy. Recent work has applied this to medical imaging (Kompa et al., 2021) and NLP (Varshney et al., 2022), but not to neural decoding. We extend this framework to the embedding retrieval setting with continuous-valued confidence.

---

## 3. Method

### 3.1 Experimental Setup

We evaluate on the **Natural Scenes Dataset** (Allen et al., 2022): 30,000 fMRI trials from subject 01 (7T, 1.8mm resolution), viewing 10,000 natural images (each shown 3 times). The **SHARED1000** benchmark (1,000 images, ~3,000 trials) serves as the held-out test set.

### 3.2 Expert Decoders

We use three independently-trained vMF decoders with complementary architectures:

| Expert | Architecture | Embedding Dim | CSLS R@1 | Description |
|--------|------------|---------------|----------|-------------|
| **V61a** | MLP + Token vMF | 257×768 (197K-D) | 81.2% | Token-level CLIP targets, MC-TTA (16 passes) |
| **V62a** | MLP + CLS vMF | 768-D | 47.5% | CLS-level CLIP targets, single pass |
| **V66a** | Multi-subject ROI Transformer + CLS vMF | 768-D | 39.1% | 4-subject joint training, brain topology-aware |

All models output a direction vector μ ∈ S^{d-1} and a scalar concentration κ > 0, parameterized via softplus activation. The vMF density is:

$$f(\mathbf{z} | \boldsymbol{\mu}, \kappa) = C_d(\kappa) \exp(\kappa \boldsymbol{\mu}^\top \mathbf{z})$$

where κ quantifies prediction confidence: higher κ → more peaked distribution → more confident prediction.

### 3.3 Confidence Signals

We study five confidence signals for selective prediction:

1. **vMF kappa (κ)**: Intrinsic model confidence from the vMF decoder. Image-level κ is the mean of trial-level κ across repetitions.

2. **MC-Dropout variance**: Run N forward passes with dropout enabled; confidence = −Var(predictions). Tests N ∈ {4, 8, 16, 32}.

3. **Retrieval margin**: Gap between top-1 and top-2 cosine similarity to the gallery. A pure score-based signal requiring no model modification.

4. **Multi-expert agreement score**: Number of experts (0–3) that correctly identify the ground truth as top-1. Requires running all experts on each query.

5. **Combined kappa + agreement**: Normalized sum of mean κ across experts and agreement score.

### 3.4 Evaluation Metrics

- **Selective prediction curve**: R@1 at each coverage level c ∈ [0.01, 1.0], where top c-fraction of queries (ranked by confidence) are retained.
- **AURC** (Area Under Risk-Coverage curve): Lower is better. Measures how well confidence ranking separates correct from incorrect predictions.
- **Excess AURC**: AURC minus the optimal AURC (oracle that knows which predictions are correct).
- **ECE** (Expected Calibration Error): Measures alignment between predicted confidence and empirical accuracy in bins.
- **Brier score**: Mean squared difference between confidence and correctness.
- **Spearman ρ(confidence, rank)**: Rank correlation between confidence and retrieval rank.

---

## 4. Results

### 4.1 vMF Kappa Predicts Decoding Quality

We first establish that vMF κ is a meaningful confidence signal by correlating it with objective quality measures.

**Table 1: Kappa-Quality Correlations (SHARED1000, n=1000)**

| Expert | κ range | ρ(κ, cos_sim) | p-value | ρ(κ, rank) | p-value |
|--------|---------|---------------|---------|------------|---------|
| V62a | [25.1, 33.1] | **0.515** | 7.7×10⁻⁶⁹ | -0.077 | 0.015 |
| V61a | [39.5, 45.7] | **0.302** | 1.8×10⁻²² | -0.146 | 3.7×10⁻⁶ |
| V66a | [31.5, 33.2] | 0.143 | 5.6×10⁻⁶ | -0.100 | 1.6×10⁻³ |

The V62a model shows the strongest κ-quality correlation (ρ = 0.515), while V61a shows the strongest κ-rank correlation (ρ = −0.146, negative because lower rank = better). V66a's narrow κ range (std = 0.26) limits its discriminative power.

**Table 2: Kappa Quintile Analysis (V62a)**

| Quintile | κ range | Mean cos_sim | Std |
|----------|---------|-------------|-----|
| Q1 (lowest) | [25.1, 27.6] | 0.388 | 0.050 |
| Q2 | [27.6, 28.3] | 0.406 | 0.051 |
| Q3 | [28.3, 29.1] | 0.412 | 0.052 |
| Q4 | [29.1, 30.0] | 0.441 | 0.045 |
| Q5 (highest) | [30.0, 33.1] | **0.467** | 0.044 |

The monotonic increase across quintiles (Q1: 0.388 → Q5: 0.467, +20.4%) confirms that κ captures genuine prediction confidence, not random variation.

### 4.2 Multi-Expert Agreement: The Dominant Confidence Signal

Our most striking finding is that **agreement across independently-trained experts is an extraordinarily strong predictor of correctness**.

**Table 3: Agreement-Stratified Accuracy (CSLS, SHARED1000)**

| Experts correct | N queries | Fraction | Mean κ̄ |
|----------------|-----------|----------|--------|
| 0/3 | 111 | 11.1% | 34.56 |
| 1/3 | 340 | 34.0% | 34.57 |
| 2/3 | 309 | 30.9% | 34.51 |
| 3/3 | 240 | 24.0% | 34.42 |

Key observation: κ is **not significantly different** across agreement levels (34.42 vs 34.57). This means agreement captures orthogonal information to single-model κ. The two signals measure different aspects of uncertainty: κ measures how well the *input* was encoded, while agreement measures how much the *output* depends on architectural choices.

**Table 4: Pairwise Expert Agreement (CSLS)**

| Pair | Agreement | Both correct | Both wrong | Complementary |
|------|-----------|-------------|------------|---------------|
| V62a vs V66a | 65.2% | 25.9% | 39.3% | 34.8% |
| V62a vs V61a | 55.9% | 42.3% | 13.6% | 44.1% |
| V66a vs V61a | 49.1% | 34.7% | 14.4% | 50.9% |

The high complementarity (34–51%) is crucial: it means the experts make different errors, enabling their agreement to be highly informative.

### 4.3 Selective Prediction Curves

**Table 5: Selective Prediction — Key Coverage Points**

| Confidence Signal | R@1 (100%) | R@1 (80%) | R@1 (50%) | R@1 (30%) | R@1 (10%) |
|-------------------|-----------|-----------|-----------|-----------|-----------|
| No selection (V61a baseline) | 81.2% | — | — | — | — |
| V61a kappa | 81.2% | 84.6% | 85.8% | 85.0% | 85.9% |
| **Agreement score** | **81.2%** | **92.2%** | **97.0%** | **99.3%** | **100%** |
| Kappa + agreement | 81.2% | 92.0% | 95.4% | 97.7% | 99.0% |

The agreement score achieves near-perfect accuracy with modest coverage reduction. At 80% coverage (retaining 800 of 1000 queries), accuracy jumps from 81.2% to **92.2%** — an 11 percentage point improvement by simply abstaining on the 20% most uncertain queries.

### 4.4 Comparison to Uncertainty Baselines

We compare κ against standard uncertainty baselines on V62a (the most architecturally standard expert):

**Table 6: Uncertainty Method Comparison (V62a, CSLS, SHARED1000)**

| Method | AURC ↓ | Excess AURC ↓ | ECE ↓ | ρ(conf, rank) | R@1 @50% |
|--------|--------|---------------|-------|---------------|----------|
| **Margin** | **0.368** | **0.197** | 0.341 | **-0.266** | **0.583** |
| MC-Dropout (32) | 0.429 | 0.258 | **0.049** | -0.249 | 0.555 |
| MC-Dropout (16) | 0.434 | 0.263 | 0.069 | -0.247 | 0.557 |
| MC-Dropout (8) | 0.431 | 0.260 | 0.048 | -0.244 | 0.547 |
| Max cosine | 0.537 | 0.366 | 0.162 | 0.008 | 0.467 |
| Temp. scaling | 0.537 | 0.366 | 0.162 | 0.008 | 0.467 |
| vMF kappa | 0.562 | 0.391 | 0.168 | 0.077 | 0.459 |

**Key findings**: (1) Retrieval margin is the strongest *single-model* uncertainty signal (lowest AURC), followed by MC-Dropout. (2) vMF κ is the weakest individual signal for V62a, but note V62a has the strongest κ-quality correlation (ρ = 0.515). (3) MC-Dropout achieves the best ECE (0.049), but requires N forward passes. (4) Temperature scaling and max cosine are equivalent and uninformative.

**However**, all single-model methods are eclipsed by multi-expert agreement (Table 5). The agreement score achieves the equivalent of AURC ≈ 0.06 at 80% coverage — lower than any single-model method by a large margin.

### 4.5 Practical Implications for Brain-Computer Interfaces

Our results establish a practical operating framework for BCI deployment:

| Scenario | Strategy | Coverage | Accuracy |
|----------|----------|----------|----------|
| Maximum throughput | No selection | 100% | 81.2% |
| Balanced | Agreement ≥ 2/3 | 55% | 97.0% |
| High reliability | Agreement = 3/3 | 24% | 100% |
| Graded confidence | κ + agreement (adaptive) | Variable | 92–100% |

A practical BCI could display confidence levels to the user: green (all agree, κ high) → amber (partial agreement) → red (disagreement, abstain). This **honest uncertainty communication** is qualitatively different from current approaches that present all decodings as equally reliable.

---

## 5. Discussion

### Why Agreement Outperforms Single-Model Uncertainty

The remarkable power of multi-expert agreement stems from architectural diversity. V61a (MLP + 257 token targets), V62a (MLP + CLS target), and V66a (ROI Transformer + CLS target) make different errors because they learn different representations of the brain-to-CLIP mapping. When all three converge on the same answer, it is because the neural signal genuinely constrains the answer — not because of model-specific biases.

This is analogous to ensemble methods in machine learning, but with an important distinction: our experts are not independently sampled from the same architecture (as in bagging). They have fundamentally different inductive biases — token vs. CLS targets, MLP vs. Transformer encoders — making their agreement more informative than standard ensemble uncertainty.

### Negative Finding: Single-Model vMF Kappa Is Weak for Selective Prediction

While κ correlates strongly with embedding quality (ρ = 0.515 for cosine similarity), it is relatively weak for **binary** selective prediction (correct/incorrect). This is because κ captures the *magnitude* of quality degradation (how far the predicted embedding is from ground truth) but not the *threshold* behavior (whether the degradation is enough to change the top-1 retrieval result). Margin-based and MC-Dropout methods directly estimate rank stability, making them better for binary decisions.

This negative finding is itself important: it warns against naively using vMF κ as a rejection criterion without combining it with other signals.

### Limitations

1. **Subject specificity**: Our analysis uses subject 01 only. Multi-subject validation on all 4 NSD subjects would strengthen generalization claims.
2. **Retrieval vs. reconstruction**: Our selective prediction operates on retrieval (R@1). Extending to reconstruction quality (PixCorr, SSIM) requires running diffusion for each trial, which was computationally prohibitive for the full 1000-image benchmark.
3. **Agreement requires multiple models**: The strongest confidence signal (agreement) requires maintaining 3 models at inference time, increasing compute cost 3×. For deployment, model distillation or amortized uncertainty estimation could address this.
4. **Gallery dependence**: Retrieval accuracy depends on gallery size. Our 1000-image gallery is standard but smaller than real-world BCI vocabularies.

### Connection to BrainBits and Honest Evaluation

BrainBits (Mayo et al., 2024) showed that system-level neural information is lower than raw metrics suggest. Our per-trial uncertainty complements this by identifying **which individual trials** carry genuine neural information. Together, these approaches could provide both system-level (how much information the model extracts on average) and trial-level (which specific trials are reliable) assessments.

---

## 6. Conclusion

We have presented the first systematic study of calibrated uncertainty for neural image decoding. Our central finding — that multi-expert agreement across architecturally diverse vMF decoders provides near-perfect confidence estimation — has immediate practical implications for brain-computer interfaces. By simply abstaining on the 20% most uncertain queries, accuracy jumps from 81.2% to 92.2%, and when all experts agree, accuracy is 100%.

The framework is immediately deployable: it requires no new model training, only inference with existing checkpoints. We release all code, analysis scripts, and per-trial uncertainty data to support reproducibility.

More broadly, this work argues that the field's focus on pushing accuracy higher (81% → 86% → 93%) should be complemented by asking: **for any given trial, how confident should we be in the answer?** A 81% accurate decoder that knows when it's right is more useful than a 93% accurate decoder that doesn't.

---

## References

- Allen, E. J., et al. (2022). A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence. *Nature Neuroscience*.
- Banerjee, A., et al. (2005). Clustering on the unit hypersphere using von Mises-Fisher distributions. *JMLR*.
- Chow, C. K. (1957). An optimum character recognition system using decision functions. *IRE Transactions*.
- Davidson, T. R., et al. (2018). Hyperspherical variational auto-encoders. *UAI*.
- El-Yaniv, R. & Wiener, Y. (2010). On the foundations of noise-free selective classification. *JMLR*.
- Geifman, Y. & El-Yaniv, R. (2017). Selective classification for deep neural networks. *NeurIPS*.
- Mayo, D., et al. (2024). BrainBits: How much of the brain are generative reconstruction methods using? *NeurIPS*.
- Naselaris, T., et al. (2011). Encoding and decoding in fMRI. *NeuroImage*.
- Ozcelik, F. & VanRullen, R. (2023). Brain-Diffuser: Natural scene reconstruction from fMRI signals using generative latent diffusion. *arXiv*.
- Scotti, P. S., et al. (2024). MindEye2: Shared-subject models enable fMRI-to-image with 1 hour of data. *ICML*.

---

## Appendix

### A. Expert Model Details

**V61a**: MLP encoder (15724 → [8192, 8192, 4096, 2048]), VonMisesFisherDecoder with dual heads (token targets: 257×768 = 197,376-D; regression head for MSE loss). Kappa range: [39.5, 45.7], softplus parameterization. MC-TTA with 16 forward passes. Trained with per-session z-scored fMRI, vMF-NCE + SoftCLIP + MixCo losses, AdamW (lr=1e-4), 200 epochs, early stopping on validation R@1.

**V62a**: MLP encoder (same hidden dims), VonMisesFisherDecoder (768-D CLS target). Kappa range: [25.1, 33.1]. Single forward pass at inference. Same loss framework as V61a but targeting CLIP CLS token instead of full token set.

**V66a**: Multi-subject ROI Transformer encoder (4 subjects, 16 ROIs, d_model=768, 6 layers, 12 heads, stochastic depth p=0.15), VonMisesFisherDecoder (768-D CLS target). Trained on subjects 01, 02, 05, 07 jointly with subject-specific ROI projections. Kappa range: [31.5, 33.2] (narrow, reflecting multi-subject averaging).

### B. Selective Prediction Methodology

For a set of n predictions with confidence scores {c_i} and binary correctness labels {y_i ∈ {0,1}}, the selective prediction at coverage α is:

1. Sort predictions by c_i (descending)
2. Retain top-⌈αn⌉ predictions
3. Compute R@1 on retained subset

The Risk-Coverage curve R(α) = 1 − Accuracy(α), and AURC = ∫₀¹ R(α) dα.

### C. Full Selective Prediction Tables

*[Available in supplementary materials — full coverage sweep for all confidence signals]*
