# Ablation Plan

## Overview

Streamlined ablation study with 6 experiments: 2 strong baselines + 4 novel contributions.
Consecutive row differences isolate each innovation — the main table IS the ablation.
All experiments use the same dataset, splits, seeds, and evaluation protocol.

## Experimental Conditions

### B0: Strong Deterministic Baseline
**Purpose**: Best possible system using only published, well-known components.

**Configuration**:
- Architecture: MLP encoder [V -> 4096 -> 2048 -> 1024], GELU, dropout 0.1
- Loss: MSE + InfoNCE (symmetric, learnable temperature)
- Preprocessing: center_pcr (k=8)
- Queue: MoCo-style (Q=8192)
- Batch size: 4, gradient accumulation 16 (effective 64), mixed precision

**Expected**:
- Good cosine similarity and decent R@1 (standard techniques all combined)
- No uncertainty estimates
- Establishes the ceiling for non-probabilistic methods

**Proves**: This is the strongest standard baseline. Every novel experiment must beat this.

---

### B1: Probabilistic Gaussian Baseline
**Purpose**: Best possible probabilistic system using Gaussian assumption.

**Changes from B0**:
- Model outputs: mu + logvar (heteroscedastic)
- Loss: Gaussian-NLL + Gaussian-NCE + KL annealing (linear, free-bits 0.5)
- Removes MSE and InfoNCE

**Expected**:
- Similar or slightly better retrieval than B0
- Calibrated uncertainty via Gaussian-NCE
- Usable AURC for selective prediction

**Proves**: Gaussian probabilistic modeling provides uncertainty but assumes wrong geometry
(Euclidean in R^d vs hyperspherical S^{d-1}).

---

### N1: vMF-NCE (Novel — Principled Hyperspherical Distribution)
**Purpose**: Replace Gaussian with vMF, which matches the data manifold.

**Changes from B1**:
- Distribution: vMF with bounded-sigmoid kappa [0.001, 500]
- Loss: vMF-NCE (Bessel-free) + kappa regularizer
- Same MLP encoder as B0/B1

**Expected**:
- **Better calibration** than B1: Coverage@95 closer to 0.95 (correct manifold)
- **Better retrieval**: vMF log-density is the principled similarity for L2-normalized keys
- **Lower AURC**: uncertainty is more decision-relevant on S^{d-1}

**Proves**: vMF is the correct distribution for hyperspherical CLIP embeddings.

**Key comparison**: B1 vs N1 isolates the effect of the distributional assumption.

---

### N2: ROI Transformer (Novel — Brain-Topology-Aware Architecture)
**Purpose**: Replace flat MLP with brain-region-aware Transformer encoder.

**Changes from N1**:
- Encoder: ROI-Tokenized Transformer (17 ROI tokens + [CLS], 4 layers, 8 heads)
- Higher LR (3e-4), longer warmup (10 epochs), weight decay 0.05
- Interpretability: attention weights reveal which ROIs drive each prediction

**Expected**:
- **Better retrieval** than N1: inductive bias from cortical topology
- **Interpretable**: V1-V4 attend to low-level features, FFA/PPA to high-level
- **ROI ablation**: masking individual ROIs reveals functional specialization

**Proves**: Brain topology is a useful inductive bias for fMRI decoding.

**Key comparison**: N1 vs N2 isolates MLP vs ROI Transformer (same vMF distribution).

---

### N3: ROI-DCF (Novel — Per-ROI Directional Consensus Fusion)
**Purpose**: Each brain region predicts its own vMF distribution; fuse via spherical consensus.

**Changes from N2**:
- model.type = vmf_dcf (per-ROI vMF heads + consensus fusion)
- Multi-task loss: fused vMF-NCE + auxiliary per-ROI vMF-NCE (lambda=0.5)
- Produces kappa_consensus and disagreement delta
- Disagreement-aware UA-CFG for inference

**Expected**:
- **Better retrieval and calibration** than N2
- **Dual uncertainty**: kappa (aleatoric) + delta (epistemic-like)
- **Per-ROI interpretability**: each region's directional prediction is inspectable

**Proves**: Per-ROI distributions > single-head distribution.

**Key comparison**: N2 vs N3 isolates single-head vs ROI-DCF consensus.

---

### N4: Full System (Flagship — All Innovations Combined)
**Purpose**: Combine all innovations for best overall performance.

**Changes from N3**:
- kappa-SPCL curriculum: self-paced contrastive learning (T: 100 -> 1, cosine)
- Noise-ceiling temperature: beta=0.5, ceiling from NCSNR
- vMF mixture sampling for generation (K=8, proportional, closest-to-consensus)
- Decomposed UA-CFG: kappa -> w, delta -> K and steps

**Expected**:
- **Best overall**: retrieval, calibration, reconstruction, interpretability
- Demonstrates the full value of decomposed uncertainty for generation
- Risk-coverage curves show principled selective prediction

**Proves**: Full system > any subset; all innovations contribute.

**Key comparison**: N3 vs N4 isolates the generation stack innovations.

---

## Ablation Matrix

| Component | B0 | B1 | **N1** | **N2** | **N3** | **N4** |
|-----------|----|----|--------|--------|--------|--------|
| **Architecture** | MLP | MLP | MLP | **ROI-Trans** | **ROI-Trans** | **ROI-Trans** |
| **Preprocessing** | center_pcr | center_pcr | center_pcr | center_pcr | center_pcr | center_pcr |
| **Queue** | 8192 | 8192 | 8192 | 8192 | 8192 | 8192 |
| **Distribution** | — | Gaussian | **vMF** | **vMF** | **vMF-DCF** | **vMF-DCF** |
| **Contrastive Loss** | InfoNCE | G-NCE | **vMF-NCE** | **vMF-NCE** | **Multi-vMF-NCE** | **kappa-SPCL** |
| **KL/Regularizer** | — | KL anneal | kappa reg | kappa reg | kappa reg | kappa reg |
| **Mixture Sampling** | — | — | — | — | — | **Yes** |
| **DUA-CFG** | — | — | — | — | UA-CFG | **Decomposed** |
| **Ceiling Temp** | — | — | — | — | — | **Yes** |

## Evaluation Metrics per Experiment

### All Experiments
- Retrieval: R@1, R@5, R@10, MeanR, MedR, MRR, nDCG@10
- Identification: 2AFC, AUC, Cohen's d, d'
- Structure: RSA, CKA
- Oracle check: Pass/Fail

### Probabilistic (B1, N1-N4)
- NLL per-dim
- Energy Score (with sampling)
- Coverage@80, Coverage@95
- ECE, AURC
- Probabilistic 2AFC

### ROI-specific (N2-N4)
- Per-ROI attention importance (bootstrap CIs)
- ROI ablation study (leave-one-out)
- Category-ROI interaction heatmaps

### Full System (N4 only)
- Risk-coverage curves (kappa-only, delta-only, combined)
- Mixture energy score
- Ceiling-normalized metrics
- Dual uncertainty quadrant analysis

## Key Comparisons

| Comparison | Isolates | Expected Winner |
|------------|----------|-----------------|
| B0 vs B1 | Probabilistic modeling (Gaussian) | B1 (adds calibrated uncertainty) |
| B1 vs N1 | Distribution choice (Gaussian vs vMF) | N1 (correct manifold) |
| N1 vs N2 | Architecture (MLP vs ROI Transformer) | N2 (topology inductive bias) |
| N2 vs N3 | Fusion (single-head vs ROI-DCF) | N3 (per-ROI distributions) |
| N3 vs N4 | Generation stack (base vs full) | N4 (all innovations) |

## Statistical Testing

For each consecutive comparison, perform:
1. **Bootstrap test** (1000 iterations) for significance of R@1 difference
2. **Paired t-test** on per-sample metrics (2AFC, RSA)
3. Report **effect sizes** (Cohen's d) for all comparisons

**Significance threshold**: p < 0.05 (after Bonferroni correction for 5 tests)

## Timeline and Resources

### Estimated Time per Experiment
- B0, B1, N1: ~2-3 hours each (MLP encoder)
- N2, N3, N4: ~4-6 hours each (Transformer encoder)
- **Total**: ~18-24 hours for all 6 experiments (was ~45-60 hours for 15)

### Compute Requirements
- GPU: 1x A100 or V100 (min 16GB VRAM)
- Storage: ~50GB (checkpoints + cached data)

### Execution Order
Sequential chain: B0 -> B1 -> N1 -> N2 -> N3 -> N4.
Can partially parallelize baselines (B0 || B1) if multiple GPUs available.

## Paper Results Table Structure

**Table 1 — Main comparison** (each row is one training run):

| Method | R@1 | R@5 | MRR | AURC | PixCorr | SSIM |
|--------|-----|-----|-----|------|---------|------|
| Ridge baseline (published) | - | - | - | - | - | - |
| MindEye (Scotti et al., 2023) | - | - | - | - | 0.309 | 0.323 |
| MindEye2 (Scotti et al., 2024) | - | - | - | - | 0.320 | 0.341 |
| B0: Deterministic MLP | x | x | x | - | x | x |
| B1: Gaussian MLP | x | x | x | x | x | x |
| **N1: vMF-NCE (ours)** | x | x | x | x | x | x |
| **N2: ROI Transformer (ours)** | x | x | x | x | x | x |
| **N3: ROI-DCF (ours)** | x | x | x | x | x | x |
| **N4: Full system (ours)** | x | x | x | x | x | x |

The chain structure means **consecutive row differences = ablation**.
No separate ablation table needed.

## Deliverables

1. **Metrics JSON** for each experiment
2. **Plots** for each experiment (in evaluation/plots/)
3. **Comparison report** (compare_experiments.py output)
4. **LaTeX table** for paper (auto-generated from metrics)
5. **Statistical test results** (p-values, effect sizes)
