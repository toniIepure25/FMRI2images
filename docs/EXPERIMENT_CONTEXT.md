# Experiment Context: fMRI-to-Image Neural Decoding

This document provides complete context for analyzing why our training experiments produce near-chance retrieval metrics despite multiple optimization iterations. It is designed to be consumed by an LLM or researcher performing deep analysis.

---

## 1. Task Definition

**Goal**: Decode visual perception from fMRI brain activity into CLIP embeddings, then reconstruct the perceived image via a diffusion model.

**Pipeline**: `fMRI (7T, NSD) -> Encoder -> CLIP ViT-L/14 768-D embedding -> Stable Diffusion 2.1 -> Reconstructed image`

**Primary metric**: R@1 retrieval accuracy -- given a predicted CLIP embedding, retrieve the correct image from the full validation gallery (~986 unique images). Chance level = 1/986 = 0.10%.

**SOTA reference points (subj01)**:

| Method | R@1 | PixCorr | SSIM | Alex(2) | Alex(5) |
|--------|-----|---------|------|---------|---------|
| MindEye (Scotti et al., 2024) | 93.2% | 0.309 | 0.323 | 0.947 | 0.978 |
| MindEye2 (Scotti et al., 2024) | 93.0% | 0.320 | 0.341 | 0.960 | 0.983 |
| Brain Diffuser (Ozcelik & VanRullen, 2023) | -- | 0.254 | 0.356 | 0.942 | 0.962 |

MindEye achieves 93%+ R@1 using the **same dataset (NSD)**, the **same subjects**, and a **simpler architecture** (MLP + InfoNCE + MSE, no vMF, no ROI Transformer). This is the critical benchmark.

---

## 2. Dataset: Natural Scenes Dataset (NSD)

**Source**: Allen et al., 2022. 7T fMRI at 1.8mm resolution.

| Property | Value |
|----------|-------|
| Subjects used | subj01, subj02, subj05, subj07 |
| Trials per subject | 30,000 (40 sessions x 750 volumes) |
| Unique images per subject | ~10,000 (each shown ~3 times) |
| shared1000 | 1,000 images seen by all 8 subjects (held out for evaluation) |
| Beta volume shape (subj01) | (81, 104, 83) per trial |
| ROI mask | `nsdgeneral` -- general visual cortex |
| nsdgeneral voxels (subj01) | ~15,724 |
| fMRI dtype | float32 (converted from NIfTI) |

**ROIs available** (17 regions for ROI Transformer):
V1v, V1d, V2v, V2d, V3v, V3d, V3A, V3B, V4, FFA1, FFA2, PPA, EBA, OFA, OPA, RSC, nsdgeneral_other (residual voxels).

ROI voxel counts are computed dynamically from NSD atlas NIfTI files at runtime via `build_roi_index()`. The YAML config `roi_dims` values are nominal placeholders only.

---

## 3. Data Pipeline

### 3.1 fMRI Feature Extraction (offline, per-subject)

```
NIfTI betas (40 sessions x 750 volumes)
  -> apply nsdgeneral.nii.gz boolean mask
  -> flatten to (N_voxels,) per trial
  -> stack all trials -> (30000, ~15724) float32
  -> save as cache/preextracted/subject={subj}/fmri_features.npy (~1.8 GB)
```

Script: `scripts/build/preextract_fmri.py`. No normalization or PCA applied here -- raw ROI-masked beta values.

### 3.2 Training-time Processing

```
fmri_features.npy (30000, ~15724)
  -> individual trials (NO repetition averaging; each fMRI repetition
     is a separate training sample mapped to the same CLIP target)
     -> ~24,000 training trials (3x more than averaged)
  -> image-level train/val split (90/10 by unique nsdId, shared1000 excluded)
     -> train: ~8881 unique images (~24k trials), val: ~986 images
  -> per-SESSION z-scoring (computed on train trials within each session)
     -> for each of 40 sessions: features_s = (features_s - mean_s) / std_s
     -> removes session-level scanner drift without destroying inter-session variance
     -> stats saved to {output_dir}/zscore_stats/session_{s}_mean.npy, session_{s}_std.npy
  -> model input: (batch_size, ~15724) float32
```

### 3.3 CLIP Target Embeddings

```
NSD stimulus images (73k COCO images in nsd_stimuli.hdf5)
  -> CLIP ViT-L/14 (openai) encode_image()
  -> L2-normalize
  -> 768-D float32 embeddings
  -> stored in outputs/clip_cache/clip.parquet (9999 unique nsdIds)
```

The CLIP embeddings are the regression/contrastive targets. The model must learn to map fMRI -> CLIP space.

### 3.4 Data Split Details

- `split_by_image: true` -- splits by unique `nsdId`, not by trial row (prevents leakage from repeated images)
- `exclude_shared1000: true` -- 1000 shared images removed from train+val entirely
- Remaining ~9000 unique images split 90/10
- `seed: 42` for reproducibility
- Individual trials: each fMRI repetition is a separate sample, all mapped to the same CLIP target (biological augmentation)

---

## 4. Model Architectures

### 4.1 Overview

All models use the `UnifiedModel` factory: `Encoder -> Decoder -> Output`. Input dim is set from data (~15,724 voxels). Output dim matches CLIP (768).

### 4.2 B0: Deterministic Baseline

```
fMRI (B, 15724)
  -> MLPEncoder [8192, 4096, 2048] with ResidualBlocks, GELU, dropout=0.15
     -> (B, 2048)
  -> DeterministicDecoder [2048] -> Linear(2048, 768) -> L2-normalize
     -> (B, 768)
```

ResidualBlock: `x + (LayerNorm -> Linear(dim,dim) -> GELU -> Dropout -> Linear(dim,dim) -> Dropout)(x)`
One ResidualBlock inserted after each MLP projection layer.

Total parameters: ~328M (dominated by first Linear(15724, 8192) = 129M params).

### 4.3 B1: Gaussian Probabilistic Baseline

Same encoder as B0. Decoder outputs `(mu, logvar)`:

```
fMRI -> MLPEncoder -> (B, 2048)
  -> GaussianDecoder: shared backbone [2048]
     -> mu_head: Linear(2048, 768) -> L2-normalize
     -> logvar_head: Linear(2048, 768) -> clamp(-10, 5)
  -> output: (mu, logvar) both (B, 768)
```

### 4.4 N1: vMF-NCE (MLP + von Mises-Fisher)

Same encoder as B0. Decoder outputs `(mu, kappa)` on the unit hypersphere S^{767}:

```
fMRI -> MLPEncoder -> (B, 2048)
  -> VonMisesFisherDecoder: backbone [2048]
     -> mu_head: Linear(2048, 768) -> L2-normalize
     -> kappa_head: Linear(2048, 1) -> bounded sigmoid
        kappa = kappa_min + (kappa_max - kappa_min) * sigmoid(raw)
        kappa_min=1.0, kappa_max=50.0
  -> output: mu (B, 768) on S^{767}, kappa (B, 1) in [1, 50]
```

### 4.5 N2: ROI-Tokenized Transformer

```
fMRI (B, 15724)
  -> ROITransformerEncoder:
     -> 17 ROI projections: per-ROI Linear(n_voxels_r, 512) -> LayerNorm -> GELU -> Dropout
     -> prepend learnable [CLS] token
     -> add positional embeddings
     -> 4-layer Transformer (d_model=512, nhead=8, dim_ff=2048, pre-norm, GELU)
     -> output: [CLS] token (B, 512)
  -> VonMisesFisherDecoder (legacy): backbone [1024]
     -> mu, log_kappa with log_kappa in [0, 4] (kappa in [1, ~55])
```

Each ROI gets its own projection layer. Voxel counts per ROI vary by subject (e.g., V1v may have 700 voxels for subj01 but differ for subj02).

### 4.6 N3: ROI-DCF (Directional Consensus Fusion)

Same ROI Transformer encoder as N2, but with `return_roi_tokens=True`:

```
fMRI -> ROITransformerEncoder -> cls_out (B, 512), roi_tokens (B, 17, 512), alphas (B, 17)
  -> ROIDCFDecoder:
     -> PerROIVMFHeads (shared weights): each token -> (mu_r, kappa_r)
        17 x [Linear(512, 768) -> L2-norm for mu, Linear(512, 1) -> bounded sigmoid for kappa]
     -> SphericalConsensusFusion:
        R_vec = sum_r (alpha_r * kappa_r * mu_r)
        mu_fused = normalize(R_vec)
        kappa_consensus = ||R_vec||
        delta = 1 - sum_{r,s} alpha_r * alpha_s * cos(mu_r, mu_s)
  -> output: mu_fused (B, 768), kappa_consensus (B, 1), delta (B, 1)
```

Alphas come from [CLS]-to-ROI attention in the Transformer's final layer, normalized over ROIs. No extra gating network.

### 4.7 N4: Full System

Identical architecture to N3. Differences are in training (SPCL curriculum, dual contrastive losses) and inference (mixture sampling, decomposed UA-CFG). See Sections 5 and 9.

---

## 5. Loss Functions

### 5.1 InfoNCE (used by B0)

```
logits_ij = (query_i . key_j) * exp(log_scale) / tau
loss = CrossEntropy(logits, diag_labels)
```

- Learnable temperature via `log_scale` parameter (initialized to `log(1/0.07)`)
- Clamped: `logit_scale <= exp(4.6052) = 100`
- Symmetric: average of query->key and key->query losses
- Memory queue (16384): negatives from previous batches concatenated to logits

### 5.2 MSE Loss (used by B0, B1)

Standard MSE between predicted embedding and GT CLIP embedding, weight=0.1.

### 5.3 Gaussian NCE (used by B1)

```
log_likelihood(query, key) = -0.5 * sum_d (logvar_d + (x_d - mu_d)^2 / exp(logvar_d) + log(2*pi))
logits = pairwise_log_likelihood_matrix / temperature
loss = CrossEntropy(logits, diag_labels)
```

- `logvar` clamped to [-10, 5]
- Temperature: 1.0 (fixed)
- Weight: 2.0

### 5.4 KL Divergence (used by B1)

Annealed from weight 0.0 to 0.001 over 10,000 steps (linear). Free-bits: 0.5 per dimension.

### 5.5 vMF-NCE (used by N1, N2, N3)

```
logit(q, k) = kappa_q * cos(mu_q, z_k) / tau
loss = CrossEntropy(logits, diag_labels)
```

- The log-normalizer `log C_d(kappa)` cancels in the softmax -- NO Bessel functions needed
- `tau = 0.07`
- Logits clamped to [-80, 80] for float16 stability
- Memory queue: 16384 entries

### 5.6 Kappa Regularizer (used by N1, N2, N3, N4)

```
L_reg = lambda_kappa * mean(kappa)
```

Prevents unbounded kappa growth. `lambda_kappa` varies: 0.05 (N1), 0.1 (N2), 0.05 (N3), 0.02 (N4).

### 5.7 KappaSPCL (used by N4)

```
w_i = softmax(kappa_i / T_curriculum)
L = -sum_i w_i * log(exp(s_ii) / sum_j exp(s_ij))
```

Self-Paced Curriculum Learning: high-confidence samples (high kappa) contribute more early in training. `T_curriculum` anneals from 50.0 to 1.0 over warmup (cosine schedule).

### 5.8 MultiTask vMF-NCE (used by N3, N4)

```
L = L_vmf_nce(mu_fused, kappa_consensus, z_GT) + lambda_aux * (1/R) * sum_r L_vmf_nce(mu_r, kappa_r, z_GT)
```

Fused term trains the consensus direction; auxiliary term trains each ROI expert independently. `lambda_aux`: 0.5 (N3), 0.3 (N4).

### 5.9 MixCo Augmentation (used by B0, B1, N1 in v4)

```
lam ~ Beta(alpha, alpha)    # alpha=0.2
perm = randperm(B)
fmri_mix = lam * fmri + (1-lam) * fmri[perm]
clip_mix = lam * clip + (1-lam) * clip[perm]
soft_labels[i,i] = lam[i], soft_labels[i,perm[i]] += (1-lam[i])

logits = normalize(model(fmri_mix)) @ normalize(clip_mix).T / 0.006
loss = -(soft_labels * log_softmax(logits)).sum(-1).mean()
loss = (loss_fwd + loss_rev) / 2
```

MixCo requires a **second forward pass** on mixed fMRI. Weight: 1.0. Inspired by MindEye.

**Phase schedule (v5)**: When SoftCLIP is also enabled, MixCo runs for the first 1/3 of epochs (warmup augmentation), then SoftCLIP takes over for the remaining 2/3 (knowledge distillation). MixCo is disabled during the SoftCLIP phase.

### 5.10 SoftCLIP Knowledge Distillation (used by all v4 experiments)

```
teacher_logits = clip_embs @ all_keys.T / tau
teacher = softmax(teacher_logits)        # CLIP-CLIP similarity distribution

student_logits = pred_embs @ all_keys.T / tau
student = log_softmax(student_logits)    # predicted-CLIP similarity distribution

loss = -sum(teacher * student, dim=-1).mean()   # KL(teacher || student)
```

- `tau = 0.07` (matches contrastive temperature; N4 uses 0.05)
- Symmetric: loss computed in both directions and averaged
- Memory queue: 16384 entries concatenated to `all_keys` for a larger contrastive pool
- **Rationale**: Standard InfoNCE treats all negatives as equally wrong; SoftCLIP preserves graded semantic similarity (a dog is more similar to a cat than to a building)

---

## 6. Training Configuration (v4 -- Current)

### 6.1 Shared Settings (all v4 experiments)

| Parameter | Value |
|-----------|-------|
| Batch size | 64 |
| Mixed precision (AMP) | True (float16 forward, float32 loss) |
| Optimizer | AdamW |
| LR scheduler | Cosine decay with warmup |
| Min LR | 1e-6 |
| Gradient clipping | 1.0 (max norm) |
| Max epochs | 300 |
| Dropout | 0.15 |
| Weight decay | 0.01 |
| Memory queue | 16384 |
| fMRI z-scoring | True (**per-session**, train stats within each session) |
| Repetition averaging | **False** (individual trials, ~24k samples) |
| Embedding preprocessing (PCR) | **Disabled** |
| MixCo | Enabled for first 1/3 epochs (alpha=0.2, tau=0.07, weight=1.0) |
| SoftCLIP | Enabled for remaining 2/3 epochs (tau=0.07, symmetric, queue) |
| Split | Image-level, 90/10, shared1000 excluded |
| Early stopping | On R@1, patience varies |
| Seed | 42 |

### 6.2 Per-Experiment Hyperparameters

| Param | B0v4 | B1v4 | N1v4 | N2v4 | N3v4 | N4v4 |
|-------|------|------|------|------|------|------|
| Model type | deterministic | gaussian | vmf | vmf | vmf_dcf | vmf_dcf |
| Encoder | MLP [8192,4096,2048] | MLP [8192,4096,2048] | MLP [8192,4096,2048] | ROI Transformer d=512 | ROI Transformer d=512 | ROI Transformer d=512 |
| Decoder hidden | [2048] | [2048] | [2048] | [1024] | (direct) | (direct) |
| LR | 1e-4 | 5e-5 | 1e-4 | 5e-5 | 1e-4 | 2e-4 |
| Warmup epochs | 10 | 10 | 10 | 10 | 15 | 15 |
| Grad accum | 1 | 1 | 1 | 1 | 1 | 2 (eff. batch 128) |
| Patience | 40 | 40 | 40 | 40 | 40 | 60 |
| Betas | (0.9, 0.999) | (0.9, 0.999) | (0.9, 0.999) | (0.9, 0.95) | (0.9, 0.95) | (0.9, 0.95) |
| Kappa range | -- | -- | [1, 50] | log [0, 4] | [1, 100] | [1, 100] |
| MixCo alpha | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.15 |

### 6.3 Loss Configuration per Experiment

| Loss | B0v4 | B1v4 | N1v4 | N2v4 | N3v4 | N4v4 |
|------|------|------|------|------|------|------|
| MSE (w=0.1) | Y | Y | -- | -- | -- | -- |
| InfoNCE (tau=0.07, learnable) | Y | -- | -- | -- | -- | -- |
| Gaussian NLL | -- | Y (w=1) | -- | -- | -- | -- |
| Gaussian NCE (tau=1.0) | -- | Y (w=2) | -- | -- | -- | -- |
| KL (annealed 0->0.001) | -- | Y | -- | -- | -- | -- |
| vMF-NCE (tau=0.07) | -- | -- | Y | Y | Y | -- |
| vMF-NCE-SPCL (tau=0.05) | -- | -- | -- | -- | -- | Y |
| MultiTask (lambda_aux) | -- | -- | -- | -- | Y (0.5) | Y (0.3) |
| Kappa reg (lambda) | -- | -- | Y (0.05) | Y (0.1) | Y (0.05) | Y (0.02) |
| SoftCLIP (tau=0.07) | Y | Y | Y | Y | Y | Y (tau=0.05) |
| MixCo (w=1.0, first 1/3 epochs) | Y | Y | Y | Y | Y | Y |

---

## 7. Experimental History and Results

### 7.1 Version Progression

| Version | Key Changes | Rationale |
|---------|-------------|-----------|
| **v1** | batch=4, grad_accum=16, 100 epochs, patience=15, dropout=0.1, queue=8192 | Initial implementation |
| **v2** | dropout=0.3, R@1 early stopping, tuned LR/WD per experiment | Address overfitting, track retrieval |
| **v3** | batch=64, residual MLP, image-level split, exclude_shared1000, 300 epochs, patience=40, queue=16384, PCR k=4 | Fix gradient signal, data leakage, underfitting |
| **v4** | Per-voxel z-scoring, disable PCR, repetition averaging, MixCo augmentation, wider MLP [8192,4096,2048] | Fix fundamental data pipeline issues, add augmentation |
| **v5** | Individual trials (no rep-avg), per-session z-scoring, SoftCLIP knowledge distillation, MixCo->SoftCLIP phase schedule | Deep research analysis: 3x more data, remove session drift, preserve semantic similarity structure |

### 7.2 v1 Results (subj01, 4 subjects total)

R@1 was not tracked. All experiments completed. Val loss only:

| Experiment | Best Val Loss | Best Epoch | Total Epochs |
|------------|--------------|------------|--------------|
| B0 | 1.408 | 2 | 17 |
| B1 | 1.018 | 1 | 16 |
| N1 | 1.366 | 7 | 22 |
| N2 | -171.3 | 5 | 25 |
| N3 | 3.508 | 2 | 22 |
| N4 | 3.495 | 2 | 22 |

Observations: Models converge in very few epochs then early-stop. N2 has deeply negative loss (vMF-NLL dominated). N3/N4 losses are not comparable across experiments.

### 7.3 v2 Results (subj01)

First version tracking R@1. Chance level = 1/3000 = 0.033% (larger val set before image-level split):

| Experiment | Best R@1 | Best Epoch | Total Epochs | Wall Time |
|------------|----------|------------|--------------|-----------|
| B0v2 | 0.13% | 9 | 24 | 16 min |
| B1v2 | 0.23% | 15 | 30 | 30 min |
| N1v2 | 0.57% | 28 | 43 | 36 min |
| N2v2 | 0.30% | 14 | 34 | 74 min |
| N3v2 | 0.33% | 49 | 69 | 343 min |

All models are within 4-17x of random chance. N1v2 is best at 17x chance but still <1%.

### 7.4 v3 Results (subj01)

Chance level = 1/~2950 = 0.034% (image-level split, shared1000 excluded):

| Experiment | Best R@1 | Best Epoch | Observation |
|------------|----------|------------|-------------|
| B0v3 | ~0.07% | 40-46 | Overfitting: train loss 11->2.6, val loss plateauing ~5.1 |
| B1v3 | ~0.10% | 15 | Val loss diverging after epoch ~10 |
| N1v3 | ~0.10% | 39 | Kappa growing 1->5.5 over 57 epochs, overfitting |
| N2v3 | ~0.10% | 39 | Slow training, kappa growing to ~4 |
| N3v3 | ~0.10% | 31 | Complex model, visible overfitting |
| N4v3 | ~0.10-0.13% | 43-44 | SPCL curriculum appears to help slightly |

All models still near chance level despite batch=64, residual MLP, image-level split, 300 epochs.

### 7.5 v4 Results and Diagnostic

Initial v4 runs showed near-chance R@1 across all experiments. Diagnostic verification on JupyterHub confirmed data integrity:

| Check | Result |
|-------|--------|
| CLIP embedding dim | **768** (ViT-L/14) -- correct |
| CLIP cache coverage | 9999/10000 nsdIds (1 missing) |
| Embedding L2 norm | 1.0000 (properly normalized) |
| Mean pairwise cosine | 0.5564 (expected shared CLIP subspace) |
| fMRI raw stats | mean=382.81, std=789.09 (confirms z-scoring essential) |

The diagnostic rules out a wrong-model CLIP cache as the cause. Three architectural/preprocessing changes were then applied (v5 fixes):

1. **Individual trials** (`average_repetitions: false`) -- 3x more training samples
2. **Per-session z-scoring** (`zscore_mode: per_session`) -- removes session drift
3. **SoftCLIP loss** -- soft targets from CLIP-CLIP similarity replace hard one-hot InfoNCE

Results with v5 fixes: **pending experiment re-run.**

---

## 8. Reconstruction Pipeline (Post-Training)

```
Trained checkpoint
  -> load model weights
  -> load test fMRI features (pre-extracted .npy)
  -> apply z-scoring (using saved voxel_mean, voxel_std)
  -> model.predict(X_test) -> predicted CLIP embedding (768-D)
  -> (optional) UA-CFG: kappa -> guidance_scale, steps per sample
  -> Stable Diffusion 2.1: embed predicted CLIP into prompt conditioning
  -> generate 768x768 image
  -> evaluate: CLIPScore, R@K, PixCorr, SSIM, AlexNet features
```

UA-CFG (Uncertainty-Aware CFG) scales diffusion guidance by model confidence:
- High kappa -> high confidence -> higher guidance scale (w_max=12)
- Low kappa -> low confidence -> lower guidance scale (w_min=1.5)
- ROI disagreement (delta) further reduces confidence

---

## 9. Novel Contributions (N-series)

| Component | What's New | Why It Matters |
|-----------|-----------|----------------|
| **N1: vMF-NCE** | von Mises-Fisher distribution on S^{d-1} instead of Gaussian in R^d. Bessel-free NCE loss. Bounded-sigmoid kappa parameterization. | Geometrically correct for L2-normalized CLIP embeddings. Kappa provides calibrated uncertainty. |
| **N2: ROI Transformer** | Brain-topology-aware tokenization. Each of 17 ROIs gets its own projection + token, processed by a Transformer. | Learns which brain regions matter for each image. Interpretable attention weights. |
| **N3: ROI-DCF** | Per-ROI vMF expert heads fused via spherical consensus (Mardia & Jupp, 2000). | Uncertainty decomposes: kappa_consensus reflects agreement, delta measures ROI disagreement. |
| **N4: Full System** | SPCL curriculum on kappa + decomposed UA-CFG for diffusion + mixture sampling. | Automatic curriculum: easy samples first. Uncertainty-aware image generation. |

---

## 10. Environment

| Resource | Value |
|----------|-------|
| GPU | NVIDIA A100-SXM4-40GB |
| RAM | ~100 GB |
| CPU | 32 cores |
| Storage | PVC ~877 GB (NFS, `/home/jovyan/work`) |
| Python | 3.13 (conda base) |
| PyTorch | >= 2.0 |
| CLIP | open_clip_torch ViT-L/14 (openai weights) |
| Diffusion | SD 2.1 via diffusers (`sd2-community/stable-diffusion-2-1`) |

Key paths on JupyterHub:
- Repo: `/home/jovyan/work/FMRI2images/` (branch: `dirbrain-vmf-uacfg`)
- NSD data: `/home/jovyan/work/data/nsd/`
- Pre-extracted features: `cache/preextracted/subject={subj}/fmri_features.npy`
- CLIP cache: `outputs/clip_cache/clip.parquet`
- Results: `experimental_results/{experiment}/{subject}/`

---

## 11. Key Observations and Open Questions

### 11.1 The Core Problem

Across v1, v2, v3, and early v4 results, **all models achieve R@1 barely above chance** (0.03-0.6% on galleries where chance is 0.03-0.10%). MindEye achieves 93%+ R@1 on the same data with a simpler architecture (MLP + InfoNCE + MSE). Something fundamental is preventing the model from learning the fMRI -> CLIP mapping.

### 11.2 What We've Ruled Out

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| Batch too small for contrastive learning | Addressed in v3 | Batch 64 with queue 16384 -- no meaningful improvement |
| Data leakage inflating metrics | Addressed in v3 | Image-level split, shared1000 excluded -- results actually got worse |
| Not enough training time | Addressed in v3 | 300 epochs, patience 40 -- models overfit well before that |
| No residual connections | Addressed in v3 | ResidualBlocks added -- no meaningful improvement |
| PCR destroying CLIP signal | Addressed in v4 | PCR disabled entirely -- results pending |
| fMRI not standardized | Addressed in v4 | Per-voxel z-scoring added -- results pending |
| Not averaging repetitions | Addressed in v4 | Repetition averaging (SNR x1.73) -- results pending |
| Model too small | Addressed in v4 | Wider MLP [8192, 4096, 2048] = 328M params -- results pending |
| No augmentation | Addressed in v4 | MixCo added -- results pending |
| Wrong CLIP cache dim | **Verified** | Diagnostic confirms 768-D ViT-L/14, norm=1.0, 9999/10000 coverage |
| Global z-scoring destroys session signal | Addressed in v5 | Per-session z-scoring preserves within-session variance |
| Repetition averaging reduces data | Addressed in v5 | Individual trials: ~24k samples instead of ~8k |
| Hard InfoNCE ignores semantic structure | Addressed in v5 | SoftCLIP uses CLIP-CLIP similarity as soft targets |

### 11.3 Hypotheses Still Open

1. **CLIP embedding quality**: Are the cached 768-D ViT-L/14 embeddings correct? Is L2-normalization applied consistently between cache and training?

2. **Embedding-target alignment**: When the model predicts a 768-D vector and we compare it to the CLIP target, are both in the same space? Could there be a preprocessing mismatch?

3. **Loss function scaling**: InfoNCE with learnable temperature -- is the temperature learning correctly? Could it be collapsing to a degenerate value?

4. **Gradient signal through the pipeline**: With AMP (float16), are gradients flowing properly through the 328M-param MLP? Could there be vanishing/exploding gradients?

5. **Pre-extraction correctness**: Are the pre-extracted features correctly aligned with the CLIP cache? Does trial N in fmri_features.npy correspond to the correct nsdId in clip.parquet?

6. **vMF-specific**: For N1-N4, the kappa values in v3 were very low (1-5). This means the model is expressing near-zero confidence. Is kappa regularization too aggressive?

7. **ROI Transformer capacity**: The ROI Transformer has only 512-D tokens and 4 layers. Is this sufficient to learn cross-region interactions?

8. **What MindEye does differently**: MindEye uses (1) a much larger effective batch through aggressive memory bank, (2) soft contrastive loss (not hard InfoNCE) -- **now addressed by SoftCLIP in v5**, (3) multi-subject pre-training, (4) per-session preprocessing -- **now addressed by per-session z-scoring in v5**. Remaining gap: multi-subject pre-training and larger effective batch.

### 11.4 Key Comparisons to MindEye Architecture

| Aspect | Our Approach (B0v4) | MindEye |
|--------|-------------------|---------|
| Encoder | MLP [8192, 4096, 2048] + ResidualBlocks | MLP with residual blocks (similar) |
| CLIP target | ViT-L/14 768-D | ViT-L/14 768-D |
| Loss | MSE + InfoNCE + **SoftCLIP** (v5) | MSE + soft contrastive (bidirectional) |
| Augmentation | MixCo warmup (1/3 epochs) -> SoftCLIP (2/3 epochs) | MixCo (similar) |
| Batch size | 64 | ~1000 (effective, with large memory bank) |
| Preprocessing | **Per-session** z-scoring (v5) | Per-voxel z-scoring |
| Multi-subject | No (single-subject) | Pre-training on all subjects |
| Training data | ~24k trials (~8881 unique images, individual reps) | ~8859 unique images |
| Queue | 16384 | Not used (large batch instead) |
| Contrastive temp | 0.07 (learnable for InfoNCE; fixed for SoftCLIP) | Different formulation |

### 11.5 Specific Numbers to Investigate

- MindEye reports R@1 > 90% on shared1000 (982 images, paired retrieval). Our val set is ~986 images. The gallery sizes are comparable.
- MindEye's MLP is ~996M params for subj01 (vs our 328M). Could model capacity be the issue?
- MindEye uses a "soft contrastive loss" with continuous similarity labels, not hard InfoNCE. **v5 addresses this with SoftCLIP knowledge distillation.**
- MindEye pre-trains on 7 subjects then fine-tunes on the target subject. We train single-subject only.

---

## 12. File Reference

| Component | File |
|-----------|------|
| Training script | `scripts/training/train_unified.py` |
| UnifiedModel | `src/fmri2img/models/unified_model.py` |
| ROI Transformer | `src/fmri2img/models/roi_transformer.py` |
| vMF Decoder | `src/fmri2img/models/vmf_decoder.py` |
| ROI-DCF Decoder | `src/fmri2img/models/roi_dcf.py` |
| InfoNCE loss | `src/fmri2img/losses/infonce_queue.py` |
| Gaussian NCE loss | `src/fmri2img/losses/gaussian_nce.py` |
| vMF-NCE/SPCL/MultiTask losses | `src/fmri2img/losses/vmf_nce.py` |
| MixCo augmentation | `src/fmri2img/losses/mixco.py` |
| SoftCLIP loss | `src/fmri2img/losses/softclip.py` |
| Memory queue | `src/fmri2img/contrastive/queue.py` |
| ROI index builder | `src/fmri2img/data/roi_utils.py` |
| fMRI pre-extraction | `scripts/build/preextract_fmri.py` |
| CLIP cache builder | `scripts/build/build_clip_cache.py` |
| Embedding preprocessor | `src/fmri2img/embedding_preproc.py` |
| Reconstruction | `scripts/reconstruction/decode_diffusion.py` |
| Evaluation | `scripts/evaluation/eval_reconstruction.py` |
| Aggregation | `scripts/evaluation/aggregate_ablation.py` |
| Ablation orchestration | `scripts/training/run_ablation_ladder.sh` |
| v4 configs | `configs/experiments/B0v4_deterministic.yaml` through `N4v4_full_system.yaml` |
| Base config | `configs/base.yaml` |
| CLIP config | `configs/system/clip.yaml` |
| Environment reference | `.cursor/rules/jupyterhub-environment.mdc` |
