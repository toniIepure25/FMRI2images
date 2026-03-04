# Experiment Context: fMRI-to-Image Neural Decoding

This document provides complete technical context for a bachelor thesis project on neural decoding of visual perception from fMRI. It is designed as a self-contained briefing for an LLM or researcher performing deep analysis.

**Project status (March 2026):** After resolving critical bugs (nsdId off-by-one, kappa collapse) and six iterative architecture versions (v4-v9), the system achieves **~50% R@1** on subj01 with v8 configs (N3v8: 50.3%, N4v8: 50.8%). The v9 iteration introduces anti-overfitting regularization (Stochastic Depth, R-Drop, label smoothing), contrastive projection head separation, CSLS retrieval correction, and MC-Dropout test-time augmentation -- targeting **65-70% R@1**. All v9 experiments run on an **NVIDIA H100 80GB HBM3** with bf16 mixed precision and effective batch size 1024.

**SOTA target:** MindEye achieves 93.2% R@1 on the same dataset. The remaining gap is attributed to (1) single-subject vs 7-subject pre-training, (2) smaller model capacity (328M vs 996M params), and (3) MindEye's OpenCLIP ViT-bigG/14 embeddings (256x1664-D) vs our ViT-L/14 (768-D).

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
     -> kappa_head: Linear(2048, 1) -> activation
  -> output: mu (B, 768) on S^{767}, kappa (B, 1)
```

**Kappa parameterization history:**
- **v4-v6**: Bounded sigmoid: `kappa = kappa_min + (kappa_max - kappa_min) * sigmoid(raw)`, range [1, 50]. Suffered gradient saturation at boundaries.
- **v7+**: Softplus: `kappa = softplus(raw) + 1.0`. Unbounded, no gradient saturation, with explicit `kappa_reg` to prevent explosion.

### 4.5 N2: ROI-Tokenized Transformer

**v5 (single-subject):**
```
fMRI (B, 15724)
  -> ROITransformerEncoder:
     -> 17 ROI projections: per-ROI Linear(n_voxels_r, d_model) -> LayerNorm -> GELU -> Dropout
     -> prepend learnable [CLS] token + positional embeddings
     -> Transformer (d_model=768, nhead=12, 6 layers, dim_ff=3072)
     -> output: [CLS] token (B, 768)
  -> VonMisesFisherDecoder: mu + kappa (softplus in v7+)
```

**v7+ (multi-subject):**
```
fMRI (B, ~15724)
  -> MultiSubjectROITransformer:
     -> per-subject ROI projections (nn.ModuleDict)
     -> subject embedding added to tokens
     -> shared Transformer backbone (d=768, 12 heads, 6 layers)
     -> [CLS] token (B, 768)
```

**v9 additions:** Stochastic Depth (`drop_path_rate=0.15`), ContrastiveProjectionHead.

Each ROI gets its own projection layer. Voxel counts per ROI vary by subject (e.g., V1v may have 700 voxels for subj01 but differ for subj02). In multi-subject mode, each subject has separate `ROIProjection` modules while sharing the Transformer backbone.

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
- **v4**: `tau = 0.07` -- this caused **kappa collapse** (see below)
- **v5**: `tau = 1.0` with `learnable_temperature: true` -- kappa IS the inverse temperature by definition; a separate tau is redundant and destructive
- Logits clamped to [-80, 80] for float16 stability (with tau=1.0 and kappa_max=50, max logit is 50 -- clamp never triggers)
- Memory queue: 16384 entries

**Kappa collapse (v4 bug, fixed in v5):** With `tau=0.07`, effective logit scaling was `kappa / 0.07 = kappa * 14.28`. For a positive pair with cos_sim near 1.0, any `kappa > 5.6` produced a logit above 80 that was clamped, zeroing the gradient w.r.t. kappa. Combined with the kappa regularizer applying constant downward pressure, kappa was mathematically trapped at ~3-5 in 768-D space -- a near-uniform distribution with no useful confidence signal. Setting `tau=1.0` restores full gradient flow up to `kappa_max`.

### 5.6 Kappa Regularizer (used by N1-N4 in v4; disabled in v5; re-enabled in v7+)

```
L_reg = lambda_kappa * mean(kappa)
```

**v4**: Enabled. `lambda_kappa` varies: 0.05 (N1), 0.1 (N2), 0.05 (N3), 0.02 (N4).

**v5-v6**: **Disabled**. With `tau=1.0` and bounded sigmoid, the softmax denominator naturally constrains kappa.

**v7+**: **Re-enabled** at `lambda_kappa=0.01` (v7) then `0.1` (v9). With softplus kappa (unbounded), explicit regularization is needed to prevent kappa explosion. The stronger `0.1` weight in v9 also acts as anti-overfitting regularization.

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

## 6. Training Configuration

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

### 6.4 v5 N-Series Configs (Current Active)

The ablation ladder now runs **B0v4, B1v4, N1v5, N2v5, N3v5, N4v5**. B-series remain at v4. N-series upgraded to v5 to fix the kappa collapse and add training improvements.

**Key v5 changes from v4 (N-series only):**

| Parameter | v4 | v5 |
|-----------|----|----|
| vMF-NCE tau | 0.07 (N1-N3) / 0.05 (N4) | **1.0** (all) |
| Kappa regularizer | Enabled | **Disabled** |
| Learnable temperature | N/A | **True** (logit_scale starts at 0 with tau=1.0) |
| Gradient accumulation | 1 (N1-N3) / 2 (N4) | **4** (effective batch 256) |
| fMRI noise augmentation | None | **0.1** (Gaussian noise std) |
| Voxel dropout | None | **0.1** (random voxel masking) |
| EMA | None | **Enabled** (decay=0.999) |
| SoftCLIP schedule | After 1/3 epochs | **From start** (simultaneous with MixCo) |
| MixCo weight | 1.0 | **0.5** (reduced since running alongside SoftCLIP) |
| Max epochs | 300 | **150** (models peak early, 300 wastes compute) |
| Patience | 40-60 | **25** |
| Dropout | 0.15 | **0.2** |
| Weight decay | 0.01 | **0.05** |
| N2 Transformer | d_model=512, 4 layers, 8 heads | **d_model=768, 6 layers, 12 heads** |

**v5 loss configuration (N-series):**

| Loss | N1v5 | N2v5 | N3v5 | N4v5 |
|------|------|------|------|------|
| vMF-NCE (tau=1.0, learnable) | Y | Y | Y | -- |
| vMF-NCE-SPCL (tau=1.0) | -- | -- | -- | Y |
| MultiTask (lambda_aux) | -- | -- | Y (0.5) | Y (0.3) |
| Kappa reg | **Disabled** | **Disabled** | **Disabled** | **Disabled** |
| SoftCLIP (tau=0.05) | Y (from start) | Y (from start) | Y (from start) | Y (from start) |
| MixCo (w=0.5) | Y | Y | Y | Y |

### 6.5 v6 N-Series Configs (Novel Losses)

**Key v6 changes from v5:** Delta-SPCL loss (kappa+delta curriculum), vMF-SoftCLIP (kappa-weighted KD on hypersphere), Slerp MixCo interpolation.

**Result:** N1v6 ~40% R@1 but kappa saturated at sigmoid ceiling. Kappa gradient saturation at bounded sigmoid boundaries identified as the next bottleneck.

### 6.6 v7 N-Series Configs (Softplus Kappa + Multi-Subject + Multi-Layer CLIP)

**Key v7 changes from v6:**
- `kappa_mode: "softplus"` -- unbounded kappa via `softplus(raw) + 1.0`, resolving sigmoid saturation
- `kappa_reg` re-enabled at `lambda_kappa=0.01` to prevent kappa explosion
- `embedding_column: "fused"` -- blended intermediate+final CLIP features from multi-layer cache
- N2/N3/N4: `encoder_type: "multi_subject_roi_transformer"` with `data.subjects: [subj01, subj02, subj05, subj07]` (4x data)
- N2/N3/N4: d_model=768, 12 heads, 6 layers, dim_ff=3072

**Result:** N1v7 achieved **49% R@1** (best single-subject result at the time).

### 6.7 v8 N-Series Configs (Hierarchical CLIP + CKA)

**Key v8 changes from v7 (N3/N4 only):**
- `hierarchical_clip` loss (weight 0.3): supervises per-ROI experts with different CLIP layer features (early, mid, high)
- `cka` loss (weight 0.5): global representational similarity (1-CKA) computed per-subject
- `vmf_nce` with `use_arctanh`, `margin_base=0.2`, `margin_kappa_ref=50.0`

**Result:** N3v8: **50.3% R@1**, N4v8: **50.8% R@1**. Clear overfitting observed after epoch 50-55.

### 6.8 v9 N-Series Configs (Anti-Overfit + Projection Head + CSLS + TTA)

**Key v9 changes (all 4 N-series):**

| Feature | Description |
|---------|-------------|
| Stochastic Depth (DropPath) | `drop_path_rate=0.15` in Transformer layers (N2-N4) |
| ContrastiveProjectionHead | 768->2048->768 MLP separates contrastive from retrieval representation |
| R-Drop | Symmetric KL between two vMF forward passes with different dropout masks (weight=0.5) |
| Label smoothing | 0.1 in vMF-NCE targets |
| Stronger kappa_reg | `lambda_kappa=0.1` (10x vs v7) |
| CSLS retrieval | Cross-domain Similarity Local Scaling corrects hubness in retrieval eval |
| MC-Dropout TTA | 8 forward passes with dropout at test time, average predictions |
| Kappa-weighted avg | Weight repetition averaging by model confidence (kappa) |
| Loss surgery | SoftCLIP weight 0.3; hierarchical_clip, CKA, multitask disabled |
| H100 optimization | batch_size=128, bf16 mixed precision, grad_accum=8 (effective batch 1024) |

**v9 loss configuration:**

| Loss | N1v9 | N2v9 | N3v9 | N4v9 |
|------|------|------|------|------|
| vMF-NCE (label_smooth=0.1) | Y | Y | Y | -- |
| vMF-NCE-SPCL | -- | -- | -- | Y |
| SoftCLIP (w=0.3) | Y | Y | Y | Y |
| Kappa reg (0.1) | Y | Y | Y | Y |
| R-Drop (w=0.5) | Y | Y | Y | Y |
| MixCo (w=0.5) | Y | Y | Y | Y |

---

## 7. Experimental History and Results

### 7.1 Version Progression

| Version | Key Changes | R@1 (subj01) |
|---------|-------------|--------------|
| **v1** | batch=4, grad_accum=16, 100 epochs, patience=15 | Near-chance (nsdId bug) |
| **v2** | dropout=0.3, R@1 early stopping, tuned LR | Near-chance (nsdId bug) |
| **v3** | batch=64, residual MLP, image-level split, 300 epochs | Near-chance (nsdId bug) |
| **v4** | nsdId fix, per-session z-score, MixCo, SoftCLIP, wider MLP | B: 20-25%, N: 30-36% |
| **v5** | tau=1.0, kappa_reg off, EMA, noise aug, grad_accum=4 | N1: 39.9%, N2: 35.9%, N3: 36.4%, N4: 37.4% |
| **v6** | Delta-SPCL, vMF-SoftCLIP, Slerp MixCo | N1: ~40% (kappa saturation) |
| **v7** | Softplus kappa, multi-layer CLIP, multi-subject training | N1: **49%** |
| **v8** | Hierarchical CLIP alignment, CKA loss (N3/N4 only) | N3: **50.3%**, N4: **50.8%** |
| **v9** | DropPath, projection head, R-Drop, CSLS, MC-TTA, bf16, batch 1024 | Pending |

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

### 7.5 v4 Results (subj01, after nsdId fix)

After fixing the nsdId off-by-one bug (Section 11.1), v4 experiments showed meaningful learning for the first time. Approximate R@1 on validation gallery (~986 images):

| Experiment | Approx. R@1 | Observation |
|------------|-------------|-------------|
| B0v4 | ~20-25% | Deterministic baseline, MLP + InfoNCE + MSE + SoftCLIP |
| B1v4 | ~20-25% | Gaussian probabilistic, comparable to B0 |
| N1v4 | ~30-36% | vMF-NCE, kappa stagnating at 3-5 (kappa collapse) |
| N2v4 | ~30-36% | ROI Transformer + vMF-NCE, similar kappa stagnation |
| N3v4 | ~30-36% | ROI-DCF, similar plateau |
| N4v4 | ~30-36% | Full system, similar plateau |

Diagnostic verification on JupyterHub confirmed data integrity after the fix:

| Check | Result |
|-------|--------|
| CLIP embedding dim | **768** (ViT-L/14) -- correct |
| CLIP cache coverage | 9999/10000 nsdIds (1 missing) |
| Embedding L2 norm | 1.0000 (properly normalized) |
| Mean pairwise cosine | 0.5564 (expected shared CLIP subspace) |
| fMRI raw stats | mean=382.81, std=789.09 (confirms z-scoring essential) |

**Key observation:** N-series models (30-36%) outperform B-series (20-25%), confirming that vMF + ROI structure provides a real advantage. However, N1-N4 plateau at a similar level despite increasing architectural complexity, strongly suggesting a shared bottleneck. This was identified as the **kappa collapse** bug (Section 13): all N-series models had their kappa gradient zeroed by the tau=0.07 / clamp interaction.

### 7.6 v5 Results (subj01)

v5 fixed kappa collapse (tau=1.0, kappa_reg disabled) and added EMA, noise aug, grad_accum=4:

| Experiment | R@1 | Observation |
|------------|-----|-------------|
| N1v5 | 39.9% | MLP vMF-NCE, kappa now actively learning (mean ~23, range 19-27) |
| N2v5 | 35.9% | ROI Transformer, underfitting vs N1 (data-starved) |
| N3v5 | 36.4% | ROI-DCF, similar to N2 |
| N4v5 | 37.4% | Full system with SPCL |

### 7.7 v6 Results (subj01)

v6 introduced novel losses (Delta-SPCL, vMF-SoftCLIP, Slerp MixCo):

| Experiment | R@1 | Observation |
|------------|-----|-------------|
| N1v6 | ~40% | Modest gain; kappa saturated at bounded sigmoid ceiling |
| N2-N4v6 | 35-38% | Kappa saturation limited all N-series uniformly |

**Diagnosis:** Bounded sigmoid kappa parameterization (`kappa_min + (kappa_max - kappa_min) * sigmoid(raw)`) had vanishing gradients near the boundaries, preventing kappa from expressing strong confidence.

### 7.8 v7 Results (subj01)

v7 introduced softplus kappa, multi-layer CLIP targets, and multi-subject training:

| Experiment | R@1 | Observation |
|------------|-----|-------------|
| N1v7 | **49%** | Best single-experiment result; softplus kappa healthy (mean ~23, good variance) |
| N2v7 | -- | Multi-subject training wiring bugs (fixed mid-v7) |
| N3v7 | -- | Multi-subject CLIP cache bugs (fixed mid-v7) |
| N4v7 | -- | Multi-subject CLIP cache bugs (fixed mid-v7) |

### 7.9 v8 Results (subj01, N3/N4 only)

v8 added hierarchical CLIP alignment and CKA loss:

| Experiment | R@1 | Observation |
|------------|-----|-------------|
| N3v8 | **50.3%** | Clear overfitting after epoch ~55 (train loss still decreasing, val R@1 declining) |
| N4v8 | **50.8%** | Same overfitting pattern; SPCL provided slight edge |

**Diagnosis:** The model memorizes training data (27k trials, 4 subjects) but fails to generalize. Key missing elements: (1) no separation between contrastive and retrieval representations, (2) insufficient regularization for the Transformer, (3) overly complex loss landscape (hierarchical + CKA + multitask).

### 7.10 v9 Results (pending)

v9 targets 65-70% R@1 with comprehensive anti-overfitting interventions, projection head separation, CSLS retrieval correction, and test-time ensembling. All experiments run on H100 80GB with bf16 and effective batch 1024.

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
| GPU | **NVIDIA H100 80GB HBM3** (single, `CUDA_VISIBLE_DEVICES=0`) |
| Driver | 570.211.01, CUDA 12.8 |
| RAM | ~100 GB |
| CPU | 32 cores |
| Storage | PVC ~877 GB (NFS, `/home/jovyan/work`) |
| Python | 3.13 (conda base) |
| PyTorch | >= 2.0 |
| Mixed precision | **bf16** (H100 native; same exponent range as fp32, no GradScaler needed) |
| CLIP | open_clip_torch ViT-L/14 (openai weights) |
| Diffusion | SD 2.1 via diffusers (`sd2-community/stable-diffusion-2-1`) |

**Note:** Previous experiments (v4-v8) ran on an NVIDIA A100-SXM4-40GB. The H100 upgrade (v9+) provides 2x VRAM (80GB vs 40GB), enabling batch_size=128 (vs 64) and native bf16 support which eliminates AMP overflow risks that previously affected kappa gradients.

Key paths on JupyterHub:
- Repo: `/home/jovyan/work/FMRI2images/` (branch: `dirbrain-vmf-uacfg`)
- NSD data: `/home/jovyan/work/data/nsd/`
- Pre-extracted features: `cache/preextracted/subject={subj}/fmri_features.npy`
- CLIP cache: `outputs/clip_cache/clip.parquet`
- Results: `experimental_results/{experiment}/{subject}/`

---

## 11. Key Observations and Open Questions

### 11.1 Bug 1 — nsdId Off-by-One (ROOT CAUSE for near-chance performance)

**Root cause: off-by-one in `build_full_index.py`.** The NSD behavioral file `responses.tsv` uses `73KID` (1-indexed, 1-73000), but the index builder renamed it directly to `nsdId` without subtracting 1. Since `nsdId` is 0-indexed (0-72999) everywhere else (imgBrick, stim_info_merged.csv), every fMRI trial was paired with the CLIP embedding of the wrong image (+1 shift). Adjacent NSD images are unrelated COCO images, so this created effectively random fMRI-CLIP pairings.

**Evidence**: `pipeline_diagnostic.py` check 1.2 showed Oracle R@1 = 33.33% (not 100%); training crashed on `nsdId=73000` (out of range, max valid = 72999).

**Fix applied**: `behav_data['nsdId'] = behav_data['73KID'] - 1` in `build_full_index.py` + range validation. Same fix in `nsd_index_builder.py`. Index and CLIP cache rebuilt on JupyterHub.

**Impact**: Before the fix, all v1/v2/v3/early-v4 models achieved R@1 barely above chance (0.03-0.6%). After the fix, v4 models immediately showed meaningful learning: B-series reached 20-25% R@1, N-series reached 30-36% R@1 (see Section 7.5).

### 11.1b Bug 2 — Kappa Collapse (ROOT CAUSE for N-series plateau at 30-36%)

**Root cause: tau=0.07 in vMF-NCE capped effective kappa at ~5.6.** In all v4 N-series configs, the vMF-NCE loss computed logits as `kappa * cos_sim / tau`. With `tau=0.07`, this becomes `kappa * cos_sim * 14.28`. The AMP-safe clamp at `[-80, 80]` meant any `kappa > 80 * 0.07 / 1.0 = 5.6` produced clamped logits receiving zero gradient for kappa. Combined with `kappa_reg` actively penalizing high kappa, the concentration parameter was mathematically trapped at 3-5, preventing the model from expressing confidence.

**Evidence**: All N1-N4 v4 runs showed kappa values stagnating at 3-5 across all epochs. Despite different architectures (MLP, ROI Transformer, ROI-DCF), all N-series models plateaued at the same ~30-36% R@1.

**Fix applied (v5)**: `tau=1.0` in all v5 N-series configs + `kappa_reg.enabled: false`. This removes the kappa ceiling (effective range: 1.0 to `kappa_max=50.0`). See Section 13 for mathematical details.

### 11.2 What We've Ruled Out

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| Batch too small for contrastive learning | Addressed in v3 | Batch 64 with queue 16384 |
| Data leakage inflating metrics | Addressed in v3 | Image-level split, shared1000 excluded |
| Not enough training time | Addressed in v3 | 300 epochs, patience 40 -- models converge well before |
| No residual connections | Addressed in v3 | ResidualBlocks added |
| PCR destroying CLIP signal | Addressed in v4 | PCR disabled; v4 achieves 20-36% R@1 |
| fMRI not standardized | Addressed in v4 | Per-voxel z-scoring; v4 results confirm improvement |
| Not averaging repetitions | Addressed in v4 | Repetition averaging (SNR x1.73) used in v4 |
| Model too small | Addressed in v4 | Wider MLP [8192, 4096, 2048] = 328M params |
| No augmentation | Addressed in v4 | MixCo added; v4 results confirm improvement |
| Wrong CLIP cache dim | **Verified correct** | Diagnostic confirms 768-D ViT-L/14, norm=1.0, 9999/10000 coverage |
| Global z-scoring destroys session signal | Addressed in v4/v5 | Per-session z-scoring preserves within-session variance |
| Repetition averaging reduces data | Addressed in v5 | Individual trials: ~24k samples instead of ~8k |
| Hard InfoNCE ignores semantic structure | Addressed in v5 | SoftCLIP uses CLIP-CLIP similarity as soft targets |
| vMF kappa cannot learn (tau/clamp interaction) | **Fixed in v5** | tau=1.0, kappa_reg disabled (see Section 13) |

### 11.3 Hypotheses and Resolution Status

| # | Hypothesis | Status | Resolution |
|---|-----------|--------|------------|
| 1 | CLIP embedding quality wrong | **Verified correct** | Diagnostic confirms 768-D, L2-normed, ViT-L/14, 9999 coverage |
| 2 | Embedding-target alignment mismatch | **Resolved (nsdId fix)** | Off-by-one bug caused wrong pairings; after fix, models learn (20-36% R@1) |
| 3 | InfoNCE temperature collapsing | **Partially resolved** | Learnable temperature works for B-series; v5 adds learnable temp for N-series |
| 4 | Gradient signal under AMP | **Monitored** | v4 shows learning; no evidence of gradient pathology for B-series. N-series had kappa collapse (separate bug) |
| 5 | Pre-extracted features misaligned | **Resolved (nsdId fix)** | Index rebuilt; Oracle R@1 now 100% |
| 6 | vMF kappa trapped by tau/clamp | **RESOLVED in v5** | tau=1.0, kappa_reg disabled |
| 7 | ROI Transformer too small | **RESOLVED in v5** | d_model=768, 6 layers, 12 heads |
| 8 | Missing MindEye techniques | **Partially addressed** | Multi-subject (v7), SoftCLIP, per-session z-scoring; still missing functional alignment |
| 9 | Kappa sigmoid saturation | **RESOLVED in v7** | Softplus kappa + kappa_reg re-enabled |
| 10 | Overfitting (Transformer) | **Addressed in v9** | DropPath, R-Drop, label smoothing, projection head separation |
| 11 | Hubness in retrieval | **Addressed in v9** | CSLS post-hoc correction |

**Remaining open questions (for deep research):**

1. **Functional alignment**: MindEye2 uses subject-specific ridge regression to align brain data to a shared space before the main network. Our multi-subject approach uses per-subject ROI projections + subject embeddings, which is architecturally different.

2. **CLIP embedding dimensionality**: MindEye2 uses OpenCLIP ViT-bigG/14 (256 tokens x 1664-D) with a separate retrieval submodule. Our ViT-L/14 (768-D single vector) may lack the representational bandwidth for very high retrieval accuracy.

3. **Model capacity**: MindEye uses 996M parameters vs our ~328M. With v9's effective batch 1024 and anti-overfitting measures, capacity may now be a more relevant bottleneck.

4. **Diffusion prior**: MindEye2 trains an explicit diffusion prior to map fMRI embeddings to CLIP-conditioned image latents. Our pipeline goes directly from predicted CLIP embedding to SD 2.1.

### 11.4 Key Comparisons to MindEye Architecture

| Aspect | Our N1v9 (best MLP) | Our N4v9 (best system) | MindEye2 |
|--------|---------------------|------------------------|----------|
| Encoder | MLP [8192, 4096, 2048] + Residual | Multi-Subject ROI Transformer (d=768, 6L) | MLP + residual |
| CLIP target | ViT-L/14 768-D (fused multi-layer) | ViT-L/14 768-D (fused multi-layer) | OpenCLIP ViT-bigG/14 (256x1664-D) |
| Loss | vMF-NCE + SoftCLIP + R-Drop + kappa_reg | vMF-NCE-SPCL + SoftCLIP + R-Drop | MSE + soft contrastive |
| Regularization | Dropout 0.25, MixCo, label smoothing | DropPath 0.15, R-Drop, label smoothing | MixCo |
| Effective batch | 1024 (128 x 8) | 1024 (128 x 8) | ~1000 |
| Mixed precision | bf16 (H100) | bf16 (H100) | fp16 |
| Multi-subject | No | 4-subject joint training | 7-subject pre-train + fine-tune |
| Uncertainty | vMF kappa (softplus, unbounded) | kappa + delta (DCF disagreement) | None |
| Retrieval post-hoc | CSLS + MC-TTA + kappa-weighted avg | CSLS + MC-TTA + kappa-weighted avg | 300-candidate shortlist |
| Best R@1 | v7: 49% (v9 pending) | v8: 50.8% (v9 pending) | 93.0% |

### 11.5 Quantitative Gap Analysis (v8 actual vs SOTA)

| Metric | Our B0v4 | Our best (N4v8) | MindEye2 | Gap |
|--------|----------|-----------------|----------|-----|
| R@1 (val) | ~22% | 50.8% | 93.0% | ~42 pp |

**Progress from v4 to v8:** +28.8 pp (from 22% to 50.8%). Gains attributed to:
1. **Kappa collapse fix (v5)**: +7-14 pp (freed vMF concentration parameter)
2. **Softplus kappa (v7)**: +9 pp (removed sigmoid saturation)
3. **Multi-subject + multi-layer CLIP (v7-v8)**: +1-2 pp (more data, richer targets)
4. **Hierarchical alignment + CKA (v8)**: +1 pp (diminishing returns, overfitting)

**Primary contributors to remaining gap (ranked by estimated impact):**
1. **CLIP embedding gap (est. 15-20 pp)**: MindEye2 uses OpenCLIP ViT-bigG/14 (256 tokens x 1664-D) vs our ViT-L/14 (768-D single vector)
2. **Functional alignment (est. 10-15 pp)**: MindEye2 uses subject-specific ridge regression; our multi-subject approach uses learned projections
3. **Diffusion prior (est. 5-10 pp)**: MindEye2 trains a separate diffusion prior for embedding refinement
4. **Anti-overfitting gap (est. 5-10 pp)**: v9 interventions (DropPath, R-Drop, proj head, CSLS, TTA) target this directly

**Expected outcome after v9:** With comprehensive anti-overfitting and retrieval improvements, targeting 65-70% R@1. Further gains would require upgrading to OpenCLIP ViT-bigG or adding a diffusion prior.

---

## 12. File Reference

### Core Components

| Component | File |
|-----------|------|
| Training script | `scripts/training/train_unified.py` |
| UnifiedModel | `src/fmri2img/models/unified_model.py` |
| ROI Transformer | `src/fmri2img/models/roi_transformer.py` |
| Multi-Subject ROI Transformer | `src/fmri2img/models/multi_subject_encoder.py` |
| vMF Decoder | `src/fmri2img/models/vmf_decoder.py` |
| ROI-DCF Decoder | `src/fmri2img/models/roi_dcf.py` |
| Contrastive Projection Head | `src/fmri2img/models/projection_head.py` |
| Multi-Subject Dataset | `src/fmri2img/data/multi_subject_dataset.py` |

### Loss Functions

| Loss | File |
|------|------|
| InfoNCE | `src/fmri2img/losses/infonce_queue.py` |
| Gaussian NCE | `src/fmri2img/losses/gaussian_nce.py` |
| vMF-NCE/SPCL/MultiTask/R-Drop | `src/fmri2img/losses/vmf_nce.py` |
| MixCo augmentation | `src/fmri2img/losses/mixco.py` |
| SoftCLIP / vMF-SoftCLIP | `src/fmri2img/losses/softclip.py` |
| Hierarchical CLIP | `src/fmri2img/losses/hierarchical_clip_loss.py` |
| CKA | `src/fmri2img/losses/cka_loss.py` |

### Evaluation

| Component | File |
|-----------|------|
| Retrieval + CSLS | `src/fmri2img/eval/embedding_eval.py` |
| Aggregation | `scripts/evaluation/aggregate_ablation.py` |
| Reconstruction eval | `scripts/evaluation/eval_reconstruction.py` |

### Scripts and Configs

| Component | File |
|-----------|------|
| Ablation orchestration | `scripts/training/run_ablation_ladder.sh` |
| Multi-layer CLIP cache | `scripts/build/build_multilayer_clip_cache.py` |
| fMRI pre-extraction | `scripts/build/preextract_fmri.py` |
| Reconstruction | `scripts/reconstruction/decode_diffusion.py` |
| v9 N-series configs (current) | `configs/experiments/N1v9_vmf_nce.yaml` through `N4v9_full_system.yaml` |
| v8 N-series configs | `configs/experiments/N3v8_roi_dcf.yaml`, `N4v8_full_system.yaml` |
| v7 N-series configs | `configs/experiments/N1v7_vmf_nce.yaml` through `N4v7_full_system.yaml` |
| v4 B-series configs | `configs/experiments/B0v4_deterministic.yaml`, `B1v4_gaussian.yaml` |
| Environment reference | `.cursor/rules/jupyterhub-environment.mdc` |

---

## 13. Kappa Collapse: Mathematical Deep-Dive

This section provides a complete derivation of the kappa collapse bug for an LLM or researcher investigating the v4-to-v5 transition.

### 13.1 vMF-NCE Loss Formulation

The von Mises-Fisher NCE loss computes logits as:

$$\text{logit}_{ij} = \frac{\kappa_i \cdot \cos(\mu_i, z_j)}{\tau}$$

where \(\mu_i \in S^{d-1}\) is the predicted mean direction, \(\kappa_i > 0\) is the concentration (confidence), \(z_j\) is the CLIP target embedding, and \(\tau\) is a temperature hyperparameter.

The loss is then standard cross-entropy over these logits (positive pair at diagonal):

$$\mathcal{L}_{\text{vMF-NCE}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{logit}_{ii})}{\sum_{j=1}^{N} \exp(\text{logit}_{ij})}$$

### 13.2 The Clamping Constraint

For AMP (float16) stability, logits are clamped before the softmax:

$$\text{logit}_{ij}^{\text{clamped}} = \text{clamp}\left(\frac{\kappa_i \cdot \cos(\mu_i, z_j)}{\tau}, -80, 80\right)$$

The maximum possible logit (for a perfect prediction where \(\cos = 1.0\)) is:

$$\text{logit}_{\max} = \frac{\kappa_i}{\tau}$$

For this to remain below the clamp threshold:

$$\frac{\kappa_i}{\tau} \leq 80 \implies \kappa_i \leq 80 \cdot \tau$$

### 13.3 The Collapse with tau = 0.07

In v4 N-series configs, \(\tau = 0.07\). Substituting:

$$\kappa_{\max}^{\text{effective}} = 80 \times 0.07 = 5.6$$

For any \(\kappa > 5.6\):
- The positive logit is clamped to 80, regardless of the actual \(\kappa\) value
- The gradient \(\partial \mathcal{L} / \partial \kappa\) becomes zero (clamp has zero gradient in the saturated region)
- The model cannot learn to increase confidence beyond this ceiling

Combined with `kappa_reg` (an L2 penalty on kappa, pushing it toward zero), kappa is squeezed from both sides: clamping prevents upward gradient, regularization provides downward gradient. The equilibrium is \(\kappa \approx 3\text{-}5\), regardless of the model's actual prediction quality.

### 13.4 Empirical Signature

The kappa collapse manifests as:
- **Kappa statistics**: mean 3-5, std < 1, all samples near-identical confidence
- **All N-series plateau at same R@1** (~30-36%): N1 (MLP), N2 (ROI Transformer), N3 (ROI-DCF), N4 (full system) all converge to the same range despite radically different architectures
- **B-series not affected**: B0/B1 use InfoNCE/Gaussian losses without kappa, so they learn normally (20-25% R@1)
- **N-series > B-series**: The 30-36% vs 20-25% gap confirms vMF provides a genuine advantage even with collapsed kappa, via the directional (cosine-based) loss structure

### 13.5 The v5 Fix

Setting \(\tau = 1.0\) changes the effective ceiling:

$$\kappa_{\max}^{\text{effective}} = 80 \times 1.0 = 80$$

With `kappa_max = 50.0` (bounded sigmoid parameterization), the model can now freely use the full range \(\kappa \in [1.0, 50.0]\) without hitting the clamp. Disabling `kappa_reg` removes the downward pressure, allowing kappa to track actual prediction certainty.

### 13.6 Why Not Remove tau from Code?

An alternative fix would be to remove \(\tau\) from the `_score()` method entirely. This was considered but rejected because:
1. **Backward compatibility**: Existing v3/v4 configs with `tau=0.07` would silently change behavior if the code stopped using tau
2. **Config-level fix is sufficient**: Setting `tau=1.0` makes the division a no-op without code changes
3. **Future flexibility**: If a researcher wants to use temperature scaling with high kappa_max, the code path remains available

---

## 14. Experiment Version Genealogy

| Version | Configs | Key Changes | R@1 (subj01) |
|---------|---------|-------------|---------------|
| v1 | B0-N4 | Initial: batch 4, grad_accum 16, 100 epochs | Near-chance (nsdId bug) |
| v2 | B0v2-N4v2 | Dropout 0.3, R@1 early stopping | Near-chance (nsdId bug) |
| v3 | B0v3-N4v3 | Batch 64, residual MLP, image-level split | Near-chance (nsdId bug) |
| v4 | B0v4-N4v4 | nsdId fix, per-session z-score, MixCo, SoftCLIP | B: 20-25%, N: 30-36% |
| v5 | N1v5-N4v5 | tau=1.0, no kappa_reg, EMA, noise aug, 256 eff. batch | N1: 39.9% |
| v6 | N1v6-N4v6 | Delta-SPCL, vMF-SoftCLIP, Slerp MixCo | N1: ~40% |
| v7 | N1v7-N4v7 | Softplus kappa, multi-layer CLIP, multi-subject | N1: **49%** |
| v8 | N3v8, N4v8 | Hierarchical CLIP alignment, CKA loss | N4: **50.8%** |
| v9 | N1v9-N4v9 | DropPath, proj head, R-Drop, CSLS, MC-TTA, bf16, 1024 eff. batch | Pending |

Notes:
- B-series stays at v4 (not affected by vMF-specific changes)
- v6-v9 only apply to N-series experiments
- v8 only had N3 and N4 configs (N1/N2 skipped that iteration)
- v9 is the first version with all 4 N-series experiments and H100 optimization
