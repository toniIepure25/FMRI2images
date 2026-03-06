# Experiment Context: fMRI-to-Image Neural Decoding

This document provides complete technical context for a bachelor thesis project on neural decoding of visual perception from fMRI. It is designed as a self-contained briefing for an LLM or researcher performing deep analysis.

**Project status (March 2026):** After 14 iterative versions (v4-v14), the system achieves **~45% raw R@1 / ~54% CSLS R@1** on subj01 with V13 configs. The persistent 8-10 pp gap between raw and CSLS R@1 identifies **hubness** as the primary remaining bottleneck. V14 attacks this directly via differentiable CSLS training loss, inverted softmax (ISF), direct cosine alignment, and CSLS-based checkpoint selection. All experiments run on an **NVIDIA H100 80GB HBM3** with bf16 mixed precision and effective batch size 512.

**SOTA target:** MindEye achieves 93.2% R@1 on the same dataset. The remaining gap is attributed to (1) single-subject vs 7-subject pre-training, (2) smaller model capacity (328M vs 996M params), (3) MindEye's OpenCLIP ViT-bigG/14 embeddings (256x1664-D) vs our ViT-L/14 (768-D), and (4) hubness in high-dimensional retrieval from single-trial fMRI noise.

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
- CLIP cache: `outputs/clip_cache/clip.parquet` (single-layer), `outputs/clip_cache/clip_multilayer.parquet` (multi-layer for hierarchical alignment)
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
| Best R@1 | v10: ~48% | v10: ~51% | 93.0% |

### 11.5 Quantitative Gap Analysis (v10 actual vs SOTA)

| Metric | Our B0v4 | Our best (N4v10) | MindEye2 | Gap |
|--------|----------|------------------|----------|-----|
| R@1 (val) | ~22% | ~51% | 93.0% | ~42 pp |

**Progress from v4 to v10:** +29 pp (from 22% to 51%). Gains attributed to:
1. **Kappa collapse fix (v5)**: +7-14 pp (freed vMF concentration parameter)
2. **Softplus kappa (v7)**: +9 pp (removed sigmoid saturation)
3. **Multi-subject + multi-layer CLIP (v7-v8)**: +1-2 pp (more data, richer targets)
4. **Hierarchical alignment + CKA (v8)**: +1 pp (then plateau due to gradient dilution)

**V9-V11 yielded no meaningful improvement** due to:
- V9: Over-regularization (DropPath, R-Drop, label smoothing, projection head) hurt more than helped
- V10: Kappa explosion (softplus ignoring kappa_max cap) causing overconfidence
- V11: `average_repetitions: true` reducing training data 3x; 7+ competing losses

**Primary contributors to remaining gap (ranked by estimated impact):**
1. **Missing regression stage (est. 10-15 pp)**: MindEye1/2 uses two-stage training (contrastive then MSE). V12 addresses this with vMF-NLL fine-tuning.
2. **CLIP embedding gap (est. 15-20 pp)**: MindEye2 uses OpenCLIP ViT-bigG/14 (256 tokens x 1664-D) vs our ViT-L/14 (768-D single vector)
3. **Kappa explosion (est. 2-3 pp)**: Unbounded softplus kappa >100 caused validation degradation. V12 caps at 50.
4. **Model capacity gap (est. 3-5 pp)**: MindEye1 uses 982M params. V12 N1 scales to ~600M.
5. **Functional alignment (est. 5-10 pp)**: MindEye2 uses subject-specific ridge regression; our approach uses learned ROI projections

**Expected outcome after v12:** Two-stage training is the single most impactful proven technique not yet tried. With kappa fix, batch scaling, and loss simplification, targeting ~75% R@1. Further gains require OpenCLIP ViT-bigG or diffusion prior.

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
| v12 N-series configs (current) | `configs/experiments/N1v12_vmf_nce.yaml` through `N4v12_full_system.yaml` |
| v10 N-series configs | `configs/experiments/N1v10_vmf_nce.yaml` through `N4v10_full_system.yaml` |
| v8 N-series configs | `configs/experiments/N3v8_roi_dcf.yaml`, `N4v8_full_system.yaml` |
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
| v9 | N1v9-N4v9 | Anti-overfitting (DropPath, R-Drop, label smoothing, proj head), CSLS eval, MC-TTA | N1: ~48%, N3/N4: ~51% |
| v10 | N1v10-N4v10 | V8 recipe + V9 eval + wider models + hard neg + model soup | N1: ~48%, N4: ~50.8% |
| v11 | N1v11-N4v11 | CSLS training loss, ISF, rep averaging, direct alignment, uniformity | ~51% (no improvement) |
| v12 | N1v12-N4v12 | Two-stage training + kappa cap fix + eff. batch 512 + auto-weighting | N1: CSLS 52%, N3/N4: 27-37% (regression) |
| v13 | N1v13-N4v13 | MSE regression from epoch 1 (MindEye-style) + hierarchical CLIP (N3/N4) | N1: 45-46%, CSLS 54%; N3: 45%, CSLS 54-55% |
| v14 | N1v14-N4v14 | Anti-hubness (CSLS training + ISF + direct alignment) + CSLS checkpoint | N1: ~43%, CSLS ~53%; N4: ~44-45%, CSLS ~55-58% |
| v15 | N1v15-N4v15 | **Fix fundamentals**: PCR re-enabled, batch 256, z-scoring bug fix, loss simplification | Pending |

Notes:
- B-series stays at v4 (not affected by vMF-specific changes)
- v6+ only apply to N-series experiments
- v8 only had N3 and N4 configs (N1/N2 skipped that iteration)
- v10 combined best of V8 (losses) and V9 (eval tricks)
- v11 attempted hubness mitigation and denoising but failed due to `average_repetitions: true` reducing data 3x
- v12 two-stage never activated for N1/N2 (early stopping before epoch 120); auto-weighting destroyed N3/N4
- v13 adds MSE regression (MindEye1's key ingredient) and hierarchical CLIP alignment for N3/N4
- v14 attacks the 8-10 pp hubness gap with three training-time mechanisms + CSLS-based checkpoint selection
- v15 fixes three fundamental bottlenecks: z-scoring bug for multi-subject, disabled CLIP PCR, small in-batch size

---

## 15. V12 Root Cause Analysis and Design

### 15.1 Four Root Causes of the 51% Ceiling

**Root Cause 1 -- Data reduction from repetition averaging:**
V11's `average_repetitions: true` collapsed ~24K training trials to ~8K unique images.
This was catastrophic for contrastive learning: the queue (65536) cycled through the dataset ~8x with stale negatives, and the natural data augmentation from 3 noisy repetitions per image was destroyed.

**Root Cause 2 -- Kappa explosion (unbounded softplus):**
In `vmf_decoder.py`, the softplus kappa activation path ignored `kappa_max` from config entirely, only clamping at `KAPPA_AMP_CEIL=5000`. V10 logs show kappa exploding to >100 by epoch 28-36, turning vMF-NCE's softmax into a hard argmax (training overconfidence, validation degradation). **Fixed in V12**: softplus now clamps at `min(kappa_max, 5000)`, with `kappa_max: 50` in configs.

**Root Cause 3 -- Contrastive-only training (no regression stage):**
MindEye1 achieves 93.2% R@1 via two-stage training: (1) contrastive learning to learn neighborhood structure, then (2) MSE/NLL fine-tuning to precisely place each embedding. We used only contrastive losses, which learn *relative* positioning but not *absolute* coordinates. **Fixed in V12**: Stage 2 transitions to vMF-NLL loss at epoch 120.

**Root Cause 4 -- Loss gradient dilution:**
V10/V11 N4 configs had 6-8 simultaneous loss terms competing for gradient bandwidth. Without dynamic weighting, the primary retrieval objective stagnated. **Fixed in V12**: Simplified to 4 core losses + homoscedastic uncertainty weighting (Kendall et al., 2018).

### 15.2 V12 Two-Stage Training Protocol

```
Stage 1 (epochs 1-119): Contrastive manifold learning
  Losses: vMF-NCE (w=1.0) + SoftCLIP (w=1.0) + MixCo (w=0.5) + kappa_reg (0.01)
  LR: 1e-4 (N1) / 7e-5 (N2-N4), cosine schedule
  Effective batch: 512 (64 x 8 grad_accum)

Stage 2 (epochs 120-200): NLL coordinate fine-tuning
  Losses: vMF-NLL (w=2.0) + SoftCLIP (w=0.3)
  LR: reduced by 10x
  MixCo: disabled
  Early stopping: reset, patience=20
```

The vMF NLL loss \(\mathcal{L}_{\text{NLL}} = -\log C_d(\kappa) - \kappa \cdot \mu^T z\) provides:
- Direct cosine alignment (the \(-\kappa \mu^T z\) term)
- Automatic kappa calibration (the normalizing constant \(C_d(\kappa)\) penalizes extreme overconfidence)
- Per-sample gradient independent of other batch elements (unlike contrastive losses)

### 15.3 V12 Config Summary

| Experiment | Encoder | Key V12 changes |
|-----------|---------|-----------------|
| N1v12 | MLP [8192, 8192, 4096, 4096, 2048] (~600M params) | Two-stage, kappa_max=50, eff. batch 512 |
| N2v12 | ROI Transformer (d=1024, FFN=8192, 6L) | Two-stage, kappa_max=50, eff. batch 512 |
| N3v12 | ROI-DCF (d=1024, FFN=8192, 6L) | Two-stage, kappa_max=50, simplified losses, auto-weighting |
| N4v12 | ROI-DCF + SPCL (d=1024, FFN=8192, 6L) | Two-stage, kappa_max=50, simplified losses, auto-weighting |

---

## 16. V12 Post-Mortem and V13 Design

### 16.1 V12 Results

| Experiment | Raw R@1 | CSLS R@1 | Epoch | Notes |
|-----------|---------|----------|-------|-------|
| N1v12 | 43.3% | **52.2%** | 83 (early-stopped) | Stage 2 never activated |
| N2v12 | 42.3% | 50.0% | 101 (early-stopped) | Stage 2 never activated |
| N3v12 | 25.2% | 37.2% | 140 | Auto-weighting + vmf_nll destroyed training |
| N4v12 | 27.3% | 37.1% | 60 (truncated) | Same auto-weight bug + numerical spikes |

### 16.2 V12 Failure Diagnosis

**Bug 1 -- Auto-weighting destroyed N3/N4:**
The homoscedastic uncertainty weighting had two fatal flaws:
1. `vmf_nll` (config weight=0.0) was included in the auto-weight set. The filter `[k for k in losses if k != "kappa_reg"]` doesn't check config weight. So vmf_nll got overridden to weight 0.5.
2. The `log_sigma` regularization gradient was always `+0.5` (from the `0.5 * log_sigma` additive term), independent of loss magnitude. The actual loss weights were computed as floats from `log_sigma.item()`, NOT as differentiable tensors. The auto-weighting learned nothing.
Result: vmf_nll at ~40,000 dominated all other losses (~4-6 range), destroying contrastive learning for N3/N4.

**Bug 2 -- Stage 2 never activated for N1/N2:**
N1 early-stopped at epoch 83 (best ~epoch 53). Stage 2 was set at epoch 120. The two-stage training -- the centerpiece of V12 -- never ran for the best-performing models.

**Root cause 3 -- vmf_nll is impractical:**
The vMF normalizing constant \(-\log C_d(\kappa)\) for d=768 adds a ~40,000 baseline. While gradients are technically correct (the constant is independent of \(\mu\)), the enormous scale makes multi-loss balancing catastrophic and loss logging meaningless.

### 16.3 V13: MSE Regression + Contrastive (MindEye-style)

**Key insight:** The fundamental missing piece is a **per-sample regression loss**. All previous versions used only contrastive losses (vMF-NCE, SoftCLIP), which provide *relative* positioning ("this pair closer than that pair") but never tell the model the *absolute* target location. MindEye1 achieves 93.2% R@1 by combining SoftCLIP with **MSE loss** from epoch 1.

For L2-normalized vectors: \(\text{MSE}(\mu, z) = \|\mu - z\|^2 = 2(1 - \cos(\mu, z))\), bounded [0, 4], gradient \(2(\mu - z)\) per sample. This directly forces each prediction toward its target.

**V13 loss recipe (all experiments):**

| Loss | Stage 1 weight | Stage 2 weight (epoch 80+) | Purpose |
|------|---------------|--------------------------|---------|
| MSE | 1.0 | **2.0** | Per-sample regression (MindEye's key ingredient) |
| vMF-NCE / SPCL | 1.0 | 0.3 | Contrastive positioning with kappa |
| SoftCLIP | 1.0 | 0.5 | Knowledge distillation |
| HierarchicalCLIP (N3/N4) | 0.5 | 0.2 | Per-tier ROI-to-CLIP-layer alignment |
| Multitask aux (N3/N4) | lambda=0.05 | disabled | Weak global coherence signal |
| kappa_reg | 0.01 | 0.01 | Kappa regularization |
| MixCo | 0.5 | disabled | Augmentation |

Changes vs V12: removed vmf_nll (replaced by bounded MSE), removed auto_weight (broken), lowered stage2 start to epoch 80, enabled hierarchical CLIP alignment for N3/N4 (BrainMCLIP-style, see 16.5).

### 16.4 V13 Config Summary

| Experiment | Encoder | Key V13 changes |
|-----------|---------|-----------------|
| N1v13 | MLP [8192, 8192, 4096, 4096, 2048] | MSE from epoch 1, stage2 at 80, no vmf_nll |
| N2v13 | ROI Transformer (d=1024, FFN=8192, 6L) | MSE from epoch 1, stage2 at 80, no vmf_nll |
| N3v13 | ROI-DCF (d=1024, FFN=8192, 6L) | MSE from epoch 1, stage2 at 80, no auto_weight, hierarchical CLIP (0.5), multitask 0.05 |
| N4v13 | ROI-DCF + SPCL (d=1024, FFN=8192, 6L) | MSE from epoch 1, stage2 at 80, no auto_weight, hierarchical CLIP (0.5), multitask 0.05 |

### 16.5 Hierarchical ROI-to-Layer Alignment (N3/N4)

**Root cause addressed:** In V12 (and prior versions), the `vmf_nce_multitask` auxiliary loss forces every ROI expert to independently predict the final, global 768-D CLIP embedding. This violates the known functional hierarchy of the visual cortex. Early visual areas (V1, V2) process low-level features -- edges, orientations, spatial frequency -- while the final CLIP layer encodes high-level semantic identity ("a dog," "a beach scene"). Forcing V1 to predict abstract semantics produces near-random gradients that conflict with the meaningful gradients from higher-level ROIs (FFA, PPA), suppressing their learning signal.

**Solution -- BrainMCLIP-style multi-layer alignment (Liu et al., 2023; Luo et al., 2024 MindHier):**
Instead of one global target for all ROIs, we align each visual hierarchy tier to the CLIP layer at the corresponding level of abstraction:

| Tier | ROIs | CLIP target | Neuroscience rationale |
|------|------|-------------|----------------------|
| Early | V1v, V1d, V2v, V2d, V3v, V3d | Layer 12 (low-level features) | V1-V3 encode edges, orientations, spatial frequency |
| Mid | V3A, V3B, V4 | Layer 18 (mid-level features) | V3A/V3B process motion and contour, V4 handles shape and color |
| High | FFA1, FFA2, PPA, EBA, OFA, OPA | Final layer (semantic identity) | FFA=faces, PPA=scenes, EBA=bodies, OFA=faces, OPA=scenes |

This is implemented by `HierarchicalCLIPLoss` (already in the codebase since V8), which computes the attention-weighted average of per-ROI \(\mu\) predictions within each tier and measures cosine distance to the corresponding CLIP layer target. The multi-layer CLIP cache (`clip_multilayer.parquet`) stores `layer_12_proj` and `layer_18_proj` embeddings (768-D, projected via CLIP's `visual.proj`).

**Weight schedule:**
- Stage 1 (epochs 1-79): `hierarchical_clip` weight = 0.5, `lambda_aux` = 0.05
- Stage 2 (epoch 80+): `hierarchical_clip` weight = 0.2, `lambda_aux` = 0.0

The `lambda_aux` is reduced from 0.2 to 0.05 because the hierarchical loss now provides per-tier supervision, making the global auxiliary signal largely redundant. A small residual maintains weak global coherence.

**Prerequisite:** Multi-layer CLIP cache must be built via `make multilayer-clip-cache` before training N3v13/N4v13.

---

## 17. V13 Results and V14 Design

### 17.1 V13 Results (subj01)

| Experiment | Raw R@1 | CSLS R@1 | R@5 | Median Rank | Key Observation |
|-----------|---------|----------|-----|-------------|-----------------|
| N1v13 | ~45-46% | **~54%** | ~75-79% | ~2 | kappa_mean at ceiling (~50); large CSLS uplift |
| N2v13 | ~44% | ~51-52% | ~76-77% | ~2 | Smoothest convergence; lowest CSLS gap |
| N3v13 | ~45% | **~54-55%** | ~73-78% | ~2-3 | Best CSLS performer; hierarchical alignment improves geometry |
| N4v13 | ~45-46% | ~53-54% | ~74-77% | ~2 | Competitive peak; SPCL curriculum + hierarchical |

**Key takeaway:** All four models converge to similar raw R@1 (~44-46%) but CSLS R@1 is consistently 8-10 pp higher. This confirms **hubness** as the dominant bottleneck -- the embedding space contains hub points that are artificially close to many queries under raw cosine similarity, penalizing raw R@1 more than the hubness-corrected CSLS metric.

The hierarchical CLIP alignment in N3v13 produces the best CSLS R@1 (54-55%), validating that tier-aware supervision improves the geometry of the embedding space and reduces hub formation compared to the global auxiliary loss.

### 17.2 Hubness Diagnosis

Hubness manifests when a few gallery embeddings become "universally popular" nearest neighbours for many queries. In high-dimensional spaces (d=768), this is a known phenomenon (Radovanovi\'{c} et al., 2010). The 8-10 pp CSLS gap quantifies the severity: CSLS corrects for hubness by penalizing embeddings with high average similarity to their k-NN, and the gap shows that removing this bias substantially improves retrieval accuracy.

Hubness is exacerbated by:
1. **Single-trial fMRI noise**: Each fMRI repetition contains irreducible physiological noise, pushing some predictions toward the gallery mean (which becomes a hub).
2. **Contrastive-only training**: Standard cross-entropy on cosine logits does not explicitly penalize hub formation -- it only requires the correct item to rank highest, not that the embedding space be uniformly distributed.
3. **Kappa at ceiling**: With kappa_mean ~50 (the configured maximum), the model is maximally confident about directions that may still be noisy, concentrating predictions in tight clusters that overlap with hubs.

### 17.3 V14: Three-Pronged Anti-Hubness Strategy

V14 enables three mechanisms already implemented in the codebase but disabled in V13:

**1. Differentiable CSLS Training Loss** (`vmf_nce.use_csls_training: true`)

Applies CSLS correction directly to the contrastive logit matrix *before* the cross-entropy computation:

\[\text{logits}_{\text{CSLS}}(x, y) = 2 \cdot s(x, y) - r_X(x) - r_Y(y)\]

where \(r_X(x) = \frac{1}{k}\sum_{y' \in \text{kNN}(x)} s(x, y')\). All operations (topk, mean, subtract) are differentiable, so gradients teach the encoder to produce embeddings that avoid hub regions. This directly attacks the root cause during training, not just at evaluation time.

**2. Inverted Softmax (ISF)** (`vmf_nce.isf_weight: 0.3`)

Standard contrastive loss normalizes over keys for each query (row-wise softmax). ISF adds a column-wise softmax component that penalizes gallery items appearing as top matches for many queries:

\[\mathcal{L}_{\text{ISF}} = -\log \frac{\exp(s_{ii})}{\sum_j \exp(s_{ji})}\]

The final loss is \((1 - w_{\text{ISF}}) \cdot \mathcal{L}_{\text{std}} + w_{\text{ISF}} \cdot \mathcal{L}_{\text{ISF}}\) with \(w_{\text{ISF}} = 0.3\). This symmetrizes the contrastive objective and explicitly discourages hub formation on the gallery side.

**3. Direct Cosine Alignment** (`direct_alignment.weight: 0.5`)

Adds a per-sample absolute alignment signal independent of batch composition:

\[\mathcal{L}_{\text{DA}} = 1 - \cos(\mu, z_{\text{GT}})\]

While MSE provides a similar signal, direct alignment operates in angular space and is less sensitive to the embedding norm. It provides a stable gradient for positioning each prediction toward its target without interacting with the contrastive objective or batch structure.

### 17.4 CSLS-Based Checkpoint Selection

V14 introduces configurable `training.checkpoint_metric: "csls_r@1"`. Previous versions always selected the best checkpoint by raw R@1, which preferentially selects models at local optima that may coincidentally benefit from hub-biased rankings. By selecting on CSLS R@1, we choose the model whose embedding space has the best hubness-corrected geometry.

Supported values: `"r@1"` (default, backward compatible), `"csls_r@1"`, `"median_rank"` (lower is better). The Stage 2 transition properly resets the best metric value.

### 17.5 OpenCLIP ViT-bigG/14 Assessment

MindEye/MindEye2 use OpenCLIP ViT-bigG/14 (2B parameters, trained on LAION-2B), which provides:
- **Pooled 1280-D embeddings**: single vector, higher-dimensional than our ViT-L/14 768-D
- **256 x 1664-D token embeddings**: full spatial token representation used by MindEye2 for multi-token decoding

**Assessment for this project:**

The 256 x 1664-D multi-token representation requires predicting 256 separate spatial tokens via a cross-attention decoder and diffusion prior -- a fundamental architecture redesign beyond the scope of this thesis.

The pooled 1280-D migration is feasible (config + cache rebuild + output_dim update) but carries risks: (a) higher-dimensional target space is harder to learn with limited single-subject data (~24K trials), (b) the hierarchical loss layer mapping must be recalibrated for bigG's 48-layer architecture, and (c) SD 2.1 reconstruction would need a 1280->1024 adapter.

**Recommendation:** Defer bigG migration until V14 anti-hubness results are available. If the hubness gap closes but raw R@1 still plateaus below 55%, the pooled 1280-D migration becomes the next logical step. The codebase is largely dimension-agnostic (embedding_dim inferred from data), making the migration a cache-rebuild + config change.

### 17.6 Diagnostic Tooling

V14 adds `scripts/evaluation/diagnose_embeddings.py` for post-training analysis:

```bash
python scripts/evaluation/diagnose_embeddings.py \
    --results-dir experimental_results/N1v14_vmf_nce/subj01
```

Produces `diagnostics/report.json` with:
- **Hubness metrics**: k-occurrence skewness, hub/antihub fraction, top hub indices
- **Similarity analysis**: positive vs negative pair cosine distributions, separability, overlap fraction
- **Retrieval gap**: raw R@1 vs CSLS R@1 with per-k breakdown
- **Kappa analysis**: per-kappa-bin R@1 (tests uncertainty-accuracy correlation)
- **Failure cases**: top-20 hardest samples with ranks and retrieved competitors

Plus two plots: `hubness_histogram.png` and `similarity_distribution.png`.

---

## 18. V14 Results and V15 Design

### 18.1 V14 Results (subj01)

| Experiment | Raw R@1 | CSLS R@1 | R@5 | R@10 | Key Observation |
|-----------|---------|----------|-----|------|-----------------|
| N1v14 | ~43% | ~52-54% | ~75-79% | ~92% | Early-stop ~epoch 40-45; kappa ramps hard early |
| N2v14 | ~38-40% | ~52-55% | ~76-77% | ~91-92% | More gradual; competitive CSLS but lower raw R@1 |
| N3v14 | ~40%+ | ~52-55% | ~73-78% | ~86-88% | Steadier ranking improvements from extra heads |
| N4v14 | **~44-45%** | **~55-58%** | ~74-77% | ~92% | Strongest overall; CSLS gains consistently strong |

**Assessment:** The V14 anti-hubness measures (CSLS training, ISF, direct alignment) did **not** improve raw R@1 compared to V13. In fact, N1v14 (43%) is slightly below N1v13 (45-46%), while N4v14 is essentially identical. The CSLS gap persists at 8-13 pp, indicating the training-time CSLS correction failed to address hubness at its source.

### 18.2 V15: Fix the Fundamentals

Deep analysis of the codebase reveals three critical bottlenecks that together explain most of the 40 pp gap to MindEye's 84% R@1 on NSD subj01:

**Bug 1 -- Z-scoring never applied to multi-subject datasets (N2/N3/N4):**

In `train_unified.py` (line 1771, pre-V15), the z-scoring block was gated by:
```python
if normalize_fmri and isinstance(full_dataset, PreextractedNSDDataset):
```

`MultiSubjectPreextractedDataset` (used by N2/N3/N4) fails this isinstance check, so per-session z-scoring silently does not run despite `normalize_fmri: true` and `zscore_mode: "per_session"` being set in the config. N2/N3/N4 have been training on **raw, unnormalized beta values** across all 40 sessions, including inter-session scanner drift and baseline shifts.

**Fix (V15):** Added an `elif` branch in `train_unified.py` that handles `MultiSubjectPreextractedDataset` by iterating over each subject's `features_list` entry and applying per-session z-scoring using training-only statistics. Stats are saved per-subject per-session to `zscore_stats/`.

**Bottleneck 2 -- CLIP embedding PCR disabled since V9:**

All configs from V5 through V14 set `preprocessing.enabled: false`. The `EmbeddingPreprocessor` (center-PCR) removes the top-k dominant principal components from CLIP embeddings. These shared components are the primary cause of hubness -- they make many gallery items appear close to many queries. Removing them spreads embeddings more uniformly on the hypersphere.

The 8-10 pp CSLS gap observed across all versions is a direct symptom of these shared components remaining in the target embeddings. Rather than adding complex training-time corrections (CSLS loss, ISF), V15 addresses the root cause by re-enabling PCR with \(k = 4\).

**Bottleneck 3 -- Only 63 in-batch contrastive negatives:**

`batch_size: 64` yields only 63 in-batch negatives per contrastive step. MindEye uses 300. The queue (65536 entries) provides additional negatives, but they are **stale** -- computed from past model states, not the current parameters. In-batch negatives produce much more informative gradients because they reflect the current model's failure modes.

On the H100 (80 GB VRAM), `batch_size: 256` is feasible for all architectures, providing 255 in-batch negatives -- a 4x improvement.

### 18.3 V15 Changes Summary

| Change | V14 Value | V15 Value | Rationale |
|--------|-----------|-----------|-----------|
| `preprocessing.enabled` | false | **true** (center_pcr, k=4) | Eliminate hubness at the embedding level |
| `training.batch_size` | 64 | **256** | 4x more in-batch contrastive negatives |
| `training.gradient_accumulation_steps` | 8 | **2** | Effective batch remains ~512 |
| `queue.size` | 65536 | **16384** | Sufficient with larger in-batch pool |
| `vmf_nce.use_csls_training` | true | **false** | PCR handles hubness at source |
| `vmf_nce.isf_weight` | 0.3 | **0.0** | Removed (didn't help in V14) |
| `direct_alignment.enabled` | true | **false** | MSE already provides per-sample alignment |
| `training.checkpoint_metric` | csls_r@1 | **r@1** | With PCR, raw and CSLS R@1 should converge |
| Multi-subject z-scoring | **broken** | **fixed** | Per-subject per-session normalization |
| Val prediction saving | absent | **implemented** | Saves .npy for `diagnose_embeddings.py` |

### 18.4 V15 Config Summary

| Config | Encoder | Key V15 changes |
|--------|---------|-----------------|
| N1v15 | MLP [8192, 8192, 4096, 4096, 2048] | PCR k=4, batch 256, drop CSLS/ISF/DA |
| N2v15 | ROI Transformer (d=1024, FFN=8192, 6L) | PCR k=4, batch 256, z-scoring fixed, drop CSLS/ISF/DA |
| N3v15 | ROI-DCF (d=1024, FFN=8192, 6L) | PCR k=4, batch 256, z-scoring fixed, drop CSLS/ISF/DA, keep hierarchical CLIP |
| N4v15 | ROI-DCF + SPCL (d=1024, FFN=8192, 6L) | PCR k=4, batch 256, z-scoring fixed, drop CSLS/ISF/DA, keep hierarchical CLIP + DUA-CFG |

### 18.5 V15 Actual Results (subj01)

V15 did NOT achieve the expected R@1 gains. Diagnostic analysis via `diagnose_embeddings.py`:

| Metric | N1v15 | N2v15 | N4v15 |
|--------|-------|-------|-------|
| Raw R@1 | 44.0% | 38.9% | 42.2% |
| CSLS R@1 | 51.4% | 45.9% | 47.2% |
| Hubness gap | 7.4 pp | 7.0 pp | 5.0 pp |
| Pos sim (mean) | 0.354 | 0.331 | 0.340 |
| Neg sim (mean) | 0.037 | 0.014 | 0.044 |
| Separability | 0.317 | 0.317 | 0.296 |
| Kappa (mean/std) | 12.1/0.4 | 17.7/0.5 | 20.4/0.4 |
| Skewness (k-occ) | 0.70 | 0.72 | 0.65 |

**Key findings:**

1. **Positive cosine similarity (0.33-0.35) is the primary bottleneck.** For 65% R@1, this needs to be 0.50+. The model's predictions are off by ~70 degrees from the targets on average.

2. **PCR reduced hubness gap slightly** (5-7 pp in V15 vs 8-10 pp in V14), but absolute R@1 did not improve -- it slightly decreased. PCR appears to remove useful variance alongside the hubness-causing components.

3. **Kappa dropped from ~50 (V14) to 12-20 (V15).** The model is appropriately less confident given the PCR-altered embedding geometry. This is a correct adaptation, not a regression.

4. **Separability (~0.30) is consistent.** The contrastive loss effectively separates positives from negatives in relative terms, but the absolute alignment (positive cosine similarity) is too weak for reliable top-1 retrieval.

5. **N2v15 (ROI Transformer, 38.9%) performs worst.** The ROI tokenization compresses ~15,000 voxels through 17 tokens of 1024 dimensions -- an information bottleneck that hurts prediction quality.

---

## 19. V16: Unlock Prediction Quality

### 19.1 Root Cause Analysis

The V15 diagnostics reveal that the ~600M-param N1v15 MLP is **underfitting** despite its substantial capacity. The root cause is **8 simultaneous regularization sources** that collectively prevent the model from learning precise fMRI-to-CLIP mappings:

| Regularization Source | V15 Value | Effect |
|----------------------|-----------|--------|
| Encoder dropout | 0.2 | Randomly drops 20% of hidden units |
| Decoder dropout | 0.2 | Same in decoder pathway |
| Weight decay | 0.05 | Aggressive L2 penalty on all params |
| fmri_noise_std | 0.1 | Additive Gaussian noise on input voxels |
| voxel_dropout | 0.1 | Randomly zeros 10% of input voxels |
| MixCo | alpha=0.15, w=0.5 | Soft-label interpolated augmentation |
| kappa_reg | lambda=0.01 | Penalizes high concentration (confidence) |
| SoftCLIP soft labels | tau=0.05 | Prevents hard positive/negative discrimination |

Combined with training on **noisy single-repetition fMRI** (average_repetitions: false), the model receives weak, noisy signals through a heavily regularized network -- a recipe for underfitting.

### 19.2 V16 Changes

**Change 1 -- Slash regularization:**

| Parameter | V15 | V16 | Rationale |
|-----------|-----|-----|-----------|
| dropout | 0.2 | **0.05** | 4x reduction; model is underfitting |
| weight_decay | 0.05 | **0.005** | 10x reduction |
| fmri_noise_std | 0.1 | **0.0** | Disable input noise entirely |
| voxel_dropout | 0.1 | **0.0** | Disable voxel masking |
| MixCo | enabled, w=0.5 | **disabled** | Remove augmentation |
| kappa_reg | lambda=0.01 | **disabled** | Stop penalizing confidence |

**Change 2 -- Enable repetition averaging:**

Average fMRI across all repetitions of the same image before training (~27,000 noisy trials to ~9,000 clean images, SNR improvement \(\sqrt{3} \approx 1.73\times\)). Implemented for both `PreextractedNSDDataset` (N1) and `MultiSubjectPreextractedDataset` (N2-N4).

**Change 3 -- Simplify loss to MSE + vMF-NCE:**

- MSE (weight 2.0) -- primary per-sample regression signal
- vMF-NCE (weight 0.5) -- contrastive structure + kappa training
- All other losses disabled (SoftCLIP, MixCo, kappa_reg, hierarchical_clip, multitask)

**Change 4 -- Remove two-stage training:**

Stage 2 abruptly shifted loss weights at epoch 80, disrupting optimization. V16 uses a single continuous schedule throughout.

**Change 5 -- Disable PCR:**

V15 PCR (k=4) slightly hurt R@1 compared to V13. Disabled to preserve full CLIP embedding variance.

**Change 6 -- Adjust training dynamics:**

| Parameter | V15 | V16 | Rationale |
|-----------|-----|-----|-----------|
| LR | 1e-4 | **2e-4** | Stronger gradients with fewer samples |
| Epochs | 200 | **300** | Compensate for fewer samples (9k vs 27k) |
| Queue size | 16384 | **8192** | Proportional to ~8,100 training images |
| kappa_max | 50 | **200** | Allow higher confidence with better predictions |
| Patience | 30 | **40** | More runway before early stopping |
| Warmup | 15 | **20** | Gentler ramp with higher peak LR |

**Change 7 -- Training R@1 monitoring:**

Added periodic training R@1 computation (every 10 epochs, ~1024-sample subset). This is critical for distinguishing overfitting (train R@1 >> val R@1) from underfitting (both low). Logged as `train_r@1` in training CSV.

### 19.3 V16 Config Summary

| Config | Encoder | Key V16 changes |
|--------|---------|-----------------|
| N1v16 | MLP [8192, 8192, 4096, 4096, 2048] | Rep-avg, dropout 0.05, MSE(2.0)+NCE(0.5), no PCR, no stage2 |
| N2v16 | ROI Transformer (d=1024, 6L, FFN=8192) | Rep-avg, dropout 0.05, MSE(2.0)+NCE(0.5), no PCR, no stage2 |
| N3v16 | ROI-DCF (d=1024, 6L, FFN=8192) | Rep-avg, dropout 0.05, MSE(2.0)+NCE(0.5), no PCR, no stage2, no multitask |
| N4v16 | ROI-DCF + SPCL (d=1024, 6L, FFN=8192) | Rep-avg, dropout 0.05, MSE(2.0)+SPCL(0.5), no PCR, no stage2, no multitask |

### 19.4 Expected Impact

| Change | Expected R@1 Gain | Rationale |
|--------|-------------------|-----------|
| Repetition averaging | +3-5 pp | 1.73x SNR improvement on training data |
| Reduced regularization | +5-8 pp | Unlock model capacity from underfitting |
| Simplified loss | +2-3 pp | Cleaner optimization landscape |
| Disabled PCR | +1-2 pp | Preserve full embedding variance |
| Higher LR + longer training | +1-2 pp | Better convergence with cleaner data |
| **Combined estimate** | **+12-20 pp** | Target: 55-65% R@1 |

**Risk:** With 9,000 clean training images and a 600M-param model at dropout=0.05, overfitting is possible. The training R@1 monitor will detect this. If train R@1 exceeds val R@1 by more than 20 pp, light regularization (dropout 0.10) should be restored.

### 19.5 Shared1000 Benchmark Evaluation

Starting with V16, every training run automatically evaluates on the **NSD shared1000 benchmark** at the end of training. This is the community-standard test set used by MindEye, MindEye2, Brain Diffuser, and other NSD decoding papers.

**What it does:**
1. Loads the raw pre-extracted fMRI features for the subject (independent of training-time averaging)
2. Filters to shared1000 trials (~3,000 trials across ~982 unique images)
3. Averages the 3 repetitions per image (standard protocol)
4. Applies the same z-scoring transform used during training
5. Runs model inference on the ~982 averaged images
6. Computes R@1, R@5, R@10, MRR, median rank (raw + CSLS) against CLIP embeddings

**Output files** (in `experimental_results/{exp}/{subj}/metrics/`):
- `shared1000_metrics.json` -- full retrieval results with metadata
- `shared1000_predictions.npy` -- (N, 768) predicted embeddings
- `shared1000_ground_truth.npy` -- (N, 768) CLIP GT embeddings

**Gallery comparison with published methods:**

| Method | Test Set | Gallery Size | CLIP Model |
|--------|----------|-------------|------------|
| MindEye2 (Scotti et al., 2024) | shared1000 | ~982 images | OpenCLIP ViT-bigG/14 (1280-D) |
| Brain Diffuser (Ozcelik & VanRullen, 2023) | shared1000 | ~982 images | CLIP ViT-L/14 (768-D) |
| **Ours (V16)** | **shared1000** | **~982 images** | CLIP ViT-L/14 (768-D) |

**Note on comparability:** Our shared1000 R@1 is directly comparable in gallery size and protocol. The CLIP model difference (ViT-L/14 vs bigG) means absolute R@1 numbers are not strictly apples-to-apples with MindEye2, but are comparable with Brain Diffuser and other ViT-L/14-based methods. Migration to OpenCLIP ViT-bigG/14 is deferred until V16 results prove the prediction quality improvements are effective.

**Config flag:** `evaluation.eval_shared1000: true` (default true in V16 configs). Set to `false` to skip.

### 19.6 Version Genealogy

```
V4 (fix 6 bugs) -> V5 (tau=1, no kappa_reg)
  -> V6 (novel losses) -> V7 (multi-subj, softplus)
    -> V8 (hierarchical CLIP) -> V9 (anti-overfit, H100)
      -> V10 (wider + hard neg + soup) -> V11 (CSLS training)
        -> V12 (two-stage, auto-weight) -> V13 (MSE regression)
          -> V14 (anti-hubness + diagnostics) -> V15 (PCR + batch + z-score fix)
            -> V16 (rep-avg + low-reg + focused loss + shared1000 eval)
```
