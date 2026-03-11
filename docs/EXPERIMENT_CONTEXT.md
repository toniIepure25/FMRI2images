# Experiment Context: fMRI-to-Image Neural Decoding

This document provides complete technical context for a bachelor thesis project on neural decoding of visual perception from fMRI. It is designed as a self-contained briefing for an LLM or researcher performing deep analysis.

**Project status (March 2026):** After 26 iterative versions, the project-best is **52.8% raw R@1 / 69.6% CSLS R@1** from N1v26a (MindEye-style 257×768 token targets, single-subject, 675M params). V26a delivered **+5.2pp raw / +17.1pp CSLS** over the previous best (N1v23a), proving that target quality was the primary remaining bottleneck. V24 (hard negatives) and V25 (structural experiments) both regressed. V25 results: N1v25_rerun 57.2% CSLS (val900), 52.5% (shared1000); N1v25a sequential 54.9% / 50.8%; N2v25c patched-ROI 46.2% / 45.1%; N1v25b crashed (optimizer bug, now fixed). V26b (cross-subject + token targets, 1.59B params) was abandoned after two CUDA OOM crashes at epoch 2 — a co-tenant process permanently holds ~34.5 GiB on the shared H100, leaving insufficient headroom for the larger model. V26c (V26a + R-Drop + label smoothing + slerp MixCo + stronger regularization, same 675M architecture) yielded 53.5% / 69.6% CSLS — marginal (+0.7pp raw, 0.0pp CSLS). The 69.6% CSLS ceiling appears hard for ViT-L/14 token targets. Kappa collapsed to ~1.54 (zero confidence differentiation); V26d (kappa_reg disabled) yielded 54.2% raw (+0.7pp) but **66.8% CSLS (-2.8pp)** — kappa_reg was helping CSLS, and the 69.6% ceiling is a fundamental ViT-L/14 representation limit, not a kappa issue. **V27a** migrates to **ViT-bigG/14** (LAION-2B, 257×1280 tokens, ~825M params) to break this ceiling; cache build + training pending. All experiments run on an **NVIDIA H100 80GB HBM3** with bf16 mixed precision.

**SOTA target:** MindEye achieves 93.2% R@1 on the same dataset. The remaining gap is attributed to (1) single-subject vs 7-subject pre-training, (2) smaller model capacity (328M vs 996M params), (3) MindEye's OpenCLIP ViT-bigG/14 embeddings (257×1280-D projected) vs our ViT-L/14 (257×768-D) — **V27a directly addresses this by switching to bigG**, and (4) hubness in high-dimensional retrieval from single-trial fMRI noise.

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
| v15 | N1v15-N4v15 | **Fix fundamentals**: PCR re-enabled, batch 256, z-scoring bug fix, loss simplification | N1: 44%, CSLS 51% |
| v16 | N1v16-N4v16 | Rep-avg, low-reg, focused loss (MSE+NCE only) | ~39% REGRESSION |
| v17 | N1v17-N4v17 | Restore V7/V8 baseline + CSLS + shared1000 + diagnostics | N1: 47%, **CSLS 56.3%** |
| v18 | N1v18-N4v18 | V17 + MSE + PCR + shared1000 z-score fix | N1: 46%, CSLS 52.3% (regression) |
| v19 | N1v19-N4v19 | V17 + CosFace margin + DirectAlign + ROI dropout | N1: 38%, CSLS 49.1% (regression) |
| v20 | N1v20-N4v20 | V17 + sequential MixCo->SoftCLIP + batch 128 | Not evaluated (superseded by V21) |
| v21 | N1v21-N4v21 | V17 + residual_mlp + vmf_nll + ISF + no kappa_reg + 3×LR | N1: 34%, CSLS 44% (**regression**); N2-N4: 0.1% (collapsed) |
| v22 | N1v22, N1v22b | V17 + MSE(sum, w=1.0) — regression test | N1v22: 38.4% raw, CSLS 49.9% (**regression**); MSE dominated 83% gradient |
| v23 | N1v23a–d | 4 isolated ablations vs V17 control (N1v23d) | **N1v23a ep51: 47.6% raw, CSLS 59.0%** (+2.7pp, running); N1v23b: 43.4%/57.0% (+0.7pp); N1v23c: 38.4%/49.9% (MSE collapse) |
| v24 | N1v24, N1v24b, N1v24c, N1v24d | N1v23a + N1v23b + hard negatives (w=0.3) | **REGRESSION**: N1v24: 55.4%; N1v24b: 57.6% (best); N1v24c: 57.2%; N1v24d: 55.4%. Hard neg raised neg_sim 0.274→0.324 |
| v25 | N1v25_rerun, N1v25a, N1v25b, N2v25c | 3 structural experiments: sequential schedule, cross-subject adapters, sub-ROI patching | N1v25_rerun: 57.2%/52.5%; N1v25a: 54.9%/50.8%; N2v25c: 46.2%/45.1%; N1v25b: **crashed** (fix applied) |
| v26a | N1v26a | MindEye-style 257×768 token targets (single-subject, 675M params) | N1v26a: **52.8%** raw, **69.6%** CSLS (+17.1pp CSLS vs V23a); best epoch 134/174 — **new project best** |
| v26b | N1v26b | Token targets + cross-subject adapters (4 subjects, 1.59B params) | **Abandoned** — CUDA OOM at epoch 2 (shared H100, co-tenant uses ~34.5 GiB; 256 MiB `exp_avg_sq_sqrt` alloc fails) |
| v26c | N1v26c | V26a + R-Drop (w=0.3, start ep50) + label smooth 0.05 + slerp MixCo + dropout 0.2/0.25 + queue 8192 + 350 epochs | N1v26c: 53.5% raw, 69.6% CSLS (+0.7pp/+0.0pp vs V26a); best epoch 116/166; kappa collapsed at ~1.54 |
| v26d | N1v26d | V26c + kappa_reg disabled (unlock kappa in 197K-D token space) | N1v26d: 54.2% raw (+0.7pp), **66.8% CSLS (-2.8pp)** vs V26c; kappa 1.54→2.5; best epoch 49/99 — **kappa_reg removal hurt CSLS** |
| **v27a** | **N1v27a** | **ViT-bigG/14** 257×1280 token targets (LAION-2B), V26a recipe, queue 2048, ~825M params | *Pending* — cache build + training required |

Notes:
- B-series stays at v4 (not affected by vMF-specific changes)
- v6+ only apply to N-series experiments
- v8 only had N3 and N4 configs (N1/N2 skipped that iteration)
- v10 combined best of V8 (losses) and V9 (eval tricks)
- v27 switches CLIP backbone from ViT-L/14 → ViT-bigG/14 (LAION-2B); requires separate token cache
- v11 attempted hubness mitigation and denoising but failed due to `average_repetitions: true` reducing data 3x
- v12 two-stage never activated for N1/N2 (early stopping before epoch 120); auto-weighting destroyed N3/N4
- v13 adds MSE regression (MindEye1's key ingredient) and hierarchical CLIP alignment for N3/N4
- v14 attacks the 8-10 pp hubness gap with three training-time mechanisms + CSLS-based checkpoint selection
- v15 fixes three fundamental bottlenecks: z-scoring bug for multi-subject, disabled CLIP PCR, small in-batch size
- v16 stripped too many losses (SoftCLIP, MixCo, kappa_reg) and over-reduced regularization
- v17 faithfully restored V7/V8 proven recipe; achieved **project-best 56.3% CSLS R@1** on N1 (confirmed reproducible by v23d)
- v18 additions (MSE, PCR, uniformity) were all net-negative; MSE only 0.008% of gradient
- v19 DirectAlignmentLoss at weight 2.0 caused embedding space collapse (pos/neg sim both spiked)
- v20 tests MindEye-style sequential scheduling (never tested on a clean baseline in 15 versions)
- v21 worst regression: vmf_nll at d=768 adds ~40K constant to loss, drowning vMF-NCE signal (~7.5)
- v22 root cause: MSE(sum, w=1.0) ≈ 87 at epoch 1 = 83% of gradient; same structural failure as vmf_nll
- v23c root cause: MSE at any weight on L2-normalized embeddings causes pos_sim collapse to 0.800; MSE minimizes Euclidean distance, pulling predictions toward the CLIP embedding mean, not angular discriminability
- **v23a breakthrough**: wider [8192,8192,4096,2048] residual MLP hits 59.0% CSLS at epoch 51; first experiment to beat V17 in 6 versions
- v23b: CSLS training (use_csls_training=True) tested cleanly in isolation for first time; kappa 28→17 (correctly expresses more uncertainty); independent of v23a
- v24: combines v23a + v23b (two orthogonal improvements) + hard_negative_weight=0.3 to target separability improvement (sep 0.221 → >0.260)
- **v24 failure**: hard_negative_weight=0.3 + use_csls_training=true created gradient conflict — CSLS penalises hubs while hard negatives reweight loss toward those same items; neg_sim went UP from 0.274→0.324; all 4 variants regressed from V23a's 59.0%
- **v24d sequential contaminated**: V24d tested sequential scheduling but WITH hard negatives + CSLS training, masking the true sequential effect
- v25: pivots to structural changes (loss surgery exhausted); V25a sequential was -4.1pp worse than joint (curriculum hurts); V25c patched-ROI at 46.2% matched N2v17 (no gain from balanced tokens); V25b crashed at epoch 31 with optimizer `add_param_group` duplicate params — fixed with `_cs_backbone_frozen` guard + `_existing_param_ids` filter
- **v26a breakthrough**: pivot to MindEye-style token targets (257 tokens × 768-D ViT-L/14 patch embeddings projected to 768-D); token cache pre-built at `outputs/clip_cache/tokens_ViT-L-14_projected.h5` (37K images × 257 × 768); R@1 52.8%, CSLS 69.6% at epoch 134 — +17.1pp CSLS confirms target quality was the primary bottleneck after all loss-level exhaustion in v16–v25
- **v26b failure**: cross-subject token-target training (4 subjects, ~100K trials, 1.59B params); two consecutive CUDA OOM crashes at `torch.optim.adam._multi_tensor_adam` epoch 2 (256 MiB `exp_avg_sq_sqrt` allocation fails); root cause is a co-tenant process permanently occupying ~34.5 GiB on the shared H100 — effective budget ~44.5 GiB is insufficient for 1.59B param Adam optimizer; abandoned; `gc.collect() + torch.cuda.empty_cache()` end-of-epoch and batch 16→8 / queue 2048→512 reductions did not resolve
- v26c: config-only improvements on V26a (same 675M param architecture, no cross-subject adapters); R-Drop (w=0.3, active from epoch 50), label smoothing 0.05, slerp MixCo, encoder dropout 0.2 / decoder dropout 0.25, queue 8192, 350 epochs, patience 50; **result: 53.5% / 69.6% CSLS — marginal** (+0.7pp raw, +0.0pp CSLS vs V26a); converged earlier (epoch 116 vs 134) but hit identical CSLS ceiling; kappa collapsed to ~1.54 (kappa_std=0.057) — kappa_reg's constant 0.01 downward gradient dominates the weak contrastive signal in 197K-D space (pos_sim≈0.16)
- v26d: single-variable ablation — V26c with kappa_reg.enabled=false; kappa rose modestly (1.54→2.5, not the 5–50 expected) but **CSLS dropped 2.8pp** (66.8% vs 69.6%); raw R@1 improved +0.7pp (54.2%); converged much faster (best epoch 49 vs 116) suggesting less regularization → earlier overfitting; kappa_reg was actually helping CSLS by dampening hub formation; the 69.6% CSLS ceiling is **not** caused by kappa collapse — it is a fundamental limit of ViT-L/14 token representation quality or single-subject data volume

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

### 19.4b V16 Actual Results (subj01)

V16 achieved **~39% R@1**, a significant regression from V8's 50.8%. The aggressive simplification went too far:

| Root Cause | V8 (51%) | V16 (39%) | Impact |
|------------|----------|-----------|--------|
| SoftCLIP | weight 1.0 | disabled | Lost soft-label contrastive signal |
| MixCo | weight 0.5 | disabled | Lost embedding-space augmentation |
| kappa_reg | 0.01 | disabled | Kappa unconstrained |
| Regularization | dropout 0.2, wd 0.05, noise 0.1 | dropout 0.05, wd 0.005, no noise | Too little for 9k samples |
| Queue | 16384 | 8192 | Fewer negatives |
| Rep averaging | false (~27k trials) | true (~9k) | 3x data reduction |
| MSE loss | none | weight 2.0 | Conflicts with contrastive on unit sphere |

**Conclusion:** V16's hypothesis that the model was underfitting was incorrect. The V8 loss recipe (SoftCLIP + MixCo + kappa_reg + proper regularization + all 27k trials) was essential. V17 reverts to the proven V7/V8 baseline.

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
| **Ours (V16+)** | **shared1000** | **~982 images** | CLIP ViT-L/14 (768-D) |

**Note on comparability:** Our shared1000 R@1 is directly comparable in gallery size and protocol. The CLIP model difference (ViT-L/14 vs bigG) means absolute R@1 numbers are not strictly apples-to-apples with MindEye2, but are comparable with Brain Diffuser and other ViT-L/14-based methods.

**Config flag:** `evaluation.eval_shared1000: true` (default true in V16+ configs). Set to `false` to skip.

### 19.6 Version Genealogy

```
V4 (fix 6 bugs) -> V5 (tau=1, no kappa_reg)
  -> V6 (novel losses) -> V7 (multi-subj, softplus)  <-- N1 best: ~49%
    -> V8 (hierarchical CLIP)                         <-- N3/N4 best: 50.3%/50.8%
      -> V9 (anti-overfit, H100) -> V10 (wider + hard neg + soup)
        -> V11 (CSLS training) -> V12 (two-stage, auto-weight)
          -> V13 (MSE regression) -> V14 (anti-hubness + diagnostics)
            -> V15 (PCR + batch + z-score fix)
              -> V16 (rep-avg + low-reg + focused loss + shared1000 eval) = 39% REGRESSION
    -> V17 (restore V7/V8 baseline + CSLS + shared1000) = 47% N1, 40% N2-N4
      -> V18 (fix shared1000 bug + MSE + PCR + kappa cap)  <-- CURRENT
```

---

## 20. V17: Restore V7/V8 Baseline + CSLS + Shared1000 + Diagnostics

### 20.1 Rationale

After V8 achieved the project's best R@1 (50.8% for N4v8), nine subsequent versions (V9-V16) attempted various improvements but none exceeded V8. V16's aggressive simplification (stripping SoftCLIP, MixCo, kappa_reg, increasing dropout to 0.05, enabling repetition averaging) caused a regression to ~39% R@1.

V17 takes a fundamentally different approach: **faithfully restore the proven V7/V8 recipe** and add only non-invasive evaluation/diagnostic capabilities from later versions.

### 20.2 Strategy

V17 configs are near-exact copies of V7 (for N1/N2) and V8 (for N3/N4), with only evaluation and diagnostic additions:

**Training recipe (unchanged from V7/V8):**
- Losses: vMF-NCE (1.0) + SoftCLIP (1.0) + kappa_reg (0.01) + MixCo (0.5)
- N3/N4 add: hierarchical_clip (0.3) + CKA (0.5) + vmf_nce_multitask
- Regularization: dropout 0.2, weight_decay 0.05, fmri_noise_std 0.1, voxel_dropout 0.1
- Queue: 16384
- Data: average_repetitions: false (all ~27k trials)
- No MSE, no PCR, no two-stage, no projection head, no R-Drop, no label smoothing

**Evaluation additions (no impact on training):**
- `eval_shared1000: true` -- community benchmark for SOTA comparison
- `use_csls: true`, `csls_k: 10` -- hubness-corrected retrieval metrics
- `mc_tta_samples: 8`, `kappa_weighted_avg: true` -- MC-Dropout TTA
- `checkpoint_metric: "r@1"` -- early-stop on raw R@1
- `train_r1_interval: 10` -- training R@1 monitoring for overfit detection
- `early_stop_min_delta: 0.002` -- prevents noise from resetting patience
- `mixed_precision_dtype: "bf16"` -- H100 optimization

### 20.3 V17 Config Summary

| Config | Base | Encoder | Expected R@1 |
|--------|------|---------|-------------|
| N1v17 | N1v7 | MLP [8192, 4096, 2048] | ~49% |
| N2v17 | N2v7 | ROI Transformer (d=768, 6L, FFN=3072) | ~48% |
| N3v17 | N3v8 | ROI-DCF (d=768, 6L, FFN=3072) | ~50% |
| N4v17 | N4v8 | ROI-DCF + SPCL (d=768, 6L, FFN=3072) | ~51% |

### 20.4 V17 Actual Results (subj01)

V17 did NOT reproduce V8-level results. N1 was close to V7, but N2/N3/N4 were ~10pp below V8.

**Validation diagnostics (900-image gallery):**

| Metric | N1v17 | N2v17 | N3v17 | N4v17 |
|--------|-------|-------|-------|-------|
| Raw R@1 | 47.0% | 40.3% | 40.1% | 40.6% |
| CSLS R@1 | 56.3% | 46.9% | 47.0% | 46.3% |
| Hubness gap | 9.3pp | 6.6pp | 6.9pp | 5.8pp |
| Pos sim (mean) | 0.495 | 0.388 | 0.501 | 0.529 |
| Neg sim (mean) | 0.274 | 0.165 | 0.291 | 0.331 |
| Separability | 0.221 | 0.223 | 0.210 | 0.198 |
| Kappa (mean/std) | 28.2/2.6 | 41.8/4.5 | 60.3/6.5 | 54.5/5.5 |

**Shared1000 benchmark (1000-image gallery):**

| Metric | N1v17 | N2v17 | N3v17 | N4v17 |
|--------|-------|-------|-------|-------|
| R@1 | 45.2% | 6.2% | 6.1% | 6.8% |
| CSLS R@1 | 52.1% | 13.3% | 12.6% | 12.8% |
| Mean pos sim | 0.387 | 0.229 | 0.359 | 0.322 |

### 20.5 V17 Post-Mortem: Three Root Causes

**Bug: Shared1000 z-scoring filename mismatch.** Multi-subject training saves per-session z-score stats as `{subj}_session_{sess}_mean.npy`, but `_evaluate_shared1000()` looked for `session_{sess}_mean.npy` (without subject prefix). This caused N2/N3/N4 shared1000 features to be evaluated WITHOUT z-scoring, producing garbage 6% R@1. N1 (single-subject) used the non-prefixed filename and worked correctly. **Fixed in V18.**

**Premature early-stopping.** `early_stop_min_delta: 0.002` (added in V17) stopped training when R@1 improvements fell below 0.2pp per patience window. Multi-subject ROI Transformer models converge slowly, and this truncated N3/N4 training prematurely. This partly explains the 40% vs V8's 50%.

**Missing regression loss.** MindEye1 achieves 93.2% R@1 using 50% MSE + 50% SoftCLIP. Our models rely entirely on contrastive losses (vMF-NCE + SoftCLIP), which learn relative ordering but not absolute coordinates. This creates the "contrastive plateau" where the correct image is nearby but not rank-1. Separability of 0.20-0.22 confirms the core bottleneck.

---

## 21. V18: Bug Fix + MSE Loss + PCR Anti-Hubness

### 21.1 Rationale

V17 diagnostics revealed three actionable bottlenecks: a shared1000 evaluation bug, premature early-stopping, and the absence of a direct regression signal. V18 addresses all three while keeping the proven V7/V8 training recipe intact.

### 21.2 Changes from V17

| Change | Scope | Rationale |
|--------|-------|-----------|
| Fix shared1000 z-score filenames | Code (train_unified.py) | N2/N3/N4 shared1000 was broken |
| Remove `early_stop_min_delta` | All V18 configs | Was 0.002, caused premature stopping |
| Add MSE loss (weight 1.0) | All V18 configs | MindEye-proven; pushes from relative to absolute coordinate alignment |
| Enable PCR (k=4) | All V18 configs | Reduces hubness by removing dominant PCs from CLIP space |
| Kappa cap = 50 | N3v18, N4v18 only | V17 kappa 60.3/54.5 indicated overconfidence |

### 21.3 MSE on the Unit Hypersphere

MSE between L2-normalized vectors equals `2(1 - cos_sim)`, so it directly maximizes cosine similarity to the correct target while being agnostic to negatives. Combined with vMF-NCE (which pushes negatives apart), this creates both attractive and repulsive forces -- the full recipe that MindEye1 proved essential for high R@1.

### 21.4 V18 Config Summary

| Config | Base | Key Additions | Expected Impact |
|--------|------|---------------|----------------|
| N1v18 | N1v17 | MSE + PCR | 47% -> 60-65% |
| N2v18 | N2v17 | MSE + PCR | 40% -> 50-55% |
| N3v18 | N3v17 | MSE + PCR + kappa_max=50 | 40% -> 55-60% |
| N4v18 | N4v17 | MSE + PCR + kappa_max=50 | 40% -> 55-65% |

### 21.5 V18 Actual Results

| Metric | N1v18 | N2v18 | N3v18 | N4v18 |
|--------|-------|-------|-------|-------|
| Val Raw R@1 | 46.0% | 40.6% | 39.6% | 40.4% |
| Val CSLS R@1 | 52.3% | 46.0% | 44.9% | 46.2% |
| S1000 Raw R@1 | 39.3% | 30.9% | 33.0% | 34.9% |
| S1000 CSLS R@1 | 46.8% | 36.0% | 40.1% | 40.7% |
| Separability | 0.287 | 0.288 | 0.276 | 0.282 |
| Kappa mean/std | 15.6/0.4 | 18.9/0.7 | 29.8/0.8 | 31.5/1.1 |
| Hubness gap | 6.3pp | 5.4pp | 5.3pp | 5.8pp |

### 21.6 V18 Post-Mortem: Regression from V17

V18 **regressed** from V17 despite identical model capacity. N1 dropped from 56.3% to 52.3% CSLS R@1 (val), and from 52.1% to 46.8% (shared1000). The z-score fix helped N2-N4 on shared1000 (from catastrophic 6% to 36-41%), but the three new additions (MSE, PCR, uniformity) were net negative.

**Root cause analysis from N3v18 loss breakdown:**

| Loss | Contribution | % of Total |
|------|-------------|------------|
| vmf_nce_multitask | 11.3 | 43.1% |
| vmf_nce | 7.5 | 28.6% |
| softclip | 5.6 | 21.5% |
| mixco | 1.5 | 5.6% |
| hier_clip + cka | 0.5 | 2.0% |
| MSE | 0.002 | 0.008% |
| uniformity | -0.3 | -1.3% |
| kappa_reg | 0.15 | 0.6% |

- **MSE contributed 0.008%** of gradient. On 768-D unit-norm vectors, `nn.MSELoss(reduction='mean')` produces values ~0.002, completely drowned out by contrastive losses (~7.5). Dead weight.
- **PCR removed useful discriminative variance.** The top principal components of CLIP embeddings carry the most discriminative semantic information.
- **Uniformity** blindly pushed apart semantically similar embeddings.
- **vmf_nce_multitask dominated at 43%** — mt_aux forces V1/V2 ROIs to predict global semantics, creating contradictory gradients.

---

## 22. V19: Revert V18 Regressions + Margin + DirectAlign + ROI Dropout

### 22.1 Rationale

V18 proved that MSE, PCR, and uniformity were counterproductive. V19 returns to the V17 foundation (best: 56.3% CSLS R@1) and makes three surgical additions: CosFace angular margin, DirectAlignmentLoss (properly scaled 1 - cos_sim), and ROI-Token Dropout. For N3/N4, the loss landscape is drastically simplified by disabling mt_aux, hierarchical_clip, and cka.

### 22.2 Changes from V17

| Change | Scope | Rationale |
|--------|-------|-----------|
| Add CosFace margin (0.15) | N1, N2, N3 vmf_nce | Carves strict angular boundaries on the hypersphere; already implemented in vmf_nce.py |
| Add DirectAlignmentLoss (w=2.0) | All V19 | `1 - cos_sim`, range [0,2]; properly scaled unlike MSE (0.002). Operates on raw decoder output |
| Add ROI-Token Dropout (10%) | N2, N3, N4 | Randomly zeros ROI tokens during training; forces Transformer robustness |
| Disable mt_aux (lambda_aux=0.0) | N3, N4 | Contributed 43% of loss but forced V1/V2 to predict global semantics |
| Disable hier_clip | N3, N4 | Contributed only 1% of total loss |
| Disable cka | N3, N4 | Contributed only 1% of total loss |
| Remove early_stop_min_delta | All V19 | Prevented convergence in V17/V18 |
| Kappa-weighted Frechet mean | Shared1000 eval | Zero-cost: fuse 3 repetitions using model confidence |
| Keep kappa curriculum (10->50) | N3, N4 | Prevents early overconfidence |

### 22.3 V19 Loss Landscape

| Config | Active Losses | Margin |
|--------|--------------|--------|
| N1v19 | vMF-NCE + SoftCLIP + MixCo + DirectAlign + kappa_reg | 0.15 |
| N2v19 | same as N1 | 0.15 |
| N3v19 | vMF-NCE + SoftCLIP + MixCo + DirectAlign + mt_fused + kappa_reg | 0.15 |
| N4v19 | vMF-NCE-SPCL + SoftCLIP + MixCo + DirectAlign + mt_fused + kappa_reg | (SPCL, no margin) |

### 22.4 Version Genealogy (Updated)

```
V4 (fix 6 bugs) -> V5 (tau=1, no kappa_reg)
  -> V6 (novel losses) -> V7 (multi-subj, softplus)  <-- N1 best: ~49%
    -> V8 (hierarchical CLIP)                         <-- N3/N4 best: 50.3%/50.8%
      -> V9 (anti-overfit, H100) -> V10 (wider + hard neg + soup)
        -> V11 (CSLS training) -> V12 (two-stage, auto-weight)
          -> V13 (MSE regression) -> V14 (anti-hubness + diagnostics)
            -> V15 (PCR + batch + z-score fix)
              -> V16 (rep-avg + low-reg + focused loss + shared1000 eval) = 39% REGRESSION
    -> V17 (restore V7/V8 baseline + CSLS + shared1000) = 56.3% N1 CSLS, 47% N2-N4 CSLS
      -> V18 (fix s1000 bug + MSE + PCR) = 52.3% N1 CSLS (REGRESSION)
        -> V19 (revert V18 regressions + margin + DirectAlign + ROI dropout) = 49.1% N1 CSLS (REGRESSION)
          -> V20 (V17 base + sequential MixCo->SoftCLIP + batch 128)  <-- CURRENT
```

---

## 23. V19 Actual Results and Post-Mortem

### 23.1 V19 Actual Results (subj01)

**Validation diagnostics (900-image gallery):**

| Metric | N1v19 | N2v19 | N3v19 | N4v19 |
|--------|-------|-------|-------|-------|
| Raw R@1 | 38.1% | 36.8% | 36.6% | 36.9% |
| CSLS R@1 | 49.1% | 44.7% | 45.3% | 44.6% |
| Hubness gap | 11.0pp | 7.9pp | 8.8pp | 7.7pp |
| Pos sim (mean) | 0.800 | 0.760 | 0.738 | 0.747 |
| Neg sim (mean) | 0.616 | 0.573 | 0.550 | 0.564 |
| Separability | 0.185 | 0.187 | 0.188 | 0.183 |
| Kappa (mean/std) | 32.3/4.5 | 62.4/4.0 | 48.5/0.3 | 48.5/0.5 |

**Shared1000 benchmark (1000-image gallery):**

| Metric | N1v19 | N2v19 | N3v19 | N4v19 |
|--------|-------|-------|-------|-------|
| R@1 | 40.5% | 36.2% | 35.3% | 37.3% |
| CSLS R@1 | 48.3% | 41.8% | 41.7% | 41.8% |
| Mean pos sim | 0.744 | 0.732 | 0.725 | 0.678 |

### 23.2 V19 Post-Mortem: Embedding Space Collapse

V19 showed **severe regression** from V17 across all metrics:

| Metric | V17 (best) | V18 | V19 | V19 Δ from V17 |
|--------|-----------|-----|-----|----------------|
| N1 Val CSLS R@1 | **56.3%** | 52.3% | 49.1% | **-7.2pp** |
| N1 S1000 CSLS R@1 | **52.1%** | 46.8% | 48.3% | **-3.8pp** |
| N1 Separability | 0.221 | 0.287 | 0.185 | **-0.036** |
| N1 Pos sim | 0.495 | 0.309 | 0.800 | +0.305 |
| N1 Neg sim | 0.274 | 0.022 | 0.616 | +0.342 |
| N1 Hubness gap | 9.3pp | 6.3pp | 11.0pp | +1.7pp |

**Root cause: DirectAlignmentLoss at weight 2.0 caused embedding space collapse.**

DirectAlignmentLoss (`1 - cos_sim`) at weight 2.0 aggressively forced predictions toward targets. Both positive AND negative cosine similarities spiked (pos: 0.49 -> 0.80, neg: 0.27 -> 0.62). The net separability plummeted from 0.22 to 0.18. Instead of making the model more discriminative, DirectAlign collapsed the embedding space -- all predictions moved toward a common region, making it harder to distinguish the correct image.

**Secondary cause: CosFace margin compounded the issue.** The angular margin subtracted from positive logits while the embedding space was already collapsing, further degrading discriminability.

**Conclusion from V17-V19:** Every loss addition since V17 has hurt performance. The V17 contrastive-only recipe (vMF-NCE + SoftCLIP + MixCo + kappa_reg) is the proven optimal loss landscape.

---

## 24. V20: MindEye-Style Sequential Loss Scheduling + Larger Batch

### 24.1 The Key Insight

Across V5 through V19 (15 versions), `softclip_from_start: true` has been set in **every** experiment except V16 (which confounded the scheduling change with 6 other drastic changes). The MindEye1 recipe -- which achieves 84% R@1 -- uses **sequential** BiMixCo -> SoftCLIP scheduling. Our codebase already implements this scheduling when `softclip_from_start: false`, but it has **never been tested in isolation on a clean baseline**.

### 24.2 V20 Strategy: Two Config-Only Changes on V17

**Change 1 -- Sequential Loss Scheduling (PRIMARY):**

Set `softclip_from_start: false`. This activates the existing code path in `train_unified.py`:
- **Phase 1 (epochs 1-67):** MixCo only + vMF-NCE + kappa_reg. Builds a noise-resistant manifold via interpolated soft-label contrastive training.
- **Phase 2 (epochs 68-200):** SoftCLIP only + vMF-NCE + kappa_reg. Refines the manifold using CLIP inter-similarity soft labels.

The transition happens at `num_epochs/3 + 1 = 68` for 200-epoch training.

**Why simultaneous scheduling hurts:** MixCo creates blurred interpolated samples (noise-resistant but fuzzy), while SoftCLIP uses exact CLIP similarities (sharp discrimination). Running both simultaneously means MixCo's "blur" fights SoftCLIP's "sharpen" in every batch, creating gradient conflict. Sequential scheduling lets each do its job properly.

**Change 2 -- Larger In-Batch Size (SECONDARY):**

Change from `batch_size: 64, gradient_accumulation_steps: 4` to `batch_size: 128, gradient_accumulation_steps: 2`.

Same effective batch (256), but 2x more **fresh in-batch negatives** per contrastive step (128 vs 64). Gradient accumulation provides stale micro-batch negatives that are less informative. On H100 80GB, batch 128 is trivially feasible for all model types.

### 24.3 What NOT to Change (learned from V17-V19)

- NO regression loss (MSE, DirectAlign -- both proven harmful)
- NO PCR preprocessing (removes useful variance)
- NO uniformity regularization (blind repulsion hurts)
- NO CosFace margin (compounds embedding collapse)
- NO ROI-token dropout (V19 showed no benefit)
- NO kappa curriculum (N3/N4 just saturate at cap)
- NO encoder width changes (keep V17 architecture exact)
- NO loss simplification for N3/N4 (V17's full loss set > V19's simplified)

### 24.4 Bug Fixes Retained from V18/V19

- Shared1000 z-score filename fix (already in codebase)
- `early_stop_min_delta` removed (already in codebase)
- Kappa-weighted Frechet mean for shared1000 eval (already in codebase from V19)

### 24.5 V20 Config Summary

| Config | Base | Key Changes | Expected Impact |
|--------|------|-------------|----------------|
| N1v20 | N1v17 | `softclip_from_start: false`, `batch_size: 128`, `grad_accum: 2` | 56% -> 60-67% CSLS R@1 |
| N2v20 | N2v17 | Same | 47% -> 50-55% |
| N3v20 | N3v17 | Same | 47% -> 51-58% |
| N4v20 | N4v17 | Same | 46% -> 51-58% |

### 24.6 Run Commands

```bash
nohup make ablation SUBJECTS="subj01" GPU=0 ONLY=N20 SAVE_CKPT=best > ablation_v20.log 2>&1 &
tail -f ablation_v20.log
```

---

## 25. V21 Results and Post-Mortem

### 25.1 V21 Design (Summary)

V21 attempted five simultaneous changes from V17:
1. `encoder_type: "residual_mlp"` (MindEye-style: project once + 4 residual blocks)
2. `learnable_temperature: false` (let kappa be sole temperature)
3. `vmf_nll` auxiliary loss (weight 0.1) for kappa calibration
4. `isf_weight: 0.3` (inverted softmax anti-hubness)
5. `batch_size: 128, grad_accum: 8` (effective 1024, 4× V17) + `lr: 2e-4` (3× V17)
6. `kappa_reg: disabled`

### 25.2 V21 Actual Results (subj01)

**Validation diagnostics (900-image gallery):**

| Metric | N1v21 | N2v21 | N3v21 | N4v21 |
|--------|-------|-------|-------|-------|
| Raw R@1 | 33.2% | 0.1% | 0.1% | 0.1% |
| CSLS R@1 | 47.2% | 0.1% | 0.1% | 0.1% |
| Hubness gap | 14.0pp | 0.0pp | — | — |
| Pos sim (mean) | 0.821 | 0.734 | NaN | NaN |
| Neg sim (mean) | 0.646 | 0.733 | NaN | NaN |
| Separability | 0.175 | 0.001 | NaN | NaN |
| Kappa (mean/std) | 50.0/0.0 | 137.4/0.2 | — | — |

**Shared1000 benchmark (1000-image gallery):**

| Metric | N1v21 | N2v21 | N3v21 | N4v21 |
|--------|-------|-------|-------|-------|
| R@1 | 34.0% | 0.1% | 0.1% | 0.1% |
| CSLS R@1 | 44.0% | 0.1% | 0.1% | 0.1% |
| Mean pos sim | 0.819 | -0.009 | NaN | NaN |

### 25.3 V21 Root Cause Analysis

**N1v21 regressed from 56.3% to 44.0% CSLS R@1 (−12.3pp):**

| Root Cause | Evidence | Impact |
|------------|----------|--------|
| vmf_nll (weight 0.1) | $-\log C_d(\kappa)$ at $d=768$ adds ~40,000 baseline. $0.1 \times 40{,}000 = 4{,}000$ effective loss vs vMF-NCE at ~7.5. Dominated training (same failure as V12). | Critical: drowned contrastive signal |
| ISF at 0.3 | V14 tested ISF: no R@1 improvement. Column-wise softmax adds noise without reducing hubness. | Moderate: net-negative |
| kappa_reg disabled | V16 showed kappa becomes unconstrained without regularization. N1v21 kappa collapsed to constant 50.0±0.0 (no uncertainty signal). | Moderate: lost calibration |
| 3× LR (2e-4 vs 7e-5) | Untested with V17 recipe. May cause overshooting with vmf_nll's enormous gradients. | Unknown (confounded) |
| residual_mlp encoder | Never tested in isolation. Different architecture than proven V17 MLP. | Unknown (confounded) |

**N2/N3/N4 collapsed to chance (0.1% R@1):**

vmf_nll's ~4,000 effective loss completely overwhelmed all contrastive losses for the ROI Transformer models. N2v21 shows separability of 0.001 (positive and negative cosine similarities are identical at ~0.73) — the model learned nothing. N3v21/N4v21 produced NaN predictions, indicating numerical failure from the Bessel function computation in the vMF normalizing constant.

**Critical lesson:** vmf_nll has now destroyed training in **V12** (auto-weight gave it 50% share) and **V21** (explicit 0.1 weight still produced 4,000 loss). The normalizing constant's ~40,000 baseline makes it fundamentally incompatible with multi-loss training at $d=768$.

### 25.4 V21 Version Genealogy Update

```
V17 (restore V7/V8 baseline) = 56.3% N1 CSLS  <-- PROJECT BEST
  -> V18 (MSE mean + PCR) = 52.3% (REGRESSION)
    -> V19 (DirectAlign + margin) = 49.1% (REGRESSION)
  -> V20 (sequential MixCo→SoftCLIP + batch 128) — pending
  -> V21 (residual_mlp + vmf_nll + ISF + no kappa_reg + 3×LR) = 44.0% (REGRESSION)
    N2-N4: collapsed to 0.1% (vmf_nll numerical failure)
```

---

## 26. V22: Properly-Scaled MSE Regression (The One Missing Piece)

### 26.1 The Core Insight: MSE Was Never Properly Tested

Across V13, V18, and the entire 22-version history, MSE loss has been tested using `nn.MSELoss(reduction='mean')`. On 768-D L2-normalized vectors:

$$\text{MSE}_{\text{mean}} = \frac{1}{768} \sum_{d=1}^{768} (\mu_d - z_d)^2 = \frac{2(1 - \cos(\mu, z))}{768} \approx 0.002$$

This is 0.008% of the total gradient (V18 loss breakdown confirmed). **MindEye's MSE is not this.** MindEye uses a per-sample MSE loss that is comparable in magnitude to the contrastive loss.

V22 uses `nn.MSELoss(reduction='sum')`:

$$\text{MSE}_{\text{sum}} = \sum_{d=1}^{768} (\mu_d - z_d)^2 = 2(1 - \cos(\mu, z)) \approx 1.5$$

At weight 1.0, this is 1.5 vs vMF-NCE's 7.5 — MSE contributes ~17% of the total gradient. This is a meaningful regression signal that pushes each prediction toward its absolute target coordinate, complementing the contrastive loss's relative positioning.

### 26.2 Code Change

In `scripts/training/train_unified.py`, the MSE loss setup was:
```python
losses["mse"] = nn.MSELoss()  # reduction='mean' by default → 0.002 per sample
```

Changed to:
```python
mse_reduction = loss_cfg["mse"].get("reduction", "mean")  # backward-compatible
losses["mse"] = nn.MSELoss(reduction=mse_reduction)
```

The `reduction` parameter is now configurable from YAML.

### 26.3 V22 Strategy: One Variable at a Time

V22 creates two N1 configs to isolate effects:

**N1v22 (PRIMARY): V17 + MSE(sum) only**
- Identical to N1v17 except: `loss.mse.enabled: true, weight: 1.0, reduction: "sum"`
- Removes `early_stop_min_delta: 0.002` (caused premature stopping)
- No encoder changes, no scheduling changes, no ISF, no vmf_nll
- Isolates the impact of properly-scaled MSE

**N1v22b (SECONDARY): V17 + MSE(sum) + sequential scheduling + batch 128**
- Same as N1v22 plus: `softclip_from_start: false` (sequential MixCo→SoftCLIP)
- `batch_size: 128, grad_accum: 2` (2× in-batch negatives, same effective 256)
- Tests all remaining MindEye techniques combined

**N2v22, N3v22, N4v22:** V17 bases + MSE(sum) + `early_stop_min_delta` removed. N3/N4 additionally reduce `lambda_aux` from 0.3-0.5 to 0.05 (V18 showed mt_aux at 0.5 contributed 43% of loss with contradictory V1/V2 gradients).

### 26.4 What NOT to Change (Lessons from V9-V21)

| Intervention | Versions Tested | Result | V22 Status |
|---|---|---|---|
| vmf_nll | V12, V21 | 40K baseline destroys training | **NEVER ENABLE** |
| ISF | V14, V21 | No R@1 improvement | Disabled |
| PCR | V15, V18 | Removes useful variance | Disabled |
| DirectAlignmentLoss | V19 | Embedding collapse (sep 0.22→0.18) | Disabled |
| CosFace margin | V19 | Compounds collapse | Disabled |
| MSE (mean reduction) | V13, V18 | 0.008% of gradient (dead weight) | **Fixed: sum reduction** |
| Disabled kappa_reg | V5-V6, V16, V21 | Kappa unconstrained | Enabled (0.01) |
| Disabled SoftCLIP | V16 | −11pp regression | Enabled |
| Disabled MixCo | V16 | Part of −11pp regression | Enabled |
| residual_mlp encoder | V21 | Confounded with other failures | Not tested (too risky) |
| Uniformity | V18 | Negative contribution | Disabled |

### 26.5 V22 Config Summary

| Config | Base | Key Additions | Expected CSLS R@1 |
|--------|------|---------------|-------------------|
| N1v22 | N1v17 | MSE(sum, w=1.0) | 56% → 62-68% |
| N1v22b | N1v17 | MSE(sum) + sequential + batch 128 | 56% → 64-72% |
| N2v22 | N2v17 | MSE(sum, w=1.0) | 47% → 52-58% |
| N3v22 | N3v17 | MSE(sum) + mt_aux 0.05 | 47% → 53-60% |
| N4v22 | N4v17 | MSE(sum) + mt_aux 0.05 | 46% → 53-60% |

### 26.6 Verification Checklist

1. **Epoch 1 loss magnitude:** MSE should be ~1.0-2.0 per step (not 0.002). If still ~0.002, the `reduction: "sum"` config is not flowing through.
2. **Loss proportion:** MSE should be 10-20% of total loss. Check with `mse / (mse + vmf_nce + softclip + mixco)`.
3. **Val R@1 trajectory:** V17 peaked at ~epoch 70-90. V22 should show higher R@1 starting from epoch 40-50.
4. **Diagnostics:** Run `diagnose_embeddings.py` — target pos_sim > 0.55 (vs 0.495 in V17), separability > 0.25.
5. **Shared1000:** Target R@1 ≥ 50%, CSLS R@1 ≥ 65%.

### 26.7 Contingency: MSE Weight Tuning

If N1v22 overshoots (MSE dominates → CSLS gap shrinks but raw R@1 drops):
- Create N1v22c with `mse.weight: 0.5`

If N1v22 undershoots (minimal improvement):
- Create N1v22c with `mse.weight: 2.0` or try `reduction: "sum"` divided by batch size via a custom wrapper

### 26.8 Run Commands

```bash
# Primary experiment (N1v22 — isolates MSE impact)
nohup make ablation SUBJECTS="subj01" GPU=0 ONLY=N1v22 SAVE_CKPT=best > ablation_v22.log 2>&1 &
tail -f ablation_v22.log

# After N1v22 completes, run N1v22b (adds sequential scheduling + batch 128)
nohup make ablation SUBJECTS="subj01" GPU=0 ONLY=N1v22b SAVE_CKPT=best > ablation_v22b.log 2>&1 &

# Once N1 results confirm MSE helps, run N2-N4
for exp in N2v22 N3v22 N4v22; do
    nohup make ablation SUBJECTS="subj01" GPU=0 ONLY=${exp} SAVE_CKPT=best > ablation_${exp}.log 2>&1 &
done

# Diagnostics after each run
for exp in N1v22_vmf_nce N1v22b_vmf_nce N2v22_roi_transformer N3v22_roi_dcf N4v22_full_system; do
    python3 scripts/evaluation/diagnose_embeddings.py \
        --results-dir experimental_results/${exp}/subj01
done

# Metrics comparison
for exp in N1v22_vmf_nce N1v22b_vmf_nce N2v22_roi_transformer N3v22_roi_dcf N4v22_full_system; do
    echo "=== ${exp} ==="
    cat experimental_results/${exp}/subj01/metrics/shared1000_metrics.json 2>/dev/null \
        || echo "  (not yet available)"
done
```

---

## Section 27: V23 Isolation Ablation Results (Complete)

### 27.1 Overview

V23 was the first version to test all changes **truly in isolation** against a verified V17 control (N1v23d). Four experiments ran:

| Exp | Description | Raw R@1 (900) | CSLS R@1 (900) | Raw R@1 (shared1000) | CSLS R@1 (shared1000) |
|-----|-------------|---------------|----------------|----------------------|-----------------------|
| N1v23d | V17 exact control | 47.0% | 56.3% | 45.5% | 53.4% |
| N1v23b | V17 + CSLS training (use_csls_training=True, csls_k=10) | 43.4% | 57.0% | 45.7% | 55.8% |
| N1v23c | V17 + MSE(sum, w=0.015) | 38.4% | 49.9% | — | — |
| N1v23a | Wider encoder [8192,8192,4096,2048], dropout 0.15, lr 5e-5, warmup 20 | 47.6% (ep51) | **59.0%** (ep51) | — | still running |

### 27.2 Key Findings

**N1v23d (Control):** V17 is **reproducible at 56.3% CSLS R@1**. This baseline is now verified. All future experiments have a clean reference point.

**N1v23a (Wider Encoder) — Project Best:**
- Wider MLP [8192,8192,4096,2048] (4 layers vs V17's 3) + lower dropout (0.15 vs 0.2) + lower LR (5e-5 vs 7e-5) + slower warmup (20 vs 15 epochs)
- At epoch 51: 47.6% raw / **59.0% CSLS** — beats V17 by **+2.7 pp**.
- Training log at epoch 51: `train_vmf_nce ≈ 9.3`, kappa ≈ 24.5, sep ≈ 0.231. Still converging.
- Wider network provides additional representation capacity; slower learning rate prevents overshooting.

**N1v23b (CSLS Training) — Independent Improvement:**
- Adding `use_csls_training: true` to V17 unchanged encoder yields **+0.7 pp CSLS** (57.0% vs 56.3%).
- Kappa dropped 28 → 17: model correctly expresses more uncertainty when CSLS penalizes hubness.
- neg_sim stable (0.270 → 0.268). Effect is orthogonal to encoder width.

**N1v23c (MSE w=0.015) — Closed Door:**
- Even at 7.8% gradient weight (calibrated via epoch-1 diagnostics), MSE caused **pos_sim collapse** to 0.800 (vs V17's 0.495).
- Root cause: MSE on L2-normalized embeddings minimizes Euclidean distance, pulling predictions toward the CLIP embedding mean. Every prediction becomes similar to everything → angular discriminability destroyed.
- **Conclusion: MSE is permanently incompatible with L2-normalized CLIP embeddings. Never re-introduce MSE in any form.**

### 27.3 Diagnostic Signals at Epoch 51 (N1v23a)

```
train_vmf_nce:     9.29   (healthy; V17 baseline ~9.1–9.5)
train_softclip:    2.31
train_mixco:       0.42
train_kappa_reg:   0.017
total_loss:        12.04
kappa (mean):      24.5
pos_sim:           0.498  (healthy; V17 = 0.495)
neg_sim:           0.271  (slight improvement vs V17 0.274)
sep:               0.228  (gap still target of V24)
hubness gap:       9.1 pp (residual)
```

### 27.4 Implications for V24

- N1v23a and N1v23b are **orthogonal improvements** (encoder width vs training loss). Combining them is expected additive.
- Remaining bottleneck: sep = 0.221–0.231, neg_sim = 0.271–0.274. Hard negatives (`hard_negative_weight=0.3, hard_neg_k=16`) directly target this.
- CSLS training reduces hubness during training (not just evaluation), complementing hard negatives.

---

## Section 28: V24 Combined Strategy (Target: CSLS 65–70%)

### 28.1 Hypothesis

V23 proved two orthogonal improvements:
1. Wider encoder (+2.7 pp CSLS)
2. CSLS training (+0.7 pp CSLS, better kappa calibration)

V24 combines both **plus** hard negatives to attack the remaining sep bottleneck. Expected trajectory from V17 (56.3%):
- Wider encoder: +2.7 pp → 59.0%
- CSLS training: +0.7 pp → 59.7%
- Hard negatives (sep improvement): +3–6 pp → **62.7–65.7%**

### 28.2 V24 Configurations

**N1v24_combined.yaml** — Primary experiment:
```yaml
# Wider encoder (from N1v23a)
hidden_dims: [8192, 8192, 4096, 2048]
dropout: 0.15
lr: 5.0e-5
warmup_epochs: 20

# CSLS training (from N1v23b)
use_csls_training: true
csls_k: 10

# Hard negatives (new in V24)
hard_negative_weight: 0.3
hard_neg_k: 16
```

**N1v24b_label_smooth.yaml** — Tests label smoothing:
- All of N1v24 + `label_smoothing: 0.05`
- Rationale: softens overconfident positives; may reduce kappa spike

**N1v24c_larger_batch.yaml** — Tests larger effective batch:
- All of N1v24 + `batch_size: 128`, `gradient_accumulation_steps: 2`
- More diverse negatives per step; effective contrastive batch = 256 in-batch + 16384 queue

### 28.3 Epoch-1 Sanity Checks for N1v24

Expected healthy ranges:
- `train_vmf_nce`: 9.0–11.0 (hard neg reweighting raises loss slightly)
- `pos_sim`: 0.48–0.52 (must NOT be >0.60 — would indicate collapse)
- `neg_sim`: 0.24–0.27 (lower than V17's 0.274 = good)
- `sep`: >0.200 (should start improving from epoch 1)
- `kappa`: 15–25 (lower than V17's 28 due to CSLS training)

If `train_vmf_nce` > 13.0 at epoch 1, reduce `hard_negative_weight` to 0.1. If `pos_sim` > 0.70, abort (MSE-like collapse).

### 28.4 Run Commands

```bash
# Launch V24 primary (N1)
nohup make ablation SUBJECTS="subj01" GPU=0 ONLY=N1v24 SAVE_CKPT=best > ablation_N1v24.log 2>&1 &
tail -f ablation_N1v24.log

# Monitor epoch 1 sanity
grep "epoch.*1\b\|vmf_nce\|pos_sim\|neg_sim\|kappa" ablation_N1v24.log | head -20

# After N1v24 confirms training health, launch variants
nohup make ablation SUBJECTS="subj01" GPU=0 ONLY=N1v24b SAVE_CKPT=best > ablation_N1v24b.log 2>&1 &
nohup make ablation SUBJECTS="subj01" GPU=0 ONLY=N1v24c SAVE_CKPT=best > ablation_N1v24c.log 2>&1 &
```

### 28.5 Success Criteria for V24

| Metric | V17 baseline | V24 target | V24 actual (best: N1v24b) |
|--------|-------------|------------|--------------------------|
| CSLS R@1 (900-way) | 56.3% | **≥65%** | 57.6% (**MISSED**) |
| CSLS R@1 (shared1000) | 53.4% | ≥62% | ~55% |
| sep | 0.221 | >0.260 | 0.205 (WORSE) |
| neg_sim | 0.274 | <0.260 | 0.324 (WORSE — opposite direction) |
| pos_sim | 0.495 | >0.510 | 0.529 |
| hubness gap | ~9 pp | <6 pp | ~10 pp |
| kappa (mean) | 28 | 15–22 | 18 |

**V24 Failure Post-Mortem:** All 4 variants regressed from V23a's 59.0% CSLS:
- N1v24 (combined): 55.4% — hard negatives + CSLS training conflict raised neg_sim
- N1v24b (label smooth): 57.6% — best V24 variant, label smoothing partially mitigated damage
- N1v24c (larger batch): 57.2% — more negatives didn't help when neg_sim already too high
- N1v24d (sequential): 55.4% — sequential scheduling contaminated by hard negatives

**Root Cause:** `hard_negative_weight=0.3` is fundamentally incompatible with `use_csls_training=true`. CSLS training penalises hub embeddings (those with high avg similarity to many neighbors), while hard negative mining reweights loss toward the hardest negatives (which are exactly the hubs). These opposing signals cancel out, and the net effect is increased neg_sim instead of decreased.

**Hard negatives permanently banned** from future experiments. Loss surgery is exhausted — V25 pivots to structural changes.

If N1v24 reaches 65%+, immediately launch N2v24, N3v24, N4v24 for all-subject evaluation.

### 28.6 V24d — Sequential Scheduling Variant

`N1v24d_sequential.yaml` adds one change to N1v24_combined: `softclip_from_start: false`.

This activates the hardcoded 1/3 phase boundary in `train_unified.py`:
- **Epochs 1–67**: MixCo only — interpolated soft-labels build noise-robust hypersphere geometry
- **Epoch 68+**: SoftCLIP only, MixCo disabled — CLIP semantic topology locks in as distillation target

Rationale: simultaneous MixCo + SoftCLIP creates a gradient conflict — blurred interpolated pairs (MixCo) and sharp semantic targets (SoftCLIP) pull the encoder in opposite directions every step. Sequential isolation was the MindEye1 approach and was planned in V20/V22b but **never cleanly evaluated in 24 versions**.

Verification at epoch 1: log must show `"SoftCLIP + MixCo schedule: MixCo epochs 1-67"` and `train_softclip` must be 0.0.
Verification at epoch 68: log must show `"Epoch 68: switching from MixCo to SoftCLIP"` and `train_mixco` must drop to 0.0.

```bash
nohup make ablation SUBJECTS="subj01" GPU=0 ONLY=N1v24d SAVE_CKPT=best > ablation_N1v24d.log 2>&1 &
grep "MixCo\|SoftCLIP\|switching" ablation_N1v24d.log
```

---

## Section 29: V25 Structural Experiments (Pivot from Loss Surgery)

### 29.1 V25 Motivation

After V24's regression, loss surgery is exhausted as a lever. The N1 MLP ceiling is ~59% CSLS (V23a). The three remaining bottlenecks are:
1. **Training data volume**: subj01 alone has ~9,000 unique images (~24K trials). MindEye uses 7 subjects × 10K images = 70K unique.
2. **MixCo/SoftCLIP interaction**: never tested in clean isolation (V24d was contaminated by hard negatives)
3. **ROI token imbalance**: nsdgeneral_other (9,374 voxels) is a single token alongside FFA1 (200 voxels) — a 47× size ratio

V25 addresses all three with isolated structural experiments.

### 29.2 V25 Experiment Map

| ID | Config | Change from V23a | Hypothesis |
|----|--------|-------------------|-----------|
| N1v25_rerun | N1v25_v23a_rerun.yaml | None (exact reproduction) | Produce checkpoint_best.pt for V25b |
| N1v25a | N1v25a_sequential.yaml | `softclip_from_start: false` | Clean sequential test: MixCo ep1-67 → SoftCLIP ep68-200 |
| N1v25b | N1v25b_cross_subject.yaml | Cross-subject linear adapters (4 subjects) | 4× data volume → +3-8pp CSLS |
| N2v25c | N2v25c_patched_roi.yaml | Sub-ROI patching (250 vox/token), 8 layers | Balanced tokens → +3-7pp CSLS |

### 29.3 N1v25_v23a_rerun — Checkpoint Reproduction

Exact copy of N1v23a_wider_encoder.yaml. Produces `checkpoint_best.pt` that V25b needs as pretrained backbone. Must run first.

### 29.4 N1v25a — Sequential MixCo→SoftCLIP Schedule

**Single change:** `softclip_from_start: false` (was `true` in V23a).

This activates the hardcoded 1/3 phase boundary in `train_unified.py`:
- **Epochs 1–67**: MixCo only — builds coarse hypersphere geometry with interpolated soft-labels
- **Epochs 68–200**: SoftCLIP only — refines with CLIP semantic topology distillation

V24d tested this but with hard_negative_weight=0.3 + use_csls_training=true contaminating the signal. V25a tests it cleanly on the proven V23a base.

**Expected:** +1-3pp if curriculum helps vs joint training. Risk: SoftCLIP starting at ep68 may not converge in remaining 133 epochs.

### 29.5 N1v25b — Cross-Subject Linear Adapters

**Architecture:**
```
subj02 fMRI (B, ~14000) → nn.Linear(14000, 15724, bias=False) → [V23a MLP backbone]
subj05 fMRI (B, ~13500) → nn.Linear(13500, 15724, bias=False) → [V23a MLP backbone]
subj07 fMRI (B, ~12800) → nn.Linear(12800, 15724, bias=False) → [V23a MLP backbone]
subj01 fMRI (B, 15724)  → identity pass-through              → [V23a MLP backbone]
```

**Training protocol:**
- **Phase 1 (epochs 1-30):** Backbone FROZEN (encoder+decoder from V23a rerun checkpoint). Only adapter `nn.Linear` layers train at full LR (5e-5). Adapters learn inter-subject voxel correspondence.
- **Phase 2 (epochs 31-200):** Backbone UNFROZEN at 0.1× base LR (5e-6). Adapters continue at full LR. Joint fine-tuning on all 4 subjects (~100K trials).

**Code changes implemented:**
- `UnifiedModel.__init__`: `self.subject_adapters = nn.ModuleDict({subj: nn.Linear(...)})` when `cross_subject.enabled`
- `UnifiedModel.forward`: routes non-canonical subjects through adapters before encoder
- `train_unified.py`: loads pretrained checkpoint, freezes backbone, creates optimizer with `requires_grad` filter only, unfreezes at `freeze_epochs+1`
- Multi-subject dataset loading triggered by `cross_subject.enabled` (not just `multi_subject_roi_transformer`)

**Adapter param count:** 3 adapters × ~15K × ~14K ≈ 634M params (bf16 = ~1.3GB, trivial on H100 80GB).

**Expected:** +3-8pp CSLS from 4× data volume. Linear adapters assume approximate functional correspondence across subjects (supported by neuroscience — shared representational geometry in visual cortex).

### 29.6 N2v25c — Sub-ROI Patched Transformer

**Problem:** V17's ROI Transformer has 17 tokens with a 47× voxel count imbalance (nsdgeneral_other: 9,374 vs FFA1: 200). The attention mechanism cannot weight these fairly.

**Solution:** `subdivide_rois(roi_dims, roi_indices, max_voxels_per_token=250)` splits any ROI > 250 voxels into uniform chunks via `np.array_split`. Result: ~80-100 balanced tokens instead of 17 imbalanced ones.

**Code changes implemented:**
- `roi_utils.py`: new `subdivide_rois()` function
- `train_unified.py`: if `encoder.roi_patch_size` is set in config, calls `subdivide_rois()` after `build_roi_index()`
- N2v25c config: `roi_patch_size: 250`, `num_layers: 8` (deeper for longer sequence)

**Expected:** +3-7pp from balanced tokenization. The Transformer can now discover finer spatial structure within large ROIs like nsdgeneral_other.

### 29.7 V25 Run Commands

```bash
# === STEP 1: V23a rerun (produces checkpoint for V25b) ===
nohup make ablation SUBJECTS="subj01" GPU=0 ONLY=N1v25_rerun SAVE_CKPT=best \
  > ablation_N1v25_rerun.log 2>&1 &
tail -f ablation_N1v25_rerun.log

# === STEP 2: V25a sequential (independent of Step 1, can run in parallel on another GPU) ===
nohup make ablation SUBJECTS="subj01" GPU=0 ONLY=N1v25a SAVE_CKPT=best \
  > ablation_N1v25a.log 2>&1 &

# === STEP 3: V25c patched ROI (independent, can run in parallel on another GPU) ===
nohup make ablation SUBJECTS="subj01" GPU=0 ONLY=N2v25c SAVE_CKPT=best \
  > ablation_N2v25c.log 2>&1 &

# === STEP 4: V25b cross-subject (REQUIRES Step 1 checkpoint — wait for rerun to complete) ===
# Verify checkpoint exists first:
ls -la experimental_results/N1v25_v23a_rerun/subj01/checkpoints/checkpoint_best.pt
nohup make ablation SUBJECTS="subj01" GPU=0 ONLY=N1v25b SAVE_CKPT=best \
  > ablation_N1v25b.log 2>&1 &
```

**Single-GPU sequential order** (if only 1 GPU available):
```bash
# Run all 4 experiments sequentially on one GPU
make ablation SUBJECTS="subj01" GPU=0 ONLY=N1v25_rerun SAVE_CKPT=best \
  && make ablation SUBJECTS="subj01" GPU=0 ONLY=N1v25a SAVE_CKPT=best \
  && make ablation SUBJECTS="subj01" GPU=0 ONLY=N1v25b SAVE_CKPT=best \
  && make ablation SUBJECTS="subj01" GPU=0 ONLY=N2v25c SAVE_CKPT=best
```

### 29.8 V25 Sanity Checks

**N1v25_rerun:** Should reproduce V23a's ~59% CSLS R@1 within ±1pp. If not, seed sensitivity is a concern.

**N1v25a:** Epoch 1 log must show `train_softclip: 0.0` and MixCo active. At epoch 68, log must show SoftCLIP switching on and MixCo off.

**N1v25b:** Epoch 1 log must show `"Cross-subject freeze: X params frozen"` and only adapter params training. At epoch 31, log must show `"[CROSS-SUBJECT] Unfreezing backbone"`. Verify with:
```bash
grep -E "frozen|unfreez|CROSS-SUBJECT" ablation_N1v25b.log
```

**N2v25c:** Log must show `"Sub-ROI patching: 17 ROIs -> N tokens (max 250 vox/token)"` where N ≈ 80-100.

### 29.9 V25 Success Criteria

| Experiment | Metric | Target | **Actual** | Verdict |
|-----------|--------|--------|-----------|--------|
| N1v25_rerun | CSLS R@1 (val900) | 58-60% | **57.2%** | -1.8pp from V23a; seed/eval variance |
| N1v25a | CSLS R@1 (val900) | ≥60% | **54.9%** | **FAIL** — sequential -4.1pp worse than joint |
| N1v25b | CSLS R@1 (val900) | ≥63% | **CRASHED** | optimizer `add_param_group` bug; fix applied, pending re-run |
| N2v25c | CSLS R@1 (val900) | ≥55% | **46.2%** | **FAIL** — matched N2v17 (46.3%), no gain from balanced tokens |

**Shared1000 results (full benchmark):**

| Experiment | R@1 | CSLS R@1 | R@5 | R@10 | MRR | Pos sim |
|-----------|-----|----------|-----|------|-----|---------|
| N1v25_rerun | 43.7% | 52.5% | 75.9% | 86.2% | 0.580 | 0.439 |
| N1v25a | 45.4% | 50.8% | 76.9% | 87.3% | 0.594 | 0.351 |
| N2v25c | 39.8% | 45.1% | 72.0% | 82.9% | 0.543 | 0.346 |

**Val900 diagnostics:**

| Experiment | Raw R@1 | CSLS R@1 | Hub gap | Pos sim | Neg sim | Sep | Kappa |
|-----------|---------|----------|---------|---------|---------|-----|-------|
| N1v25_rerun | 46.3% | 57.2% | 10.9% | 0.514 | 0.293 | 0.221 | 30.6±2.6 |
| N1v25a | 45.6% | 54.9% | 9.3% | 0.483 | 0.267 | 0.216 | 27.7±2.4 |
| N2v25c | 39.6% | 46.2% | 6.7% | 0.343 | 0.125 | 0.218 | 52.9±5.2 |

### 29.9.1 V25 Failure Post-Mortem

**N1v25_rerun (57.2% vs expected 59.0%):** The 1.8pp drop is within random seed variance but notable. The shared1000 result (52.5%) is significantly below val900 (57.2%), consistent with all prior experiments showing val900 overestimates by 3-5pp.

**N1v25a sequential (54.9%, -4.1pp):** Clean evidence that **joint MixCo+SoftCLIP training is better than sequential**. The MixCo-only phase (ep1-67) builds a coarse geometry but SoftCLIP starting at ep68 cannot fully recover — it has 133 fewer epochs of SoftCLIP signal. Lower kappa (27.7 vs 30.6) indicates less confident predictions. This confirms V23a's `softclip_from_start: true` is the correct default.

**N2v25c patched-ROI (46.2%):** Matched N2v17 (46.3%) exactly — sub-ROI patching did NOT help. The Transformer with 69 balanced tokens performs identically to 16 imbalanced tokens. This suggests the ROI Transformer's limitation is not token imbalance but rather the architecture itself — the MLP (59%) fundamentally outperforms the Transformer (46%) on this task. Possible reasons: (a) 27K training trials insufficient for attention to learn spatial relationships, (b) per-ROI linear projections lose information that the MLP's full-vector processing retains. Notably, N2v25c has very high kappa (52.9 vs 30.6 for MLP) suggesting overconfidence in wrong predictions.

**N1v25b (CRASHED):** The optimizer crash occurred at epoch 31 (unfreeze point). Root cause: when checkpoint loaded and backbone frozen, the optimizer was created with only `requires_grad=True` params (adapters). At unfreeze, `add_param_group` tried to add backbone params, but some were already tracked. **Fix applied**: (1) `_cs_backbone_frozen` flag tracks whether freeze actually happened, (2) unfreeze block filters `_existing_param_ids` before calling `add_param_group`.

If N1v25b reaches ≥65% after re-run, proceed to full evaluation on shared1000.

---

### 29.10 Forward Roadmap — V26

### V26: 3D Spatial Patch Tokenization (Architecture Research Track)

**Target:** Replace the rigid anatomical ROI tokenization (N2–N4) with uniform 4×4×4 voxel patches, following the ViT conceptual leap from CNNs.

**Problem it solves:** In `N2v17_roi_transformer.yaml`, `nsdgeneral_other: 9374` voxels are projected into the same token dimension as `FFA1: 200`. This creates a 47× imbalance — the attention mechanism cannot weight these tokens fairly, and spatial locality is completely lost.

**Architecture:** 
- Unmask 1D array back into native 3D beta volume (81×104×83)
- Apply `nn.Conv3d(1, d_model, kernel_size=4, stride=4)` → ~500 patch tokens of uniform spatial coverage
- Standard Transformer backbone (12 layers, 768 d_model, 12 heads) processes patch tokens
- Per-subject: voxel counts differ, but 3D brain volume is the same space — cross-subject alignment is natural

**Prerequisites (not yet met):**
1. Raw NSD beta NIfTI files accessible (currently pipeline only stores 1D `.npy` via `make preextract`)
2. Brain mask NIfTI per subject (81×104×83 binary) to unmask/remask
3. New `BrainPatchDataset` replacing `fmri_features.npy` with 3D volumes or a masking utility
4. New `BrainPatchEncoder(nn.Module)` with `Conv3d` patch embedder
5. Cold-start from scratch — no warm-start from V17/V24 (different input shape)

**Implementation estimate:** 2–3 weeks (data pipeline + architecture + hyperparameter sweep). Do not begin until V25 adapter results are known — V26 is only worthwhile if the N1 MLP ceiling is confirmed below 70%.

**Expected gain:** If successful, V26 closes the last 3–7 pp gap to state-of-the-art MindEye2 (66–72% on NSD). The attention mechanism's ability to recover spatial patterns that the MLP loses in the 1D flattening step is the key hypothesis.

---

## 30. Formal Verification Appendix (Lean 4)

### 30.1 Scope and Motivation

Three mathematical claims in the thesis are load-bearing enough, and crisp enough, to admit machine-checked proofs. The proofs live in `lean/FMRIDecoding/Certificates.lean` and are verified by the Lean 4 kernel against Mathlib4 v4.14.0. A Lean proof is not an informal argument — it is a type-checked term; correctness is unconditional.

The scope is deliberately narrow. The Mathlib4 library (as of 2026) does not contain:
- The modified Bessel function $I_\nu$, blocking a formal proof of the vMF density or its normalizing constant $C_d(\kappa)$.
- The NCE mutual information bound (Poole et al., 2019), which requires measure-theoretic mutual information primitives that are sparse in Mathlib.
- Anything about training dynamics, convergence, or gradient interaction — these are empirical phenomena outside the scope of any proof assistant.

What Mathlib *does* contain, and what we exploit: `Real.exp`, `Real.log`, their derivatives and algebraic identities; `Finset.sum` and its interaction with multiplication; `InnerProductSpace` including the Cauchy-Schwarz inequality via `abs_inner_le_norm`; `EuclideanSpace ℝ (Fin d)` as the ambient space for L2-normalised CLIP embeddings.

### 30.2 Certificate 1 — Kappa Collapse Arithmetic Bound

**Claim (Thesis Section 13.3):** With $\tau = 0.07$ and AMP clamp $[-80, 80]$, any $\kappa > 5.6$ saturates the positive logit, zeroing $\partial \mathcal{L} / \partial \kappa$.

The bound reduces to $\kappa / 0.07 > 80 \iff \kappa > 5.6$. The Lean statement:

```lean
theorem kappa_collapse_ceiling (κ : ℝ) (hκ : κ > 5.6) : κ / 0.07 > 80
theorem kappa_ceil_tight : (5.6 : ℝ) / 0.07 = 80
theorem kappa_fix_clears_ceiling (κ : ℝ) (hκ : κ ≤ 50) : κ / 1.0 ≤ 80
```

`kappa_collapse_ceiling` is proved by `ring`-rewriting to `(κ − 5.6) / 0.07 > 0` then `div_pos`.  
`kappa_ceil_tight` is closed by `norm_num` (rational arithmetic decision procedure).  
`kappa_fix_clears_ceiling` certifies that the v5 fix ($\tau = 1.0$, `kappa_max = 50`) clears the ceiling: `linarith` suffices.

### 30.3 Certificate 2 — Bessel-Free Normalizer Cancellation

**Claim (Thesis §2.3):** The vMF log-partition $\log C_d(\kappa_i)$ is constant across all gallery items $z_j$ (it depends only on $\kappa_i$, not $z_j$), and therefore cancels out of the NCE softmax. The implemented loss is mathematically equivalent to the full vMF-NCE loss — no Bessel function need ever be computed.

Formally, for any $f : \text{Fin}\, n \to \mathbb{R}$ and constant $c$:

$$\log \frac{e^{f(i) + c}}{\sum_j e^{f(j) + c}} = \log \frac{e^{f(i)}}{\sum_j e^{f(j)}}$$

```lean
theorem softmax_const_cancel {n : ℕ} (f : Fin n → ℝ) (c : ℝ) (i : Fin n) :
    Real.log (Real.exp (f i + c) / ∑ j : Fin n, Real.exp (f j + c)) =
    Real.log (Real.exp (f i)     / ∑ j : Fin n, Real.exp (f j))
```

Proof strategy: factor $e^c$ out of the sum via `simp_rw [Real.exp_add]` and `Finset.mul_sum`, then apply `field_simp [hec, hsum]` + `ring` where `hec : exp c ≠ 0` and `hsum : Σ_j exp(f j) ≠ 0` (positivity from `i : Fin n`).

A second theorem `vmf_partition_cancels_in_nce` instantiates this with `c := log_Cd` and `f j := κ · μᵀzⱼ`, making the architectural justification explicit.

### 30.4 Certificate 3 — Cosine Similarity Bound on $S^{d-1}$

**Claim:** For any two L2-normalised vectors $\mu, z \in S^{d-1}$, $|\langle \mu, z \rangle| \leq 1$.

This is the Cauchy-Schwarz inequality on unit vectors. It is geometrically obvious but formally certifying it pins down the premise on which the clamp analysis in §30.2 is tight.

```lean
theorem cosine_bounded {d : ℕ} (μ z : EuclideanSpace ℝ (Fin d))
    (hμ : ‖μ‖ = 1) (hz : ‖z‖ = 1) : |⟪μ, z⟫_ℝ| ≤ 1
```

Proved in three lines via `abs_inner_le_norm` (Cauchy-Schwarz) followed by norm substitution.

A directional corollary `logit_bound` establishes $|\kappa \langle \mu, z \rangle| \leq \kappa$, confirming that the raw NCE logit range is $[-\kappa, \kappa]$ and the clamp $\pm 80$ is only active when $\kappa / \tau > 80$.

### 30.5 Building the Proofs

```bash
cd lean
# First build: downloads Mathlib (~30 min, cached thereafter)
lake exe cache get
lake build
```

Expected output: `Build completed successfully` with no warnings. Each `theorem` is independently checkable with `#check @kappa_collapse_ceiling`.

### 30.6 Mathlib Coverage Assessment

| Claim | Lean status | Blocking gap |
|---|---|---|
| Kappa collapse bound | ✅ Fully proved | — |
| Normalizer cancellation | ✅ Fully proved | — |
| Cosine bound on S^{d-1} | ✅ Fully proved | — |
| vMF density (full) | ❌ | Modified Bessel $I_\nu$ absent from Mathlib |
| NCE ≥ mutual information | ❌ | Donsker-Varadhan variational formula sparse in Mathlib |
| Training convergence | ❌ | Empirical; outside proof assistant scope |
| CSLS correctness | ❌ | Empirical heuristic; no formal statement in literature |

---

## 31. V27: ViT-bigG/14 Migration

### 31.1 Motivation

V26a–d exhausted config-level optimization at 69.6% CSLS R@1. Three ablations (R-Drop, label smoothing, kappa_reg removal) yielded marginal or negative delta. The ceiling is a fundamental **ViT-L/14 representation limit**: 768-D projected tokens don't carry enough information for single-subject retrieval to match MindEye's 93.2%.

MindEye1 uses OpenCLIP ViT-bigG/14 (LAION-2B) with 1280-D projected tokens. Switching to bigG is the single highest-leverage change remaining.

### 31.2 Architecture Changes

| Dimension | V26a (ViT-L/14) | V27a (ViT-bigG/14) | Ratio |
|---|---|---|---|
| Token dim (projected) | 768 | 1280 | 1.67× |
| Flat output dim | 197,376 | 328,960 | 1.67× |
| mu_head params | 404M | 674M | 1.67× |
| Total model params | ~675M | ~825M | 1.22× |
| Queue entry size (fp32) | 790 KB | 1.3 MB | 1.67× |

### 31.3 Memory Budget (H100 80 GB, ~45.5 GiB usable)

| Component | Estimate |
|---|---|
| Model weights (bf16) | ~1.6 GB |
| AdamW optimizer states (fp32) | ~9.9 GB |
| Queue (2048 × 328960 × fp32) | ~2.6 GB |
| Activations (batch 24, 4 layers) | ~5.0 GB |
| **Total** | **~19.1 GB** |

Fits comfortably within the 45.5 GiB budget. Queue reduced from 4096 → 2048 to maintain headroom.

### 31.4 Code Changes

1. **`configs/system/clip_bigg.yaml`** — new CLIP system config (ViT-bigG-14, laion2b_s39b_b160k, 1280-D)
2. **`configs/experiments/N1v27a_bigg_tokens.yaml`** — V26a recipe adapted for bigG tokens
3. **`scripts/training/train_unified.py`** — fixed hardcoded `"ViT-L/14"` in shared1000 metrics; fixed vmf_nll dim default to auto-detect from decoder config
4. **`scripts/training/run_ablation_ladder.sh`** — added N1v27a to experiment order and configs
5. **`Makefile`** — added `bigg-token-cache` target and N1v27a help text

All core ML components (vmf_decoder, queue, embedding_eval, clip_utils) are **dimension-agnostic** — no changes needed.

### 31.5 Execution Plan

```bash
# Step 1: Build bigG token cache (~30-40 min on H100)
make bigg-token-cache

# Step 2: Train V27a
make ablation ONLY=N1v27a SUBJECTS=subj01 GPU=0
```

### 31.6 Key Monitoring Targets

- **CSLS R@1 > 69.6%** = bigG representation breaks ViT-L/14 ceiling
- **pos_sim** will be lower in 329K-D space (~0.10-0.14 vs 0.16 for V26a)
- **kappa** should be monitored for collapse (expect ~1.5–3.0 range)
- **GPU memory** should peak under 25 GiB
