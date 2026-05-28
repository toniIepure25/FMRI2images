# NeuroBridge-OT: Architecture

## Data Flow

```
fMRI (B, V_subj) ─────────────────────────────┐
                                                │
Subject ID ──── SubjectFingerprint ──── HyperNetwork ──── Adapters
                     │                                       │
ROI Indices ────┐    │                                       │
                │    │                                       ▼
                ▼    ▼                                       │
          ROI Tokenizer (A) ◄───────────────────────────────┘
                │
         (B, n_rois, d_model)
                │
                ▼
    Optimal Transport Alignment (C)
    ┌─ Canonical Token Bank (K, d_model)
    ├─ Sinkhorn(cost, ε, iters)
    └─ Aligned Tokens (B, K, d_model)
                │
                ▼
    Shared Semantic Transformer (D)
    ┌─ [BRAIN_CLS] + Aligned Tokens
    ├─ Pre-Norm Transformer (N layers)
    └─ CLS Embedding + Contextualized ROI Tokens
                │
        ┌───────┴───────┐
        ▼               ▼
   CLS (B, d)    ROI (B, K, d)
        │               │
        ▼               ▼
  Decoding Heads (E)
  ├── CLIP Global (768-D normalized)
  ├── Token Target (257×768, optional)
  ├── vMF Head (mu + kappa)
  └── Calibration (scalar confidence)
```

## Module Specifications

### A. ROI Tokenizer

**Input**: `(B, V_subj)` raw fMRI + ROI indices mapping voxels to ROIs.

**Output**: `(B, n_rois, d_model)` ROI token tensor + `(B, n_rois)` validity mask.

**Variants**:
- `FixedROISummaryTokenizer`: Statistical pooling (mean/std/max) → linear projection.
- `LearnedROITokenizer`: Per-voxel learned embedding → attention pooling per ROI.
- `SubjectAdaptiveROITokenizer`: Learned + per-subject LoRA adapters.
- `HyperAdapterROITokenizer`: Learned + FiLM modulation from fingerprint.

### B. Subject Fingerprint

**Input**: Per-ROI statistics (voxel count, mean, std, tSNR, missing flag) + optional subject ID.

**Output**: `(B, fingerprint_dim)` compact vector.

**Key constraint**: For zero-shot protocols, fingerprint uses only unsupervised statistics.

### C. Optimal Transport Alignment

**Input**: ROI tokens `(B, M, d_model)` from tokenizer.

**Learnable**: Canonical token bank `(K, d_model)`.

**Algorithm**: Sinkhorn iterations on cost matrix (cosine or Euclidean distance).

**Output**: Aligned tokens `(B, K, d_model)`, transport matrix `(B, M, K)`, OT cost scalar.

**Ablation modes**: `identity` | `linear` | `ot` | `ot_prior`.

### D. Semantic Transformer

**Input**: [BRAIN_CLS] token + aligned ROI tokens.

**Architecture**: Pre-LayerNorm Transformer with stochastic depth.

**Output**: CLS embedding `(B, d_model)` + per-ROI contextualized tokens `(B, K, d_model)`.

### E. Multi-Target Decoding Heads

All heads operate on the CLS embedding (and optionally ROI tokens):
- **CLIP Global**: MLP → L2-normalize. Output: `(B, 768)`.
- **Token Target**: Cross-attention queries → project. Output: `(B, 257, 768)`.
- **vMF Uncertainty**: MLP → normalized mu + softplus kappa. Output: `(B, 768)` + `(B,)`.
- **Calibration**: MLP → scalar logit. Output: `(B,)`.

## Tensor Shapes (Full System, d_model=768, 17 ROIs)

| Tensor | Shape |
|---|---|
| Input fMRI (subj01) | (B, 15724) |
| ROI tokens | (B, 17, 768) |
| Canonical bank | (17, 768) |
| Transport matrix | (B, 17, 17) |
| Aligned tokens | (B, 17, 768) |
| Transformer input | (B, 18, 768) (CLS + 17 tokens) |
| CLS embedding | (B, 768) |
| CLIP prediction | (B, 768) |
| vMF mu | (B, 768) |
| vMF kappa | (B,) |

## Parameter Counts (Approximate)

| Component | Parameters |
|---|---|
| LearnedROITokenizer (17 ROIs) | ~1.5M |
| SubjectFingerprint | ~0.3M |
| OptimalTransportAlignment | ~0.6M |
| SemanticTransformer (6 layers) | ~28M |
| DecodingHeads (CLIP + vMF) | ~4.7M |
| HyperNetwork (FiLM) | ~2.8M |
| **Total (full system)** | **~38M** |
