# Von Mises-Fisher Decoder and Uncertainty-Aware CFG

Guide for training vMF models, calibrating kappa, and running
uncertainty-aware diffusion inference.

---

## Overview

The standard Gaussian posterior is geometrically wrong for L2-normalised
CLIP embeddings on the unit hypersphere. The **vMF (von Mises-Fisher)**
posterior is the natural distribution on S^{d-1}:

    p(z | mu, kappa) = C_d(kappa) * exp(kappa * mu^T z)

- **mu** (unit norm) is the mean direction
- **kappa** (positive scalar) is the concentration -- higher means more
  confident

At inference time, kappa drives **per-sample adaptive guidance** in
Stable Diffusion: confident samples get higher CFG scale and more
steps; uncertain samples get conservative guidance to avoid
hallucination.

---

## 1. Training: Gaussian Baseline vs vMF

### Gaussian Baseline (B1)

```bash
python3 scripts/training/train_unified.py \
    --config configs/experiments/B1v4_gaussian.yaml
```

### vMF (N1)

```bash
python3 scripts/training/train_unified.py \
    --config configs/experiments/N1v5_vmf_nce.yaml
```

Key config differences in the latest `N1v9_vmf_nce.yaml`:

```yaml
model:
  type: "vmf"
  posterior: "vmf"
  decoder:
    kappa_mode: "softplus"      # unbounded kappa via softplus(raw) + 1.0
  projection_head:
    enabled: true               # separate contrastive from retrieval
    hidden_dim: 2048
    out_dim: 768

loss:
  vmf_nce:
    enabled: true
    tau: 1.0                    # kappa IS the inverse temperature
    use_queue: true
    learnable_temperature: true
    label_smoothing: 0.1        # V9: soft targets
  kappa_reg:
    enabled: true
    lambda_kappa: 0.1           # V9: needed with softplus (unbounded kappa)
  r_drop:
    enabled: true               # V9: consistency between two dropout passes
    weight: 0.5
```

### Kappa Parameterization History

| Version | Kappa Mode | Range | Kappa Reg | Issue |
|---------|-----------|-------|-----------|-------|
| v4 | bounded sigmoid | [1, 50] | 0.05 | **Kappa collapse** (tau=0.07 capped at ~5.6) |
| v5 | bounded sigmoid | [1, 50] | Disabled | Fixed collapse (tau=1.0), but sigmoid saturated |
| v6 | bounded sigmoid | [1, 50] | Disabled | Kappa hit sigmoid ceiling (~50) |
| v7+ | **softplus** | [1, inf) | 0.01 | Unbounded, healthy gradient flow |
| v9 | **softplus** | [1, inf) | **0.1** | Stronger reg as anti-overfitting |

**Why tau=1.0 (kappa collapse fix, v5):** In the vMF-NCE loss, the logit
is `kappa * cos_sim / tau`. With `tau=0.07`, any `kappa > 5.6` hit the
float16 clamp at 80, zeroing the gradient. Combined with kappa_reg pushing
down, kappa was trapped at ~3-5 in 768-D space.

**Why softplus (v7+):** Even with tau=1.0, the bounded sigmoid
`kappa_min + (kappa_max - kappa_min) * sigmoid(raw)` had vanishing
gradients at its boundaries. Softplus (`softplus(raw) + 1.0`) has
everywhere-positive gradient, requiring explicit `kappa_reg` to prevent
unbounded growth.

**Why bf16 helps (v9):** bf16 has the same exponent range as fp32
(8-bit exponent), eliminating the float16 overflow risks that originally
caused the kappa collapse. The [-80, 80] clamp is essentially never
triggered with bf16.

During training, kappa statistics are logged every epoch:

- `kappa_mean`, `kappa_std`, `kappa_min`, `kappa_max`
- `kappa_q10`, `kappa_q50`, `kappa_q90`

Warnings fire automatically if kappa collapses (std < 0.01) or
saturates at the upper bound.

---

## 2. Full System: ROI-DCF with Dual Uncertainty

The full system (N3/N4) produces **two uncertainty signals**:

- **kappa** (concentration) -- within-region measurement noise (aleatoric)
- **delta** (disagreement) -- between-region directional conflict (epistemic-like)

```bash
python3 scripts/training/train_unified.py \
    --config configs/experiments/N4v5_full_system.yaml
```

These are used by the Decomposed UA-CFG (see Section 4).

---

## 3. Kappa Calibration

After training, compute kappa quantiles on the **validation set**:

```python
from fmri2img.eval.kappa_calibration import (
    compute_kappa_calibration,
    save_calibration,
)

cal = compute_kappa_calibration(model, val_loader, device)
save_calibration(cal, "experimental_results/N1v5_vmf_nce/subj01/kappa_calibration.json")
```

This produces a JSON file:

```json
{
  "q10": 12.3,
  "q50": 85.1,
  "q90": 340.5,
  "mean": 120.7,
  "std": 95.3,
  "min": 0.5,
  "max": 498.2,
  "n_samples": 4500
}
```

---

## 4. Uncertainty-Aware Diffusion Inference

### Fixed Policy (Baseline)

```bash
python3 scripts/reconstruction/decode_diffusion.py \
    --config configs/inference/production.yaml \
    --checkpoint experimental_results/N1v5_vmf_nce/subj01/checkpoint_best.pt \
    --subject subj01
```

### UA-CFG (Dynamic Guidance per Sample)

```bash
python3 scripts/reconstruction/decode_diffusion.py \
    --config configs/inference/production.yaml \
    --checkpoint experimental_results/N4v5_full_system/subj01/checkpoint_best.pt \
    --subject subj01 \
    --override "ua_cfg.enabled=true"
```

### Guidance Mapping

For single-uncertainty models (N1):

```
kappa_norm = clamp((kappa - q10) / (q90 - q10), 0, 1)
guidance   = w_min + (kappa_norm ^ gamma) * (w_max - w_min)
steps      = steps_min + round(kappa_norm * (steps_max - steps_min))
```

For dual-uncertainty models (N3, N4):

| kappa | delta | Interpretation | Strategy |
|-------|-------|----------------|----------|
| High | Low | Confident, regions agree | Strong CFG, few steps |
| Low | Low | Noisy but consistent | Moderate CFG |
| High | High | Confident but conflicting | Multiple mixture samples |
| Low | High | Everything uncertain | Abstain or conservative |

---

## 5. Outputs

| File | Description |
|------|-------------|
| `kappa_calibration.json` | Quantile artifact (next to checkpoint) |
| `risk_coverage.csv` | Per-sample: nsd_id, kappa, guidance, steps, cosine |
| `triplets/triplet_info.json` | Highest and lowest kappa reconstructions |
| `decode_summary.json` | Full run metadata including inference policy |

---

## 6. Experiment Progression

| Experiment | Distribution | Uncertainty | UA-CFG | V9 Additions |
|-----------|-------------|------------|--------|--------------|
| B0 | Deterministic | None | Fixed | -- |
| B1 | Gaussian | sigma (Euclidean) | Fixed | -- |
| **N1** | **vMF** | **kappa (softplus)** | kappa -> w | Proj head, R-Drop, CSLS, TTA |
| **N2** | **vMF** | **kappa (softplus)** | kappa -> w | DropPath, Proj head, R-Drop, CSLS, TTA |
| **N3** | **vMF-DCF** | **kappa + delta** | Decomposed | DropPath, Proj head, R-Drop, CSLS, TTA |
| **N4** | **vMF-DCF** | **kappa + delta** | Decomposed + Mixture | DropPath, Proj head, R-Drop, CSLS, TTA, SPCL |

---

## 7. Running Tests

```bash
pytest tests/test_vmf.py tests/test_vmf_mixture.py tests/test_decomposed_ua_cfg.py -v
```

Tests cover:
- Decoder: mu unit-norm, kappa bounds, gradient flow
- vMF-NCE: no NaN, no Bessel in logits, gradient flow
- Mixture sampling: concentration ordering, energy score
- Decomposed UA-CFG: monotonicity, boundary conditions
- Kappa calibration: normalize + save/load roundtrip
