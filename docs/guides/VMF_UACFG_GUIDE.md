# Von Mises-Fisher Decoder & Uncertainty-Aware CFG

This guide explains how to train a vMF model, calibrate kappa, and run
uncertainty-aware diffusion inference (UA-CFG).

---

## Overview

The standard Gaussian posterior is geometrically wrong for L2-normalised
CLIP embeddings on the unit hypersphere.  The **vMF (von Mises-Fisher)**
posterior is the natural distribution on S^{d-1}:

    p(z | mu, kappa) = C_d(kappa) * exp(kappa * mu^T z)

- **mu** (unit norm) is the mean direction
- **kappa** (positive scalar) is the concentration — higher means more
  confident

At inference time, kappa drives **per-sample adaptive guidance** in
Stable Diffusion:  confident samples get higher CFG scale and more
steps; uncertain samples get conservative guidance to avoid
hallucination.

---

## 1. Training: Gaussian baseline vs vMF

### Gaussian (existing, unchanged)

```bash
python scripts/training/train_unified.py \
    --config configs/experiments/exp4_gaussian_nce.yaml
```

### vMF (new)

```bash
python scripts/training/train_unified.py \
    --config configs/experiments/exp7_vmf_nce.yaml
```

Key config differences in `exp7_vmf_nce.yaml`:

```yaml
model:
  type: "vmf"
  posterior: "vmf"          # selects bounded-sigmoid decoder
  decoder:
    kappa_min: 0.001        # lower bound for concentration
    kappa_max: 500.0        # upper bound for concentration

loss:
  vmf_nce:
    enabled: true
    tau: 0.07               # temperature (same as InfoNCE)
    use_queue: true
  kappa_reg:
    enabled: true
    lambda_kappa: 0.01      # penalises unbounded kappa growth
```

During training, kappa statistics are logged every epoch:

- `kappa_mean`, `kappa_std`, `kappa_min`, `kappa_max`
- `kappa_q10`, `kappa_q50`, `kappa_q90`

Warnings fire automatically if kappa collapses (std < 0.01) or
saturates at the upper bound.

---

## 2. Kappa calibration

After training, compute kappa quantiles on the **validation set**:

```python
from fmri2img.eval.kappa_calibration import (
    compute_kappa_calibration,
    save_calibration,
)

cal = compute_kappa_calibration(model, val_loader, device)
save_calibration(cal, "experimental_results/exp7_vmf_nce/kappa_calibration.json")
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

## 3. Uncertainty-aware diffusion inference (UA-CFG)

### Fixed policy (baseline — identical to before)

```bash
python scripts/reconstruction/decode_diffusion.py \
    --encoder prob --ckpt path/to/checkpoint.pt \
    --inference-policy fixed \
    --guidance 7.5 --steps 50 \
    --clip-cache outputs/clip_cache/clip.parquet
```

### UA-CFG (dynamic guidance per sample)

```bash
python scripts/reconstruction/decode_diffusion.py \
    --encoder prob --ckpt path/to/checkpoint.pt \
    --inference-policy ua_cfg \
    --kappa-calibration path/to/kappa_calibration.json \
    --kappa-values path/to/kappa_per_sample.npy \
    --w-min 1.5 --w-max 12.0 --gamma 1.0 \
    --clip-cache outputs/clip_cache/clip.parquet
```

### UA-CFG + dynamic steps

```bash
python scripts/reconstruction/decode_diffusion.py \
    --encoder prob --ckpt path/to/checkpoint.pt \
    --inference-policy ua_cfg_steps \
    --kappa-calibration path/to/kappa_calibration.json \
    --kappa-values path/to/kappa_per_sample.npy \
    --w-min 1.5 --w-max 12.0 --gamma 1.0 \
    --steps-min 10 --steps-max 50 \
    --clip-cache outputs/clip_cache/clip.parquet
```

The mapping is:

```
kappa_norm = clamp((kappa - q10) / (q90 - q10), 0, 1)
guidance   = w_min + (kappa_norm ^ gamma) * (w_max - w_min)
steps      = steps_min + round(kappa_norm * (steps_max - steps_min))
```

Per-sample values are logged and saved to `risk_coverage.csv`.

---

## 4. Outputs

| File | Description |
|------|-------------|
| `kappa_calibration.json` | Quantile artifact (next to checkpoint) |
| `risk_coverage.csv` | Per-sample: nsd_id, kappa, guidance, steps, cosine |
| `triplets/triplet_info.json` | Highest and lowest kappa reconstructions |
| `decode_summary.json` | Full run metadata including inference policy |

---

## 5. Backward compatibility

- Old Gaussian configs (`exp0`–`exp6`) are completely untouched
- Old checkpoints with `log_kappa_min`/`log_kappa_max` load via the
  legacy decoder (`VonMisesFisherDecoderLegacy`)
- The `--inference-policy fixed` default reproduces the original
  fixed-guidance behaviour
- All new features are behind config flags

---

## 6. Running tests

```bash
pytest tests/test_vmf.py -v
```

Tests cover:
- Decoder: mu unit-norm, kappa bounds, gradient flow
- vMF-NCE: no NaN, no Bessel in logits, gradient flow, alignment
- Legacy decoder: backward compat with old configs
- Kappa calibration: normalise + save/load roundtrip
- CSLS + hubness: shape checks, self-retrieval sanity (requires sklearn)
