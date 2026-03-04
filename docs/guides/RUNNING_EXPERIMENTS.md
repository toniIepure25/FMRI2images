# Running Experiments Guide

Complete guide for training, evaluating, and analyzing experiments.

---

## Quick Start

### Single Experiment

```bash
python3 scripts/training/train_unified.py \
    --config configs/experiments/N1v9_vmf_nce.yaml --gpu 0
```

### Full V9 Ablation (N-series only)

```bash
nohup make ablation SUBJECTS="subj01" GPU=0 ONLY=N9 > ablation_v9.log 2>&1 &
tail -f ablation_v9.log
```

The ladder runs: N1v9, N2v9, N3v9, N4v9 (in order). Each experiment is skipped if `checkpoint_best.pt` or `summary.json` already exists.

Results are saved to `experimental_results/{experiment_name}/{subject}/`.

---

## Experiment Configurations

### Current Ablation Ladder (V9 — Latest)

| ID | Config | Description | Novel? |
|----|--------|-------------|--------|
| **N1v9** | `N1v9_vmf_nce.yaml` | MLP + vMF-NCE + proj head + R-Drop + CSLS + TTA | Yes |
| **N2v9** | `N2v9_roi_transformer.yaml` | Multi-Subj ROI Transformer + DropPath + proj head + CSLS | Yes |
| **N3v9** | `N3v9_roi_dcf.yaml` | ROI-DCF + proj head + R-Drop + CSLS + MC-TTA | Yes |
| **N4v9** | `N4v9_full_system.yaml` | N3 + SPCL + DUA-CFG + MC-TTA (flagship) | Yes |

### Baseline Experiments (B-series, v4)

| ID | Config | Description |
|----|--------|-------------|
| **B0v4** | `B0v4_deterministic.yaml` | MLP + MSE + InfoNCE + SoftCLIP + MixCo |
| **B1v4** | `B1v4_gaussian.yaml` | MLP + Gaussian-NCE + KL + SoftCLIP + MixCo |

### Ablation Ladder Filters

```bash
make ablation SUBJECTS="subj01" GPU=0 ONLY=N9    # V9 N-series only
make ablation SUBJECTS="subj01" GPU=0 ONLY=N      # All N-series (v5-v9)
make ablation SUBJECTS="subj01" GPU=0 ONLY=B      # B-series only
make ablation SUBJECTS="subj01" GPU=0              # Everything
make ablation SUBJECTS="subj01" GPU=0 SAVE_CKPT=0 # Skip saving checkpoints
```

All configs are in `configs/experiments/`. Previous versions (v1-v8) are preserved for reproducibility.

---

## Training Workflow

### Step 1: Prepare Environment

```bash
# On JupyterHub (H100):
set -a && source .env && set +a
pip install -e ".[train,diffusion]"
make preflight
```

### Step 2: Pre-extract Features (First Time Only)

```bash
make preextract SUBJECT=subj01
```

Creates: `cache/preextracted/subject=subj01/fmri_features.npy` (~1.8 GB).

### Step 3: Build CLIP Cache (First Time Only)

```bash
# Standard single-layer cache:
make clip-cache

# Multi-layer cache (for fused targets, required by v7+ configs):
python3 scripts/build/build_multilayer_clip_cache.py \
    --subjects subj01 subj02 subj05 subj07 \
    --layers 12 18 24 --fuse-alpha 0.3
```

### Step 4: Run Training

**Option A: V9 ablation (recommended)**

```bash
nohup make ablation SUBJECTS="subj01" GPU=0 ONLY=N9 > ablation_v9.log 2>&1 &
```

**Option B: Single experiment**

```bash
python3 scripts/training/train_unified.py \
    --config configs/experiments/N1v9_vmf_nce.yaml --gpu 0
```

**Option C: Without saving checkpoints (saves disk space)**

```bash
nohup make ablation SUBJECTS="subj01" GPU=0 ONLY=N9 SAVE_CKPT=0 > ablation_v9.log 2>&1 &
```

### Step 5: Monitor Training

```bash
tail -f ablation_v9.log
```

---

## Training Parameters (V9)

All V9 experiments share these H100-optimized settings:

```yaml
training:
  batch_size: 128                     # H100 80GB (was 64 on A100)
  gradient_accumulation_steps: 8      # Effective batch = 1024
  mixed_precision: true
  mixed_precision_dtype: "bf16"       # H100 native bf16 (no GradScaler)
  num_epochs: 200
  fmri_noise_std: 0.1
  voxel_dropout: 0.1
  gradient_clip: 1.0
  early_stop_patience: 30
  ema:
    enabled: true
    decay: 0.999
```

### GPU Memory Guide

| GPU | Batch Size | Grad Accum | Effective Batch | Mixed Precision |
|-----|-----------|-----------|----------------|-----------------|
| H100 80GB | 128 | 8 | 1024 | bf16 (native) |
| A100 40GB | 64 | 8 | 512 | fp16 (GradScaler) |

---

## Evaluation

### During Training (Automatic)

Validation runs every epoch: R@1, R@5, MRR, loss values. V9 adds:
- CSLS-corrected retrieval metrics (csls_r@1, csls_r@5, csls_r@10)
- MC-Dropout TTA (8 samples, averaged predictions)
- Kappa-weighted repetition averaging

### After Training

```bash
# Aggregate ablation results
python3 scripts/evaluation/aggregate_ablation.py \
    --results-dir experimental_results \
    --subjects subj01

# Compare specific experiments
python3 scripts/evaluation/compare_experiments.py \
    --exp-dirs experimental_results/N1v9_vmf_nce \
              experimental_results/N4v9_full_system \
    --output experimental_results/v9_comparison.md
```

### Metrics by Experiment Type

**All experiments (B0-N4):**
- Retrieval: R@1, R@5, R@10, MRR, MeanR, MedR
- V9+: CSLS-corrected variants (csls_r@1, csls_r@5, csls_r@10)

**Probabilistic experiments (N1-N4):**
- Kappa statistics: mean, std, min, max, q10, q50, q90
- R-Drop loss (consistency penalty)

**ROI experiments (N2-N4):**
- Per-ROI attention importance
- ROI ablation study (leave-one-out)

---

## Version History

| Version | Experiments | Best R@1 | Key Innovation |
|---------|------------|----------|----------------|
| v4 | B0-N4 | N: 30-36% | nsdId fix, z-scoring, MixCo |
| v5 | N1-N4 | N1: 39.9% | tau=1.0 (kappa collapse fix) |
| v6 | N1-N4 | N1: ~40% | Delta-SPCL, vMF-SoftCLIP |
| v7 | N1-N4 | N1: 49% | Softplus kappa, multi-subject, multi-layer CLIP |
| v8 | N3-N4 | N4: 50.8% | Hierarchical CLIP, CKA |
| v9 | N1-N4 | Pending | DropPath, proj head, R-Drop, CSLS, TTA, bf16 |

---

## Troubleshooting

**Out of GPU memory:** Reduce `batch_size` in config, increase `gradient_accumulation_steps` proportionally.

**NaN losses:** Check `gradient_clip: 1.0` is set. With bf16 on H100, kappa overflow is essentially eliminated (bf16 has same exponent range as fp32).

**bf16 not available:** Fall back to fp16 by removing `mixed_precision_dtype: "bf16"` from config. GradScaler will be auto-enabled.

**Disk space:** Use `SAVE_CKPT=0` to skip checkpoint saving. Each checkpoint is ~1.3 GB.

**Module not found:** Re-run `pip install -e ".[train,diffusion]"` after pulling changes.

---

## Additional Resources

- **Experiment context:** [EXPERIMENT_CONTEXT.md](../EXPERIMENT_CONTEXT.md)
- **vMF and UA-CFG:** [VMF_UACFG_GUIDE.md](VMF_UACFG_GUIDE.md)
- **JupyterHub reference:** [JUPYTERHUB_REFERENCE.md](JUPYTERHUB_REFERENCE.md)
