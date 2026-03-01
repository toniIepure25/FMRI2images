# Running Experiments Guide

Complete guide for training, evaluating, and analyzing Phase 2 experiments.

---

## Quick Start

### Single Experiment

```bash
python3 scripts/training/train_unified.py \
    --config configs/experiments/N1v5_vmf_nce.yaml --gpu 0
```

### Full Ablation Ladder (B v4 + N v5)

```bash
nohup make ablation SUBJECTS="subj01" GPU=0 > ablation_v5.log 2>&1 &
tail -f ablation_v5.log
```

The ladder runs: B0v4, B1v4, N1v5, N2v5, N3v5, N4v5 (in order). Each experiment is skipped if `checkpoint_best.pt` already exists.

Results are saved to `experimental_results/{experiment_name}/{subject}/`.

---

## Experiment Configurations

### Current Ablation Ladder (B v4 + N v5)

| ID | Config | Description | Novel? |
|----|--------|-------------|--------|
| **B0v4** | `B0v4_deterministic.yaml` | MLP + MSE + InfoNCE + SoftCLIP + MixCo | No |
| **B1v4** | `B1v4_gaussian.yaml` | MLP + Gaussian-NCE + KL + SoftCLIP + MixCo | No |
| **N1v5** | `N1v5_vmf_nce.yaml` | MLP + vMF-NCE (tau=1.0) + SoftCLIP + MixCo + EMA | Yes |
| **N2v5** | `N2v5_roi_transformer.yaml` | ROI Transformer (d=768, 6L) + vMF-NCE + EMA | Yes |
| **N3v5** | `N3v5_roi_dcf.yaml` | Per-ROI vMF experts + consensus fusion + EMA | Yes |
| **N4v5** | `N4v5_full_system.yaml` | N3 + SPCL curriculum + DUA-CFG + EMA | Yes |

Consecutive row differences isolate each innovation:

- **B1 vs N1** = Gaussian vs vMF (distributional choice)
- **N1 vs N2** = MLP vs ROI Transformer (architecture)
- **N2 vs N3** = Single-head vs ROI-DCF (fusion strategy)
- **N3 vs N4** = Base system vs full innovations (generation stack)

All configs are in `configs/experiments/`. Previous versions (v1-v4) are preserved for reproducibility.

---

## Training Workflow

### Step 1: Prepare Environment

```bash
python3 scripts/utils/preflight_check.py
```

Requirements: Python 3.10+, PyTorch with CUDA, 16GB+ VRAM, NSD data files.

### Step 2: Build Preprocessing Cache (First Time Only)

```bash
python3 scripts/build/fit_preprocessing.py \
    --subject subj01 --mode center_pcr --k 8
```

Creates: `cache/embedding_preproc/subj01_center_pcr_k8.pkl` (~500MB).
Required once per subject.

### Step 3: Run Training

**Option A: Single experiment**

```bash
python3 scripts/training/train_unified.py \
    --config configs/experiments/N1v5_vmf_nce.yaml \
    --gpu 0
```

**Option B: Full ablation ladder (B v4 + N v5)**

```bash
nohup make ablation SUBJECTS="subj01 subj02 subj05 subj07" GPU=0 > ablation_v5.log 2>&1 &
```

**Option C: Resume from a specific experiment**

```bash
bash scripts/training/run_ablation_ladder.sh --start N1v5 --gpu 0
```

### Step 4: Monitor Training

```bash
tensorboard --logdir experimental_results/ --port 6006
```

---

## Training Parameters

All v5 N-series experiments share these settings (B-series at v4 differ slightly):

```yaml
model:
  encoder:
    hidden_dims: [8192, 4096, 2048]    # MLP layers (B0/B1/N1); N2-N4 use ROI Transformer
    activation: "gelu"
    dropout: 0.2                        # v5 (was 0.15 in v4)
    use_residual: true
  decoder:
    output_dim: 768                     # CLIP ViT-L/14

training:
  batch_size: 64
  gradient_accumulation_steps: 4        # Effective batch size = 256 (v5)
  mixed_precision: true
  num_epochs: 150                       # v5 (was 300 in v4)
  fmri_noise_std: 0.1                   # v5: Gaussian noise augmentation
  voxel_dropout: 0.1                    # v5: random voxel masking
  optimizer:
    type: "adamw"
    lr: 7.0e-5                          # v5 (varies per experiment)
    weight_decay: 0.05                  # v5 (was 0.01 in v4)
  gradient_clip: 1.0
  early_stop_patience: 25              # v5 (was 40 in v4)
  ema:
    enabled: true                       # v5: model averaging
    decay: 0.999
```

### GPU Memory Guide

| GPU | Batch Size | Grad Accum | Effective Batch | Approx. Time/Epoch |
|-----|-----------|-----------|----------------|-------------------|
| 40GB (A100) | 64 | 4 | 256 | ~2-5 min (MLP), ~5-10 min (Transformer) |

Out of memory? Reduce `batch_size` and increase `gradient_accumulation_steps`
to maintain the same effective batch size.

---

## Evaluation

### During Training (Automatic)

Validation runs every epoch: R@1, R@5, MRR, loss values.
Results saved to `experimental_results/<exp_name>/training_info.json`.

### After Training

```bash
python3 scripts/evaluation/evaluate_experiment.py \
    --config configs/experiments/N1v5_vmf_nce.yaml \
    --checkpoint experimental_results/N1v5_vmf_nce/subj01/checkpoint_best.pt \
    --output experimental_results/N1v5_vmf_nce/subj01/evaluation/
```

### Metrics by Experiment Type

**All experiments (B0-N4):**
- Retrieval: R@1, R@5, R@10, MRR, MeanR, MedR
- Identification: 2AFC with bootstrap CI
- Separability: ROC AUC, Cohen's d

**Probabilistic experiments (B1, N1-N4):**
- NLL, Energy Score
- Coverage@80, Coverage@95, ECE
- Risk-coverage curves, AURC

**ROI experiments (N2-N4):**
- Per-ROI attention importance (bootstrap CIs)
- ROI ablation study (leave-one-out)
- Category-ROI interaction heatmaps

### Cross-Experiment Comparison

```bash
python3 scripts/evaluation/compare_experiments.py \
    --exp-dirs experimental_results/B0v4_deterministic \
              experimental_results/N1v5_vmf_nce \
              experimental_results/N4v5_full_system \
    --output experimental_results/comparison_report.md \
    --paired-test
```

---

## Output Structure

### Training Outputs

```
experimental_results/N1_vmf_nce/
  config.yaml              # Frozen config snapshot
  training_info.json       # Epoch, loss curves, checkpoint path
  notes.md                 # Per-experiment analysis (fill after run)
  best_model.pt            # Best model checkpoint
  evaluation/
    embedding_metrics.json
    probabilistic_metrics.json
    summary_report.md
```

---

## Time Estimates

| Task | A100 (24GB) | V100 (16GB) |
|------|------------|-------------|
| Preprocessing (per subject) | 30 min | 1 hour |
| B0 or B1 (MLP, 100 epochs) | 2-3 hours | 4-5 hours |
| N1 (MLP + vMF, 100 epochs) | 2-3 hours | 4-5 hours |
| N2-N4 (Transformer, 100 epochs) | 4-6 hours | 8-10 hours |
| Full ladder (6 exp x 4 subj) | 18-24 hours | 36-48 hours |
| Evaluation (per experiment) | 15-30 min | 20-40 min |

---

## Troubleshooting

**Out of GPU memory:** Reduce `batch_size` in config, increase
`gradient_accumulation_steps` proportionally.

**NaN losses:** Check `gradient_clip: 1.0` is set. For vMF models in v5, `kappa_reg` is intentionally disabled -- kappa is bounded by a sigmoid to `[kappa_min, kappa_max]` and the contrastive loss naturally constrains it. If NaN occurs, check that `kappa_max` keeps max logits below 80 (with `tau=1.0`, max logit = kappa_max).

**Training not converging:** For Transformer models (N2-N4 v5), ensure `warmup_epochs: 15` and an appropriate LR (5e-5 to 7e-5). MLP models (B0/B1/N1) use `lr: 7e-5` to `1e-4`.

**Data not found:** Run `python3 scripts/utils/preflight_check.py` to verify
NSD data files and paths.

---

## Additional Resources

- **Environment setup:** [SETUP.md](SETUP.md)
- **Evaluation details:** [EVALUATION_SUITE_GUIDE.md](EVALUATION_SUITE_GUIDE.md)
- **vMF and UA-CFG:** [VMF_UACFG_GUIDE.md](VMF_UACFG_GUIDE.md)
- **Troubleshooting:** [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
