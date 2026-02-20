# Running Experiments Guide

Complete guide for training, evaluating, and analyzing Phase 2 experiments.

---

## Quick Start

### Single Experiment

```bash
source .venv/bin/activate

python3 scripts/training/train_unified.py \
    --config configs/experiments/N1_vmf_nce.yaml --gpu 0
```

### Full Ablation Ladder

```bash
bash scripts/training/run_ablation_ladder.sh \
    --subjects "subj01 subj02 subj05 subj07" --gpu 0
```

Results are saved to `experimental_results/{B0,B1,N1,N2,N3,N4}_*/`.

---

## Experiment Configurations

### Ablation Ladder (6 Experiments)

| ID | Config | Description | Novel? |
|----|--------|-------------|--------|
| **B0** | `B0_deterministic.yaml` | MLP + PCR + queue + MSE + InfoNCE | No |
| **B1** | `B1_gaussian.yaml` | MLP + PCR + queue + Gaussian-NCE + KL | No |
| **N1** | `N1_vmf_nce.yaml` | MLP + vMF-NCE (replaces Gaussian) | Yes |
| **N2** | `N2_roi_transformer.yaml` | ROI Transformer + vMF-NCE | Yes |
| **N3** | `N3_roi_dcf.yaml` | Per-ROI vMF experts + consensus fusion | Yes |
| **N4** | `N4_full_system.yaml` | N3 + DUA-CFG + mixture + ceiling-temp + SPCL | Yes |

Consecutive row differences isolate each innovation:

- **B1 vs N1** = Gaussian vs vMF (distributional choice)
- **N1 vs N2** = MLP vs ROI Transformer (architecture)
- **N2 vs N3** = Single-head vs ROI-DCF (fusion strategy)
- **N3 vs N4** = Base system vs full innovations (generation stack)

All configs are in `configs/experiments/`.

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
    --config configs/experiments/N1_vmf_nce.yaml \
    --gpu 0
```

**Option B: Full ablation ladder (all 6 experiments x 4 subjects)**

```bash
bash scripts/training/run_ablation_ladder.sh \
    --subjects "subj01 subj02 subj05 subj07" --gpu 0
```

**Option C: Resume from a specific experiment**

```bash
bash scripts/training/run_ablation_ladder.sh --start N2 --gpu 0
```

### Step 4: Monitor Training

```bash
tensorboard --logdir experimental_results/ --port 6006
```

---

## Training Parameters

All experiments share a common schema. Key parameters:

```yaml
model:
  encoder:
    hidden_dims: [4096, 2048, 1024]    # MLP layers (B0/B1/N1)
    activation: "gelu"
    dropout: 0.1
  decoder:
    output_dim: 768                     # CLIP ViT-L/14

training:
  batch_size: 4
  gradient_accumulation_steps: 16       # Effective batch size = 64
  mixed_precision: true
  num_epochs: 100
  optimizer:
    type: "adamw"
    lr: 1.0e-4                          # 3e-4 for Transformer (N2-N4)
    weight_decay: 0.01
  gradient_clip: 1.0
  early_stop_patience: 15               # 20 for Transformer (N2-N4)
```

### GPU Memory Guide

| GPU | Batch Size | Grad Accum | Effective Batch | Approx. Time/Epoch |
|-----|-----------|-----------|----------------|-------------------|
| 16GB (V100) | 4 | 16 | 64 | ~15 min |
| 24GB (A100) | 4 | 16 | 64 | ~10 min |
| 40GB+ (A100) | 8 | 8 | 64 | ~7 min |

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
    --config configs/experiments/N1_vmf_nce.yaml \
    --checkpoint experimental_results/N1_vmf_nce/best_model.pt \
    --output experimental_results/N1_vmf_nce/evaluation/
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
    --exp-dirs experimental_results/B0_deterministic \
              experimental_results/N1_vmf_nce \
              experimental_results/N4_full_system \
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

**NaN losses:** Check `gradient_clip: 1.0` is set. For vMF models, verify
`kappa_reg.enabled: true` to prevent unbounded kappa growth.

**Training not converging:** For Transformer models (N2-N4), ensure
`warmup_epochs: 10` and `lr: 3e-4`. MLP models (B0-N1) use `lr: 1e-4`.

**Data not found:** Run `python3 scripts/utils/preflight_check.py` to verify
NSD data files and paths.

---

## Additional Resources

- **Environment setup:** [SETUP.md](SETUP.md)
- **Evaluation details:** [EVALUATION_SUITE_GUIDE.md](EVALUATION_SUITE_GUIDE.md)
- **vMF and UA-CFG:** [VMF_UACFG_GUIDE.md](VMF_UACFG_GUIDE.md)
- **Troubleshooting:** [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
