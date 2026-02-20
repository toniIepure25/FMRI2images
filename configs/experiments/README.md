# Experiment Configurations

Self-contained configs for the Phase 2 ablation ladder (EXP0-EXP14).
Each file fully specifies one experiment — no implicit dependencies.

## Running Experiments

```bash
# Single experiment
python3 scripts/training/train_unified.py \
    --config configs/experiments/exp7_vmf_nce.yaml --gpu 0

# Full ablation ladder (all 15 experiments x 4 subjects)
bash scripts/training/run_ablation_ladder.sh \
    --subjects "subj01 subj02 subj05 subj07" --gpu 0
```

## Ablation Ladder

### EXP0-EXP6: Foundation (MLP Encoder)

| Config | Key Addition | Model Type |
|--------|-------------|-----------|
| `exp0_baseline.yaml` | Deterministic MLP + MSE + InfoNCE | deterministic |
| `exp1_preproc.yaml` | + center_pcr (k=8) | deterministic |
| `exp2_queue.yaml` | + memory queue (Q=8192) | deterministic |
| `exp3_gaussian_nll.yaml` | + Gaussian NLL (mu, logvar) | gaussian |
| `exp4_gaussian_nce.yaml` | + Gaussian-NCE contrastive | gaussian |
| `exp5_kl_anneal.yaml` | + KL annealing + free-bits | gaussian |
| `exp6_whiten.yaml` | Whitening ablation (vs PCR) | gaussian |

### EXP7-EXP9: Core Novel Contributions

| Config | Key Addition | Model Type |
|--------|-------------|-----------|
| `exp7_vmf_nce.yaml` | vMF decoder + vMF-NCE loss | vmf |
| `exp8_roi_transformer.yaml` | ROI-Tokenized Transformer encoder | vmf |
| `exp9_roi_dcf.yaml` | Per-ROI vMF experts + consensus fusion | vmf_dcf |

### EXP10-EXP14: Extended Innovations

| Config | Key Addition | Model Type |
|--------|-------------|-----------|
| `exp10_vmf_mixture.yaml` | vMF mixture posterior sampling | vmf_dcf |
| `exp11_dual_ua_cfg.yaml` | Decomposed UA-CFG (kappa->w, delta->K) | vmf_dcf |
| `exp12_ceiling_temperature.yaml` | Noise-ceiling contrastive temperature | vmf_dcf |
| `exp13_kappa_spcl.yaml` | kappa-SPCL curriculum learning | vmf_dcf |
| `exp14_full_system.yaml` | All innovations combined (flagship) | vmf_dcf |

## Config Schema

Every experiment config follows this structure:

```yaml
experiment:
  name: "exp7_vmf_nce"
  description: "..."
  tags: [...]
  parent: "exp4_gaussian_nce"     # Which experiment this builds on

data:
  subject: "subj01"               # Overridden by --subject CLI arg
  roi: "nsdgeneral"
  train_split: 0.70
  val_split: 0.15
  test_split: 0.15
  seed: 42

model:
  type: "vmf"                     # deterministic | gaussian | vmf | vmf_dcf
  encoder: { ... }
  decoder: { ... }

preprocessing:
  enabled: true/false
  mode: "center_pcr"

loss:
  vmf_nce: { enabled: true, ... }
  # All unused losses: enabled: false

queue:
  enabled: true/false
  size: 8192

training:
  batch_size: 4
  gradient_accumulation_steps: 16
  mixed_precision: true
  num_epochs: 100
  optimizer: { type: "adamw", lr: ..., weight_decay: ..., betas: [...] }
  warmup_epochs: 5-10
  min_lr: 1.0e-6
  gradient_clip: 1.0
  early_stop_patience: 15-20

paths:
  output_dir: "experimental_results/<exp_name>"
```

## ROI Dimensions

EXP8-EXP14 use placeholder ROI dimensions. These are populated at runtime
from the NSD atlas masks for the specified subject. The values in the configs
(V1v: 700, V2v: 600, etc.) are approximate and serve as documentation only.

## Output Convention

All experiments write to `experimental_results/<exp_name>/` with:
- `config.yaml` — frozen config snapshot
- `training_info.json` — epoch, loss curves, checkpoint path
- `notes.md` — per-experiment analysis
- `evaluation/` — metric JSONs and summary reports
