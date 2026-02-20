# Experiment Configurations

Self-contained configs for the Phase 2 ablation ladder (B0-B1-N1-N2-N3-N4).
Each file fully specifies one experiment — no implicit dependencies.

## Running Experiments

```bash
# Single experiment
python3 scripts/training/train_unified.py \
    --config configs/experiments/N1_vmf_nce.yaml --gpu 0

# Full ablation ladder (6 experiments x 4 subjects)
bash scripts/training/run_ablation_ladder.sh \
    --subjects "subj01 subj02 subj05 subj07" --gpu 0

# Resume from a specific experiment
bash scripts/training/run_ablation_ladder.sh --start N1
```

## Ablation Ladder

The ladder is structured so that **consecutive row differences = ablation**.
No separate ablation table needed — the main table IS the ablation.

### Baselines (B0-B1): Best Known Standard Techniques

| Config | Description | Model Type |
|--------|-------------|-----------|
| `B0_deterministic.yaml` | MLP + PCR + queue + MSE + InfoNCE | deterministic |
| `B1_gaussian.yaml` | MLP + PCR + queue + Gaussian-NCE + KL annealing | gaussian |

### Novel Contributions (N1-N4): Our Innovations

| Config | Description | Model Type |
|--------|-------------|-----------|
| `N1_vmf_nce.yaml` | vMF decoder + vMF-NCE loss (replaces Gaussian) | vmf |
| `N2_roi_transformer.yaml` | ROI-Tokenized Transformer encoder + vMF-NCE | vmf |
| `N3_roi_dcf.yaml` | Per-ROI vMF experts + consensus fusion | vmf_dcf |
| `N4_full_system.yaml` | All innovations: DCF + Mixture + DUA-CFG + Ceiling-Temp + SPCL | vmf_dcf |

### What Each Comparison Proves

| Comparison | Isolates |
|------------|----------|
| B1 vs N1 | Gaussian vs vMF (distributional choice) |
| N1 vs N2 | Flat MLP vs ROI Transformer (architecture) |
| N2 vs N3 | Single-head vs ROI-DCF (fusion strategy) |
| N3 vs N4 | Base system vs full innovations (generation stack) |

## Config Schema

Every experiment config follows this structure:

```yaml
experiment:
  name: "N1_vmf_nce"
  description: "..."
  tags: [...]
  parent: "B1_gaussian"              # Which experiment this builds on

data:
  subject: "subj01"                  # Overridden by --subject CLI arg
  roi: "nsdgeneral"
  train_split: 0.70
  val_split: 0.15
  test_split: 0.15
  seed: 42

model:
  type: "vmf"                        # deterministic | gaussian | vmf | vmf_dcf
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

N2-N4 use placeholder ROI dimensions. These are populated at runtime
from the NSD atlas masks for the specified subject. The values in the configs
(V1v: 700, V2v: 600, etc.) are approximate and serve as documentation only.

## Output Convention

All experiments write to `experimental_results/<exp_name>/` with:
- `config.yaml` — frozen config snapshot
- `training_info.json` — epoch, loss curves, checkpoint path
- `notes.md` — per-experiment analysis
- `evaluation/` — metric JSONs and summary reports
