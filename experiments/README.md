# Experiment Run Documentation

This directory contains reproducibility documentation for experiment execution.

> **Note**: Experiment *configurations* are in `configs/experiments/`. This directory holds
> runtime policies and historical run logs.

## Contents

| File | Purpose |
|------|---------|
| `SEEDS.md` | Seed and determinism policy for reproducible runs |
| `RUNS.md` | Historical run log from Phase 1 (probabilistic inference pipeline) |

## Phase History

### Phase 1 (Completed — Results in `RaportPaper3/`)

Soft Reliability Weighting, InfoNCE, MC Dropout Uncertainty.
CLIP ViT-B/32 (512-D), 4 NSD subjects. Configs used a separate runner
(`scripts/orchestration/run_experiment.py`). Results archived in
`experimental_results/exp001_baseline_ultimate/`.

### Phase 2 (Current — Configs in `configs/experiments/`)

vMF-NCE, ROI-DCF, Dual Uncertainty, full ablation ladder EXP0-EXP14.
CLIP ViT-L/14 (768-D). All configs use `scripts/training/train_unified.py`.

To run Phase 2 experiments:

```bash
# Single experiment
python3 scripts/training/train_unified.py --config configs/experiments/exp7_vmf_nce.yaml --gpu 0

# Full ablation ladder (4 subjects x 15 experiments)
bash scripts/training/run_ablation_ladder.sh --subjects "subj01 subj02 subj05 subj07" --gpu 0
```
