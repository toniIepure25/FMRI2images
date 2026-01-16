# Experiment Tracking System

This directory contains completed experiments with full documentation for reproducibility and comparison.

## Structure

```
experiments_archive/
├── exp001_ultimate_baseline/
│   ├── README.md              # Experiment description
│   ├── config.yaml            # Full configuration
│   ├── checkpoints/           # Model checkpoints
│   │   ├── best_model.pt
│   │   └── epoch_XX.pt
│   ├── metrics/               # Training metrics
│   │   ├── training_log.json
│   │   └── evaluation_results.json
│   ├── visualizations/        # Plots and reconstructions
│   │   ├── loss_curves.png
│   │   └── sample_reconstructions/
│   └── notes.md              # Observations, issues, insights
├── exp002_modified_lr/
│   └── ...
└── comparison_report.md      # Cross-experiment comparison
```

## Experiment Naming Convention

`expXXX_descriptive_name/`
- XXX: Zero-padded experiment number
- descriptive_name: Brief description (e.g., `baseline`, `double_lr`, `add_validation`)

## Required Documentation

Each experiment directory must contain:

1. **README.md**: 
   - Goal/hypothesis
   - Key changes from baseline
   - Expected outcomes

2. **config.yaml**: Complete config used for training

3. **metrics/**:
   - `training_log.json`: Per-epoch metrics
   - `evaluation_results.json`: Final evaluation on test set

4. **notes.md**: 
   - Observations during training
   - Unexpected behaviors
   - Ideas for next experiments

## Current Experiments

### exp001_ultimate_baseline (2026-01-15)
- **Status**: ✅ Completed epoch 33/50
- **Run ID**: `20260115_172845_ultimate_novel_subj01`
- **Key Features**: All 7 novel contributions, LR=1e-5, grad_clip=0.3
- **Checkpoints**: epoch_31, epoch_32, epoch_33, best_model
- **Next**: Continue to epoch 50

## Quick Start

### 1. Archive Completed Experiment
```bash
python scripts/archive_experiment.py --run-dir runs/20260115_172845_ultimate_novel_subj01 \
                                     --exp-name exp001_ultimate_baseline \
                                     --description "Baseline training with all 7 contributions"
```

### 2. Evaluate Archived Experiment
```bash
python scripts/evaluate_archived_exp.py --exp-name exp001_ultimate_baseline \
                                         --num-samples 1000
```

### 3. Compare Experiments
```bash
python scripts/compare_experiments.py --experiments exp001_ultimate_baseline exp002_modified_lr \
                                       --output comparison_report.md
```
