# Orchestration Scripts

Experiment orchestration, execution, and monitoring.

## Scripts

- **`run_all_experiments.sh`** - Run all experiments (ablation study)
  - Sequential execution of all experiment configs
  - Handles failures and continues
  - Comprehensive logging
  
- `run_experiment.py` - Single experiment runner
  - Run one experiment configuration
  - Better error handling than direct training
  
- `run_full_pipeline.py` - Full end-to-end pipeline
  - Data preparation → Training → Evaluation → Reconstruction
  - Complete workflow automation
  
- `run_reconstruct_and_eval.py` - Reconstruction + evaluation
  - Generate images and evaluate quality
  - Combined workflow
  
- `run_stage34_recon_eval.py` - Stage 3/4 reconstruction evaluation
  - Multi-stage reconstruction pipeline
  
- `monitor_experiments.sh` - Monitor running experiments
  - Watch GPU usage, logs, progress
  - Real-time monitoring
  
- `check_experiment_status.sh` - Check experiment status
  - Summarize completed/running/failed experiments
  - Quick status overview

## Usage

```bash
# Run all experiments (ablation study)
bash scripts/orchestration/run_all_experiments.sh 0

# Run single experiment
python scripts/orchestration/run_experiment.py \
    --config configs/experiments/exp0_baseline.yaml

# Monitor experiments
bash scripts/orchestration/monitor_experiments.sh

# Check status
bash scripts/orchestration/check_experiment_status.sh
```
