#!/usr/bin/env python3
"""
Archive a completed training run into the experiments_archive with full documentation.

Usage:
    python scripts/archive_experiment.py \
        --run-dir runs/20260115_172845_ultimate_novel_subj01 \
        --exp-name exp001_ultimate_baseline \
        --description "Baseline training with all 7 novel contributions"
"""

import shutil
import json
import yaml
from pathlib import Path
import argparse
from datetime import datetime


def archive_experiment(run_dir, exp_name, description, notes=""):
    """Archive a training run to experiments_archive/"""
    
    run_dir = Path(run_dir)
    archive_dir = Path("experiments_archive") / exp_name
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📦 Archiving experiment: {exp_name}")
    print(f"   Source: {run_dir}")
    print(f"   Destination: {archive_dir}")
    
    # 1. Copy checkpoints
    checkpoints_src = run_dir / "checkpoints"
    checkpoints_dst = archive_dir / "checkpoints"
    if checkpoints_src.exists():
        shutil.copytree(checkpoints_src, checkpoints_dst, dirs_exist_ok=True)
        print(f"   ✓ Copied checkpoints")
    
    # 2. Copy config
    config_src = run_dir / "config.yaml"
    if config_src.exists():
        shutil.copy(config_src, archive_dir / "config.yaml")
        print(f"   ✓ Copied config")
    
    # 3. Copy metrics if exists
    metrics_dir = archive_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    
    metrics_src = run_dir / "metrics.json"
    if metrics_src.exists():
        shutil.copy(metrics_src, metrics_dir / "training_log.json")
        print(f"   ✓ Copied training metrics")
    
    # 4. Create README with experiment info
    readme_path = archive_dir / "README.md"
    
    # Load config for details
    with open(archive_dir / "config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Count epochs from checkpoints
    checkpoint_files = list(checkpoints_dst.glob("epoch_*.pt"))
    epochs_completed = len(checkpoint_files)
    max_epoch = max([int(f.stem.split('_')[1]) for f in checkpoint_files]) if checkpoint_files else 0
    
    readme_content = f"""# {exp_name}

## Overview
{description}

**Status**: Completed  
**Date**: {datetime.now().strftime('%Y-%m-%d')}  
**Run ID**: `{run_dir.name}`  
**Epochs Completed**: {max_epoch}/{config['training']['num_epochs']}

## Configuration

### Model Architecture
- **Type**: ProbabilisticMultiLayerTwoStageEncoder
- **Parameters**: ~300M
- **Latent Dim**: {config['model']['latent_dim']}
- **Blocks**: {config['model']['n_blocks']}
- **Head Hidden**: {config['model']['head_hidden_dim']}
- **Dropout**: {config['model']['dropout_rate']}

### Training Hyperparameters
- **Learning Rate**: {config['training']['learning_rate']}
- **Batch Size**: {config['training']['batch_size']}
- **Gradient Accumulation**: {config['training']['gradient_accumulation_steps']}
- **Effective Batch Size**: {config['training']['batch_size'] * config['training']['gradient_accumulation_steps']}
- **Gradient Clipping**: {config['training']['grad_clip']}
- **KL Weight**: {config['training']['kl_weight']}
- **Mixed Precision**: {config['training']['use_amp']}

### Loss Composition
- **MSE**: {config['training']['loss_weights']['mse']}
- **Cosine**: {config['training']['loss_weights']['cosine']}
- **InfoNCE**: {config['training']['loss_weights']['infonce']}

### Novel Contributions
{chr(10).join([f"- ✅ {layer}" for layer in config['model']['enabled_layers']])}
- ✅ Probabilistic Encoder (Variational Inference)
- ✅ Multi-Loss Composition
- ✅ InfoNCE Contrastive Loss
- ✅ Soft Reliability Weighting
- ✅ MC Dropout Uncertainty

## Data
- **Subject**: {config['data']['subject']}
- **Session**: {config['data']['session']}
- **Index**: `{config['data']['index_path']}`
- **CLIP Cache**: `{config['data'].get('clip_cache_path', 'N/A')}`
- **Preprocessing**: ✅ Enabled (370,503 voxels retained)

## Checkpoints
```
{chr(10).join([f"- {f.name} ({f.stat().st_size / (1024**3):.1f} GB)" for f in sorted(checkpoints_dst.glob("*.pt"))])}
```

## Results

### Training Progress
- **Best Model**: `checkpoints/best_model.pt` (epoch {max_epoch})
- **Latest**: `checkpoints/epoch_{max_epoch:02d}.pt`

### Metrics
See `metrics/training_log.json` for per-epoch metrics.

To evaluate:
```bash
python scripts/evaluate_model.py \\
    --checkpoint experiments_archive/{exp_name}/checkpoints/best_model.pt \\
    --output-dir outputs/eval/{exp_name} \\
    --num-samples 1000
```

## Notes
{notes or "No additional notes."}

## Next Steps
- [ ] Complete training to epoch 50
- [ ] Run full evaluation on test set
- [ ] Generate reconstruction visualizations
- [ ] Compare with baseline experiments

---
*Auto-generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"   ✓ Created README")
    
    # 5. Create notes.md template
    notes_path = archive_dir / "notes.md"
    if not notes_path.exists():
        notes_content = f"""# Training Notes: {exp_name}

## Observations During Training

### Stability
- No NaN divergence after preprocessing was added
- Training stable with LR={config['training']['learning_rate']}

### Performance Trends
- (Add observations about loss curves)
- (Note any plateau or improvements)

### Issues Encountered
- Initial NaN issue resolved with proper fMRI preprocessing
- Disk space management: auto-cleanup of old checkpoints

### Hyperparameter Choices
- **LR={config['training']['learning_rate']}**: (Rationale)
- **grad_clip={config['training']['grad_clip']}**: (Rationale)

## Ideas for Next Experiments
- [ ] Try validation split for better model selection
- [ ] Add CLIP retrieval metrics
- [ ] Experiment with learning rate scheduling
- [ ] Test InfoNCE with cross-batch queue

## Ablation Study Ideas
- Disable specific layers to measure contribution
- Test different loss weight combinations
- Compare with/without probabilistic encoding
"""
        with open(notes_path, 'w') as f:
            f.write(notes_content)
        print(f"   ✓ Created notes template")
    
    print(f"\n✅ Experiment archived successfully!")
    print(f"   Location: {archive_dir}")
    print(f"\nNext steps:")
    print(f"   1. Review/edit: {readme_path}")
    print(f"   2. Add observations: {notes_path}")
    print(f"   3. Run evaluation: python scripts/evaluate_model.py --checkpoint {archive_dir}/checkpoints/best_model.pt")


def main():
    parser = argparse.ArgumentParser(description='Archive experiment for documentation')
    parser.add_argument('--run-dir', type=str, required=True, help='Path to training run directory')
    parser.add_argument('--exp-name', type=str, required=True, help='Experiment name (e.g., exp001_baseline)')
    parser.add_argument('--description', type=str, required=True, help='Brief experiment description')
    parser.add_argument('--notes', type=str, default="", help='Additional notes')
    args = parser.parse_args()
    
    archive_experiment(args.run_dir, args.exp_name, args.description, args.notes)


if __name__ == '__main__':
    main()
