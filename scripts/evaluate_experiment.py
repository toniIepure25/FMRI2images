#!/usr/bin/env python3
"""
Complete evaluation workflow for a trained experiment.
This is the FIRST step after training completes.

Usage:
    python scripts/evaluate_experiment.py \
        --checkpoint runs/20260115_172845_ultimate_novel_subj01/checkpoints/best_model.pt \
        --config experiments/ultimate_novel_subj01.yaml \
        --exp-name exp001_baseline_ultimate \
        --num-samples 1000

This will:
1. Create experimental_results/exp001_baseline_ultimate/
2. Run comprehensive evaluation with all CLIP metrics
3. Save results to evaluation/ subfolder
4. Copy config for reference
5. Generate summary report

Author: Bachelor Thesis - Evaluation Workflow
"""

import argparse
import sys
import json
import yaml
import shutil
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import evaluation logic - need to import as module
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import evaluate_ultimate_model
load_checkpoint = evaluate_ultimate_model.load_checkpoint
prepare_dataloader = evaluate_ultimate_model.prepare_dataloader
evaluate_model = evaluate_ultimate_model.evaluate_model
print_results = evaluate_ultimate_model.print_results

import torch


def create_experiment_structure(exp_name: str) -> Path:
    """Create folder structure for experiment results."""
    base_dir = Path("experimental_results") / exp_name
    
    # Create subdirectories
    (base_dir / "evaluation").mkdir(parents=True, exist_ok=True)
    (base_dir / "reconstructions").mkdir(parents=True, exist_ok=True)
    
    return base_dir


def save_training_info(exp_dir: Path, checkpoint_path: Path, config: dict):
    """Save training information."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    training_info = {
        'checkpoint_path': str(checkpoint_path),
        'epoch': checkpoint.get('epoch', 'N/A'),
        'date_evaluated': datetime.now().isoformat(),
        'num_epochs_trained': checkpoint.get('epoch', 0),
        'target_epochs': config['training'].get('num_epochs', config['training'].get('epochs', 'N/A')),
        'training_completed': checkpoint.get('epoch', 0) >= config['training'].get('num_epochs', config['training'].get('epochs', 999))
    }
    
    if 'metrics' in checkpoint:
        training_info['final_training_metrics'] = checkpoint['metrics']
    
    output_path = exp_dir / "training_info.json"
    with open(output_path, 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print(f"   ✓ Saved training info: {output_path}")


def generate_summary_report(exp_dir: Path, results: dict, config: dict, checkpoint_info: dict):
    """Generate human-readable summary report."""
    
    report = f"""# Experiment Evaluation Summary: {exp_dir.name}

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Epoch**: {checkpoint_info.get('epoch', 'N/A')} / {config['training'].get('num_epochs', config['training'].get('epochs', 'N/A'))}  
**Samples Evaluated**: {results['num_samples']}

---

## Quick Summary

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Cosine Similarity** | {results['global']['cosine_similarity']['cosine_mean']:.4f} ± {results['global']['cosine_similarity']['cosine_std']:.4f} | {'✅ Excellent' if results['global']['cosine_similarity']['cosine_mean'] > 0.50 else '✅ Good' if results['global']['cosine_similarity']['cosine_mean'] > 0.40 else '⚠️ Moderate' if results['global']['cosine_similarity']['cosine_mean'] > 0.30 else '❌ Poor'} |
| **Top-1 Retrieval** | {results['global']['retrieval']['top1_accuracy']:.2%} | {'✅ Excellent' if results['global']['retrieval']['top1_accuracy'] > 0.20 else '✅ Good' if results['global']['retrieval']['top1_accuracy'] > 0.10 else '⚠️ Moderate' if results['global']['retrieval']['top1_accuracy'] > 0.05 else '❌ Poor'} |
| **Top-5 Retrieval** | {results['global']['retrieval']['top5_accuracy']:.2%} | {'✅ Excellent' if results['global']['retrieval']['top5_accuracy'] > 0.50 else '✅ Good' if results['global']['retrieval']['top5_accuracy'] > 0.40 else '⚠️ Moderate' if results['global']['retrieval']['top5_accuracy'] > 0.25 else '❌ Poor'} |
| **Mean Rank** | {results['global']['retrieval']['mean_rank']:.1f} | {'✅ Excellent' if results['global']['retrieval']['mean_rank'] < 5 else '✅ Good' if results['global']['retrieval']['mean_rank'] < 10 else '⚠️ Moderate' if results['global']['retrieval']['mean_rank'] < 20 else '❌ Poor'} |
| **KL Divergence** | {results['kl_divergence']['mean']:.4f} | {'✅ Good balance' if 0.1 <= results['kl_divergence']['mean'] <= 0.3 else '⚠️ Check regularization'} |

---

## Detailed Metrics

### CLIP Embedding Quality

**Cosine Similarity Distribution**:
- Mean: {results['global']['cosine_similarity']['cosine_mean']:.4f}
- Median: {results['global']['cosine_similarity']['cosine_median']:.4f}
- Std: {results['global']['cosine_similarity']['cosine_std']:.4f}
- Min: {results['global']['cosine_similarity']['cosine_min']:.4f}
- Max: {results['global']['cosine_similarity']['cosine_max']:.4f}

**Interpretation**: 
- Values closer to 1.0 indicate better alignment with ground truth CLIP embeddings
- Median similar to mean suggests consistent performance
- Low std (<0.15) indicates stable predictions across samples

### Retrieval Performance

**Top-K Accuracy**:
- Top-1: {results['global']['retrieval']['top1_accuracy']:.2%} (correct image ranked #1)
- Top-5: {results['global']['retrieval']['top5_accuracy']:.2%} (correct image in top-5)
- Top-10: {results['global']['retrieval']['top10_accuracy']:.2%} (correct image in top-10)

**Ranking Statistics**:
- Mean Rank: {results['global']['retrieval']['mean_rank']:.1f}
- Median Rank: {results['global']['retrieval']['median_rank']:.1f}

**Interpretation**:
- Top-5 >40% indicates strong practical retrieval capability
- Lower mean rank indicates more predictions near ground truth
- Median < Mean suggests some outliers with very poor predictions

### Reconstruction Error

**Embedding Space Distance**:
- MSE: {results['global']['mse']:.4f}
- RMSE: {results['global']['rmse']:.4f}
- L2 Distance (mean): {results['global']['l2_distance_mean']:.4f} ± {results['global']['l2_distance_std']:.4f}

### Probabilistic Component

**KL Divergence (Variational Regularization)**:
- Mean: {results['kl_divergence']['mean']:.4f}
- Std: {results['kl_divergence']['std']:.4f}

**Interpretation**:
- KL ~0.1-0.3: Good balance between reconstruction and regularization
- KL <0.05: Posterior collapsed to prior (underfitting)
- KL >0.5: High divergence (may need higher KL weight)

---

## Configuration

**Training Hyperparameters**:
- Learning Rate: {config['training']['learning_rate']}
- Batch Size: {config['training']['batch_size']}
- Gradient Accumulation: {config['training'].get('gradient_accumulation_steps', config.get('advanced', {}).get('gradient_accumulation_steps', 1))}
- Effective Batch: {config['training']['batch_size'] * config['training'].get('gradient_accumulation_steps', config.get('advanced', {}).get('gradient_accumulation_steps', 1))}
- Gradient Clip: {config['training']['grad_clip']}
- KL Weight: {config['training']['kl_weight']}

**Model Architecture**:
- Latent Dim: {config['model']['latent_dim']}
- Blocks: {config['model']['n_blocks']}
- Head Hidden: {config['model']['head_hidden_dim']}
- Dropout: {config['model'].get('dropout_rate', config['model'].get('dropout', 'N/A'))}
- Enabled Layers: {', '.join(config['model']['enabled_layers'])}

**Loss Weights**:
- MSE: {config['training']['loss_weights']['mse']}
- Cosine: {config['training']['loss_weights']['cosine']}
- InfoNCE: {config['training']['loss_weights']['infonce']}

---

## Decision

"""
    
    # Add recommendation based on metrics
    cosine_mean = results['global']['cosine_similarity']['cosine_mean']
    top5 = results['global']['retrieval']['top5_accuracy']
    
    if cosine_mean > 0.50 and top5 > 0.50:
        report += """### ✅ EXCELLENT - Proceed with Confidence
- Metrics indicate strong embedding quality
- **Recommendation**: Archive this model, generate reconstructions for thesis
- Consider this as a strong baseline or even final model

"""
    elif cosine_mean > 0.40 and top5 > 0.40:
        report += """### ✅ GOOD - Acceptable Performance
- Metrics show solid learning
- **Recommendation**: Archive for comparison, may generate reconstructions
- Could try further hyperparameter optimization

"""
    elif cosine_mean > 0.30 and top5 > 0.25:
        report += """### ⚠️ MODERATE - Room for Improvement
- Model is learning but not optimal
- **Recommendation**: Try modifications before reconstructing images
- Consider: higher LR, more epochs, architecture changes

"""
    else:
        report += """### ❌ NEEDS IMPROVEMENT
- Metrics below expectations
- **Recommendation**: Modify hyperparameters before continuing
- Check: data preprocessing, loss weights, learning rate

"""
    
    report += """
### Next Steps

1. **If satisfied with metrics**:
   ```bash
   # Generate reconstructions (optional)
   python scripts/run_stage34_recon_eval.py \\
       --checkpoint CHECKPOINT_PATH \\
       --config CONFIG_PATH \\
       --output-dir experimental_results/{exp_name}/reconstructions \\
       --num-images 50
   ```

2. **If trying new experiment**:
   - Modify hyperparameters in config
   - Train new model
   - Evaluate with this workflow again

3. **Compare with other experiments**:
   ```bash
   python scripts/compare_experimental_results.py \\
       --exp-dirs experimental_results/{exp_name} experimental_results/exp002_... \\
       --output experimental_results/comparison_reports/comparison.md
   ```

---

*Evaluation completed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
""".format(exp_name=exp_dir.name)
    
    # Save report
    output_path = exp_dir / "evaluation" / "summary_report.md"
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"   ✓ Generated summary report: {output_path}")


def create_notes_template(exp_dir: Path):
    """Create notes template for observations."""
    notes_path = exp_dir / "notes.md"
    
    if not notes_path.exists():
        notes = f"""# Experiment Notes: {exp_dir.name}

## Overview
_(Brief description of this experiment)_

## Motivation
_(Why did you run this experiment? What hypothesis are you testing?)_

## Changes from Previous Experiments
_(What's different compared to baseline or previous runs?)_

## Training Observations
- Stability: 
- Convergence: 
- Training time: 
- Issues encountered: 

## Evaluation Results
- Key findings: 
- Unexpected results: 
- Comparison with expectations: 

## Insights
_(What did you learn from this experiment?)_

## Next Experiment Ideas
- [ ] Try ...
- [ ] Test ...
- [ ] Compare ...

---
*Date: {datetime.now().strftime('%Y-%m-%d')}*
"""
        with open(notes_path, 'w') as f:
            f.write(notes)
        
        print(f"   ✓ Created notes template: {notes_path}")


def main():
    parser = argparse.ArgumentParser(description='Complete experiment evaluation workflow')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--exp-name', type=str, required=True, help='Experiment name (e.g., exp001_baseline_ultimate)')
    parser.add_argument('--num-samples', type=int, default=1000, help='Number of samples to evaluate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()
    
    print("="*100)
    print(" " * 30 + "EXPERIMENT EVALUATION WORKFLOW")
    print("="*100)
    print(f"Experiment: {args.exp_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create experiment structure
    print("📁 Creating experiment structure...")
    exp_dir = create_experiment_structure(args.exp_name)
    print(f"   ✓ Created: {exp_dir}")
    
    # Copy config
    config_dest = exp_dir / "config.yaml"
    shutil.copy(args.config, config_dest)
    print(f"   ✓ Copied config")
    
    # Save training info
    save_training_info(exp_dir, Path(args.checkpoint), config)
    
    # Run evaluation (using imported functions)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    checkpoint_path = Path(args.checkpoint)
    
    print("\n📥 Loading model...")
    model, checkpoint_info = load_checkpoint(checkpoint_path, config, device)
    
    print("\n📊 Preparing evaluation dataset...")
    loader, num_samples = prepare_dataloader(config, device, args.num_samples)
    
    print("\n🔍 Running evaluation...")
    results = evaluate_model(model, loader, device, config)
    
    # Print results
    print_results(results, checkpoint_info)
    
    # Save detailed results
    output_path = exp_dir / "evaluation" / f"eval_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Detailed results saved: {output_path}")
    
    # Save CLIP metrics separately
    clip_metrics_path = exp_dir / "evaluation" / "clip_metrics.json"
    with open(clip_metrics_path, 'w') as f:
        json.dump({
            'cosine_similarity': results['global']['cosine_similarity'],
            'retrieval': results['global']['retrieval']
        }, f, indent=2)
    print(f"   ✓ CLIP metrics: {clip_metrics_path}")
    
    # Save probabilistic metrics separately
    prob_metrics_path = exp_dir / "evaluation" / "probabilistic_metrics.json"
    with open(prob_metrics_path, 'w') as f:
        json.dump({
            'kl_divergence': results['kl_divergence'],
            'reconstruction_error': {
                'mse': results['global']['mse'],
                'rmse': results['global']['rmse'],
                'l2_distance_mean': results['global']['l2_distance_mean'],
                'l2_distance_std': results['global']['l2_distance_std']
            }
        }, f, indent=2)
    print(f"   ✓ Probabilistic metrics: {prob_metrics_path}")
    
    # Generate summary report
    print("\n📝 Generating summary report...")
    generate_summary_report(exp_dir, results, config, checkpoint_info)
    
    # Create notes template
    create_notes_template(exp_dir)
    
    print("\n" + "="*100)
    print("✅ EVALUATION COMPLETE!")
    print("="*100)
    print(f"\nResults location: {exp_dir}")
    print(f"\nNext steps:")
    print(f"  1. Review: {exp_dir / 'evaluation' / 'summary_report.md'}")
    print(f"  2. Add notes: {exp_dir / 'notes.md'}")
    print(f"  3. Decide: Generate reconstructions or try new experiment")
    print()


if __name__ == '__main__':
    main()
