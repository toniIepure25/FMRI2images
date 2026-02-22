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

# Import evaluation logic from the library
from fmri2img.eval.eval_comprehensive import (
    load_encoder,
    predict_clip_embeddings,
    compute_retrieval_metrics,
    compute_perceptual_metrics,
)


import os
import numpy as np
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
    parser.add_argument('--exp-name', type=str, required=True, help='Experiment name')
    parser.add_argument('--subject', type=str, default='subj01', help='NSD subject id')
    parser.add_argument('--encoder-type', type=str, default='unified',
                        choices=['ridge', 'mlp', 'two_stage', 'unified'])
    parser.add_argument('--nsd-root', type=str, default=None,
                        help='NSD data root (default: $NSD_DATA_ROOT)')
    parser.add_argument('--clip-cache', type=str, default=None,
                        help='Path to CLIP cache parquet')
    parser.add_argument('--fmri-path', type=str, default=None,
                        help='Path to preprocessed fMRI .npy')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()

    print("=" * 100)
    print(" " * 30 + "EXPERIMENT EVALUATION WORKFLOW")
    print("=" * 100)
    print(f"Experiment : {args.exp_name}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Subject    : {args.subject}")
    print()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    exp_dir = create_experiment_structure(args.exp_name)
    shutil.copy(args.config, exp_dir / "config.yaml")
    save_training_info(exp_dir, Path(args.checkpoint), config)

    device = args.device if torch.cuda.is_available() else 'cpu'
    nsd_root = Path(args.nsd_root or os.environ.get('NSD_DATA_ROOT', 'data/nsd'))

    print("Loading encoder...")
    encoder = load_encoder(args.encoder_type, args.checkpoint, device)

    stim_info_path = nsd_root / 'nsddata' / 'experiments' / 'nsd' / 'nsd_stim_info_merged.csv'
    from fmri2img.eval.eval_comprehensive import load_nsd_shared_1000, get_shared_1000_trials, average_fmri_reps

    shared_df = load_nsd_shared_1000(str(stim_info_path))
    trials, nsd_ids = get_shared_1000_trials(shared_df, args.subject, average_reps=True)

    fmri_path = args.fmri_path
    if fmri_path is None:
        fmri_path = f"data/preprocessed/{args.subject}/fmri_masked.npy"
    print(f"Loading fMRI data from {fmri_path}...")
    fmri_all = np.load(fmri_path)
    fmri_avg = average_fmri_reps(fmri_all, trials)

    clip_cache_path = args.clip_cache
    if clip_cache_path is None:
        clip_cache_path = str(nsd_root / 'clip_cache' / 'clip_cache.parquet')
    print(f"Loading CLIP targets from {clip_cache_path}...")
    import pandas as pd
    clip_df = pd.read_parquet(clip_cache_path)
    if 'nsdId' in clip_df.columns and clip_df.index.name != 'nsdId':
        clip_df = clip_df.set_index('nsdId')
    clip_col = 'clip_embedding' if 'clip_embedding' in clip_df.columns else clip_df.columns[-1]
    gt_embeddings = np.stack(clip_df.loc[nsd_ids, clip_col].values).astype(np.float32)

    print("Predicting CLIP embeddings...")
    pred_embeddings = predict_clip_embeddings(encoder, args.encoder_type, fmri_avg, device)

    print("Computing retrieval metrics...")
    retrieval = compute_retrieval_metrics(pred_embeddings, gt_embeddings)
    perceptual = compute_perceptual_metrics(pred_embeddings, gt_embeddings)

    results = {
        "subject": args.subject,
        "encoder_type": args.encoder_type,
        "num_samples": len(pred_embeddings),
        "retrieval": retrieval,
        "perceptual": perceptual,
    }

    for k, v in retrieval.items():
        if k != "per_sample_ranks":
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    for k, v in perceptual.items():
        if k != "per_sample_cosine":
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    output_path = exp_dir / "evaluation" / "eval_results.json"
    serializable = {k: v for k, v in results.items()
                    if k not in {"per_sample_ranks", "per_sample_cosine"}}
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"Saved results: {output_path}")

    create_notes_template(exp_dir)

    print("\n" + "=" * 100)
    print("EVALUATION COMPLETE")
    print("=" * 100)
    print(f"\nResults: {exp_dir}")
    print(f"  eval   : {output_path}")
    print(f"  config : {exp_dir / 'config.yaml'}")


if __name__ == '__main__':
    main()
