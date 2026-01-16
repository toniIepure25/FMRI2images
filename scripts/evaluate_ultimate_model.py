#!/usr/bin/env python3
"""
Evaluate ProbabilisticMultiLayerTwoStageEncoder model with comprehensive CLIP metrics.

This script evaluates the trained model using:
- Cosine similarity (mean, median, std)
- CLIP retrieval accuracy (Top-1, Top-5, Top-10)
- Mean rank in retrieval task
- Per-layer output quality (if enabled)

Usage:
    python scripts/evaluate_ultimate_model.py \
        --checkpoint checkpoints/best_model.pt \
        --config experiments/ultimate_novel_subj01.yaml \
        --output-dir outputs/eval/exp001_baseline \
        --num-samples 1000

Author: Bachelor Thesis - Ultimate Model Evaluation
"""

import argparse
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import mean_squared_error

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fmri2img.models.encoders import ProbabilisticMultiLayerTwoStageEncoder
from fmri2img.data.torch_dataset import NSDIterableDataset
from fmri2img.data.preprocess import NSDPreprocessor


def load_checkpoint(checkpoint_path: Path, config: dict, device: str) -> Tuple[torch.nn.Module, dict]:
    """Load model from checkpoint."""
    print(f"\n📥 Loading checkpoint: {checkpoint_path.name}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Auto-detect input dimension from checkpoint (try multiple possible keys)
    input_dim = None
    state_dict = checkpoint['model_state_dict']
    
    # Try different possible input layer names
    for key in ['input_projection.weight', 'encoder.0.weight', 'fc1.weight', 'input_layer.weight']:
        if key in state_dict:
            input_dim = state_dict[key].shape[1]
            break
    
    # Fallback: find first layer with 'weight' that looks like input
    if input_dim is None:
        for key in sorted(state_dict.keys()):
            if 'weight' in key and len(state_dict[key].shape) == 2:
                input_dim = state_dict[key].shape[1]
                print(f"   Auto-detected input_dim from: {key}")
                break
    
    if input_dim is None:
        raise ValueError("Could not auto-detect input dimension from checkpoint")
    
    # Initialize model
    model = ProbabilisticMultiLayerTwoStageEncoder(
        input_dim=input_dim,
        latent_dim=config['model']['latent_dim'],
        output_dim=config['model']['output_dim'],
        n_blocks=config['model']['n_blocks'],
        head_hidden_dim=config['model']['head_hidden_dim'],
        dropout_rate=config['model']['dropout_rate'],
        enabled_layers=config['model']['enabled_layers'],
        predict_text_clip=config['model'].get('predict_text_clip', False)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    epoch = checkpoint.get('epoch', 'N/A')
    print(f"   ✓ Loaded from epoch: {epoch}")
    print(f"   ✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   ✓ Input dimension: {model.input_projection.weight.shape[1]:,}")
    
    return model, checkpoint


def prepare_dataloader(config: dict, device: str, num_samples: int = None):
    """Prepare evaluation dataloader with preprocessing."""
    print(f"\n📊 Preparing evaluation dataset")
    
    # Load preprocessing
    subject = config['data']['subject']
    preprocessor = None
    preprocessor_dir = Path(f"outputs/preproc/{subject}")
    if preprocessor_dir.exists():
        preprocessor = NSDPreprocessor(subject=subject, out_dir="outputs/preproc")
        if preprocessor.load_artifacts():
            print(f"   ✓ Preprocessing loaded ({preprocessor.n_voxels_kept:,} voxels)")
        else:
            preprocessor = None
    
    # Load index
    index_df = pd.read_csv(config['data']['index_path'])
    print(f"   ✓ Loaded index: {len(index_df)} samples")
    
    # Use validation split (last 10%)
    val_size = int(len(index_df) * 0.1)
    val_df = index_df.iloc[-val_size:]
    
    if num_samples and num_samples < len(val_df):
        val_df = val_df.sample(n=num_samples, random_state=42)
    
    print(f"   ✓ Evaluation samples: {len(val_df)}")
    
    # Create dataset
    dataset = NSDIterableDataset(
        index_df=val_df,
        clip_cache_path=config['data']['clip_cache_path'],
        preprocessor=preprocessor,
        augment=False  # No augmentation for evaluation
    )
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        num_workers=4,
        pin_memory=True
    )
    
    return loader, len(val_df)


def compute_cosine_similarity(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Compute cosine similarity metrics."""
    pred_norm = F.normalize(predictions, p=2, dim=1)
    target_norm = F.normalize(targets, p=2, dim=1)
    
    cosine_sims = (pred_norm * target_norm).sum(dim=1)
    
    return {
        'cosine_mean': cosine_sims.mean().item(),
        'cosine_median': cosine_sims.median().item(),
        'cosine_std': cosine_sims.std().item(),
        'cosine_min': cosine_sims.min().item(),
        'cosine_max': cosine_sims.max().item()
    }


def compute_retrieval_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Compute retrieval metrics (Top-K accuracy, mean rank).
    
    For each prediction, find its rank among all targets based on cosine similarity.
    """
    pred_norm = F.normalize(predictions, p=2, dim=1)
    target_norm = F.normalize(targets, p=2, dim=1)
    
    # Compute similarity matrix (batch_size x batch_size)
    similarities = torch.mm(pred_norm, target_norm.t())
    
    # For each row (prediction), find rank of correct target (diagonal)
    ranks = []
    top1_correct = 0
    top5_correct = 0
    top10_correct = 0
    
    for i in range(similarities.shape[0]):
        # Get similarities for this prediction
        sims = similarities[i]
        
        # Rank targets by similarity (higher = better)
        sorted_indices = torch.argsort(sims, descending=True)
        
        # Find rank of correct target (index i)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1  # 1-indexed
        ranks.append(rank)
        
        if rank == 1:
            top1_correct += 1
        if rank <= 5:
            top5_correct += 1
        if rank <= 10:
            top10_correct += 1
    
    batch_size = similarities.shape[0]
    
    return {
        'top1_accuracy': top1_correct / batch_size,
        'top5_accuracy': top5_correct / batch_size,
        'top10_accuracy': top10_correct / batch_size,
        'mean_rank': np.mean(ranks),
        'median_rank': np.median(ranks)
    }


def evaluate_model(model: torch.nn.Module, loader, device: str, config: dict) -> Dict[str, any]:
    """Run full evaluation."""
    print(f"\n🔍 Running evaluation...")
    
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_kl_divs = []
    batch_metrics = []
    
    with torch.no_grad():
        for batch_idx, (fmri, clip_img, clip_text) in enumerate(tqdm(loader, desc="Evaluating")):
            fmri = fmri.to(device)
            clip_img = clip_img.to(device)
            
            # Forward pass
            outputs = model(fmri)
            pred_clip = outputs['final']['mu']  # Use mean prediction (no sampling)
            
            # Store KL divergence
            if 'kl_div' in outputs['final']:
                all_kl_divs.append(outputs['final']['kl_div'].mean().item())
            
            # Compute batch-level metrics
            batch_cos = compute_cosine_similarity(pred_clip, clip_img)
            batch_retrieval = compute_retrieval_metrics(pred_clip, clip_img)
            
            batch_metrics.append({
                **batch_cos,
                **batch_retrieval,
                'batch_size': pred_clip.shape[0]
            })
            
            # Accumulate for global metrics
            all_predictions.append(pred_clip.cpu())
            all_targets.append(clip_img.cpu())
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    print(f"   ✓ Evaluated {all_predictions.shape[0]} samples")
    
    # Compute global metrics
    print(f"\n📈 Computing global metrics...")
    
    global_cosine = compute_cosine_similarity(all_predictions, all_targets)
    
    # Global retrieval (more expensive, sample if too large)
    if all_predictions.shape[0] > 2000:
        print(f"   ⚠️  Large dataset ({all_predictions.shape[0]} samples), sampling 2000 for global retrieval")
        sample_indices = torch.randperm(all_predictions.shape[0])[:2000]
        global_retrieval = compute_retrieval_metrics(
            all_predictions[sample_indices], 
            all_targets[sample_indices]
        )
    else:
        global_retrieval = compute_retrieval_metrics(all_predictions, all_targets)
    
    # Aggregate batch metrics
    batch_cosine_mean = np.mean([m['cosine_mean'] for m in batch_metrics])
    batch_top1_mean = np.mean([m['top1_accuracy'] for m in batch_metrics])
    batch_top5_mean = np.mean([m['top5_accuracy'] for m in batch_metrics])
    
    # MSE and L2 distance
    mse = F.mse_loss(all_predictions, all_targets).item()
    l2_distances = torch.norm(all_predictions - all_targets, p=2, dim=1)
    
    results = {
        'global': {
            'cosine_similarity': global_cosine,
            'retrieval': global_retrieval,
            'mse': mse,
            'rmse': np.sqrt(mse),
            'l2_distance_mean': l2_distances.mean().item(),
            'l2_distance_std': l2_distances.std().item()
        },
        'batch_averaged': {
            'cosine_mean': batch_cosine_mean,
            'top1_accuracy': batch_top1_mean,
            'top5_accuracy': batch_top5_mean
        },
        'kl_divergence': {
            'mean': np.mean(all_kl_divs) if all_kl_divs else None,
            'std': np.std(all_kl_divs) if all_kl_divs else None
        },
        'num_samples': all_predictions.shape[0]
    }
    
    return results


def print_results(results: Dict, checkpoint_info: Dict):
    """Print evaluation results in formatted table."""
    print("\n" + "="*100)
    print(" " * 35 + "EVALUATION RESULTS")
    print("="*100)
    
    print(f"\nCheckpoint Info:")
    print(f"  Epoch: {checkpoint_info.get('epoch', 'N/A')}")
    print(f"  Samples: {results['num_samples']}")
    
    print(f"\n{'GLOBAL METRICS':-^100}")
    
    print(f"\nCosine Similarity:")
    for key, val in results['global']['cosine_similarity'].items():
        print(f"  {key:<30} {val:>10.4f}")
    
    print(f"\nRetrieval Accuracy:")
    for key, val in results['global']['retrieval'].items():
        print(f"  {key:<30} {val:>10.4f}")
    
    print(f"\nReconstruction Error:")
    print(f"  {'MSE':<30} {results['global']['mse']:>10.4f}")
    print(f"  {'RMSE':<30} {results['global']['rmse']:>10.4f}")
    print(f"  {'L2 Distance (mean)':<30} {results['global']['l2_distance_mean']:>10.4f}")
    print(f"  {'L2 Distance (std)':<30} {results['global']['l2_distance_std']:>10.4f}")
    
    if results['kl_divergence']['mean'] is not None:
        print(f"\nKL Divergence:")
        print(f"  {'Mean':<30} {results['kl_divergence']['mean']:>10.4f}")
        print(f"  {'Std':<30} {results['kl_divergence']['std']:>10.4f}")
    
    print("\n" + "="*100 + "\n")
    
    # Quick summary
    print("📊 SUMMARY:")
    print(f"   Cosine Similarity: {results['global']['cosine_similarity']['cosine_mean']:.4f} ± {results['global']['cosine_similarity']['cosine_std']:.4f}")
    print(f"   Top-1 Retrieval:   {results['global']['retrieval']['top1_accuracy']:.2%}")
    print(f"   Top-5 Retrieval:   {results['global']['retrieval']['top5_accuracy']:.2%}")
    print(f"   Mean Rank:         {results['global']['retrieval']['mean_rank']:.1f}")
    print()


def save_results(results: Dict, output_dir: Path, checkpoint_name: str):
    """Save results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"eval_{checkpoint_name}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Ultimate Novel Model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output-dir', type=str, default='outputs/eval', help='Output directory')
    parser.add_argument('--num-samples', type=int, default=None, help='Number of samples to evaluate (default: all validation)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    
    print("="*100)
    print(" " * 30 + "ULTIMATE MODEL EVALUATION")
    print("="*100)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    
    # Load model
    model, checkpoint_info = load_checkpoint(checkpoint_path, config, device)
    
    # Prepare data
    loader, num_samples = prepare_dataloader(config, device, args.num_samples)
    
    # Evaluate
    results = evaluate_model(model, loader, device, config)
    
    # Print results
    print_results(results, checkpoint_info)
    
    # Save results
    save_results(results, output_dir, checkpoint_path.stem)
    
    print(f"✅ Evaluation complete!")


if __name__ == '__main__':
    main()
