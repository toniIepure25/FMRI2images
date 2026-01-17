#!/usr/bin/env python3
"""
Stage 1 Embedding Evaluation CLI
=================================

Evaluates fMRI → CLIP embedding predictions using standard metrics.

Usage:
    python scripts/eval_stage1_embeddings.py \
        --checkpoint runs/exp001/checkpoints/best_model.pt \
        --config experiments/ultimate_novel_subj01.yaml \
        --split val \
        --output-dir experimental_results/exp001/evaluation \
        --num-samples 1000 \
        --gallery-sizes 2 10 50 100 1000 \
        --seed 42

Author: Research-grade evaluation suite
Date: January 2026
"""

import argparse
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fmri2img.models.encoders import ProbabilisticMultiLayerTwoStageEncoder, load_probabilistic_encoder
from fmri2img.data.torch_dataset import NSDIterableDataset
from fmri2img.data.preprocess import NSDPreprocessor
from fmri2img.eval.embedding_eval import evaluate_embeddings, EmbeddingEvalResults

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: Path, device: str) -> Tuple[torch.nn.Module, Dict]:
    """Load probabilistic model from checkpoint."""
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    model, meta = load_probabilistic_encoder(str(checkpoint_path), map_location=device)
    model.eval()
    
    logger.info(f"Loaded model from epoch {meta.get('epoch', 'N/A')}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, meta


def load_config(config_path: Path) -> Dict:
    """Load experiment config."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def prepare_dataset(
    config: Dict,
    split: str,
    num_samples: Optional[int],
    device: str
):
    """Prepare evaluation dataset."""
    logger.info(f"Preparing {split} dataset")
    
    subject = config['data']['subject']
    
    # Load preprocessing
    preprocessor = None
    preprocessor_dir = Path(f"outputs/preproc/{subject}")
    if preprocessor_dir.exists():
        preprocessor = NSDPreprocessor(subject=subject, out_dir="outputs/preproc")
        if preprocessor.load_artifacts():
            logger.info("✓ Preprocessing loaded")
    
    # Load index
    index_path = config['data']['index_path']
    
    # Determine split
    import pandas as pd
    if index_path.endswith('.parquet'):
        index_df = pd.read_parquet(index_path)
    else:
        index_df = pd.read_csv(index_path)
    
    if split == 'val':
        val_size = int(len(index_df) * 0.1)
        index_df = index_df.iloc[-val_size:]
    elif split == 'test':
        # Use last 10% as test (adjust based on your split strategy)
        test_size = int(len(index_df) * 0.1)
        index_df = index_df.iloc[-test_size:]
    
    if num_samples and num_samples < len(index_df):
        index_df = index_df.iloc[:num_samples]
    
    logger.info(f"Dataset: {len(index_df)} samples")
    
    # Create dataset
    dataset = NSDIterableDataset(
        index_path_or_root=index_path,
        subject=subject,
        shuffle=False,
        limit=len(index_df),
        preprocessor=preprocessor,
        clip_cache=config['data']['clip_cache_path']
    )
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        num_workers=4,
        pin_memory=(device == 'cuda')
    )
    
    return loader, len(index_df)


def extract_embeddings(
    model: torch.nn.Module,
    dataloader,
    device: str,
    use_mean: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract predicted and ground truth embeddings.
    
    Args:
        model: Probabilistic model
        dataloader: Data loader
        device: Device
        use_mean: If True, use mean (μ); if False, sample from q(z|x)
    
    Returns:
        predictions, ground_truth (both N x D numpy arrays)
    """
    logger.info("Extracting embeddings...")
    
    all_predictions = []
    all_ground_truth = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting")):
            # Handle different batch formats
            if len(batch) == 2:
                fmri, clip_gt = batch
            elif len(batch) == 3:
                fmri, clip_gt, _ = batch  # Ignore metadata/index
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements")
            
            fmri = fmri.to(device)
            clip_gt = clip_gt.to(device)
            
            # Forward pass
            outputs, _ = model(fmri, sample=(not use_mean), return_kl=False)
            
            # Get 'final' output (main CLIP embedding)
            if use_mean:
                pred = outputs['final'].mu
            else:
                pred = outputs['final'].z
            
            all_predictions.append(pred.cpu().numpy())
            all_ground_truth.append(clip_gt.cpu().numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)
    ground_truth = np.concatenate(all_ground_truth, axis=0)
    
    logger.info(f"Extracted: predictions {predictions.shape}, GT {ground_truth.shape}")
    
    return predictions, ground_truth


def save_results(
    results: EmbeddingEvalResults,
    output_dir: Path,
    config: Dict,
    checkpoint_path: Path,
    split: str,
    args
):
    """Save evaluation results to JSON and create summary report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics as JSON
    metrics_path = output_dir / f"embedding_metrics_{split}.json"
    metrics_dict = results.to_dict()
    metrics_dict['metadata'] = {
        'checkpoint': str(checkpoint_path),
        'config': str(args.config),
        'split': split,
        'num_samples': args.num_samples,
        'seed': args.seed,
        'gallery_sizes': args.gallery_sizes if args.gallery_sizes else [],
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    logger.info(f"✓ Saved metrics to {metrics_path}")
    
    # Create markdown summary
    summary_path = output_dir / f"embedding_summary_{split}.md"
    
    with open(summary_path, 'w') as f:
        f.write(f"# Stage 1 Embedding Evaluation Summary\n\n")
        f.write(f"**Split**: {split}  \n")
        f.write(f"**Samples**: {metrics_dict['metadata']['num_samples']}  \n")
        f.write(f"**Checkpoint**: `{checkpoint_path.name}`  \n\n")
        
        f.write(f"## Retrieval Metrics\n\n")
        f.write(f"| Metric | Value | Chance |\n")
        f.write(f"|--------|-------|--------|\n")
        N = args.num_samples or 1000
        f.write(f"| Top-1 Accuracy | {results.top1_accuracy:.3f} | {1.0/N:.3f} |\n")
        f.write(f"| Top-5 Accuracy | {results.top5_accuracy:.3f} | {5.0/N:.3f} |\n")
        f.write(f"| Top-10 Accuracy | {results.top10_accuracy:.3f} | {10.0/N:.3f} |\n")
        f.write(f"| Mean Rank | {results.mean_rank:.1f} | {N/2:.1f} |\n")
        f.write(f"| Median Rank | {results.median_rank:.1f} | {N/2:.1f} |\n")
        f.write(f"| MRR | {results.mrr:.3f} | - |\n\n")
        
        f.write(f"## Identification\n\n")
        f.write(f"| Metric | Value | Chance |\n")
        f.write(f"|--------|-------|--------|\n")
        f.write(f"| 2AFC Accuracy | {results.twoafc_accuracy:.3f} (95% CI: [{results.twoafc_ci_lower:.3f}, {results.twoafc_ci_upper:.3f}]) | 0.500 |\n\n")
        
        f.write(f"## Separability\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| ROC AUC | {results.separability_auc:.3f} |\n")
        f.write(f"| Cohen's d | {results.cohens_d:.3f} |\n\n")
        
        f.write(f"## Representational Similarity (RSA)\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Spearman r | {results.rsa_spearman:.3f} (p={results.rsa_pvalue:.2e}) |\n\n")
        
        f.write(f"## Collapse Diagnostics\n\n")
        f.write(f"| Metric | Value | Interpretation |\n")
        f.write(f"|--------|-------|----------------|\n")
        f.write(f"| Collapse Ratio | {results.collapse_ratio:.3f} | Should be ~1.0 |\n")
        f.write(f"| Avg Pairwise Sim | {results.avg_pairwise_sim:.3f} | Lower is better |\n")
        f.write(f"| Mean Centering | {results.mean_centering:.3f} | Lower is better |\n\n")
        
        if results.retrieval_by_gallery_size:
            f.write(f"## Retrieval vs Gallery Size\n\n")
            f.write(f"| Gallery Size | Top-1 | Top-5 | Mean Rank | Chance |\n")
            f.write(f"|--------------|-------|-------|-----------|--------|\n")
            for size in sorted(results.retrieval_by_gallery_size.keys()):
                metrics = results.retrieval_by_gallery_size[size]
                f.write(f"| {size} | {metrics['top1_accuracy']:.3f} | {metrics['top5_accuracy']:.3f} | {metrics['mean_rank']:.1f} | {metrics['chance_top1']:.3f} |\n")
            f.write("\n")
        
        f.write(f"## Interpretation\n\n")
        
        # Provide interpretation
        if results.top1_accuracy > 0.10:
            f.write(f"✅ **Top-1 retrieval ({results.top1_accuracy:.1%})** is strong (>10%).\n\n")
        elif results.top1_accuracy > 0.05:
            f.write(f"⚠️ **Top-1 retrieval ({results.top1_accuracy:.1%})** is moderate (5-10%).\n\n")
        else:
            f.write(f"❌ **Top-1 retrieval ({results.top1_accuracy:.1%})** is weak (<5%).\n\n")
        
        if results.twoafc_accuracy > 0.65:
            f.write(f"✅ **2AFC accuracy ({results.twoafc_accuracy:.1%})** is well above chance (50%).\n\n")
        elif results.twoafc_accuracy > 0.55:
            f.write(f"⚠️ **2AFC accuracy ({results.twoafc_accuracy:.1%})** is above chance but modest.\n\n")
        else:
            f.write(f"❌ **2AFC accuracy ({results.twoafc_accuracy:.1%})** is near chance.\n\n")
        
        if results.collapse_ratio > 0.8 and results.collapse_ratio < 1.2:
            f.write(f"✅ **No collapse detected** (ratio={results.collapse_ratio:.2f} is close to 1.0).\n\n")
        else:
            f.write(f"⚠️ **Possible collapse** (ratio={results.collapse_ratio:.2f} deviates from 1.0).\n\n")
    
    logger.info(f"✓ Saved summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Stage 1 fMRI → CLIP embedding predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python scripts/eval_stage1_embeddings.py \\
      --checkpoint runs/exp001/checkpoints/best_model.pt \\
      --config experiments/ultimate_novel_subj01.yaml \\
      --output-dir experimental_results/exp001/evaluation

  # With gallery size analysis
  python scripts/eval_stage1_embeddings.py \\
      --checkpoint runs/exp001/checkpoints/best_model.pt \\
      --config experiments/ultimate_novel_subj01.yaml \\
      --output-dir experimental_results/exp001/evaluation \\
      --gallery-sizes 2 10 50 100 1000 \\
      --num-samples 2000
        """
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to experiment config (.yaml file)')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of samples to evaluate (default: all)')
    parser.add_argument('--gallery-sizes', type=int, nargs='+', default=None,
                        help='Gallery sizes for scaling analysis (e.g., 2 10 100 1000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda, cpu, or auto)')
    
    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Load config
    config = load_config(Path(args.config))
    
    # Load model
    model, meta = load_model(Path(args.checkpoint), device)
    
    # Prepare dataset
    dataloader, num_samples = prepare_dataset(config, args.split, args.num_samples, device)
    
    # Extract embeddings
    predictions, ground_truth = extract_embeddings(model, dataloader, device, use_mean=True)
    
    # Run evaluation
    logger.info("Running embedding evaluation...")
    results = evaluate_embeddings(
        predictions=predictions,
        ground_truth=ground_truth,
        gallery_sizes=args.gallery_sizes,
        normalize=True,
        seed=args.seed
    )
    
    logger.info("Evaluation complete!")
    
    # Print summary
    print("\n" + "="*60)
    print("STAGE 1 EMBEDDING EVALUATION RESULTS")
    print("="*60)
    print(f"Top-1 Retrieval:  {results.top1_accuracy:.1%}")
    print(f"Top-5 Retrieval:  {results.top5_accuracy:.1%}")
    print(f"Top-10 Retrieval: {results.top10_accuracy:.1%}")
    print(f"Mean Rank:        {results.mean_rank:.1f}")
    print(f"2AFC Accuracy:    {results.twoafc_accuracy:.1%} (CI: [{results.twoafc_ci_lower:.1%}, {results.twoafc_ci_upper:.1%}])")
    print(f"ROC AUC:          {results.separability_auc:.3f}")
    print(f"RSA Spearman:     {results.rsa_spearman:.3f} (p={results.rsa_pvalue:.2e})")
    print(f"Collapse Ratio:   {results.collapse_ratio:.3f}")
    print("="*60 + "\n")
    
    # Save results
    save_results(results, Path(args.output_dir), config, Path(args.checkpoint), args.split, args)
    
    logger.info("✓ Done! Results saved to " + args.output_dir)


if __name__ == '__main__':
    main()
