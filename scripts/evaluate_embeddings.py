#!/usr/bin/env python3
"""
Evaluate reconstruction quality at the CLIP embedding level.

This evaluates the fMRI → CLIP embedding prediction quality by comparing
predicted embeddings with ground truth CLIP embeddings from the cache.

Metrics:
- Cosine Similarity (primary metric)
- L2 Distance
- Top-K Retrieval Accuracy
- Correlation

This is more robust than image-level evaluation because:
1. Works for all test samples (not just those with GT images)
2. Measures semantic quality in CLIP space
3. Directly evaluates the encoder's learned mapping
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_predicted_embeddings(encoder_ckpt: Path, test_fmri: torch.Tensor, 
                               device: str = "cuda") -> torch.Tensor:
    """
    Load encoder and predict CLIP embeddings from fMRI.
    
    Args:
        encoder_ckpt: Path to encoder checkpoint
        test_fmri: Test fMRI data (N, 512) - already preprocessed
        device: Device to use
    
    Returns:
        Predicted CLIP embeddings (N, 512)
    """
    from fmri2img.models.encoders import TwoStageEncoder
    
    logger.info(f"Loading encoder from {encoder_ckpt}...")
    
    # Load checkpoint
    ckpt = torch.load(encoder_ckpt, map_location=device)
    
    # Get architecture config
    if 'config' in ckpt:
        config = ckpt['config']
    else:
        # Default config
        config = {
            'input_dim': 512,
            'latent_dim': 512,
            'output_dim': 512,
            'hidden_dims': [1024, 1024],
            'use_residual': True,
            'dropout': 0.3
        }
    
    # Create encoder
    encoder = TwoStageEncoder(**config).to(device)
    encoder.load_state_dict(ckpt['model_state_dict'])
    encoder.eval()
    
    logger.info("✓ Encoder loaded")
    
    # Predict embeddings
    logger.info("Predicting embeddings...")
    with torch.no_grad():
        test_fmri = test_fmri.to(device)
        predictions = encoder(test_fmri)
    
    logger.info(f"✓ Predicted embeddings: {predictions.shape}")
    return predictions.cpu()


def load_ground_truth_embeddings(clip_cache_path: Path, test_indices: np.ndarray) -> torch.Tensor:
    """
    Load ground truth CLIP embeddings from cache.
    
    Args:
        clip_cache_path: Path to CLIP cache (.npy file)
        test_indices: Indices of test samples in the full dataset
    
    Returns:
        Ground truth CLIP embeddings (N, 512)
    """
    logger.info(f"Loading ground truth embeddings from {clip_cache_path}...")
    
    # Load full CLIP cache
    clip_cache = np.load(clip_cache_path)
    logger.info(f"CLIP cache shape: {clip_cache.shape}")
    
    # Extract test embeddings
    gt_embeddings = clip_cache[test_indices]
    logger.info(f"✓ Loaded {len(gt_embeddings)} ground truth embeddings")
    
    return torch.from_numpy(gt_embeddings).float()


def compute_embedding_metrics(pred_emb: torch.Tensor, gt_emb: torch.Tensor) -> Dict[str, float]:
    """
    Compute metrics between predicted and ground truth embeddings.
    
    Args:
        pred_emb: Predicted embeddings (N, D)
        gt_emb: Ground truth embeddings (N, D)
    
    Returns:
        Dictionary of metrics
    """
    pred_np = pred_emb.numpy()
    gt_np = gt_emb.numpy()
    
    # Normalize embeddings
    pred_norm = pred_np / (np.linalg.norm(pred_np, axis=1, keepdims=True) + 1e-8)
    gt_norm = gt_np / (np.linalg.norm(gt_np, axis=1, keepdims=True) + 1e-8)
    
    # Cosine similarity (per sample)
    cos_sims = np.sum(pred_norm * gt_norm, axis=1)
    
    # L2 distance
    l2_dists = np.linalg.norm(pred_np - gt_np, axis=1)
    
    # Correlation (element-wise across all dimensions)
    pearson_corr, _ = pearsonr(pred_np.flatten(), gt_np.flatten())
    spearman_corr, _ = spearmanr(pred_np.flatten(), gt_np.flatten())
    
    metrics = {
        'mean_cosine_similarity': float(np.mean(cos_sims)),
        'median_cosine_similarity': float(np.median(cos_sims)),
        'std_cosine_similarity': float(np.std(cos_sims)),
        'min_cosine_similarity': float(np.min(cos_sims)),
        'max_cosine_similarity': float(np.max(cos_sims)),
        'mean_l2_distance': float(np.mean(l2_dists)),
        'median_l2_distance': float(np.median(l2_dists)),
        'pearson_correlation': float(pearson_corr),
        'spearman_correlation': float(spearman_corr),
    }
    
    return metrics, cos_sims, l2_dists


def compute_retrieval_metrics(pred_emb: torch.Tensor, gt_emb: torch.Tensor, 
                               k_values: List[int] = [1, 5, 10, 50]) -> Dict[str, float]:
    """
    Compute top-K retrieval accuracy.
    
    For each prediction, find the K nearest ground truth embeddings
    and check if the correct one is in the top K.
    
    Args:
        pred_emb: Predicted embeddings (N, D)
        gt_emb: Ground truth embeddings (N, D)
        k_values: List of K values to evaluate
    
    Returns:
        Dictionary of top-K accuracies
    """
    logger.info("Computing retrieval metrics...")
    
    pred_np = pred_emb.numpy()
    gt_np = gt_emb.numpy()
    
    # Normalize
    pred_norm = pred_np / (np.linalg.norm(pred_np, axis=1, keepdims=True) + 1e-8)
    gt_norm = gt_np / (np.linalg.norm(gt_np, axis=1, keepdims=True) + 1e-8)
    
    # Compute similarity matrix (N x N)
    # For each predicted embedding, compute similarity to all GT embeddings
    sim_matrix = cosine_similarity(pred_norm, gt_norm)
    
    # For each sample, get top-K most similar GT embeddings
    retrieval_metrics = {}
    
    for k in k_values:
        correct = 0
        for i in range(len(pred_np)):
            # Get indices of top-K most similar GT embeddings
            top_k_indices = np.argsort(sim_matrix[i])[-k:]
            
            # Check if correct index (i) is in top-K
            if i in top_k_indices:
                correct += 1
        
        accuracy = correct / len(pred_np)
        retrieval_metrics[f'top{k}_accuracy'] = float(accuracy)
        logger.info(f"Top-{k} Accuracy: {accuracy:.4f}")
    
    return retrieval_metrics


def create_visualizations(cos_sims: np.ndarray, l2_dists: np.ndarray, 
                         output_dir: Path):
    """Create visualization plots for embedding evaluation."""
    logger.info("Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Cosine similarity distribution
    ax = axes[0, 0]
    ax.hist(cos_sims, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(cos_sims), color='red', linestyle='--', 
               label=f'Mean: {np.mean(cos_sims):.4f}')
    ax.axvline(np.median(cos_sims), color='green', linestyle='--', 
               label=f'Median: {np.median(cos_sims):.4f}')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Frequency')
    ax.set_title('Cosine Similarity Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # L2 distance distribution
    ax = axes[0, 1]
    ax.hist(l2_dists, bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax.axvline(np.mean(l2_dists), color='red', linestyle='--', 
               label=f'Mean: {np.mean(l2_dists):.2f}')
    ax.set_xlabel('L2 Distance')
    ax.set_ylabel('Frequency')
    ax.set_title('L2 Distance Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Cosine similarity vs L2 distance scatter
    ax = axes[1, 0]
    ax.scatter(cos_sims, l2_dists, alpha=0.5, s=10)
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('L2 Distance')
    ax.set_title('Cosine Similarity vs L2 Distance')
    ax.grid(True, alpha=0.3)
    
    # Cumulative distribution
    ax = axes[1, 1]
    sorted_sims = np.sort(cos_sims)
    cumulative = np.arange(1, len(sorted_sims) + 1) / len(sorted_sims)
    ax.plot(sorted_sims, cumulative, linewidth=2)
    ax.axvline(np.median(cos_sims), color='green', linestyle='--', 
               label=f'Median: {np.median(cos_sims):.4f}')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution Function')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "embedding_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved visualization to {output_dir / 'embedding_metrics.png'}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate embeddings at CLIP level")
    parser.add_argument("--ckpt", type=str, required=True, help="Encoder checkpoint")
    parser.add_argument("--subject", type=str, required=True, help="Subject ID")
    parser.add_argument("--index-root", type=str, required=True, help="Index root directory")
    parser.add_argument("--preproc-dir", type=str, required=True, help="Preprocessing directory")
    parser.add_argument("--clip-cache", type=str, required=True, help="CLIP cache file (.npy)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit test samples")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("EMBEDDING-LEVEL EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {args.ckpt}")
    logger.info(f"Subject: {args.subject}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 80)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load index and create test split
    logger.info("Loading index...")
    from fmri2img.data.nsd_index_reader import read_subject_index
    index_data = read_subject_index(args.index_root, args.subject)
    index_df = pd.DataFrame(index_data)
    
    # Apply same split (80/10/10) with seed 42
    n_total = len(index_df)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    
    index_shuffled = index_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = index_shuffled[n_train + n_val:].reset_index(drop=True)
    
    # Get original indices (before shuffling)
    test_original_indices = test_df.index.values
    
    if args.limit:
        test_df = test_df.head(args.limit)
        test_original_indices = test_original_indices[:args.limit]
    
    logger.info(f"Total samples: {n_total}")
    logger.info(f"Test samples: {len(test_df)}")
    
    # Load preprocessing and fMRI data
    logger.info("Loading fMRI data...")
    from fmri2img.data.preprocess import NSDPreprocessor
    from fmri2img.io.s3 import get_s3_filesystem, NIfTILoader
    
    # NSDPreprocessor expects base_dir, it adds subject internally
    preproc_base = Path(args.preproc_dir).parent if Path(args.preproc_dir).name == args.subject else args.preproc_dir
    preprocessor = NSDPreprocessor(args.subject, str(preproc_base))
    if not preprocessor.load_artifacts():
        raise RuntimeError(f"Failed to load preprocessing from {preproc_base}/{args.subject}")
    
    s3_fs = get_s3_filesystem()
    nifti_loader = NIfTILoader(s3_fs)
    
    fmri_data = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Loading fMRI"):
        beta_path = row.get("beta_path", row.get("beta_file"))
        beta_index = int(row.get("beta_index", row.get("volume_index", 0)))
        
        img = nifti_loader.load(beta_path)
        vol = img.slicer[..., beta_index].get_fdata().astype(np.float32)
        
        fmri_vec = preprocessor.transform(vol)
        fmri_data.append(fmri_vec)
    
    fmri_data = torch.from_numpy(np.vstack(fmri_data)).float()
    logger.info(f"✓ Loaded fMRI data: {fmri_data.shape}")
    
    # Load ground truth CLIP embeddings
    gt_embeddings = load_ground_truth_embeddings(Path(args.clip_cache), test_original_indices)
    
    # Predict embeddings
    pred_embeddings = load_predicted_embeddings(Path(args.ckpt), fmri_data, args.device)
    
    # Compute metrics
    logger.info("\nComputing embedding metrics...")
    metrics, cos_sims, l2_dists = compute_embedding_metrics(pred_embeddings, gt_embeddings)
    
    # Compute retrieval metrics
    retrieval_metrics = compute_retrieval_metrics(pred_embeddings, gt_embeddings)
    metrics.update(retrieval_metrics)
    
    # Log summary
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"n_samples: {len(test_df)}")
    logger.info(f"mean_cosine_similarity: {metrics['mean_cosine_similarity']:.4f}")
    logger.info(f"median_cosine_similarity: {metrics['median_cosine_similarity']:.4f}")
    logger.info(f"mean_l2_distance: {metrics['mean_l2_distance']:.4f}")
    logger.info(f"pearson_correlation: {metrics['pearson_correlation']:.4f}")
    logger.info(f"top1_accuracy: {metrics['top1_accuracy']:.4f}")
    logger.info(f"top5_accuracy: {metrics['top5_accuracy']:.4f}")
    logger.info(f"top10_accuracy: {metrics['top10_accuracy']:.4f}")
    logger.info("=" * 80)
    
    # Save results
    with open(output_dir / "embedding_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save per-sample metrics
    per_sample_df = pd.DataFrame({
        'sample_idx': range(len(cos_sims)),
        'cosine_similarity': cos_sims,
        'l2_distance': l2_dists,
    })
    per_sample_df.to_csv(output_dir / "per_sample_metrics.csv", index=False)
    
    # Create visualizations
    create_visualizations(cos_sims, l2_dists, output_dir)
    
    logger.info(f"\n✓ Results saved to {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
