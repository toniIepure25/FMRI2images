#!/usr/bin/env python3
"""
Comprehensive Evaluation Suite for NSD fMRI → Image Reconstruction
==================================================================

Complete evaluation pipeline for fMRI reconstruction models including:
1. **NSD Shared 1000 Evaluation** - Standard benchmark with 3 fMRI repetitions
2. **Multi-strategy Generation** - Compare single/best-of-N/BOI-lite
3. **Retrieval Metrics** - R@K, ranking statistics
4. **Perceptual Metrics** - CLIPScore, SSIM, LPIPS
5. **Brain Alignment** - Encoding model correlation
6. **Statistical Testing** - Significance tests across strategies

The NSD Shared 1000 is a standard test set where all 8 subjects viewed the same
1000 images, each with 3 fMRI repetitions. This allows for:
- Averaging fMRI across repetitions (higher SNR)
- Direct comparison across subjects
- Comparison with published results (MindEye2, Brain-Diffuser)

Usage:
    # Full evaluation with all strategies
    python scripts/eval_comprehensive.py \\
        --subject subj01 \\
        --encoder-checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \\
        --encoder-type two_stage \\
        --output-dir outputs/eval_comprehensive \\
        --strategies single best_of_8 boi_lite \\
        --clip-cache outputs/clip_cache/clip.parquet
    
    # Quick evaluation (single strategy only)
    python scripts/eval_comprehensive.py \\
        --subject subj01 \\
        --encoder-checkpoint checkpoints/mlp/subj01/mlp.pt \\
        --encoder-type mlp \\
        --output-dir outputs/eval_quick \\
        --strategies single \\
        --no-brain-alignment

Scientific Context:
- NSD Shared 1000: Standard benchmark (Allen et al. 2022)
- CLIPScore: Perceptual similarity metric (Hessel et al. 2021)
- Brain alignment: Encoding model correlation (Naselaris et al. 2011)

References:
- Allen et al. (2022). "A massive 7T fMRI dataset to bridge cognitive neuroscience and AI"
- Scotti et al. (2024). "Reconstructing the Mind's Eye: fMRI to Image with Contrastive Learning"
- Ozcelik & VanRullen (2023). "Brain-optimized inference via diffusion models"
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import stats
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import project modules
from fmri2img.data.nsd_index_reader import read_subject_index
from fmri2img.data.preprocess import NSDPreprocessor
from fmri2img.data.clip_cache import CLIPCache
from fmri2img.io.s3 import get_s3_filesystem, NIfTILoader
from fmri2img.models.ridge import RidgeEncoder
from fmri2img.models.mlp import load_mlp
from fmri2img.models.encoders import load_two_stage_encoder
from fmri2img.models.encoding_model import load_encoding_model
from fmri2img.models.train_utils import train_val_test_split, extract_features_and_targets
from fmri2img.eval.retrieval import retrieval_at_k, compute_ranking_metrics, cosine_sim
from fmri2img.generation.advanced_diffusion import (
    generate_best_of_n,
    refine_with_boi_lite,
    generate_with_all_strategies
)

# Optionally import perceptual metrics if available
try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    logger.warning("LPIPS not available. Install with: pip install lpips")

try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False
    logger.warning("SSIM not available. Install with: pip install torchmetrics")


def load_nsd_shared_1000(stim_info_path: str) -> pd.DataFrame:
    """
    Load NSD Shared 1000 stimulus metadata.
    
    The NSD Shared 1000 are 1000 images shown to all 8 subjects with 3 repetitions.
    This is the standard benchmark for cross-subject comparison.
    
    Args:
        stim_info_path: Path to nsd_stim_info_merged.csv
        
    Returns:
        DataFrame with shared1000=True rows, containing:
        - nsdId: NSD stimulus ID (0-72999)
        - cocoId: COCO image ID
        - subject{1-8}_rep{0,1,2}: Trial indices for each repetition
        
    Example:
        >>> shared = load_nsd_shared_1000("cache/nsd_stim_info_merged.csv")
        >>> print(f"Found {len(shared)} shared images")
        Found 1000 shared images
        >>> # Get trial indices for subj01, all 3 reps
        >>> trials_rep0 = shared["subject1_rep0"].values
        >>> trials_rep1 = shared["subject1_rep1"].values
        >>> trials_rep2 = shared["subject1_rep2"].values
    """
    logger.info(f"Loading NSD stimulus info from {stim_info_path}")
    df = pd.read_csv(stim_info_path)
    
    # Filter to shared 1000
    shared = df[df["shared1000"] == True].copy()
    logger.info(f"Found {len(shared)} shared images")
    
    if len(shared) != 1000:
        logger.warning(f"Expected 1000 shared images, found {len(shared)}")
    
    return shared


def get_shared_1000_trials(
    shared_df: pd.DataFrame,
    subject: str,
    average_reps: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get trial indices and nsdIds for NSD Shared 1000.
    
    Args:
        shared_df: Shared 1000 metadata from load_nsd_shared_1000()
        subject: Subject ID (e.g., "subj01")
        average_reps: If True, return all 3 repetitions for averaging
        
    Returns:
        trials: Trial indices, shape (1000,) or (1000, 3) if average_reps
        nsd_ids: NSD stimulus IDs, shape (1000,)
        
    Example:
        >>> shared = load_nsd_shared_1000("cache/nsd_stim_info_merged.csv")
        >>> trials, nsd_ids = get_shared_1000_trials(shared, "subj01", average_reps=True)
        >>> print(trials.shape)  # (1000, 3) - 3 repetitions
    """
    subj_num = int(subject.replace("subj", "").replace("0", ""))
    
    if average_reps:
        # Get all 3 repetitions
        rep0 = shared_df[f"subject{subj_num}_rep0"].values
        rep1 = shared_df[f"subject{subj_num}_rep1"].values
        rep2 = shared_df[f"subject{subj_num}_rep2"].values
        
        # Stack into (1000, 3)
        trials = np.stack([rep0, rep1, rep2], axis=1)
        logger.info(f"Loaded {len(trials)} shared images with 3 repetitions each")
    else:
        # Just use first repetition
        trials = shared_df[f"subject{subj_num}_rep0"].values
        logger.info(f"Loaded {len(trials)} shared images (rep 0 only)")
    
    nsd_ids = shared_df["nsdId"].values
    
    return trials, nsd_ids


def average_fmri_reps(
    fmri_data: np.ndarray,
    trial_indices: np.ndarray
) -> np.ndarray:
    """
    Average fMRI across repetitions for higher SNR.
    
    Args:
        fmri_data: All fMRI trials, shape (n_trials, n_voxels)
        trial_indices: Trial indices for each repetition, shape (n_images, n_reps)
        
    Returns:
        averaged: Averaged fMRI, shape (n_images, n_voxels)
        
    Example:
        >>> fmri = np.random.randn(30000, 15724)  # All trials
        >>> trials = np.array([[100, 200, 300], [150, 250, 350]])  # 2 images, 3 reps each
        >>> avg = average_fmri_reps(fmri, trials)
        >>> print(avg.shape)  # (2, 15724)
    """
    n_images, n_reps = trial_indices.shape
    _, n_voxels = fmri_data.shape
    
    averaged = np.zeros((n_images, n_voxels), dtype=np.float32)
    
    for i in range(n_images):
        reps = trial_indices[i]  # (n_reps,)
        # Average across repetitions
        averaged[i] = fmri_data[reps].mean(axis=0)
    
    return averaged


def load_encoder(encoder_type: str, checkpoint_path: str, device: str):
    """Load encoder (Ridge, MLP, or TwoStage) from checkpoint."""
    logger.info(f"Loading {encoder_type} encoder from {checkpoint_path}")
    
    if encoder_type == "ridge":
        import pickle
        with open(checkpoint_path, "rb") as f:
            encoder = pickle.load(f)
    elif encoder_type == "mlp":
        encoder = load_mlp(checkpoint_path, device=device)
        encoder.eval()
    elif encoder_type == "two_stage":
        encoder = load_two_stage_encoder(checkpoint_path, device=device)
        encoder.eval()
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    return encoder


def predict_clip_embeddings(
    encoder,
    encoder_type: str,
    fmri_features: np.ndarray,
    device: str,
    batch_size: int = 64
) -> np.ndarray:
    """
    Predict CLIP embeddings from fMRI features.
    
    Args:
        encoder: Ridge/MLP/TwoStage encoder
        encoder_type: "ridge", "mlp", or "two_stage"
        fmri_features: fMRI features, shape (n_samples, n_features)
        device: Device for computation
        batch_size: Batch size for neural models
        
    Returns:
        predictions: CLIP embeddings, shape (n_samples, 512), L2-normalized
    """
    n_samples = len(fmri_features)
    
    if encoder_type == "ridge":
        # Ridge is sklearn, operates on numpy
        predictions = encoder.predict(fmri_features)
        # Normalize
        predictions = predictions / np.linalg.norm(predictions, axis=1, keepdims=True)
        return predictions
    
    # Neural models (MLP/TwoStage)
    predictions = []
    encoder.eval()
    
    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc="Predicting"):
            batch = fmri_features[i:i+batch_size]
            batch_t = torch.from_numpy(batch).float().to(device)
            
            # Get predictions
            pred_t = encoder(batch_t)
            pred_np = pred_t.cpu().numpy()
            predictions.append(pred_np)
    
    predictions = np.concatenate(predictions, axis=0)
    
    # Ensure normalized (should already be from model)
    predictions = predictions / np.linalg.norm(predictions, axis=1, keepdims=True)
    
    return predictions


def compute_retrieval_metrics(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    k_values: List[int] = [1, 5, 10, 20, 50]
) -> Dict[str, float]:
    """
    Compute retrieval metrics.
    
    Args:
        query_embeddings: Query CLIP embeddings, shape (n_queries, 512)
        gallery_embeddings: Gallery CLIP embeddings, shape (n_gallery, 512)
        k_values: K values for R@K computation
        
    Returns:
        metrics: Dict with R@K, mean_rank, median_rank, MRR
    """
    # Compute similarity
    sim = cosine_sim(query_embeddings, gallery_embeddings)
    
    # Get rankings (argsort in descending order)
    ranks = np.argsort(-sim, axis=1)
    
    # True index is i (diagonal)
    true_indices = np.arange(len(query_embeddings))
    
    # Find rank of true image for each query
    true_ranks = np.zeros(len(query_embeddings), dtype=np.int32)
    for i in range(len(query_embeddings)):
        true_ranks[i] = np.where(ranks[i] == true_indices[i])[0][0]
    
    metrics = {}
    
    # R@K
    for k in k_values:
        r_at_k = (true_ranks < k).mean() * 100
        metrics[f"R@{k}"] = r_at_k
    
    # Ranking statistics
    metrics["mean_rank"] = float(true_ranks.mean())
    metrics["median_rank"] = float(np.median(true_ranks))
    metrics["MRR"] = float((1.0 / (true_ranks + 1)).mean())
    
    # Top-1 cosine similarity
    metrics["top1_cosine"] = float(np.diag(sim).mean())
    
    return metrics


def compute_perceptual_metrics(
    generated_images: List[Image.Image],
    ground_truth_images: List[Image.Image],
    clip_model,
    device: str
) -> Dict[str, float]:
    """
    Compute perceptual metrics (CLIPScore, SSIM, LPIPS).
    
    Args:
        generated_images: List of generated PIL images
        ground_truth_images: List of ground truth PIL images
        clip_model: CLIP model for CLIPScore
        device: Device for computation
        
    Returns:
        metrics: Dict with CLIPScore, SSIM, LPIPS
    """
    metrics = {}
    
    # CLIPScore
    logger.info("Computing CLIPScore...")
    from fmri2img.eval.image_metrics import clip_score
    clip_scores = []
    for gen_img, gt_img in tqdm(zip(generated_images, ground_truth_images), 
                                  total=len(generated_images)):
        score = clip_score(gen_img, gt_img, clip_model, device)
        clip_scores.append(score)
    metrics["CLIPScore"] = float(np.mean(clip_scores))
    metrics["CLIPScore_std"] = float(np.std(clip_scores))
    
    # SSIM (if available)
    if HAS_SSIM:
        logger.info("Computing SSIM...")
        ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        ssim_scores = []
        
        for gen_img, gt_img in tqdm(zip(generated_images, ground_truth_images),
                                     total=len(generated_images)):
            # Convert to tensors (C, H, W) normalized to [0, 1]
            gen_t = torch.from_numpy(np.array(gen_img)).permute(2, 0, 1).float() / 255.0
            gt_t = torch.from_numpy(np.array(gt_img)).permute(2, 0, 1).float() / 255.0
            
            # Add batch dim and move to device
            gen_t = gen_t.unsqueeze(0).to(device)
            gt_t = gt_t.unsqueeze(0).to(device)
            
            score = ssim_fn(gen_t, gt_t).item()
            ssim_scores.append(score)
        
        metrics["SSIM"] = float(np.mean(ssim_scores))
        metrics["SSIM_std"] = float(np.std(ssim_scores))
    
    # LPIPS (if available)
    if HAS_LPIPS:
        logger.info("Computing LPIPS...")
        lpips_fn = lpips.LPIPS(net='alex').to(device)
        lpips_scores = []
        
        for gen_img, gt_img in tqdm(zip(generated_images, ground_truth_images),
                                     total=len(generated_images)):
            # Convert to tensors (C, H, W) normalized to [-1, 1]
            gen_t = torch.from_numpy(np.array(gen_img)).permute(2, 0, 1).float() / 127.5 - 1.0
            gt_t = torch.from_numpy(np.array(gt_img)).permute(2, 0, 1).float() / 127.5 - 1.0
            
            # Add batch dim and move to device
            gen_t = gen_t.unsqueeze(0).to(device)
            gt_t = gt_t.unsqueeze(0).to(device)
            
            score = lpips_fn(gen_t, gt_t).item()
            lpips_scores.append(score)
        
        metrics["LPIPS"] = float(np.mean(lpips_scores))
        metrics["LPIPS_std"] = float(np.std(lpips_scores))
    
    return metrics


def compute_brain_alignment(
    generated_images: List[Image.Image],
    true_fmri: np.ndarray,
    encoding_model,
    device: str
) -> Dict[str, float]:
    """
    Compute brain alignment: correlation between encoding model predictions
    and true fMRI for generated images.
    
    This measures how well the generated images capture brain activity patterns.
    Higher correlation = better neural fidelity.
    
    Args:
        generated_images: List of generated PIL images
        true_fmri: True fMRI features, shape (n_images, n_features)
        encoding_model: Trained EncodingModel (Image → fMRI)
        device: Device for computation
        
    Returns:
        metrics: Dict with correlation statistics
        
    Scientific Context:
        This is inspired by Brain-Optimized Inference (Ozcelik & VanRullen 2023).
        Images that evoke similar brain activity to the true stimulus are more
        faithful reconstructions, even if pixel-level metrics are imperfect.
    """
    logger.info("Computing brain alignment (encoding model correlation)...")
    
    # Predict fMRI from generated images
    predicted_fmri = []
    
    encoding_model.eval()
    with torch.no_grad():
        for img in tqdm(generated_images, desc="Encoding images"):
            pred = encoding_model.predict(img)  # Returns numpy array
            predicted_fmri.append(pred)
    
    predicted_fmri = np.array(predicted_fmri)  # (n_images, n_features)
    
    # Compute per-sample correlation
    correlations = []
    for i in range(len(true_fmri)):
        corr = np.corrcoef(true_fmri[i], predicted_fmri[i])[0, 1]
        correlations.append(corr)
    
    correlations = np.array(correlations)
    
    metrics = {
        "brain_correlation": float(correlations.mean()),
        "brain_correlation_std": float(correlations.std()),
        "brain_correlation_median": float(np.median(correlations)),
        "brain_correlation_min": float(correlations.min()),
        "brain_correlation_max": float(correlations.max())
    }
    
    return metrics


def statistical_comparison(
    results: Dict[str, Dict[str, Any]],
    metric_name: str
) -> Dict[str, Any]:
    """
    Perform statistical tests comparing strategies.
    
    Args:
        results: Results dict with per-strategy metrics
        metric_name: Metric to compare (e.g., "CLIPScore")
        
    Returns:
        comparison: Dict with pairwise t-test results
    """
    strategies = list(results.keys())
    
    if len(strategies) < 2:
        return {}
    
    comparison = {}
    
    # Pairwise comparisons
    for i in range(len(strategies)):
        for j in range(i + 1, len(strategies)):
            strat1 = strategies[i]
            strat2 = strategies[j]
            
            # Get per-sample scores (if available)
            if f"{metric_name}_samples" in results[strat1]:
                samples1 = results[strat1][f"{metric_name}_samples"]
                samples2 = results[strat2][f"{metric_name}_samples"]
                
                # Paired t-test
                t_stat, p_value = stats.ttest_rel(samples1, samples2)
                
                comparison[f"{strat1}_vs_{strat2}"] = {
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                    "mean_diff": float(np.mean(samples1) - np.mean(samples2))
                }
    
    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation on NSD Shared 1000",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument("--subject", type=str, required=True,
                        help="Subject ID (e.g., subj01)")
    parser.add_argument("--encoder-checkpoint", type=str, required=True,
                        help="Path to encoder checkpoint")
    parser.add_argument("--encoder-type", type=str, required=True,
                        choices=["ridge", "mlp", "two_stage"],
                        help="Encoder type")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for results")
    
    # Data paths
    parser.add_argument("--data-root", type=str, default="s3://natural-scenes-dataset",
                        help="NSD data root (S3 or local)")
    parser.add_argument("--cache-root", type=str, default="cache",
                        help="Local cache directory")
    parser.add_argument("--stim-info", type=str, 
                        default="cache/nsd_stim_info_merged.csv",
                        help="Path to nsd_stim_info_merged.csv")
    parser.add_argument("--clip-cache", type=str,
                        default="outputs/clip_cache/clip.parquet",
                        help="Path to CLIP cache")
    
    # Evaluation options
    parser.add_argument("--strategies", nargs="+", 
                        default=["single", "best_of_8", "boi_lite"],
                        choices=["single", "best_of_4", "best_of_8", "best_of_16", 
                                 "boi_lite"],
                        help="Generation strategies to evaluate")
    parser.add_argument("--average-reps", action="store_true", default=True,
                        help="Average fMRI across 3 repetitions (higher SNR)")
    parser.add_argument("--no-brain-alignment", action="store_true",
                        help="Skip brain alignment computation (faster)")
    parser.add_argument("--encoding-model-checkpoint", type=str, default=None,
                        help="Path to encoding model checkpoint (for brain alignment)")
    
    # Generation parameters
    parser.add_argument("--num-inference-steps", type=int, default=250,
                        help="Number of diffusion steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5,
                        help="Classifier-free guidance scale")
    parser.add_argument("--boi-steps", type=int, default=3,
                        help="BOI-lite refinement steps")
    parser.add_argument("--boi-candidates", type=int, default=4,
                        help="BOI-lite candidates per step")
    
    # Compute options
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for encoder predictions")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Subset for testing
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples to evaluate (for testing)")
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging to file
    fh = logging.FileHandler(output_dir / "eval_comprehensive.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(fh)
    
    logger.info("=" * 80)
    logger.info("NSD Shared 1000 Comprehensive Evaluation")
    logger.info("=" * 80)
    logger.info(f"Subject: {args.subject}")
    logger.info(f"Encoder: {args.encoder_type} from {args.encoder_checkpoint}")
    logger.info(f"Strategies: {args.strategies}")
    logger.info(f"Output: {output_dir}")
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # =========================================================================
    # 1. Load NSD Shared 1000 metadata
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Loading NSD Shared 1000 metadata")
    logger.info("=" * 80)
    
    shared_df = load_nsd_shared_1000(args.stim_info)
    trials, nsd_ids = get_shared_1000_trials(
        shared_df, args.subject, average_reps=args.average_reps
    )
    
    if args.max_samples is not None:
        logger.info(f"Limiting to {args.max_samples} samples for testing")
        trials = trials[:args.max_samples]
        nsd_ids = nsd_ids[:args.max_samples]
    
    n_samples = len(nsd_ids)
    logger.info(f"Evaluating on {n_samples} shared images")
    
    # =========================================================================
    # 2. Load and preprocess fMRI data
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Loading and preprocessing fMRI data")
    logger.info("=" * 80)
    
    # Load subject index
    index_df = read_subject_index(args.subject, args.data_root, args.cache_root)
    
    # Load fMRI data
    fs = get_s3_filesystem() if args.data_root.startswith("s3://") else None
    nifti_loader = NIfTILoader(fs)
    
    logger.info(f"Loading fMRI from {len(index_df)} trials...")
    all_fmri = nifti_loader.load_all_trials(index_df, verbose=True)
    
    # Average across repetitions if requested
    if args.average_reps:
        logger.info("Averaging fMRI across 3 repetitions...")
        fmri_data = average_fmri_reps(all_fmri, trials)
    else:
        # Just extract the trials
        fmri_data = all_fmri[trials]
    
    logger.info(f"fMRI shape: {fmri_data.shape}")
    
    # Preprocess fMRI (T0/T1/T2 pipeline)
    logger.info("Preprocessing fMRI (T0/T1/T2)...")
    preprocessor = NSDPreprocessor(
        subject=args.subject,
        cache_dir=args.cache_root,
        pca_k=512  # Use same as training
    )
    
    # Fit on training data (from index)
    train_indices, val_indices, test_indices = train_val_test_split(index_df)
    train_fmri = all_fmri[train_indices]
    
    logger.info("Fitting preprocessor on training data...")
    preprocessor.fit(train_fmri)
    
    # Transform shared 1000 data
    logger.info("Transforming shared 1000 fMRI...")
    fmri_features = preprocessor.transform(fmri_data)
    
    logger.info(f"Preprocessed fMRI shape: {fmri_features.shape}")
    
    # =========================================================================
    # 3. Load encoder and predict CLIP embeddings
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Predicting CLIP embeddings from fMRI")
    logger.info("=" * 80)
    
    encoder = load_encoder(args.encoder_type, args.encoder_checkpoint, args.device)
    
    predicted_embeddings = predict_clip_embeddings(
        encoder, args.encoder_type, fmri_features, 
        args.device, args.batch_size
    )
    
    logger.info(f"Predicted embeddings shape: {predicted_embeddings.shape}")
    
    # =========================================================================
    # 4. Load ground truth CLIP embeddings
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Loading ground truth CLIP embeddings")
    logger.info("=" * 80)
    
    clip_cache = CLIPCache(args.clip_cache)
    
    # Get ground truth embeddings for shared 1000
    gt_embeddings = []
    for nsd_id in nsd_ids:
        emb = clip_cache.get_embedding(nsd_id)
        if emb is None:
            logger.error(f"Missing CLIP embedding for nsdId={nsd_id}")
            raise ValueError(f"Missing embedding for nsdId={nsd_id}")
        gt_embeddings.append(emb)
    
    gt_embeddings = np.array(gt_embeddings)
    logger.info(f"Ground truth embeddings shape: {gt_embeddings.shape}")
    
    # =========================================================================
    # 5. Compute retrieval metrics
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Step 5: Computing retrieval metrics")
    logger.info("=" * 80)
    
    retrieval_metrics = compute_retrieval_metrics(
        predicted_embeddings, gt_embeddings,
        k_values=[1, 5, 10, 20, 50, 100]
    )
    
    logger.info("Retrieval Results:")
    for k, v in retrieval_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    
    # Save retrieval results
    with open(output_dir / "retrieval_metrics.json", "w") as f:
        json.dump(retrieval_metrics, f, indent=2)
    
    logger.info(f"Retrieval metrics saved to {output_dir / 'retrieval_metrics.json'}")
    
    # =========================================================================
    # 6. Generate images with all strategies (TODO: Next implementation)
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Step 6: Image generation with multiple strategies")
    logger.info("=" * 80)
    logger.info("Image generation not yet implemented in this phase.")
    logger.info("Will be added in next iteration with:")
    logger.info("  - Single sample generation")
    logger.info("  - Best-of-N sampling")
    logger.info("  - BOI-lite refinement")
    logger.info("  - Perceptual metrics (CLIPScore, SSIM, LPIPS)")
    logger.info("  - Brain alignment (if encoding model provided)")
    
    # =========================================================================
    # 7. Summary
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Evaluation Complete!")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Evaluated {n_samples} shared images")
    logger.info(f"Top-1 Cosine Similarity: {retrieval_metrics['top1_cosine']:.4f}")
    logger.info(f"R@1: {retrieval_metrics['R@1']:.2f}%")
    logger.info(f"R@5: {retrieval_metrics['R@5']:.2f}%")
    logger.info(f"R@10: {retrieval_metrics['R@10']:.2f}%")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
