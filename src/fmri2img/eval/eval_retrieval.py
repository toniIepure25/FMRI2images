#!/usr/bin/env python3
"""
Comprehensive Retrieval Evaluation Script
=========================================

Evaluates fMRI → CLIP encoders using retrieval metrics on various gallery sizes:
- Train gallery (all training images)
- Val gallery (validation images)
- Test gallery (test images)
- Full gallery (all images)
- NSD shared 1000 (if available)

Metrics:
- Retrieval@K (K=1, 5, 10, 20, 50)
- Mean/median rank
- Mean reciprocal rank (MRR)

Usage:
    # Evaluate on test set with test gallery
    python scripts/eval_retrieval.py \\
        --subject subj01 \\
        --encoder-type mlp \\
        --checkpoint checkpoints/mlp/subj01/mlp.pt \\
        --split test \\
        --gallery test \\
        --clip-cache outputs/clip_cache/clip.parquet
    
    # Evaluate on test set with full gallery (harder)
    python scripts/eval_retrieval.py \\
        --subject subj01 \\
        --encoder-type two_stage \\
        --checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \\
        --split test \\
        --gallery full \\
        --clip-cache outputs/clip_cache/clip.parquet
    
    # Evaluate Ridge baseline
    python scripts/eval_retrieval.py \\
        --subject subj01 \\
        --encoder-type ridge \\
        --checkpoint checkpoints/ridge/subj01/ridge.pkl \\
        --split test \\
        --gallery test
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
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
from fmri2img.models.train_utils import train_val_test_split, extract_features_and_targets
from fmri2img.eval.retrieval import retrieval_at_k, compute_ranking_metrics, cosine_sim


def load_encoder(encoder_type: str, checkpoint_path: str, device: str):
    """Load encoder (Ridge, MLP, or TwoStage) from checkpoint."""
    logger.info(f"Loading {encoder_type} encoder from {checkpoint_path}")
    
    if encoder_type == "ridge":
        encoder = RidgeEncoder.load(checkpoint_path)
        logger.info(f"✅ Loaded Ridge encoder (alpha={encoder.alpha:.1f})")
        
        # Wrap in common interface
        class EncoderWrapper:
            def __init__(self, model):
                self.model = model
            
            def predict(self, X: np.ndarray) -> np.ndarray:
                return self.model.predict(X)
        
        return EncoderWrapper(encoder)
    
    elif encoder_type == "mlp":
        import torch
        model, meta = load_mlp(checkpoint_path, map_location=device)
        model = model.to(device)
        model.eval()
        logger.info(f"✅ Loaded MLP encoder (best_epoch={meta.get('best_epoch', 'N/A')})")
        
        class MLPWrapper:
            def __init__(self, model, device):
                self.model = model
                self.device = device
            
            def predict(self, X: np.ndarray) -> np.ndarray:
                import torch
                with torch.no_grad():
                    X_tensor = torch.from_numpy(X).float().to(self.device)
                    pred = self.model(X_tensor)
                    return pred.cpu().numpy()
        
        return MLPWrapper(model, device)
    
    elif encoder_type == "two_stage":
        import torch
        model, meta = load_two_stage_encoder(checkpoint_path, map_location=device)
        model = model.to(device)
        model.eval()
        logger.info(f"✅ Loaded TwoStageEncoder (latent_dim={meta.get('latent_dim')}, n_blocks={meta.get('n_blocks')})")
        
        class TwoStageWrapper:
            def __init__(self, model, device):
                self.model = model
                self.device = device
            
            def predict(self, X: np.ndarray) -> np.ndarray:
                import torch
                with torch.no_grad():
                    X_tensor = torch.from_numpy(X).float().to(self.device)
                    pred = self.model(X_tensor)
                    return pred.cpu().numpy()
        
        return TwoStageWrapper(model, device)
    
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def build_gallery_embeddings(
    clip_cache: CLIPCache,
    gallery_nsd_ids: np.ndarray
) -> np.ndarray:
    """Build gallery of CLIP embeddings from NSD IDs."""
    logger.info(f"Building gallery of {len(gallery_nsd_ids)} embeddings...")
    
    embeddings = []
    missing_ids = []
    
    for nsd_id in tqdm(gallery_nsd_ids, desc="Loading gallery embeddings"):
        emb_dict = clip_cache.get([int(nsd_id)])
        emb = emb_dict.get(int(nsd_id))
        
        if emb is not None:
            embeddings.append(emb)
        else:
            missing_ids.append(nsd_id)
    
    if missing_ids:
        logger.warning(f"Missing {len(missing_ids)}/{len(gallery_nsd_ids)} embeddings from gallery")
    
    if not embeddings:
        raise ValueError("No valid embeddings found in gallery!")
    
    gallery_embeddings = np.vstack(embeddings)
    logger.info(f"✅ Built gallery: {gallery_embeddings.shape}")
    
    return gallery_embeddings


def evaluate_retrieval(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    gt_indices: np.ndarray,
    ks: Tuple[int, ...] = (1, 5, 10, 20, 50)
) -> Dict:
    """
    Evaluate retrieval metrics.
    
    Args:
        query_embeddings: Predicted embeddings (N, 512), L2-normalized
        gallery_embeddings: Gallery embeddings (M, 512), L2-normalized
        gt_indices: Ground truth gallery indices (N,)
        ks: K values for retrieval@K
    
    Returns:
        Dictionary with all metrics
    """
    logger.info(f"Evaluating retrieval: {len(query_embeddings)} queries, {len(gallery_embeddings)} gallery")
    
    # Retrieval@K metrics
    retrieval_metrics = retrieval_at_k(
        query_embeddings,
        gallery_embeddings,
        gt_indices,
        ks=ks
    )
    
    # Ranking metrics
    ranking_metrics = compute_ranking_metrics(
        query_embeddings,
        gallery_embeddings,
        gt_indices
    )
    
    # Combine
    metrics = {**retrieval_metrics, **ranking_metrics}
    
    # Additional stats
    sim_matrix = cosine_sim(query_embeddings, gallery_embeddings)
    metrics["mean_sim_to_gt"] = float(sim_matrix[np.arange(len(gt_indices)), gt_indices].mean())
    metrics["std_sim_to_gt"] = float(sim_matrix[np.arange(len(gt_indices)), gt_indices].std())
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Retrieval evaluation for fMRI → CLIP encoders")
    
    # Data
    parser.add_argument("--subject", default="subj01")
    parser.add_argument("--index-root", default="data/indices/nsd_index")
    parser.add_argument("--clip-cache", required=True, help="Path to CLIP cache parquet")
    
    # Model
    parser.add_argument("--encoder-type", required=True, 
                       choices=["ridge", "mlp", "two_stage"])
    parser.add_argument("--checkpoint", required=True, help="Path to encoder checkpoint")
    
    # Preprocessing
    parser.add_argument("--use-preproc", action="store_true")
    parser.add_argument("--preproc-dir", default="outputs/preproc")
    
    # Evaluation
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                       help="Which split to evaluate on")
    parser.add_argument("--gallery", default="test",
                       choices=["train", "val", "test", "full", "matched"],
                       help="Gallery to retrieve from")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10, 20, 50],
                       help="K values for retrieval@K")
    
    # Output
    parser.add_argument("--output-json", help="Path to save results JSON")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, help="Limit samples for testing")
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Using device: {device}")
    
    # Load index
    logger.info(f"Loading index for {args.subject}...")
    df = read_subject_index(args.index_root, args.subject)
    
    if args.limit:
        df = df.head(args.limit)
        logger.info(f"Limited to {len(df)} samples for testing")
    
    # Train/val/test split (same seed as training)
    train_df, val_df, test_df = train_val_test_split(df, random_seed=42)
    
    # Select evaluation split
    if args.split == "train":
        eval_df = train_df
    elif args.split == "val":
        eval_df = val_df
    else:
        eval_df = test_df
    
    logger.info(f"Evaluating on {args.split} split: {len(eval_df)} samples")
    
    # Select gallery split
    if args.gallery == "train":
        gallery_df = train_df
    elif args.gallery == "val":
        gallery_df = val_df
    elif args.gallery == "test":
        gallery_df = test_df
    elif args.gallery == "full":
        gallery_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    elif args.gallery == "matched":
        gallery_df = eval_df  # Same as evaluation split
    
    logger.info(f"Gallery: {args.gallery} ({len(gallery_df)} images)")
    
    # Load CLIP cache
    logger.info("Loading CLIP cache...")
    clip_cache = CLIPCache(args.clip_cache)
    
    # Setup preprocessing if needed
    preprocessor = None
    if args.use_preproc:
        logger.info("Setting up preprocessing...")
        preprocessor = NSDPreprocessor(args.subject, out_dir=args.preproc_dir)
        
        if not preprocessor.meta_path.exists():
            logger.error(f"Preprocessing artifacts not found at {preprocessor.out_dir}")
            logger.error("Please run preprocessing first")
            sys.exit(1)
        
        preprocessor.load_artifacts()
        logger.info(f"Loaded preprocessing: PCA k={preprocessor.pca_info_.get('n_components_eff', 'N/A')}")
    
    # Setup NIfTI loader
    fs = get_s3_filesystem()
    nifti_loader = NIfTILoader(fs)
    
    # Extract evaluation features
    logger.info(f"Extracting {args.split} split features...")
    X_eval, Y_eval, eval_nsd_ids = extract_features_and_targets(
        eval_df, nifti_loader, preprocessor, clip_cache, desc=args.split
    )
    
    logger.info(f"✅ Extracted {len(X_eval)} samples")
    
    # Load encoder
    encoder = load_encoder(args.encoder_type, args.checkpoint, device)
    
    # Predict CLIP embeddings
    logger.info("Predicting CLIP embeddings...")
    pred_embeddings = encoder.predict(X_eval)  # (N, 512)
    
    # Normalize predictions
    pred_embeddings = pred_embeddings / (np.linalg.norm(pred_embeddings, axis=1, keepdims=True) + 1e-8)
    
    logger.info(f"✅ Predicted embeddings: {pred_embeddings.shape}")
    
    # Build gallery embeddings
    gallery_nsd_ids = gallery_df["nsdId"].values
    gallery_embeddings = build_gallery_embeddings(clip_cache, gallery_nsd_ids)
    
    # Normalize gallery
    gallery_embeddings = gallery_embeddings / (np.linalg.norm(gallery_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Map eval nsd_ids to gallery indices
    logger.info("Mapping ground truth indices...")
    gallery_nsd_id_to_idx = {int(nsd_id): idx for idx, nsd_id in enumerate(gallery_nsd_ids)}
    
    gt_indices = []
    valid_mask = []
    
    for nsd_id in eval_nsd_ids:
        if int(nsd_id) in gallery_nsd_id_to_idx:
            gt_indices.append(gallery_nsd_id_to_idx[int(nsd_id)])
            valid_mask.append(True)
        else:
            valid_mask.append(False)
    
    valid_mask = np.array(valid_mask)
    n_valid = valid_mask.sum()
    
    if n_valid < len(eval_nsd_ids):
        logger.warning(f"Only {n_valid}/{len(eval_nsd_ids)} samples have GT in gallery")
        # Filter to valid samples
        pred_embeddings = pred_embeddings[valid_mask]
        eval_nsd_ids = eval_nsd_ids[valid_mask]
    
    gt_indices = np.array(gt_indices)
    
    logger.info(f"✅ Mapped {len(gt_indices)} ground truth indices")
    
    # Evaluate retrieval
    logger.info("=" * 80)
    logger.info("RETRIEVAL EVALUATION")
    logger.info("=" * 80)
    
    metrics = evaluate_retrieval(
        pred_embeddings,
        gallery_embeddings,
        gt_indices,
        ks=tuple(args.ks)
    )
    
    # Print results
    logger.info(f"Gallery size: {len(gallery_embeddings)}")
    logger.info(f"Query samples: {len(pred_embeddings)}")
    logger.info("")
    logger.info("Retrieval@K:")
    for k in args.ks:
        if f"R@{k}" in metrics:
            logger.info(f"  R@{k}: {metrics[f'R@{k}']:.4f} ({metrics[f'R@{k}'] * 100:.2f}%)")
    
    logger.info("")
    logger.info("Ranking Metrics:")
    logger.info(f"  Mean Rank: {metrics['mean_rank']:.2f}")
    logger.info(f"  Median Rank: {metrics['median_rank']:.0f}")
    logger.info(f"  MRR: {metrics['mrr']:.4f}")
    
    logger.info("")
    logger.info("Similarity to GT:")
    logger.info(f"  Mean: {metrics['mean_sim_to_gt']:.4f}")
    logger.info(f"  Std: {metrics['std_sim_to_gt']:.4f}")
    
    # Save results
    results = {
        "subject": args.subject,
        "encoder_type": args.encoder_type,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "gallery": args.gallery,
        "gallery_size": len(gallery_embeddings),
        "n_queries": len(pred_embeddings),
        "n_valid": n_valid,
        "metrics": metrics
    }
    
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ Saved results to {output_path}")
    
    logger.info("=" * 80)
    logger.info("Retrieval evaluation complete!")


if __name__ == "__main__":
    main()
