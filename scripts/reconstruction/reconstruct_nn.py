#!/usr/bin/env python3
"""
Nearest-Neighbor Reconstruction Baseline
=========================================

Evaluates fMRI → image reconstruction via nearest-neighbor retrieval in CLIP space.

Given an encoder (Ridge or MLP) that maps fMRI → CLIP embeddings, this script:
1. Loads test fMRI data and predicts CLIP embeddings
2. Retrieves nearest neighbors from a gallery of CLIP embeddings
3. Computes retrieval metrics (R@K, cosine similarity)
4. Visualizes reconstruction quality (ground truth vs top-K neighbors)

This provides a strong baseline for reconstruction quality without diffusion models.

Scientific Context:
- NN retrieval is standard baseline for neural decoding (Ozcelik & VanRullen 2023)
- Uses encoder's learned mapping without additional training
- Evaluates semantic alignment in CLIP space (same as training objective)

Usage:
    # Ridge encoder
    python scripts/reconstruct_nn.py \\
        --subject subj01 \\
        --encoder ridge \\
        --ckpt checkpoints/ridge/subj01/ridge.pkl \\
        --clip-cache outputs/clip_cache/clip.parquet \\
        --use-preproc \\
        --gallery-limit 5000 \\
        --topk 10 \\
        --limit 256
    
    # MLP encoder
    python scripts/reconstruct_nn.py \\
        --subject subj01 \\
        --encoder mlp \\
        --ckpt checkpoints/mlp/subj01/mlp.pt \\
        --clip-cache outputs/clip_cache/clip.parquet \\
        --use-preproc \\
        --gallery-limit 5000 \\
        --topk 10 \\
        --limit 256
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import project modules
from fmri2img.data.nsd_index_reader import read_subject_index
from fmri2img.data.preprocess import NSDPreprocessor
from fmri2img.data.clip_cache import CLIPCache
from fmri2img.io.s3 import get_s3_filesystem, NIfTILoader
from fmri2img.models.ridge import RidgeEncoder
from fmri2img.models.mlp import load_mlp
from fmri2img.models.train_utils import train_val_test_split
from fmri2img.eval.retrieval import cosine_sim, retrieval_at_k, compute_ranking_metrics


def load_encoder(encoder_type: str, ckpt_path: Path, device: str = "cpu"):
    """
    Load encoder (Ridge or MLP) from checkpoint.
    
    Args:
        encoder_type: "ridge" or "mlp"
        ckpt_path: Path to checkpoint
        device: Device for MLP ("cpu" or "cuda")
    
    Returns:
        Encoder model with predict() method
    """
    logger.info(f"Loading {encoder_type} encoder from {ckpt_path}")
    
    if encoder_type == "ridge":
        # Ridge uses pickle
        import pickle
        with open(ckpt_path, "rb") as f:
            encoder = pickle.load(f)
        logger.info(f"✅ Loaded Ridge encoder (alpha={encoder.alpha:.1f})")
        return encoder
    
    elif encoder_type == "mlp":
        # MLP uses PyTorch
        import torch
        model, meta = load_mlp(str(ckpt_path), map_location=device)
        model.eval()
        
        # Wrap in Ridge-like interface with predict()
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
        
        logger.info(f"✅ Loaded MLP encoder (best_epoch={meta.get('best_epoch', 'N/A')})")
        return MLPWrapper(model, device)
    
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def extract_features_and_targets(
    df: pd.DataFrame,
    nifti_loader: NIfTILoader,
    preprocessor: NSDPreprocessor,
    clip_cache: CLIPCache
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract fMRI features (X) and CLIP targets (Y) from DataFrame.
    
    Reuses preprocessing pipeline from training (apples-to-apples).
    
    Args:
        df: DataFrame with beta_path, beta_index, nsdId columns
        nifti_loader: NIfTI data loader
        preprocessor: Fitted preprocessor
        clip_cache: CLIP cache
    
    Returns:
        X (n_samples, n_features), Y (n_samples, 512), nsdIds (n_samples,)
    """
    logger.info(f"Extracting features from {len(df)} samples...")
    
    X_list = []
    Y_list = []
    nsdIds = []
    
    for idx, row in df.iterrows():
        try:
            # Load fMRI volume
            beta_path = row["beta_path"]
            beta_index = int(row.get("beta_index", 0))
            img = nifti_loader.load(beta_path)
            data_4d = img.get_fdata()
            vol = data_4d[..., beta_index].astype(np.float32)
            
            # Apply preprocessing
            x = preprocessor.transform(vol)
            
            # Get CLIP embedding
            nsd_id = int(row["nsdId"])
            y = clip_cache.get(nsd_id)
            
            if x is not None and y is not None:
                X_list.append(x)
                Y_list.append(y)
                nsdIds.append(nsd_id)
        
        except Exception as e:
            logger.warning(f"Failed to process row {idx}: {e}")
            continue
    
    X = np.array(X_list)
    Y = np.array(Y_list)
    nsdIds = np.array(nsdIds)
    
    logger.info(f"✅ Extracted {len(X)} valid samples")
    logger.info(f"   Features: {X.shape}, Targets: {Y.shape}")
    
    return X, Y, nsdIds


def build_gallery(
    clip_cache: CLIPCache,
    exclude_nsd_ids: np.ndarray,
    limit: int = 5000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build gallery of CLIP embeddings for retrieval.
    
    Args:
        clip_cache: CLIP cache
        exclude_nsd_ids: NSD IDs to exclude (test set)
        limit: Maximum gallery size
    
    Returns:
        gallery_embeddings (n_gallery, 512), gallery_nsd_ids (n_gallery,)
    """
    logger.info(f"Building gallery (limit={limit}, excluding {len(exclude_nsd_ids)} test samples)...")
    
    # Get all embeddings
    all_nsd_ids = clip_cache.get_all_ids()
    all_embeddings = clip_cache.get_batch(all_nsd_ids)
    
    # Exclude test samples
    mask = ~np.isin(all_nsd_ids, exclude_nsd_ids)
    gallery_nsd_ids = all_nsd_ids[mask]
    gallery_embeddings = all_embeddings[mask]
    
    # Apply limit
    if len(gallery_nsd_ids) > limit:
        indices = np.random.choice(len(gallery_nsd_ids), size=limit, replace=False)
        gallery_nsd_ids = gallery_nsd_ids[indices]
        gallery_embeddings = gallery_embeddings[indices]
    
    logger.info(f"✅ Gallery size: {len(gallery_embeddings)}")
    
    # Warn if gallery is tiny
    if len(gallery_embeddings) < 100:
        logger.warning(f"⚠️  Gallery is very small ({len(gallery_embeddings)}), R@K will be trivial!")
    
    return gallery_embeddings, gallery_nsd_ids


def retrieve_nearest_neighbors(
    query: np.ndarray,
    gallery: np.ndarray,
    k: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieve top-K nearest neighbors for each query.
    
    Args:
        query: Query embeddings (n_queries, d)
        gallery: Gallery embeddings (n_gallery, d)
        k: Number of neighbors to retrieve
    
    Returns:
        indices (n_queries, k), similarities (n_queries, k)
    """
    # Compute similarity matrix
    sim = cosine_sim(query, gallery)  # (n_queries, n_gallery)
    
    # Get top-K indices and similarities
    topk_indices = np.argsort(-sim, axis=1)[:, :k]  # (n_queries, k)
    topk_similarities = np.take_along_axis(sim, topk_indices, axis=1)  # (n_queries, k)
    
    return topk_indices, topk_similarities


def visualize_reconstruction_grid(
    test_nsd_ids: np.ndarray,
    topk_nsd_ids: np.ndarray,
    topk_similarities: np.ndarray,
    clip_cache: CLIPCache,
    out_dir: Path,
    n_samples: int = 4
) -> None:
    """
    Visualize reconstruction quality: ground truth vs top-K neighbors.
    
    Creates grid: each row is one test sample, columns are GT + top-K neighbors.
    
    Args:
        test_nsd_ids: Test NSD IDs (n_test,)
        topk_nsd_ids: Top-K retrieved NSD IDs (n_test, k)
        topk_similarities: Top-K cosine similarities (n_test, k)
        clip_cache: CLIP cache (for loading images)
        out_dir: Output directory for figures
        n_samples: Number of test samples to visualize
    """
    logger.info(f"Generating reconstruction grids for {n_samples} samples...")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Select random test samples
    indices = np.random.choice(len(test_nsd_ids), size=min(n_samples, len(test_nsd_ids)), replace=False)
    
    for idx in indices:
        nsd_id_gt = test_nsd_ids[idx]
        nsd_ids_topk = topk_nsd_ids[idx]
        sims_topk = topk_similarities[idx]
        
        k = len(nsd_ids_topk)
        
        # Create figure: 1 row, (k+1) columns (GT + top-K)
        fig = plt.figure(figsize=(2 * (k + 1), 2.5))
        gs = GridSpec(1, k + 1, figure=fig, wspace=0.3)
        
        # Ground truth
        ax = fig.add_subplot(gs[0, 0])
        ax.text(0.5, 0.5, f"GT\nNSD ID: {nsd_id_gt}", ha="center", va="center", fontsize=10)
        ax.set_title("Ground Truth", fontsize=10, fontweight="bold")
        ax.axis("off")
        
        # Top-K neighbors
        for i in range(k):
            ax = fig.add_subplot(gs[0, i + 1])
            nsd_id = nsd_ids_topk[i]
            sim = sims_topk[i]
            
            # Check if retrieved item is ground truth (correct retrieval)
            is_correct = (nsd_id == nsd_id_gt)
            color = "green" if is_correct else "black"
            
            ax.text(0.5, 0.5, f"NSD ID: {nsd_id}\nCos: {sim:.3f}", ha="center", va="center", fontsize=9)
            ax.set_title(f"Rank {i+1}", fontsize=10, color=color, fontweight="bold" if is_correct else "normal")
            ax.axis("off")
        
        plt.suptitle(f"Test Sample: NSD ID {nsd_id_gt}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        
        # Save
        fig_path = out_dir / f"reconstruction_nsd{nsd_id_gt}.png"
        plt.savefig(fig_path, dpi=100, bbox_inches="tight")
        plt.close()
        
        logger.info(f"✅ Saved {fig_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Nearest-neighbor reconstruction baseline"
    )
    
    # Data paths
    parser.add_argument("--index-root", default="data/indices/nsd_index",
                       help="NSD index root directory")
    parser.add_argument("--subject", default="subj01", help="Subject ID")
    parser.add_argument("--clip-cache", default="outputs/clip_cache/clip.parquet",
                       help="Path to CLIP cache")
    
    # Encoder
    parser.add_argument("--encoder", choices=["ridge", "mlp"], required=True,
                       help="Encoder type")
    parser.add_argument("--ckpt", required=True, help="Path to encoder checkpoint")
    
    # Preprocessing
    parser.add_argument("--use-preproc", action="store_true",
                       help="Use preprocessing (T0/T1/T2)")
    parser.add_argument("--preproc-dir", help="Preprocessing directory (if using non-default)")
    
    # Evaluation
    parser.add_argument("--gallery-limit", type=int, default=5000,
                       help="Maximum gallery size for retrieval")
    parser.add_argument("--topk", type=int, default=10,
                       help="Number of top-K neighbors to retrieve")
    parser.add_argument("--limit", type=int, help="Limit number of test samples")
    
    # Output
    parser.add_argument("--out-csv", help="Output CSV path (default: outputs/reports/{subject}/nn_eval.csv)")
    parser.add_argument("--out-fig-dir", help="Output figure directory (default: outputs/reports/{subject}/nn_figs)")
    
    # System
    parser.add_argument("--device", default="cpu", help="Device for MLP (cpu or cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Default output paths
    out_csv = Path(args.out_csv) if args.out_csv else Path(f"outputs/reports/{args.subject}/nn_eval.csv")
    out_fig_dir = Path(args.out_fig_dir) if args.out_fig_dir else Path(f"outputs/reports/{args.subject}/nn_figs")
    
    try:
        np.random.seed(args.seed)
        
        logger.info("=" * 80)
        logger.info("NEAREST-NEIGHBOR RECONSTRUCTION BASELINE")
        logger.info("=" * 80)
        logger.info(f"Subject: {args.subject}")
        logger.info(f"Encoder: {args.encoder}")
        logger.info(f"Checkpoint: {args.ckpt}")
        logger.info(f"Gallery limit: {args.gallery_limit}")
        logger.info(f"Top-K: {args.topk}")
        
        # Load encoder
        encoder = load_encoder(args.encoder, Path(args.ckpt), args.device)
        
        # Load index
        logger.info(f"Loading index for {args.subject}...")
        df = read_subject_index(args.index_root, args.subject)
        
        if args.limit:
            df = df.head(args.limit)
            logger.info(f"Limited to {len(df)} samples")
        
        # Split data (same as training)
        _, _, test_df = train_val_test_split(df, random_seed=args.seed)
        logger.info(f"Test set: {len(test_df)} samples")
        
        # Load CLIP cache
        logger.info(f"Loading CLIP cache from {args.clip_cache}")
        clip_cache = CLIPCache(args.clip_cache).load()
        stats = clip_cache.stats()
        logger.info(f"✅ CLIP cache loaded: {stats['cache_size']} embeddings")
        
        # Load preprocessing
        if args.use_preproc:
            logger.info("Loading preprocessing artifacts...")
            preproc_dir = args.preproc_dir if args.preproc_dir else f"outputs/preproc/{args.subject}"
            preprocessor = NSDPreprocessor(subject=args.subject, out_dir=preproc_dir)
            preprocessor.load_artifacts()
            logger.info(f"✅ Preprocessing loaded: {preprocessor.summary()}")
        else:
            # No preprocessing (use raw voxels)
            logger.warning("No preprocessing specified; this may not match training setup!")
            preprocessor = None
        
        # Initialize NIfTI loader
        s3_fs = get_s3_filesystem()
        nifti_loader = NIfTILoader(s3_fs)
        
        # Extract test features and targets
        X_test, Y_test, test_nsd_ids = extract_features_and_targets(
            test_df, nifti_loader, preprocessor, clip_cache
        )
        
        if len(X_test) == 0:
            logger.error("No valid test samples extracted!")
            return 1
        
        # Predict CLIP embeddings
        logger.info("Predicting CLIP embeddings from test fMRI...")
        Y_pred = encoder.predict(X_test)
        logger.info(f"✅ Predictions: {Y_pred.shape}")
        
        # Build gallery (exclude test samples)
        gallery_embeddings, gallery_nsd_ids = build_gallery(
            clip_cache, test_nsd_ids, args.gallery_limit
        )
        
        # Compute retrieval metrics (using ground truth CLIP embeddings)
        # Find ground truth indices in gallery
        gt_gallery_indices = []
        for nsd_id in test_nsd_ids:
            try:
                idx = np.where(gallery_nsd_ids == nsd_id)[0][0]
                gt_gallery_indices.append(idx)
            except IndexError:
                # Ground truth not in gallery (excluded)
                gt_gallery_indices.append(-1)
        
        gt_gallery_indices = np.array(gt_gallery_indices)
        valid_mask = gt_gallery_indices >= 0
        
        logger.info(f"Computing retrieval metrics ({valid_mask.sum()} samples with GT in gallery)...")
        
        if valid_mask.sum() > 0:
            metrics_retrieval = retrieval_at_k(
                Y_pred[valid_mask],
                gallery_embeddings,
                gt_gallery_indices[valid_mask],
                ks=(1, 5, 10, 20)
            )
            metrics_ranking = compute_ranking_metrics(
                Y_pred[valid_mask],
                gallery_embeddings,
                gt_gallery_indices[valid_mask]
            )
        else:
            logger.warning("No test samples have GT in gallery (all excluded); skipping retrieval metrics")
            metrics_retrieval = {}
            metrics_ranking = {}
        
        # Compute cosine similarity between predictions and ground truth
        cosine_scores = (Y_pred * Y_test).sum(axis=1)  # Assumes both are L2-normalized
        mean_cosine = cosine_scores.mean()
        std_cosine = cosine_scores.std()
        
        logger.info(f"✅ Mean cosine: {mean_cosine:.4f} ± {std_cosine:.4f}")
        
        for k, v in metrics_retrieval.items():
            logger.info(f"   {k}: {v:.4f}")
        
        for k, v in metrics_ranking.items():
            logger.info(f"   {k}: {v:.4f}")
        
        # Retrieve top-K neighbors for visualization
        topk_gallery_indices, topk_similarities = retrieve_nearest_neighbors(
            Y_pred, gallery_embeddings, k=args.topk
        )
        
        # Map gallery indices to NSD IDs
        topk_nsd_ids = gallery_nsd_ids[topk_gallery_indices]
        
        # Visualize reconstruction grids
        visualize_reconstruction_grid(
            test_nsd_ids,
            topk_nsd_ids,
            topk_similarities,
            clip_cache,
            out_fig_dir,
            n_samples=min(10, len(test_nsd_ids))
        )
        
        # Save results CSV
        results = {
            "subject": args.subject,
            "encoder": args.encoder,
            "checkpoint": args.ckpt,
            "n_test": len(X_test),
            "gallery_size": len(gallery_embeddings),
            "mean_cosine": mean_cosine,
            "std_cosine": std_cosine,
            **metrics_retrieval,
            **metrics_ranking
        }
        
        results_df = pd.DataFrame([results])
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out_csv, index=False)
        
        logger.info("\n" + "=" * 80)
        logger.info("RECONSTRUCTION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Results CSV: {out_csv}")
        logger.info(f"Figures: {out_fig_dir}/")
        
        return 0
        
    except Exception as e:
        logger.error(f"Reconstruction failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
