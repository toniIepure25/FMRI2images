#!/usr/bin/env python3
"""
Ridge Baseline Training Script
==============================

Train a Ridge regression model for fMRI → CLIP embedding mapping.

Pipeline:
1. Load canonical index and split train/val/test
2. Load preprocessing artifacts (T0+T1+T2)
3. Extract fMRI features and CLIP embeddings
4. Hyperparameter selection: choose α on validation set
5. Retrain on train+val with best α
6. Evaluate on test set (cosine, MSE, retrieval@K)
7. Save model and evaluation report

Scientific Design:
- Hyperparameter selection on validation only (no test leakage)
- L2-normalize CLIP embeddings and predictions (standard practice)
- Retrain on train+val before final test (maximizes data usage)
- Report retrieval@K on test set (standard NSD evaluation)

Usage:
    # Quick test with tiny PCA
    python scripts/train_ridge.py --subject subj01 --limit 256 --alpha-grid "1,10"
    
    # Full run
    python scripts/train_ridge.py \
        --subject subj01 \
        --use-preproc --pca-k 4096 \
        --clip-cache outputs/clip_cache/clip.parquet \
        --alpha-grid "0.1,1,3,10,30,100" \
        --limit 2048
"""

import argparse
import json
import logging
import sys
import yaml
from pathlib import Path

import numpy as np
import pandas as pd

# Silence nibabel warnings
logging.getLogger("nibabel.global").setLevel(logging.WARNING)

from fmri2img.data.nsd_index_reader import read_subject_index
from fmri2img.data.preprocess import NSDPreprocessor
from fmri2img.data.clip_cache import CLIPCache
from fmri2img.io.s3 import NIfTILoader, get_s3_filesystem
from fmri2img.models.ridge import RidgeEncoder, evaluate_predictions
from fmri2img.eval.retrieval import retrieval_at_k, compute_ranking_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data_config(config_path="configs/data.yaml"):
    """Load data configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def split_dataframe(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """Split dataframe into train/val/test."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    
    # Shuffle with fixed seed
    df_shuffled = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    n_total = len(df_shuffled)
    
    # Ensure minimum samples for each split (at least 1 for val/test if n >= 3)
    if n_total < 3:
        raise ValueError(f"Need at least 3 samples for train/val/test split, got {n_total}")
    
    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio))
    n_test = n_total - n_train - n_val
    
    if n_test < 1:
        # Adjust to ensure at least 1 test sample
        n_test = 1
        n_val = max(1, n_total - n_train - n_test)
        n_train = n_total - n_val - n_test
    
    train_df = df_shuffled[:n_train]
    val_df = df_shuffled[n_train:n_train + n_val]
    test_df = df_shuffled[n_train + n_val:]
    
    logger.info(f"Split {n_total} trials: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    return train_df, val_df, test_df


def extract_features_and_targets(
    df: pd.DataFrame,
    nifti_loader: NIfTILoader,
    preprocessor: NSDPreprocessor,
    clip_cache: CLIPCache,
    desc: str = "data"
) -> tuple:
    """
    Extract fMRI features and CLIP targets from DataFrame.
    
    Returns:
        X: fMRI features (n_samples, n_features)
        Y: CLIP embeddings (n_samples, 512), L2-normalized
        nsd_ids: NSD stimulus IDs (n_samples,)
    """
    X_list = []
    Y_list = []
    nsd_ids = []
    
    logger.info(f"Extracting {desc}: {len(df)} samples")
    
    for idx, row in df.iterrows():
        try:
            # Load fMRI volume
            beta_path = row["beta_path"]
            beta_index = int(row["beta_index"])
            nsd_id = int(row["nsdId"])
            
            img = nifti_loader.load(beta_path)
            data_4d = img.get_fdata()
            vol = data_4d[..., beta_index].astype(np.float32)
            
            # Apply preprocessing (T0+T1+T2)
            fmri_features = preprocessor.transform(vol)
            
            # Get CLIP embedding
            clip_emb = clip_cache.get([nsd_id])
            if nsd_id not in clip_emb:
                logger.warning(f"CLIP embedding missing for nsdId={nsd_id}, skipping")
                continue
            
            clip_vec = clip_emb[nsd_id]
            
            # Verify L2 normalization
            clip_norm = np.linalg.norm(clip_vec)
            if not np.isclose(clip_norm, 1.0, atol=1e-3):
                logger.warning(f"CLIP embedding not normalized (norm={clip_norm:.3f}), normalizing")
                clip_vec = clip_vec / clip_norm
            
            X_list.append(fmri_features)
            Y_list.append(clip_vec)
            nsd_ids.append(nsd_id)
            
        except Exception as e:
            logger.warning(f"Failed to process row {idx}: {e}")
            continue
    
    if len(X_list) == 0:
        raise ValueError(f"No valid samples extracted from {desc}")
    
    X = np.stack(X_list).astype(np.float32)
    Y = np.stack(Y_list).astype(np.float32)
    nsd_ids = np.array(nsd_ids)
    
    logger.info(f"✅ Extracted {desc}: X {X.shape}, Y {Y.shape}")
    
    return X, Y, nsd_ids


def select_alpha(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    alpha_grid: list
) -> tuple:
    """
    Select best alpha by validation cosine similarity.
    
    Returns:
        best_alpha: Best alpha value
        results: Dict of alpha -> validation metrics
    """
    logger.info(f"Alpha selection: testing {len(alpha_grid)} values on validation set")
    
    results = {}
    best_alpha = None
    best_cosine = -np.inf
    
    for alpha in alpha_grid:
        # Train model
        model = RidgeEncoder(alpha=alpha)
        model.fit(X_train, Y_train)
        
        # Evaluate on validation
        Y_val_pred = model.predict(X_val, normalize=True)
        metrics = evaluate_predictions(Y_val, Y_val_pred, normalize=True)
        
        results[alpha] = metrics
        logger.info(f"  α={alpha:8.3f}: cosine={metrics['cosine']:.4f} ± {metrics['cosine_std']:.4f}, MSE={metrics['mse']:.4f}")
        
        if metrics['cosine'] > best_cosine:
            best_cosine = metrics['cosine']
            best_alpha = alpha
    
    logger.info(f"✅ Best α={best_alpha:.3f} (val cosine={best_cosine:.4f})")
    
    return best_alpha, results


def main():
    parser = argparse.ArgumentParser(description="Train Ridge baseline for fMRI → CLIP")
    parser.add_argument("--index-root", default="data/indices/nsd_index",
                       help="NSD index root directory")
    parser.add_argument("--subject", default="subj01", help="Subject to train on")
    parser.add_argument("--alpha-grid", default="0.1,1,3,10,30,100",
                       help="Comma-separated alpha values for grid search")
    parser.add_argument("--use-preproc", action="store_true",
                       help="Use preprocessing pipeline")
    parser.add_argument("--pca-k", type=int, help="PCA components (implies --use-preproc)")
    parser.add_argument("--clip-cache", default="outputs/clip_cache/clip.parquet",
                       help="Path to CLIP cache")
    parser.add_argument("--limit", type=int, help="Limit number of samples (for testing)")
    parser.add_argument("--config", default="configs/data.yaml", help="Data config file")
    parser.add_argument("--preproc-dir", default="outputs/preproc",
                       help="Preprocessing artifacts directory")
    parser.add_argument("--checkpoint-dir", default="checkpoints/ridge",
                       help="Model checkpoint directory")
    parser.add_argument("--report-dir", default="outputs/reports",
                       help="Evaluation report directory")
    
    args = parser.parse_args()
    
    # Parse alpha grid
    alpha_grid = [float(x.strip()) for x in args.alpha_grid.split(",")]
    
    try:
        # Load configuration
        config = load_data_config(args.config)
        splits = config.get("splits", {})
        
        # Load subject index
        logger.info(f"Loading index for {args.subject} from {args.index_root}")
        df = read_subject_index(args.index_root, args.subject)
        
        if args.limit:
            df = df.head(args.limit)
            logger.info(f"Limited to {len(df)} samples")
        
        # Split data
        train_df, val_df, test_df = split_dataframe(
            df,
            train_ratio=splits.get("train_ratio", 0.8),
            val_ratio=splits.get("val_ratio", 0.1),
            test_ratio=splits.get("test_ratio", 0.1),
            random_seed=splits.get("random_seed", 42)
        )
        
        # Setup preprocessor
        preprocessor = NSDPreprocessor(args.subject, args.preproc_dir)
        if args.use_preproc or args.pca_k:
            if not preprocessor.load_artifacts():
                logger.error("Preprocessing artifacts not found. Run nsd_fit_preproc.py first!")
                return 1
            summary = preprocessor.summary()
            logger.info(f"Loaded preprocessing: {summary['n_voxels_kept']:,} voxels")
            if summary.get('pca_fitted'):
                logger.info(f"  PCA: {summary['pca_components']} components")
        
        # Load CLIP cache
        logger.info(f"Loading CLIP cache from {args.clip_cache}")
        clip_cache = CLIPCache(args.clip_cache).load()
        stats = clip_cache.stats()
        logger.info(f"✅ CLIP cache loaded: {stats['cache_size']} embeddings")
        
        # Initialize NIfTI loader
        s3_fs = get_s3_filesystem()
        nifti_loader = NIfTILoader(s3_fs)
        
        # Extract features
        X_train, Y_train, train_ids = extract_features_and_targets(
            train_df, nifti_loader, preprocessor, clip_cache, "train"
        )
        X_val, Y_val, val_ids = extract_features_and_targets(
            val_df, nifti_loader, preprocessor, clip_cache, "validation"
        )
        X_test, Y_test, test_ids = extract_features_and_targets(
            test_df, nifti_loader, preprocessor, clip_cache, "test"
        )
        
        # Alpha selection on validation set
        logger.info("=" * 80)
        logger.info("ALPHA SELECTION (validation set)")
        logger.info("=" * 80)
        best_alpha, alpha_results = select_alpha(X_train, Y_train, X_val, Y_val, alpha_grid)
        
        # Retrain on train+val with best alpha
        logger.info("=" * 80)
        logger.info(f"RETRAINING on train+val with α={best_alpha:.3f}")
        logger.info("=" * 80)
        X_trainval = np.vstack([X_train, X_val])
        Y_trainval = np.vstack([Y_train, Y_val])
        
        final_model = RidgeEncoder(alpha=best_alpha)
        final_model.fit(X_trainval, Y_trainval)
        
        # Evaluate on test set
        logger.info("=" * 80)
        logger.info("TEST SET EVALUATION")
        logger.info("=" * 80)
        
        Y_test_pred = final_model.predict(X_test, normalize=True)
        test_metrics = evaluate_predictions(Y_test, Y_test_pred, normalize=True)
        
        logger.info(f"Cosine: {test_metrics['cosine']:.4f} ± {test_metrics['cosine_std']:.4f}")
        logger.info(f"MSE: {test_metrics['mse']:.4f}")
        
        # Retrieval evaluation
        # Build gallery from test set (in practice, could be larger)
        logger.info("\nRetrieval evaluation (test set as gallery)...")
        gt_indices = np.arange(len(Y_test))  # Each query matches its own index
        
        retrieval_metrics = retrieval_at_k(
            Y_test_pred, Y_test, gt_indices, ks=(1, 5, 10)
        )
        
        for k, v in retrieval_metrics.items():
            logger.info(f"{k}: {v:.4f} ({v*100:.2f}%)")
        
        # Additional ranking metrics
        ranking_metrics = compute_ranking_metrics(Y_test_pred, Y_test, gt_indices)
        logger.info(f"Mean rank: {ranking_metrics['mean_rank']:.2f}")
        logger.info(f"Median rank: {ranking_metrics['median_rank']:.2f}")
        logger.info(f"MRR: {ranking_metrics['mrr']:.4f}")
        
        # Save model
        checkpoint_path = Path(args.checkpoint_dir) / args.subject / "ridge.pkl"
        final_model.save(checkpoint_path)
        
        # Save evaluation report
        report = {
            "subject": args.subject,
            "preprocessing": {
                "used": args.use_preproc or args.pca_k is not None,
                "pca_k": preprocessor.summary().get("pca_components", None) if preprocessor.is_fitted_ else None,
                "n_voxels_kept": preprocessor.summary().get("n_voxels_kept", None) if preprocessor.is_fitted_ else None,
            },
            "data_splits": {
                "n_train": len(train_df),
                "n_val": len(val_df),
                "n_test": len(test_df),
                "n_train_valid": len(X_train),
                "n_val_valid": len(X_val),
                "n_test_valid": len(X_test),
            },
            "hyperparameters": {
                "alpha_grid": alpha_grid,
                "best_alpha": best_alpha,
                "alpha_selection_results": {str(k): v for k, v in alpha_results.items()},
            },
            "validation_metrics": alpha_results[best_alpha],
            "test_metrics": {
                **test_metrics,
                **retrieval_metrics,
                **ranking_metrics,
            },
            "model_checkpoint": str(checkpoint_path),
        }
        
        report_path = Path(args.report_dir) / args.subject / "ridge_eval.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("=" * 80)
        logger.info(f"✅ Training complete!")
        logger.info(f"Model: {checkpoint_path}")
        logger.info(f"Report: {report_path}")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
