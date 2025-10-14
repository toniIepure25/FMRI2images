#!/usr/bin/env python3
"""
NSD Preprocessing Fitting Script
================================

Fits preprocessing pipeline (scaler + optional PCA) on training data split.

Usage:
    python scripts/nsd_fit_preproc.py --subject subj01 --k 4096 --reliability-thr 0.1

This script:
1. Reads the subject index and splits train/val/test per configs/data.yaml
2. Fits NSDPreprocessor on train split (T1: scaler + reliability mask)
3. Optionally fits PCA for dimensionality reduction (T2)
4. Saves artifacts to outputs/preproc/{subject}/
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path

import numpy as np
import pandas as pd

# Silence nibabel qfac warnings
logging.getLogger("nibabel.global").setLevel(logging.WARNING)

# Import our modules
from fmri2img.data.nsd_index_reader import read_subject_index
from fmri2img.data.preprocess import NSDPreprocessor
from fmri2img.io.s3 import NIfTILoader, get_s3_filesystem

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
    
    # Shuffle with fixed seed for reproducibility
    df_shuffled = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    n_total = len(df_shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_df = df_shuffled[:n_train]
    val_df = df_shuffled[n_train:n_train + n_val]
    test_df = df_shuffled[n_train + n_val:]
    
    logger.info(f"Split {n_total} trials: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    return train_df, val_df, test_df


def create_loader_factory():
    """Create a factory function that returns loader and volume extraction function."""
    def factory():
        s3_fs = get_s3_filesystem()
        nifti_loader = NIfTILoader(s3_fs)
        
        def get_volume(loader, row):
            """Extract volume from a DataFrame row."""
            try:
                if "beta_file" in row:
                    beta_path = row["beta_file"]
                    beta_index = int(row.get("volume_index", 0))
                else:
                    beta_path = row["beta_path"]
                    beta_index = int(row.get("beta_index", 0))
                img = loader.load(beta_path)
                vol = img.slicer[..., beta_index].get_fdata().astype(np.float32)
                return vol
            except Exception as e:
                logger.warning(f"Failed to load volume: {e}")
                return None
        
        return nifti_loader, get_volume
    
    return factory


def main():
    parser = argparse.ArgumentParser(description="Fit NSD preprocessing pipeline")
    parser.add_argument("--index-root", default="data/indices/nsd_index", 
                       help="Root directory or file path for NSD index")
    parser.add_argument("--subject", default="subj01", help="Subject to process")
    parser.add_argument("--session", type=int, help="Specific session to use (optional)")
    parser.add_argument("--k", type=int, default=4096, help="Number of PCA components")
    parser.add_argument("--reliability-thr", type=float, default=0.1, 
                       help="Test-retest reliability threshold")
    parser.add_argument("--min-variance", type=float, default=1e-6,
                       help="Minimum variance threshold")
    parser.add_argument("--no-pca", action="store_true", help="Skip PCA fitting")
    parser.add_argument("--roi-mode", choices=["pool"], help="Enable ROI pooling mode")
    parser.add_argument("--config", default="configs/data.yaml", help="Data config file")
    parser.add_argument("--out-dir", default="outputs/preproc", help="Output directory")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_data_config(args.config)
        splits = config.get("splits", {})
        
        # Read subject index
        logger.info(f"Reading index for {args.subject} from {args.index_root}")
        df = read_subject_index(args.index_root, args.subject)
        
        if args.session is not None and "session" in df.columns:
            df = df[df["session"] == args.session]
            logger.info(f"Filtered to session {args.session}: {len(df)} trials")
        
        if len(df) == 0:
            logger.error("No trials found after filtering")
            return 1
        
        # Split data
        train_df, val_df, test_df = split_dataframe(
            df,
            train_ratio=splits.get("train_ratio", 0.8),
            val_ratio=splits.get("val_ratio", 0.1), 
            test_ratio=splits.get("test_ratio", 0.1),
            random_seed=splits.get("random_seed", 42)
        )
        
        # Initialize preprocessor
        preprocessor = NSDPreprocessor(args.subject, args.out_dir, roi_mode=args.roi_mode)
        
        # Create loader factory
        loader_factory = create_loader_factory()
        
        # Fit scaler on training data
        logger.info("Fitting scaler and reliability mask...")
        preprocessor.fit(
            train_df, 
            loader_factory,
            reliability_threshold=args.reliability_thr,
            min_variance=args.min_variance
        )
        
        # Fit PCA if requested
        if not args.no_pca and args.k > 0:
            logger.info(f"Fitting PCA with {args.k} components...")
            preprocessor.fit_pca(train_df, loader_factory, k=args.k)
        
        # Print summary
        summary = preprocessor.summary()
        logger.info("Preprocessing fitted successfully!")
        logger.info(f"Subject: {summary['subject']}")
        logger.info(f"Voxels kept: {summary.get('n_voxels_kept', 'N/A'):,} / {summary.get('n_voxels_total', 'N/A'):,} "
                   f"({summary.get('voxel_retention_rate', 0):.1%})")
        
        if summary.get('pca_fitted', False):
            logger.info(f"PCA components: {summary.get('pca_components', 'N/A')}")
            logger.info(f"Explained variance: {summary.get('explained_variance_ratio', 0):.1%}")
            
        if summary.get('roi_fitted', False):
            logger.info(f"ROI pooling: {summary.get('n_rois', 'N/A')} regions")
            roi_names = summary.get('roi_names', [])
            if roi_names:
                logger.info(f"ROI names: {', '.join(roi_names[:5])}{'...' if len(roi_names) > 5 else ''}")
        
        # Print artifacts locations
        artifacts_dir = Path(args.out_dir) / args.subject
        logger.info(f"Artifacts saved to: {artifacts_dir}")
        for artifact in artifacts_dir.glob("*"):
            logger.info(f"  - {artifact.name}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to fit preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())