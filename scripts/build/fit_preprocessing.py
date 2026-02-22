#!/usr/bin/env python3
"""
Command-line interface for fMRI preprocessing pipeline.

This script fits the preprocessing pipeline (z-score, reliability masking/weighting, PCA)
on training data and saves the artifacts for later use.
"""

import argparse
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fmri2img.data.preprocess import NSDPreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Fit and save fMRI preprocessing pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--subject",
        type=str,
        required=True,
        help="Subject ID (e.g., subj01)"
    )
    
    parser.add_argument(
        "--index-file",
        type=str,
        required=True,
        help="Path to NSD index parquet file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for preprocessing artifacts"
    )
    
    # Reliability options
    parser.add_argument(
        "--reliability-mode",
        type=str,
        default="hard_threshold",
        choices=["hard_threshold", "soft_weight", "none"],
        help="Reliability weighting mode"
    )
    
    parser.add_argument(
        "--reliability-threshold",
        type=float,
        default=0.1,
        help="Reliability threshold"
    )
    
    parser.add_argument(
        "--reliability-curve",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "linear"],
        help="Soft weighting curve type (only used if mode=soft_weight)"
    )
    
    parser.add_argument(
        "--reliability-temperature",
        type=float,
        default=0.1,
        help="Temperature for sigmoid curve (only used if mode=soft_weight)"
    )
    
    # PCA options
    parser.add_argument(
        "--n-components",
        type=int,
        default=3072,
        help="Number of PCA components"
    )
    
    parser.add_argument(
        "--pca-whiten",
        action="store_true",
        help="Apply whitening in PCA"
    )
    
    args = parser.parse_args()
    
    try:
        # Setup output directory
        output_dir = Path(args.output_dir)
        subject_dir = output_dir / args.subject
        subject_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Preprocessing subject: {args.subject}")
        logger.info(f"Index file: {args.index_file}")
        logger.info(f"Output directory: {subject_dir}")
        logger.info(f"Reliability mode: {args.reliability_mode}")
        logger.info(f"PCA components: {args.n_components}")
        
        # Load index to get training data
        import pandas as pd
        index_df = pd.read_parquet(args.index_file)
        
        # Check if 'split' column exists, otherwise create simple train/val/test split
        if 'split' not in index_df.columns:
            logger.warning("Index doesn't have 'split' column, creating 80/10/10 train/val/test split")
            n = len(index_df)
            train_end = int(0.8 * n)
            val_end = int(0.9 * n)
            index_df['split'] = 'test'
            index_df.loc[:train_end-1, 'split'] = 'train'
            index_df.loc[train_end:val_end-1, 'split'] = 'val'
        
        train_df = index_df[index_df['split'] == 'train'].copy()
        
        logger.info(f"Loaded {len(train_df)} training samples from {len(index_df)} total")
        
        # Create preprocessor
        preprocessor = NSDPreprocessor(
            subject=args.subject,
            out_dir=args.output_dir
        )
        
        # Create loader factory
        from fmri2img.io.s3 import NIfTILoader, get_s3_filesystem
        def loader_factory():
            s3_fs = get_s3_filesystem()
            loader = NIfTILoader(s3_fs)
            def get_volume(loader, row):
                # Load the NIfTI file and extract the specific volume
                img = loader.load(row['beta_path'])
                data = img.get_fdata()
                # Extract the specific beta volume using beta_index
                return data[..., row['beta_index']]
            return loader, get_volume
        
        # Fit on training data
        logger.info("Fitting preprocessing pipeline on training data...")
        preprocessor.fit(
            train_df=train_df,
            loader_factory=loader_factory,
            reliability_threshold=args.reliability_threshold,
            reliability_mode=args.reliability_mode,
            reliability_curve=args.reliability_curve,
            reliability_temperature=args.reliability_temperature
        )
        
        # Fit PCA if requested
        if args.n_components > 0:
            logger.info(f"Fitting PCA with {args.n_components} components...")
            preprocessor.fit_pca(
                train_df=train_df,
                loader_factory=loader_factory,
                k=args.n_components
            )
        
        # Print summary
        logger.info("\nPreprocessing Summary:")
        logger.info(f"  Subject: {args.subject}")
        logger.info(f"  Training samples: {len(train_df)}")
        if preprocessor.mask_ is not None:
            logger.info(f"  Reliable voxels: {preprocessor.mask_.sum():,}")
        if args.reliability_mode == "soft_weight" and preprocessor.weights_ is not None:
            nonzero_weights = preprocessor.weights_[preprocessor.weights_ > 0]
            logger.info(f"  Weighted voxels: {len(nonzero_weights):,}")
            if len(nonzero_weights) > 0:
                logger.info(f"  Mean weight: {nonzero_weights.mean():.4f}")
        if preprocessor.pca_fitted_ and preprocessor.pca_ is not None:
            logger.info(f"  PCA components: {preprocessor.pca_.n_components_}")
            if hasattr(preprocessor.pca_, 'explained_variance_ratio_'):
                logger.info(f"  Explained variance: {preprocessor.pca_.explained_variance_ratio_.sum():.4f}")
        
        logger.info("\n✓ Preprocessing completed successfully!")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
