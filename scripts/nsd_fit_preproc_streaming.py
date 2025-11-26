#!/usr/bin/env python3
"""
Production NSD Preprocessing with Streaming PCA
===============================================

Professional preprocessing pipeline that:
- Streams fMRI volumes without loading all into memory
- Uses IncrementalPCA for memory-efficient dimensionality reduction
- Computes proper voxel-wise statistics with Welford's algorithm
- Provides real-time progress updates
- Handles large datasets (24K+ volumes) efficiently

Usage:
    python scripts/nsd_fit_preproc_streaming.py \\
        --subject subj01 \\
        --k 512 \\
        --reliability-thr 0.0 \\
        --out-dir outputs/preproc
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Callable, Tuple, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from sklearn.decomposition import IncrementalPCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.error("scikit-learn not available! Install with: pip install scikit-learn")
    sys.exit(1)

from fmri2img.data.nsd_index_reader import read_subject_index
from fmri2img.io.s3 import get_s3_filesystem, NIfTILoader


def split_dataframe(df: pd.DataFrame, train_ratio: float = 0.8, 
                   val_ratio: float = 0.1, test_ratio: float = 0.1,
                   random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train/val/test."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    
    df_shuffled = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    n_total = len(df_shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_df = df_shuffled[:n_train]
    val_df = df_shuffled[n_train:n_train + n_val]
    test_df = df_shuffled[n_train + n_val:]
    
    return train_df, val_df, test_df


def create_loader_factory() -> Callable:
    """Create factory for NIfTI loader."""
    def factory():
        s3_fs = get_s3_filesystem()
        nifti_loader = NIfTILoader(s3_fs)
        
        def get_volume(loader, row):
            try:
                beta_path = row.get("beta_path") or row.get("beta_file")
                beta_index = int(row.get("beta_index", row.get("volume_index", 0)))
                img = loader.load(beta_path)
                vol = img.slicer[..., beta_index].get_fdata().astype(np.float32)
                return vol
            except Exception as e:
                logger.warning(f"Failed to load volume: {e}")
                return None
        
        return nifti_loader, get_volume
    
    return factory


def fit_scaler_streaming(train_df: pd.DataFrame, loader_factory: Callable,
                        min_variance: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit voxel-wise scaler using Welford's online algorithm (streaming).
    Optimized: Groups trials by beta file to minimize file loading.
    
    Returns:
        mean: Voxelwise mean (H, W, D)
        std: Voxelwise std (H, W, D)
        mask: Reliability mask (H, W, D)
    """
    logger.info(f"Fitting scaler on {len(train_df)} training volumes (streaming)")
    
    # Group by beta file to load each file only once
    beta_col = "beta_path" if "beta_path" in train_df.columns else "beta_file"
    index_col = "beta_index" if "beta_index" in train_df.columns else "volume_index"
    
    grouped = train_df.groupby(beta_col)
    n_files = len(grouped)
    
    logger.info(f"Grouped {len(train_df)} trials into {n_files} unique beta files")
    logger.info("Loading files in batches (much faster!)...")
    
    nifti_loader, _ = loader_factory()
    
    # Welford's algorithm accumulators
    count = 0
    mean = None
    M2 = None
    
    # Progress bar over files (not individual volumes)
    pbar = tqdm(grouped, total=n_files, desc="Loading beta files", unit="file")
    
    for beta_path, group_df in pbar:
        try:
            # Load the 4D NIfTI file once
            img = nifti_loader.load(beta_path)
            
            # Extract all volumes from this file
            for _, row in group_df.iterrows():
                try:
                    beta_index = int(row[index_col])
                    vol = img.slicer[..., beta_index].get_fdata().astype(np.float32)
                    
                    # Initialize on first volume
                    if mean is None:
                        mean = np.zeros_like(vol)
                        M2 = np.zeros_like(vol)
                    
                    # Welford's update
                    count += 1
                    delta = vol - mean
                    mean += delta / count
                    delta2 = vol - mean
                    M2 += delta * delta2
                    
                except Exception as e:
                    logger.debug(f"Failed to extract volume {beta_index}: {e}")
                    continue
            
            # Update progress
            pbar.set_postfix({
                "volumes": count,
                "avg_per_file": f"{count/pbar.n:.0f}",
                "mem_mb": int(mean.nbytes * 2 / 1024 / 1024) if mean is not None else 0
            })
            
        except Exception as e:
            logger.warning(f"Failed to load file {beta_path}: {e}")
            continue
    
    pbar.close()
    
    if count == 0:
        raise ValueError("No valid volumes loaded")
    
    logger.info(f"✓ Processed {count} volumes")
    
    # Compute variance and std
    variance = M2 / (count - 1) if count > 1 else M2
    std = np.sqrt(variance)
    std = np.maximum(std, min_variance)
    
    # Create mask: keep voxels with variance above threshold
    mask = variance >= min_variance
    
    n_voxels_total = mask.size
    n_voxels_kept = mask.sum()
    retention_rate = n_voxels_kept / n_voxels_total
    
    logger.info(f"✓ Voxel retention: {n_voxels_kept:,} / {n_voxels_total:,} ({retention_rate:.1%})")
    
    return mean, std, mask, count


def fit_pca_streaming(train_df: pd.DataFrame, loader_factory: Callable,
                     mask: np.ndarray, mean: np.ndarray, std: np.ndarray,
                     k: int, batch_size: int = 100) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Fit PCA using IncrementalPCA (streaming, memory-efficient).
    Optimized: Groups trials by beta file to minimize file loading.
    
    Args:
        train_df: Training dataframe
        loader_factory: Loader factory function
        mask: Voxel mask (H, W, D)
        mean: Voxel mean (H, W, D)
        std: Voxel std (H, W, D)
        k: Number of PCA components
        batch_size: Batch size for incremental fitting
    
    Returns:
        components: PCA components (k_eff, n_features)
        pca_mean: PCA mean (n_features,)
        k_eff: Effective number of components
    """
    n_features = mask.sum()
    n_samples = len(train_df)
    k_eff = min(k, n_features, n_samples)
    
    if k_eff < k:
        logger.warning(f"Requested k={k}, but capping to k_eff={k_eff} "
                      f"(n_features={n_features}, n_samples={n_samples})")
    
    logger.info(f"Fitting IncrementalPCA: k={k_eff}, n_features={n_features}, batch_size={batch_size}")
    
    # Initialize IncrementalPCA
    pca = IncrementalPCA(n_components=k_eff, batch_size=batch_size)
    
    # Group by beta file to load each file only once
    beta_col = "beta_path" if "beta_path" in train_df.columns else "beta_file"
    index_col = "beta_index" if "beta_index" in train_df.columns else "volume_index"
    
    grouped = train_df.groupby(beta_col)
    n_files = len(grouped)
    
    logger.info(f"Grouped {len(train_df)} trials into {n_files} unique beta files")
    
    nifti_loader, _ = loader_factory()
    voxel_indices = np.where(mask.ravel())[0]
    
    # Fit PCA in batches
    batch = []
    n_processed = 0
    pbar = tqdm(grouped, total=n_files, desc="Fitting PCA on beta files", unit="file")
    
    for beta_path, group_df in pbar:
        try:
            # Load the 4D NIfTI file once
            img = nifti_loader.load(beta_path)
            
            # Extract all volumes from this file
            for _, row in group_df.iterrows():
                try:
                    beta_index = int(row[index_col])
                    vol = img.slicer[..., beta_index].get_fdata().astype(np.float32)
                    
                    # Standardize and mask
                    vol_std = (vol - mean) / std
                    vol_flat = vol_std.ravel()[voxel_indices]
                    
                    batch.append(vol_flat)
                    
                    # Fit batch when full
                    if len(batch) >= batch_size:
                        X_batch = np.array(batch)
                        pca.partial_fit(X_batch)
                        n_processed += len(batch)
                        batch = []
                        
                except Exception as e:
                    logger.debug(f"Failed to extract volume {beta_index}: {e}")
                    continue
            
            # Update progress
            pbar.set_postfix({
                "processed": n_processed,
                "batches": pca.n_samples_seen_ // batch_size if pca.n_samples_seen_ > 0 else 0,
                "var_exp": f"{pca.explained_variance_ratio_.sum():.1%}" if hasattr(pca, 'explained_variance_ratio_') else "N/A"
            })
            
        except Exception as e:
            logger.warning(f"Failed to load file {beta_path}: {e}")
            continue
    
    # Fit remaining samples
    if batch:
        X_batch = np.array(batch)
        pca.partial_fit(X_batch)
    
    pbar.close()
    
    # Get results
    components = pca.components_.astype(np.float32)
    pca_mean = pca.mean_.astype(np.float32)
    explained_var = pca.explained_variance_ratio_.sum()
    
    logger.info(f"✓ PCA fitted: {k_eff} components, {explained_var:.1%} variance explained")
    
    return components, pca_mean, k_eff, explained_var


def main():
    parser = argparse.ArgumentParser(
        description="Production NSD preprocessing with streaming PCA",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--index-root", default="data/indices/nsd_index",
                       help="Root directory for NSD index")
    parser.add_argument("--subject", default="subj01", help="Subject to process")
    parser.add_argument("--k", type=int, default=512, help="Number of PCA components")
    parser.add_argument("--reliability-thr", type=float, default=0.0,
                       help="Reliability threshold (not used in streaming version)")
    parser.add_argument("--min-variance", type=float, default=1e-6,
                       help="Minimum variance threshold for voxel retention")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Batch size for incremental PCA")
    parser.add_argument("--out-dir", default="outputs/preproc",
                       help="Output directory")
    parser.add_argument("--config", default="configs/data.yaml",
                       help="Data config file (for split ratios)")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        splits = config.get("splits", {})
        
        # Create output directory
        out_dir = Path(args.out_dir) / args.subject
        out_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("="*80)
        logger.info("NSD PREPROCESSING - STREAMING VERSION")
        logger.info("="*80)
        logger.info(f"Subject: {args.subject}")
        logger.info(f"PCA components: {args.k}")
        logger.info(f"Min variance: {args.min_variance}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Output: {out_dir}")
        logger.info("="*80)
        
        # Read and split data
        logger.info("Loading index...")
        df = read_subject_index(args.index_root, args.subject)
        
        train_df, val_df, test_df = split_dataframe(
            df,
            train_ratio=splits.get("train_ratio", 0.8),
            val_ratio=splits.get("val_ratio", 0.1),
            test_ratio=splits.get("test_ratio", 0.1),
            random_seed=splits.get("random_seed", 42)
        )
        
        logger.info(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        # Create loader factory
        loader_factory = create_loader_factory()
        
        # Phase 1: Fit scaler (streaming)
        logger.info("\n" + "="*80)
        logger.info("PHASE 1: Fitting Scaler (Streaming)")
        logger.info("="*80)
        
        mean, std, mask, n_train_samples = fit_scaler_streaming(
            train_df, loader_factory, args.min_variance
        )
        
        # Save scaler artifacts
        np.save(out_dir / "scaler_mean.npy", mean)
        np.save(out_dir / "scaler_std.npy", std)
        np.save(out_dir / "reliability_mask.npy", mask)
        
        voxel_indices = np.where(mask.ravel())[0]
        np.save(out_dir / "voxel_indices.npy", voxel_indices)
        
        # Phase 2: Fit PCA (streaming)
        logger.info("\n" + "="*80)
        logger.info("PHASE 2: Fitting PCA (Incremental)")
        logger.info("="*80)
        
        pca_components, pca_mean, k_eff, explained_var = fit_pca_streaming(
            train_df, loader_factory, mask, mean, std,
            args.k, args.batch_size
        )
        
        # Save PCA artifacts
        np.save(out_dir / "pca_components.npy", pca_components)
        np.save(out_dir / "pca_mean.npy", pca_mean)
        
        # Save metadata
        meta = {
            "subject": args.subject,
            "roi_mode": None,
            "n_train_samples": n_train_samples,
            "n_voxels_total": int(mask.size),
            "n_voxels_kept": int(mask.sum()),
            "voxel_retention_rate": float(mask.mean()),
            "reliability_method": "variance",
            "reliability_threshold": args.min_variance,
            "split_half_seed": None,
            "pca_fitted": True,
            "pca_components": k_eff,
            "explained_variance_ratio": float(explained_var),
            "preprocessing_version": "streaming"
        }
        
        with open(out_dir / "meta.json", 'w') as f:
            json.dump(meta, f, indent=2)
        
        rel_meta = {
            "method": "variance",
            "reliability_threshold": args.min_variance,
            "n_repeated_ids": 0,
            "seed": None,
            "mean_r_retained": 0.0
        }
        
        with open(out_dir / "reliability_meta.json", 'w') as f:
            json.dump(rel_meta, f, indent=2)
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("✅ PREPROCESSING COMPLETE")
        logger.info("="*80)
        logger.info(f"Output directory: {out_dir}")
        logger.info(f"Train samples: {n_train_samples:,}")
        logger.info(f"Voxels: {mask.sum():,} / {mask.size:,} ({100*mask.mean():.1f}%)")
        logger.info(f"PCA: {k_eff} components ({explained_var:.1%} variance)")
        logger.info("="*80)
        
        for artifact in sorted(out_dir.glob("*")):
            if artifact.is_file():
                size_mb = artifact.stat().st_size / (1024 * 1024)
                logger.info(f"  ✓ {artifact.name:30s} ({size_mb:6.2f} MB)")
        
        logger.info("="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
