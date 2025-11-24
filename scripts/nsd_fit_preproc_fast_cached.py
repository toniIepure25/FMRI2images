#!/usr/bin/env python3
"""
Ultra-fast preprocessing using cached files directly.
Bypasses S3 filesystem for maximum speed.
"""
import argparse
import json
import logging
import hashlib
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress nibabel warnings
logging.getLogger('nibabel').setLevel(logging.ERROR)


def get_cache_path(s3_path: str, cache_dir: Path) -> Path:
    """Get local cache path for S3 file."""
    # Try with .nii.gz extension
    cache_hash = hashlib.sha256(s3_path.encode()).hexdigest()
    cached_file = cache_dir / f"{cache_hash}.nii.gz"
    if cached_file.exists():
        return cached_file
    
    # Try without extension
    cached_file = cache_dir / cache_hash
    if cached_file.exists():
        return cached_file
    
    raise FileNotFoundError(f"Cache not found for {s3_path}")


def fit_scaler_fast(train_df: pd.DataFrame, cache_dir: Path, 
                    min_variance: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit scaler using Welford's algorithm, loading cached 4D files."""
    logger.info(f"Fitting scaler on {len(train_df)} volumes")
    
    # Group by beta file
    beta_col = "beta_path" if "beta_path" in train_df.columns else "beta_file"
    index_col = "beta_index" if "beta_index" in train_df.columns else "volume_index"
    
    grouped = train_df.groupby(beta_col)
    logger.info(f"Processing {len(grouped)} unique beta files")
    
    # Welford's algorithm
    count = 0
    mean = None
    M2 = None
    
    pbar = tqdm(grouped, desc="Loading beta files", unit="file")
    
    for beta_path, group_df in pbar:
        try:
            # Get cached file path
            cached_file = get_cache_path(beta_path, cache_dir)
            
            # Load 4D NIfTI (memory-mapped, don't load all data)
            img = nib.load(str(cached_file))
            
            # Extract needed volumes ONE AT A TIME (don't load entire 4D array)
            indices = group_df[index_col].values.astype(int)
            
            for idx in indices:
                # Use slicer to load only this volume (more efficient than get_fdata)
                vol = img.slicer[..., idx].get_fdata().astype(np.float32)
                
                if mean is None:
                    mean = np.zeros_like(vol)
                    M2 = np.zeros_like(vol)
                
                count += 1
                delta = vol - mean
                mean += delta / count
                delta2 = vol - mean
                M2 += delta * delta2
            
            pbar.set_postfix({"volumes": count})
            
        except Exception as e:
            logger.warning(f"Failed to load {beta_path}: {e}")
            continue
    
    pbar.close()
    
    if count == 0:
        raise ValueError("No valid volumes loaded")
    
    logger.info(f"✓ Processed {count} volumes")
    
    # Compute variance and mask
    variance = M2 / (count - 1) if count > 1 else M2
    std = np.sqrt(variance)
    std = np.maximum(std, min_variance)
    mask = variance >= min_variance
    
    logger.info(f"✓ Voxel retention: {mask.sum():,} / {mask.size:,} ({mask.sum()/mask.size:.1%})")
    
    return mean, std, mask


def fit_pca_fast(train_df: pd.DataFrame, cache_dir: Path,
                 mask: np.ndarray, mean: np.ndarray, std: np.ndarray,
                 k: int, batch_size: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Fit PCA using IncrementalPCA on cached files."""
    n_features = mask.sum()
    n_samples = len(train_df)
    k_eff = min(k, n_features, n_samples)
    
    logger.info(f"Fitting PCA: k={k_eff}, n_features={n_features}")
    
    pca = IncrementalPCA(n_components=k_eff, batch_size=batch_size)
    
    # Group by beta file
    beta_col = "beta_path" if "beta_path" in train_df.columns else "beta_file"
    index_col = "beta_index" if "beta_index" in train_df.columns else "volume_index"
    
    grouped = train_df.groupby(beta_col)
    voxel_indices = np.where(mask.ravel())[0]
    
    batch = []
    n_processed = 0
    
    pbar = tqdm(grouped, desc="Fitting PCA", unit="file")
    
    for beta_path, group_df in pbar:
        try:
            cached_file = get_cache_path(beta_path, cache_dir)
            img = nib.load(str(cached_file))
            data_4d = img.get_fdata().astype(np.float32)
            
            indices = group_df[index_col].values.astype(int)
            
            for idx in indices:
                vol = data_4d[..., idx]
                vol_std = (vol - mean) / std
                vol_flat = vol_std.ravel()[voxel_indices]
                
                batch.append(vol_flat)
                
                if len(batch) >= batch_size:
                    X_batch = np.array(batch)
                    pca.partial_fit(X_batch)
                    n_processed += len(batch)
                    batch = []
            
            pbar.set_postfix({
                "processed": n_processed,
                "var": f"{pca.explained_variance_ratio_.sum():.1%}" if hasattr(pca, 'explained_variance_ratio_') else "N/A"
            })
            
        except Exception as e:
            logger.warning(f"Failed to load {beta_path}: {e}")
            continue
    
    # Fit remaining
    if batch:
        X_batch = np.array(batch)
        pca.partial_fit(X_batch)
        n_processed += len(batch)
    
    pbar.close()
    
    logger.info(f"✓ Fitted PCA on {n_processed} volumes")
    logger.info(f"✓ Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    return pca.components_, pca.mean_


def main():
    parser = argparse.ArgumentParser(description="Ultra-fast preprocessing using cached files")
    parser.add_argument("--subject", type=str, required=True, help="Subject ID (e.g., subj01)")
    parser.add_argument("--k", type=int, default=512, help="Number of PCA components")
    parser.add_argument("--min-variance", type=float, default=1e-6, help="Minimum variance threshold")
    parser.add_argument("--batch-size", type=int, default=100, help="PCA batch size")
    parser.add_argument("--out-dir", type=str, default="outputs/preproc", help="Output directory")
    parser.add_argument("--cache-dir", type=str, default="cache/s3_cache", help="Cache directory")
    parser.add_argument("--index-root", type=str, default="data/indices", help="Index root directory")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("ULTRA-FAST PREPROCESSING (Direct Cache Access)")
    logger.info("=" * 80)
    logger.info(f"Subject: {args.subject}")
    logger.info(f"Cache dir: {args.cache_dir}")
    logger.info(f"Output: {args.out_dir}/{args.subject}")
    logger.info("=" * 80)
    
    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}")
    
    # Load index
    from fmri2img.data.nsd_index_reader import read_subject_index
    logger.info("Loading index...")
    index = read_subject_index(args.index_root, args.subject)
    
    # Split data
    df = pd.DataFrame(index)
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_train = int(len(df) * 0.8)
    n_val = int(len(df) * 0.1)
    
    train_df = df_shuffled[:n_train]
    val_df = df_shuffled[n_train:n_train + n_val]
    test_df = df_shuffled[n_train + n_val:]
    
    logger.info(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    # Phase 1: Fit scaler
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: Fitting Scaler")
    logger.info("=" * 80)
    mean, std, mask = fit_scaler_fast(train_df, cache_dir, args.min_variance)
    
    # Phase 2: Fit PCA
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: Fitting PCA")
    logger.info("=" * 80)
    components, pca_mean = fit_pca_fast(train_df, cache_dir, mask, mean, std, 
                                        args.k, args.batch_size)
    
    # Save outputs
    out_dir = Path(args.out_dir) / args.subject
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n" + "=" * 80)
    logger.info("Saving outputs...")
    logger.info("=" * 80)
    
    np.save(out_dir / "scaler_mean.npy", mean)
    np.save(out_dir / "scaler_std.npy", std)
    np.save(out_dir / "reliability_mask.npy", mask)
    np.save(out_dir / "pca_components.npy", components)
    np.save(out_dir / "pca_mean.npy", pca_mean)
    np.save(out_dir / "voxel_indices.npy", np.where(mask.ravel())[0])
    
    # Save metadata
    meta = {
        "subject": args.subject,
        "k": args.k,
        "k_effective": components.shape[0],
        "n_voxels_total": mask.size,
        "n_voxels_kept": int(mask.sum()),
        "retention_rate": float(mask.sum() / mask.size),
        "n_train_samples": len(train_df),
        "preprocessing_version": "fast_cached"
    }
    
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    logger.info(f"✓ Saved to {out_dir}")
    logger.info("=" * 80)
    logger.info("PREPROCESSING COMPLETE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
