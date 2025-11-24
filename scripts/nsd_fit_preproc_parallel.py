#!/usr/bin/env python3
"""
Professional parallel preprocessing for NSD data.
Uses multithreading for I/O-bound NIfTI decompression.
Implements proper streaming algorithms for memory efficiency.
"""
import argparse
import json
import logging
import hashlib
from pathlib import Path
from typing import Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm
import threading

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress nibabel warnings
logging.getLogger('nibabel').setLevel(logging.ERROR)


def get_cache_path(s3_path: str, cache_dir: Path) -> Path:
    """Get local cache path for S3 file."""
    cache_hash = hashlib.sha256(s3_path.encode()).hexdigest()
    
    # Try with .nii.gz extension
    cached_file = cache_dir / f"{cache_hash}.nii.gz"
    if cached_file.exists():
        return cached_file
    
    # Try without extension
    cached_file = cache_dir / cache_hash
    if cached_file.exists():
        return cached_file
    
    raise FileNotFoundError(f"Cache not found for {s3_path}")


def load_volumes_from_file(beta_path: str, indices: np.ndarray, 
                           cache_dir: Path) -> Tuple[str, List[np.ndarray]]:
    """
    Load specific volumes from a single beta file.
    This is the I/O-bound operation we parallelize.
    
    Returns:
        (beta_path, list of volumes)
    """
    try:
        cached_file = get_cache_path(beta_path, cache_dir)
        
        # Load 4D NIfTI (decompression happens here - this is the bottleneck)
        img = nib.load(str(cached_file))
        data_4d = img.get_fdata().astype(np.float32)  # (H, W, D, N)
        
        # Extract requested volumes
        volumes = [data_4d[..., idx] for idx in indices]
        
        return (beta_path, volumes)
        
    except Exception as e:
        logger.warning(f"Failed to load {beta_path}: {e}")
        return (beta_path, [])


class WelfordAccumulator:
    """Thread-safe Welford's online algorithm for mean/variance."""
    
    def __init__(self, shape: Tuple[int, ...]):
        self.count = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.M2 = np.zeros(shape, dtype=np.float64)
        self.lock = threading.Lock()
    
    def update(self, volumes: List[np.ndarray]):
        """Update statistics with a batch of volumes."""
        with self.lock:
            for vol in volumes:
                self.count += 1
                delta = vol - self.mean
                self.mean += delta / self.count
                delta2 = vol - self.mean
                self.M2 += delta * delta2
    
    def get_statistics(self, min_variance: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get final mean, std, and mask."""
        variance = self.M2 / (self.count - 1) if self.count > 1 else self.M2
        std = np.sqrt(variance)
        std = np.maximum(std, min_variance)
        mask = variance >= min_variance
        return self.mean.astype(np.float32), std.astype(np.float32), mask


def fit_scaler_parallel(train_df: pd.DataFrame, cache_dir: Path, 
                       min_variance: float = 1e-6, n_workers: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit scaler using Welford's algorithm with parallel file loading.
    
    Args:
        train_df: Training dataframe
        cache_dir: Cache directory
        min_variance: Minimum variance threshold
        n_workers: Number of parallel workers for file loading
    
    Returns:
        mean, std, mask
    """
    logger.info(f"Fitting scaler on {len(train_df)} volumes with {n_workers} workers")
    
    # Group by beta file
    beta_col = "beta_path" if "beta_path" in train_df.columns else "beta_file"
    index_col = "beta_index" if "beta_index" in train_df.columns else "volume_index"
    
    grouped = train_df.groupby(beta_col)
    logger.info(f"Processing {len(grouped)} unique beta files in parallel")
    
    # Prepare tasks: (beta_path, indices)
    tasks = [(beta_path, group_df[index_col].values.astype(int)) 
             for beta_path, group_df in grouped]
    
    # Initialize accumulator (will get shape from first volume)
    accumulator = None
    total_volumes = 0
    
    # Parallel loading with progress bar
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(load_volumes_from_file, beta_path, indices, cache_dir): beta_path 
                  for beta_path, indices in tasks}
        
        pbar = tqdm(total=len(tasks), desc="Loading beta files (parallel)", unit="file")
        
        for future in as_completed(futures):
            beta_path, volumes = future.result()
            
            if volumes:
                # Initialize accumulator on first batch
                if accumulator is None:
                    accumulator = WelfordAccumulator(volumes[0].shape)
                
                # Update statistics
                accumulator.update(volumes)
                total_volumes += len(volumes)
            
            pbar.update(1)
            pbar.set_postfix({"volumes": total_volumes})
        
        pbar.close()
    
    if accumulator is None or accumulator.count == 0:
        raise ValueError("No valid volumes loaded")
    
    logger.info(f"✓ Processed {accumulator.count} volumes")
    
    # Get final statistics
    mean, std, mask = accumulator.get_statistics(min_variance)
    
    logger.info(f"✓ Voxel retention: {mask.sum():,} / {mask.size:,} ({mask.sum()/mask.size:.1%})")
    
    return mean, std, mask


def fit_pca_parallel(train_df: pd.DataFrame, cache_dir: Path,
                    mask: np.ndarray, mean: np.ndarray, std: np.ndarray,
                    k: int, batch_size: int = 100, n_workers: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit PCA using IncrementalPCA with parallel file loading.
    
    Args:
        train_df: Training dataframe
        cache_dir: Cache directory
        mask: Voxel mask
        mean: Voxel mean
        std: Voxel std
        k: Number of PCA components
        batch_size: Batch size for IncrementalPCA
        n_workers: Number of parallel workers
    
    Returns:
        components, pca_mean
    """
    n_features = mask.sum()
    n_samples = len(train_df)
    k_eff = min(k, n_features, n_samples)
    
    logger.info(f"Fitting PCA: k={k_eff}, n_features={n_features}, {n_workers} workers")
    
    pca = IncrementalPCA(n_components=k_eff, batch_size=batch_size)
    
    # Group by beta file
    beta_col = "beta_path" if "beta_path" in train_df.columns else "beta_file"
    index_col = "beta_index" if "beta_index" in train_df.columns else "volume_index"
    
    grouped = train_df.groupby(beta_col)
    
    # Prepare tasks
    tasks = [(beta_path, group_df[index_col].values.astype(int)) 
             for beta_path, group_df in grouped]
    
    voxel_indices = np.where(mask.ravel())[0]
    batch = []
    n_processed = 0
    
    # Parallel loading
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(load_volumes_from_file, beta_path, indices, cache_dir): beta_path 
                  for beta_path, indices in tasks}
        
        pbar = tqdm(total=len(tasks), desc="Fitting PCA (parallel)", unit="file")
        
        for future in as_completed(futures):
            beta_path, volumes = future.result()
            
            if volumes:
                # Process each volume
                for vol in volumes:
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
            
            pbar.update(1)
            pbar.set_postfix({
                "processed": n_processed,
                "var": f"{pca.explained_variance_ratio_.sum():.1%}" if hasattr(pca, 'explained_variance_ratio_') else "N/A"
            })
        
        pbar.close()
    
    # Fit remaining
    if batch:
        X_batch = np.array(batch)
        pca.partial_fit(X_batch)
        n_processed += len(batch)
    
    logger.info(f"✓ Fitted PCA on {n_processed} volumes")
    logger.info(f"✓ Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    return pca.components_, pca.mean_


def main():
    parser = argparse.ArgumentParser(description="Professional parallel preprocessing for NSD")
    parser.add_argument("--subject", type=str, required=True, help="Subject ID (e.g., subj01)")
    parser.add_argument("--k", type=int, default=512, help="Number of PCA components")
    parser.add_argument("--min-variance", type=float, default=1e-6, help="Minimum variance threshold")
    parser.add_argument("--batch-size", type=int, default=100, help="PCA batch size")
    parser.add_argument("--n-workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--out-dir", type=str, default="outputs/preproc", help="Output directory")
    parser.add_argument("--cache-dir", type=str, default="cache/s3_cache", help="Cache directory")
    parser.add_argument("--index-root", type=str, default="data/indices/nsd_index", help="Index root")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("PROFESSIONAL PARALLEL PREPROCESSING")
    logger.info("=" * 80)
    logger.info(f"Subject: {args.subject}")
    logger.info(f"PCA components: {args.k}")
    logger.info(f"Parallel workers: {args.n_workers}")
    logger.info(f"Batch size: {args.batch_size}")
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
    logger.info("PHASE 1: Fitting Scaler (Parallel)")
    logger.info("=" * 80)
    mean, std, mask = fit_scaler_parallel(train_df, cache_dir, args.min_variance, args.n_workers)
    
    # Phase 2: Fit PCA
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: Fitting PCA (Parallel)")
    logger.info("=" * 80)
    components, pca_mean = fit_pca_parallel(train_df, cache_dir, mask, mean, std, 
                                            args.k, args.batch_size, args.n_workers)
    
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
        "n_voxels_total": int(mask.size),
        "n_voxels_kept": int(mask.sum()),
        "retention_rate": float(mask.sum() / mask.size),
        "n_train_samples": len(train_df),
        "n_workers": args.n_workers,
        "preprocessing_version": "parallel_scientific"
    }
    
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    logger.info(f"✓ Saved to {out_dir}")
    logger.info("=" * 80)
    logger.info("PREPROCESSING COMPLETE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
