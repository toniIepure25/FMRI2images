"""
NSD Preprocessing Pipeline
==========================

Production-grade preprocessing stack for NSD fMRI data with three-stage transformation:

T0 (Per-Volume Z-Score): Online normalization applied to each volume independently.
    - Justification: Removes session-specific drift and intensity variations.
    - Computation: (vol - vol.mean()) / (vol.std() + 1e-8)
    - Applied: At load time, always (even without fitting)

T1 (Subject-Level Standardization + Reliability Masking): Train-fitted voxelwise scaler with 
    split-half reliability or variance-based masking.
    - Justification: Establishes subject-specific voxel statistics while excluding unreliable voxels.
    - Scaler: Voxelwise mean/std computed via Welford's online algorithm (prevents numerical instability).
    - Reliability Mask: For repeated stimuli, computes split-half Pearson correlation per voxel.
      Only voxels with r >= reliability_threshold are retained. Falls back to variance threshold
      if insufficient repeats exist.
    - Leakage Prevention: Only train_df volumes are used for fitting. Val/test are never accessed.
    - Output: Flat float32 vector of masked voxels in fixed ordering.

T2 (PCA Dimensionality Reduction): Optional IncrementalPCA for further compression.
    - Justification: Reduces feature dimensionality while preserving variance structure.
    - Auto-capping: k_eff = min(k, n_train_samples, n_features_kept)
    - Warning: Logs when requested k exceeds constraints.
    - Output: (k_eff,) float32 vector of PCA coefficients.

All fitted parameters are persisted to outputs/preproc/{subject}/ for consistent transforms
across training, validation, and test phases.

Artifacts Layout:
    outputs/preproc/{subject}/
        scaler_mean.npy       - Voxelwise mean (H, W, D)
        scaler_std.npy        - Voxelwise std (H, W, D)
        reliability_mask.npy  - Boolean mask (H, W, D)
        voxel_indices.npy     - Masked voxel flat indices (optional)
        pca_components.npy    - PCA components matrix (k_eff, n_features)
        pca_mean.npy          - PCA mean vector (n_features,)
        meta.json             - Metadata (k_eff, explained_variance_ratio, etc.)
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Literal, Union
import logging

logger = logging.getLogger(__name__)

try:
    import joblib
    from sklearn.decomposition import IncrementalPCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn and/or joblib not available. PCA functionality will be disabled.")

try:
    from .roi import ROIPooler
    ROI_AVAILABLE = True
except ImportError:
    ROI_AVAILABLE = False
    logger.warning("ROI functionality not available.")


class _NumpyPCA:
    """
    Lightweight numpy-based PCA wrapper for transform-only operations.
    
    This avoids dependency on sklearn IncrementalPCA internals at inference time.
    Only implements transform() method using saved components and mean.
    """
    
    def __init__(self, components: np.ndarray, mean: np.ndarray):
        """
        Initialize PCA transformer from saved artifacts.
        
        Args:
            components: PCA components matrix (k, n_features)
            mean: PCA mean vector (n_features,)
        """
        self.components_ = components
        self.mean_ = mean
        self.n_components_ = components.shape[0]
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using saved PCA components.
        
        Args:
            X: Input data, shape (n_features,) or (n_samples, n_features)
            
        Returns:
            Transformed data, shape (k,) or (n_samples, k)
        """
        # Handle both 1D and 2D input
        if X.ndim == 1:
            # Single sample: (n_features,) -> (k,)
            return (X - self.mean_) @ self.components_.T
        else:
            # Batch: (n_samples, n_features) -> (n_samples, k)
            return (X - self.mean_) @ self.components_.T


class NSDPreprocessor:
    """
    Production-grade preprocessing for NSD fMRI data with T0/T1/T2 pipeline.
    
    Architecture:
    - T0 (Always): Per-volume z-score normalization
    - T1 (Fitted): Subject-level voxelwise scaler + split-half reliability masking
    - T2 (Optional): PCA dimensionality reduction or ROI pooling
    
    Leakage Control:
    All fitting (fit, fit_pca) operates ONLY on train_df provided by caller.
    Validation and test sets are never accessed during parameter estimation.
    
    Artifacts I/O:
    Fitted parameters are saved to outputs/preproc/{subject}/ and can be loaded
    via load_artifacts() for consistent transforms across splits.
    """
    
    def __init__(self, subject: str, out_dir: str = "outputs/preproc", 
                 roi_mode: Optional[Literal["pool"]] = None):
        """
        Initialize preprocessor for a subject.
        
        Args:
            subject: Subject ID (e.g., "subj01")
            out_dir: Root directory for artifacts
            roi_mode: ROI processing mode ("pool" for ROI pooling, None for voxelwise)
        """
        self.subject = subject
        self.out_dir = Path(out_dir) / subject
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.preproc_dir = self.out_dir  # Store for metadata
        self.roi_mode = roi_mode
        
        # Artifact paths (new naming convention)
        self.scaler_mean_path = self.out_dir / "scaler_mean.npy"
        self.scaler_std_path = self.out_dir / "scaler_std.npy"
        self.reliability_mask_path = self.out_dir / "reliability_mask.npy"
        self.reliability_weights_path = self.out_dir / "reliability_weights.npy"
        self.voxel_indices_path = self.out_dir / "voxel_indices.npy"
        self.pca_components_path = self.out_dir / "pca_components.npy"
        self.pca_mean_path = self.out_dir / "pca_mean.npy"
        self.meta_path = self.out_dir / "meta.json"
        self.roi_path = self.out_dir / "roi.pkl"
        
        # Fitted parameters
        self.mean_ = None
        self.std_ = None
        self.mask_ = None
        self.weights_ = None  # Continuous reliability weights (for soft weighting mode)
        self.voxel_indices_ = None
        self.pca_ = None  # Will be _NumpyPCA or sklearn IncrementalPCA
        self.roi_pooler_ = None
        
        # State flags
        self.is_fitted_ = False
        self.pca_fitted_ = False
        self.roi_fitted_ = False
        
        # PCA metadata (for logging, not used in transform)
        self.pca_info_ = {}
        
        # Metadata
        self.meta_ = {
            "subject": subject,
            "roi_mode": roi_mode,
            "n_train_samples": 0,
        }
    
    def set_out_dir(self, path: str) -> "NSDPreprocessor":
        """
        Set custom output directory for artifacts (useful for ablation studies).
        
        Args:
            path: New output directory path
            
        Returns:
            self (for chaining)
        """
        self.out_dir = Path(path)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.preproc_dir = self.out_dir  # Store for metadata
        
        # Update all artifact paths
        self.scaler_mean_path = self.out_dir / "scaler_mean.npy"
        self.scaler_std_path = self.out_dir / "scaler_std.npy"
        self.reliability_mask_path = self.out_dir / "reliability_mask.npy"
        self.reliability_weights_path = self.out_dir / "reliability_weights.npy"
        self.voxel_indices_path = self.out_dir / "voxel_indices.npy"
        self.pca_components_path = self.out_dir / "pca_components.npy"
        self.pca_mean_path = self.out_dir / "pca_mean.npy"
        self.meta_path = self.out_dir / "meta.json"
        self.roi_path = self.out_dir / "roi.pkl"
        
        logger.info(f"Preprocessor output directory changed to: {self.out_dir}")
        return self
        
    def fit(self, train_df: pd.DataFrame, loader_factory: Callable, 
            reliability_threshold: float = 0.1, min_variance: float = 1e-6,
            min_repeat_ids: int = 20, seed: int = 42,
            reliability_mode: str = "hard_threshold",
            reliability_curve: str = "sigmoid",
            reliability_temperature: float = 0.1):
        """
        Fit T1 scaler and reliability mask on training data ONLY.
        
        Leakage Prevention: Only processes volumes from train_df. Val/test are never accessed.
        
        Algorithm:
        1. Compute voxelwise mean/std using Welford's online algorithm for numerical stability
        2. For repeated stimuli (same nsdId), compute split-half reliability:
           - Group trials by nsdId and split each repeated ID into random halves
           - Compute per-voxel Pearson correlation between half-averaged responses
           - Apply reliability weighting based on mode (hard_threshold, soft_weight, or none)
        3. If insufficient repeats (< min_repeat_ids), fall back to variance filter (var >= min_variance)
        4. Save artifacts: scaler_mean.npy, scaler_std.npy, reliability_mask.npy (or weights), reliability_meta.json
        
        Args:
            train_df: Training DataFrame (already split by caller)
            loader_factory: Callable returning (NIfTILoader, get_volume_func)
            reliability_threshold: Minimum split-half correlation (default: 0.1)
            min_variance: Fallback variance threshold (default: 1e-6)
            min_repeat_ids: Minimum number of repeated IDs for split-half (default: 20)
            seed: Random seed for split-half reliability (default: 42)
            reliability_mode: Weighting mode - "hard_threshold" (default, binary mask),
                             "soft_weight" (continuous weights), or "none" (all voxels equal)
            reliability_curve: Curve for soft_weight mode - "sigmoid" (default) or "linear"
            reliability_temperature: Temperature for sigmoid curve (default: 0.1)
        """
        logger.info(f"Fitting T1 scaler for {self.subject} on {len(train_df)} train samples")
        logger.info(f"Reliability threshold: {reliability_threshold}, Min variance: {min_variance}")
        logger.info(f"Minimum repeat IDs: {min_repeat_ids}, Seed: {seed}")
        
        # Get loader
        nifti_loader, get_volume = loader_factory()
        
        # Welford's online algorithm accumulators
        count = 0
        mean = None
        M2 = None  # Sum of squared differences from current mean
        
        # Track volumes and nsdIds for split-half reliability
        volumes_list = []  # All volumes in order
        nsd_ids_list = []  # Corresponding nsdIds
        
        # Add progress logging
        total_samples = len(train_df)
        log_interval = max(1000, total_samples // 20)  # Log every 5%
        
        for idx, row in train_df.iterrows():
            try:
                vol = get_volume(nifti_loader, row)
                if vol is None:
                    continue
                    
                vol = vol.astype(np.float32)
                
                # Welford's online update
                count += 1
                if mean is None:
                    mean = np.zeros_like(vol)
                    M2 = np.zeros_like(vol)
                
                delta = vol - mean
                mean += delta / count
                delta2 = vol - mean
                M2 += delta * delta2
                
                # Store volume and nsdId for reliability computation
                volumes_list.append(vol.copy())
                if "nsdId" in row:
                    nsd_ids_list.append(int(row["nsdId"]))
                else:
                    nsd_ids_list.append(-1)  # Placeholder for missing nsdId
                
                # Log progress
                if count % log_interval == 0:
                    progress_pct = 100 * count / total_samples
                    logger.info(f"  Progress: {count}/{total_samples} volumes ({progress_pct:.1f}%)")
                    
            except Exception as e:
                logger.warning(f"Failed to load trial {idx}: {e}")
                continue
        
        if count == 0:
            raise ValueError("No valid volumes loaded for fitting")
        
        self.meta_["n_train_samples"] = count
        logger.info(f"Processed {count} train volumes")
        
        # Compute variance and std (Bessel's correction)
        variance = M2 / (count - 1) if count > 1 else M2
        std = np.sqrt(variance)
        
        # Prevent division by zero in transform
        std = np.maximum(std, min_variance)
        
        self.mean_ = mean
        self.std_ = std
        
        # Build reliability mask/weights using new robust module
        self.mask_, self.weights_, reliability_meta = self._build_reliability_mask(
            volumes_list, nsd_ids_list, reliability_threshold, 
            min_variance, min_repeat_ids, seed, reliability_mode,
            reliability_curve, reliability_temperature
        )
        
        # Save reliability metadata
        reliability_meta_path = self.out_dir / "reliability_meta.json"
        with open(reliability_meta_path, 'w') as f:
            json.dump(reliability_meta, f, indent=2)
        
        # Extract flat voxel indices for masked voxels (fixed ordering)
        # For soft weighting, include all voxels with weight > 0
        self.voxel_indices_ = np.where(self.mask_.ravel())[0]
        
        # Save artifacts
        np.save(self.scaler_mean_path, self.mean_)
        np.save(self.scaler_std_path, self.std_)
        np.save(self.reliability_mask_path, self.mask_)
        np.save(self.reliability_weights_path, self.weights_)
        np.save(self.voxel_indices_path, self.voxel_indices_)
        
        self.is_fitted_ = True
        
        # Update metadata
        self.meta_["n_voxels_total"] = int(self.mask_.size)
        self.meta_["n_voxels_kept"] = int(self.mask_.sum())
        self.meta_["voxel_retention_rate"] = float(self.mask_.mean())
        self.meta_["reliability_method"] = reliability_meta.get("method", "unknown")
        self.meta_["reliability_threshold"] = reliability_meta.get("reliability_threshold", reliability_threshold)
        self.meta_["split_half_seed"] = reliability_meta.get("seed", None)
        
        # Fit ROI pooler if requested
        if self.roi_mode == "pool":
            self._fit_roi_pooler(train_df, loader_factory)
        
        # Save metadata
        with open(self.meta_path, 'w') as f:
            json.dump(self.meta_, f, indent=2)
        
        logger.info(f"✅ T1 fitted: {self.mask_.sum():,} / {self.mask_.size:,} voxels "
                   f"({100 * self.mask_.mean():.1f}% retained)")
        if reliability_meta.get("method") == "split_half":
            mean_r = reliability_meta.get("mean_r_retained", 0)
            logger.info(f"   Mean reliability (retained voxels): r={mean_r:.3f}")

        
    def _build_reliability_mask(
        self, 
        volumes: list, 
        nsd_ids: list, 
        threshold: float, 
        min_var: float,
        min_repeat_ids: int,
        seed: int,
        mode: str = "hard_threshold",
        curve: str = "sigmoid",
        temperature: float = 0.1
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Build mask and weights based on split-half reliability for repeated stimuli.
        
        Uses robust implementation from reliability.py module that handles:
        - Repeat-aware trial grouping by nsdId
        - Balanced random splits per repeated ID
        - Per-voxel Pearson correlation computation
        - Combined reliability + variance thresholding OR soft weighting
        
        Falls back to variance threshold if insufficient repeats exist.
        
        Novel contribution: Supports soft reliability weighting (continuous weights)
        instead of binary hard thresholding. This allows the model to learn optimal
        voxel weighting rather than discarding voxels completely.
        
        Args:
            volumes: List of volumes (each H×W×D)
            nsd_ids: List of corresponding nsdIds
            threshold: Minimum split-half correlation (or midpoint for soft weighting)
            min_var: Fallback variance threshold
            min_repeat_ids: Minimum number of repeated IDs required
            seed: Random seed for split-half reliability
            mode: Weighting mode - "hard_threshold", "soft_weight", or "none"
            curve: Curve for soft_weight mode - "sigmoid" or "linear"
            temperature: Temperature for sigmoid curve
            
        Returns:
            Tuple of (mask, weights, metadata_dict):
                - mask: Boolean array (H, W, D) indicating voxels with weight > 0
                - weights: Float array (H, W, D) with continuous reliability weights [0, 1]
                - metadata: Dict with method, n_ids, mean_r, effective_voxels, etc.
        """
        from .reliability import (
            compute_split_half_reliability, 
            filter_voxels_by_reliability,
            compute_soft_reliability_weights
        )
        
        if not volumes or not nsd_ids:
            logger.info("No data for reliability computation, using variance threshold")
            mask, stats = self._build_mask_from_variance_with_stats(min_var)
            weights = mask.astype(np.float32)  # Binary weights for variance fallback
            return mask, weights, {
                "method": "variance_fallback",
                "reason": "no_data",
                "min_variance": min_var,
                "n_voxels_retained": int(mask.sum()),
                "retention_rate": float(mask.mean()),
                "reliability_mode": mode
            }
        
        # Stack volumes into (n_trials, H, W, D) then reshape to (n_trials, n_voxels)
        X_4d = np.stack(volumes, axis=0)  # (n_trials, H, W, D)
        original_shape = X_4d.shape[1:]  # (H, W, D)
        X = X_4d.reshape(X_4d.shape[0], -1)  # (n_trials, n_voxels)
        nsd_ids_arr = np.array(nsd_ids, dtype=np.int32)
        
        # Check for sufficient repeated IDs
        from collections import Counter
        id_counts = Counter(nsd_ids_arr[nsd_ids_arr >= 0])  # Exclude placeholders
        repeated_ids = [nsd_id for nsd_id, count in id_counts.items() if count >= 2]
        
        if len(repeated_ids) == 0:
            logger.warning("No repeated stimuli found - using variance-based masking only")
            mask, stats = self._build_mask_from_variance_with_stats(min_var)
            weights = mask.astype(np.float32)  # Binary weights for variance fallback
            return mask, weights, {
                "method": "none",
                "note": "no repeated IDs; variance fallback",
                "n_repeat_stimuli": 0,
                "mean_trials_per_id": 0.0,
                "min_variance": min_var,
                "n_voxels_retained": int(mask.sum()),
                "retention_rate": float(mask.mean()),
                "n_ids_with_repeats": 0,
                "ids_used": [],
                "median_trials_per_id": 0.0,
                "reliability_mode": mode
            }
        
        if len(repeated_ids) < min_repeat_ids:
            logger.info(f"Only {len(repeated_ids)} repeated IDs found (need >= {min_repeat_ids}), "
                       f"falling back to variance threshold")
            mask, stats = self._build_mask_from_variance_with_stats(min_var)
            weights = mask.astype(np.float32)  # Binary weights for variance fallback
            return mask, weights, {
                "method": "variance_fallback",
                "reason": "insufficient_repeats",
                "n_repeated_ids_found": len(repeated_ids),
                "min_repeat_ids_required": min_repeat_ids,
                "min_variance": min_var,
                "n_voxels_retained": int(mask.sum()),
                "retention_rate": float(mask.mean()),
                "n_repeat_stimuli": len(repeated_ids),
                "mean_trials_per_id": float(np.mean([c for c in id_counts.values()])) if id_counts else 0.0,
                "n_ids_with_repeats": len(repeated_ids),
                "ids_used": repeated_ids[:10] if repeated_ids else [],
                "median_trials_per_id": float(np.median([c for c in id_counts.values()])) if id_counts else 0.0,
                "reliability_mode": mode
            }
        
        logger.info(f"Computing split-half reliability from {len(repeated_ids)} stimuli with repeats")
        
        # Compute split-half reliability
        r, r_meta = compute_split_half_reliability(
            X=X,
            nsd_ids=nsd_ids_arr,
            seed=seed,
            min_repeats=2
        )
        
        # Reshape r back to volume shape
        r_3d = r.reshape(original_shape)  # (H, W, D)
        
        # Compute voxel variance for combined thresholding/weighting
        voxel_var = self.std_.flatten() ** 2
        
        # Compute weights using new soft weighting function
        weights_flat, weight_stats = compute_soft_reliability_weights(
            r=r,
            voxel_variance=voxel_var,
            mode=mode,
            reliability_thr=threshold,
            min_var=min_var,
            curve=curve,
            temperature=temperature
        )
        
        # Create binary mask from weights (weight > 0)
        mask_flat = (weights_flat > 0).astype(bool)
        
        # Reshape mask and weights back to volume shape
        mask = mask_flat.reshape(original_shape)
        weights = weights_flat.reshape(original_shape)
        
        # Build comprehensive metadata
        metadata = {
            "method": "split_half",
            "reliability_mode": mode,
            "reliability_curve": curve if mode == "soft_weight" else None,
            "reliability_temperature": temperature if mode == "soft_weight" and curve == "sigmoid" else None,
            "seed": seed,
            "reliability_threshold": threshold,
            "min_variance": min_var,
            "n_trials": int(X.shape[0]),
            "n_repeated_ids": len(repeated_ids),
            "n_ids_with_repeats": r_meta["n_ids_with_repeats"],
            "ids_used": r_meta["ids_used"],
            "mean_trials_per_id": float(r_meta["mean_trials_per_id"]),
            "median_trials_per_id": float(r_meta["median_trials_per_id"]),
            "n_voxels_total": int(X.shape[1]),
            "n_voxels_retained": weight_stats["n_nonzero_weights"],
            "n_voxels_rejected": int(X.shape[1]) - weight_stats["n_nonzero_weights"],
            "retention_rate": weight_stats["retention_rate"],
            "effective_voxels": weight_stats["effective_voxels"],
            "mean_weight": weight_stats["mean_weight"],
            "median_weight": weight_stats["median_weight"],
            "weight_percentiles": weight_stats["weight_percentiles"],
        }
        
        if mode == "hard_threshold":
            logger.info(f"Split-half reliability (hard threshold): {mask.sum():,} voxels retained")
        elif mode == "soft_weight":
            logger.info(f"Split-half reliability (soft {curve}): "
                       f"{mask.sum():,} voxels with weight > 0, "
                       f"effective voxels: {weight_stats['effective_voxels']:.1f}")
        else:  # mode == "none"
            logger.info(f"Split-half reliability (no weighting): all {mask.sum():,} voxels retained")
        
        return mask, weights, metadata
        
    def _build_mask_from_variance_with_stats(self, min_var: float) -> tuple[np.ndarray, dict]:
        """Build mask from variance threshold with statistics."""
        if self.std_ is None:
            raise ValueError("Must fit scaler before building variance mask")
            
        variance = self.std_ ** 2
        mask = variance >= min_var
        
        logger.info(f"Variance masking: {mask.sum():,} voxels with var >= {min_var}")
        
        stats = {
            "n_retained": int(mask.sum()),
            "n_rejected": int((~mask).sum()),
            "retention_rate": float(mask.mean())
        }
        
        return mask, stats
        
    def _fit_roi_pooler(self, train_df: pd.DataFrame, loader_factory: Callable):
        """
        Fit ROI pooler using first beta volume for shape inference.
        
        Gracefully handles missing ROI masks by falling back to whole-brain.
        """
        if not ROI_AVAILABLE:
            raise ImportError("ROI functionality not available")
            
        logger.info(f"Fitting ROI pooler for {self.subject}")
        
        # Get first valid beta path for shape inference
        nifti_loader, get_volume = loader_factory()
        sample_path = None
        
        for idx, row in train_df.head(10).iterrows():  # Try first few rows
            try:
                vol = get_volume(nifti_loader, row)
                if vol is not None:
                    # Extract the path from the row or reconstruct it
                    if hasattr(row, 'beta_path'):
                        sample_path = row.beta_path
                    else:
                        # Reconstruct path from row data
                        from fmri2img.io.nsd_layout import NSDLayout
                        layout = NSDLayout("configs/data.yaml")
                        sample_path = layout.session_beta_path(
                            subject=self.subject, 
                            session=row.get('session', 1), 
                            full_url=True
                        )
                    break
            except Exception as e:
                continue
                
        if sample_path is None:
            raise ValueError("Could not find valid beta file for ROI fitting")
            
        # Initialize and fit ROI pooler (handles missing masks gracefully)
        self.roi_pooler_ = ROIPooler(self.subject)
        self.roi_pooler_.fit(sample_path)
        
        # Check if ROI masks were found
        if len(self.roi_pooler_.rois) == 0:
            logger.warning(
                f"No ROI masks found for {self.subject} → ROI pooling disabled. "
                "Preprocessing will use full masked volume instead."
            )
        else:
            # Save ROI pooler
            if SKLEARN_AVAILABLE:
                joblib.dump(self.roi_pooler_, self.roi_path)
        
        self.roi_fitted_ = True
        logger.info(f"ROI pooler fitted with {len(self.roi_pooler_.rois)} ROIs")
    
    def fit_pca(self, train_df: pd.DataFrame, loader_factory: Callable, k: int = 4096, 
                batch_size: int = 512):
        """
        Fit T2 PCA on masked training data ONLY.
        
        Leakage Prevention: Only processes train_df volumes. Val/test never accessed.
        
        Auto-Capping: Components are capped to k_eff = min(k, n_train, n_features_kept)
        to satisfy sklearn constraints. Logs warning when k is reduced.
        
        Args:
            train_df: Training DataFrame (already split by caller)
            loader_factory: Callable returning (NIfTILoader, get_volume_func)
            k: Desired number of PCA components (will be auto-capped)
            batch_size: Batch size for IncrementalPCA fitting (default: 512)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for PCA. Install with: pip install scikit-learn")
            
        if not self.is_fitted_:
            raise ValueError("Must call fit() before fit_pca()")
        
        # Auto-cap PCA components
        n_train = len(train_df)
        n_features = int(self.mask_.sum())  # Number of kept voxels
        # Rank constraint: components <= min(n_samples - 1, n_features)
        k_eff = int(min(k, n_features, max(1, n_train - 1)))
        
        if k_eff < k:
            logger.warning(
                f"PCA auto-capping: requested k={k} but using k_eff={k_eff} "
                f"(limited by n_train={n_train}, n_features={n_features}, rank constraint)"
            )
        
        logger.info(f"Fitting T2 PCA: k_eff={k_eff} components, n_train={n_train}, n_features={n_features}")
        
        # Initialize IncrementalPCA
        # Batch size must be >= n_components for sklearn
        batch_size_eff = max(batch_size, k_eff + 1)
        self.pca_ = IncrementalPCA(n_components=k_eff, batch_size=batch_size_eff)
        
        # Get loader
        nifti_loader, get_volume = loader_factory()
        
        # Collect batches for incremental fitting
        batch_data = []
        n_processed = 0
        
        for idx, row in train_df.iterrows():
            try:
                vol = get_volume(nifti_loader, row)
                if vol is None:
                    continue
                    
                # Apply T0 + T1 (but not T2)
                vol_t1 = self.transform_T0(vol)
                vol_t1 = self.transform_T1(vol_t1)
                
                batch_data.append(vol_t1)
                n_processed += 1
                
                # Fit batch when ready
                if len(batch_data) >= batch_size_eff:
                    X_batch = np.stack(batch_data)
                    self.pca_.partial_fit(X_batch)
                    logger.info(f"PCA partial_fit: {n_processed}/{n_train} samples processed")
                    batch_data = []
                    
            except Exception as e:
                logger.warning(f"Failed to process trial {idx} for PCA: {e}")
                continue
        
        # Fit remaining data
        if batch_data:
            X_batch = np.stack(batch_data)
            self.pca_.partial_fit(X_batch)
            logger.info(f"PCA partial_fit: {n_processed}/{n_train} samples processed (final batch)")
        
        if n_processed == 0:
            raise ValueError("No valid samples for PCA fitting")
        
        # Save PCA artifacts
        np.save(self.pca_components_path, self.pca_.components_)
        np.save(self.pca_mean_path, self.pca_.mean_)
        
        self.pca_fitted_ = True
        
        # Update metadata
        explained_var = float(self.pca_.explained_variance_ratio_.sum())
        self.meta_["pca_fitted"] = True
        self.meta_["pca_components"] = int(k_eff)
        self.meta_["explained_variance_ratio"] = explained_var
        
        # Save updated metadata
        with open(self.meta_path, 'w') as f:
            json.dump(self.meta_, f, indent=2)
        
        logger.info(f"✅ T2 PCA fitted:")
        logger.info(f"   Components (k_eff): {k_eff}")
        logger.info(f"   Mean-centering: enabled (PCA subtracts fitted mean)")
        logger.info(f"   Explained variance: {100*explained_var:.2f}%")
        logger.info(f"   Artifacts saved to: {self.out_dir}")

        
    def transform_T0(self, vol: np.ndarray) -> np.ndarray:
        """
        T0: Per-volume z-score normalization (always applied, even without fitting).
        
        Removes session-specific intensity drift and normalizes each volume independently.
        
        Args:
            vol: Input volume (H, W, D) float32
            
        Returns:
            Z-scored volume (H, W, D) float32
        """
        vol_mean = vol.mean()
        vol_std = vol.std()
        return ((vol - vol_mean) / (vol_std + 1e-8)).astype(np.float32)
    
    def transform_T1(self, vol: np.ndarray) -> np.ndarray:
        """
        T1: Apply subject-level scaler, reliability weighting, and mask (requires fitting).
        
        Returns flat float32 vector of masked voxels in fixed ordering.
        
        Novel contribution: Applies sqrt(reliability_weight) to each voxel after scaling.
        This downweights less reliable voxels rather than discarding them completely,
        allowing the model to potentially learn from weak signals while emphasizing
        strong signals. The sqrt ensures proper variance scaling for subsequent PCA.
        
        Args:
            vol: Input volume after T0 (H, W, D) float32
            
        Returns:
            Flat masked vector (n_voxels_kept,) float32
            
        Raises:
            ValueError: If preprocessor not fitted
        """
        if not self.is_fitted_:
            raise ValueError("Cannot apply T1 transform: preprocessor not fitted. Call fit() first.")
            
        # Apply voxelwise standardization
        vol_scaled = (vol - self.mean_) / self.std_
        
        # Apply reliability weighting (sqrt for proper variance scaling)
        if self.weights_ is not None:
            vol_weighted = vol_scaled * np.sqrt(self.weights_)
        else:
            # Backward compatibility: if weights not available, use mask as binary weights
            vol_weighted = vol_scaled
        
        # Apply mask and flatten to vector
        masked_voxels = vol_weighted[self.mask_]
        
        return masked_voxels.astype(np.float32)
    
    def transform_T2(self, vec: np.ndarray) -> np.ndarray:
        """
        T2: Apply PCA transformation (requires PCA fitting).
        
        Args:
            vec: Input vector after T1 (n_voxels_kept,) float32
            
        Returns:
            PCA coefficients (k_eff,) float32
            
        Raises:
            ValueError: If PCA not fitted
        """
        if not self.pca_fitted_ or self.pca_ is None:
            raise ValueError("Cannot apply T2 transform: PCA not fitted. Call fit_pca() first.")
            
        # Reshape for sklearn (expects 2D)
        vec_2d = vec.reshape(1, -1)
        pca_out = self.pca_.transform(vec_2d)
        
        return pca_out.flatten().astype(np.float32)
    
    def transform(self, vol: np.ndarray) -> np.ndarray:
        """
        Transform a volume through the full preprocessing pipeline.
        
        Pipeline:
        - Always applies T0 (z-score)
        - If fitted, applies T1 (scaler + mask) → returns flat vector
        - If PCA fitted, applies T2 (PCA) → returns PCA coefficients
        - If ROI mode enabled, applies ROI pooling instead of T2
        
        Args:
            vol: Input volume (H, W, D) float32
            
        Returns:
            Preprocessed output:
            - (H, W, D) if only T0 applied (unfitted)
            - (n_voxels_kept,) if T0+T1 applied
            - (k_eff,) if T0+T1+T2 applied
            - (n_rois,) if ROI pooling applied
        """
        # T0: Always apply z-score
        vol = self.transform_T0(vol)
        
        # If not fitted, return T0 only
        if not self.is_fitted_:
            return vol
        
        # T1: Apply scaler and mask
        vec_t1 = self.transform_T1(vol)
        
        # T2: Apply ROI pooling if enabled (takes precedence over PCA)
        if self.roi_mode == "pool" and self.roi_fitted_ and self.roi_pooler_ is not None:
            roi_means = self.roi_pooler_.pool(vol)  # Use T0 volume
            if roi_means.size == 0:
                logger.warning("No ROIs available, falling back to T1 output")
                return vec_t1
            return roi_means.astype(np.float32)
        
        # T2: Apply PCA if fitted
        if self.pca_fitted_ and self.pca_ is not None:
            return self.transform_T2(vec_t1)
        
        # Otherwise return T1 output
        return vec_t1
        
    def load_artifacts(self) -> bool:
        """
        Load saved preprocessing artifacts from disk.
        
        Loads T1 scaler artifacts (mean, std, mask) and optionally T2 PCA artifacts.
        
        Returns:
            True if T1 artifacts successfully loaded, False otherwise
        """
        try:
            # Load T1 artifacts (required)
            if not (self.scaler_mean_path.exists() and self.scaler_std_path.exists() 
                    and self.reliability_mask_path.exists()):
                logger.debug(f"T1 artifacts not found in {self.out_dir}")
                return False
            
            self.mean_ = np.load(self.scaler_mean_path)
            self.std_ = np.load(self.scaler_std_path)
            self.mask_ = np.load(self.reliability_mask_path)
            
            # Load reliability weights (optional, for soft weighting mode)
            if self.reliability_weights_path.exists():
                self.weights_ = np.load(self.reliability_weights_path)
            else:
                # Backward compatibility: if weights file doesn't exist, use binary mask
                self.weights_ = self.mask_.astype(np.float32)
            
            if self.voxel_indices_path.exists():
                self.voxel_indices_ = np.load(self.voxel_indices_path)
            
            self.is_fitted_ = True
            logger.info(f"✅ Loaded T1 scaler for {self.subject}: {self.mask_.sum():,} voxels retained")
            
            # Load T2 PCA artifacts (optional) - use lightweight numpy-based wrapper
            if self.pca_components_path.exists() and self.pca_mean_path.exists():
                components = np.load(self.pca_components_path)
                mean = np.load(self.pca_mean_path)
                
                # Use lightweight _NumpyPCA wrapper (no sklearn dependency at inference)
                self.pca_ = _NumpyPCA(components, mean)
                self.pca_fitted_ = True
                
                # Load PCA info from metadata for logging only (not used in transform)
                if self.meta_path.exists():
                    with open(self.meta_path, 'r') as f:
                        meta = json.load(f)
                        self.pca_info_ = {
                            "k_eff": int(components.shape[0]),
                            "explained_variance_ratio": meta.get("explained_variance_ratio", 0.0)
                        }
                else:
                    self.pca_info_ = {"k_eff": int(components.shape[0]), "explained_variance_ratio": 0.0}
                
                logger.info(f"✅ Loaded T2 PCA: k={self.pca_info_['k_eff']} components "
                           f"(numpy-based, no sklearn dependency)")
            elif SKLEARN_AVAILABLE and self.pca_components_path.exists():
                # Fallback warning if only one file exists
                logger.warning("PCA components found but mean missing; PCA not loaded")
            
            # Load ROI pooler (optional)
            if (self.roi_mode == "pool" and SKLEARN_AVAILABLE and 
                ROI_AVAILABLE and self.roi_path.exists()):
                self.roi_pooler_ = joblib.load(self.roi_path)
                self.roi_fitted_ = True
                logger.info(f"✅ Loaded ROI pooler: {len(self.roi_pooler_.rois)} regions")
            
            # Load metadata
            if self.meta_path.exists():
                with open(self.meta_path, 'r') as f:
                    self.meta_ = json.load(f)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load artifacts from {self.out_dir}: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def get_output_shape(self, input_shape: tuple) -> tuple:
        """
        Get expected output shape after preprocessing.
        
        Args:
            input_shape: Input volume shape (H, W, D)
            
        Returns:
            Output shape tuple
        """
        if self.roi_mode == "pool" and self.roi_fitted_ and self.roi_pooler_ is not None:
            return (len(self.roi_pooler_.rois),)
        elif self.pca_fitted_ and self.pca_ is not None:
            return (self.pca_.n_components_,)
        elif self.is_fitted_:
            return (int(self.mask_.sum()),)  # Flat masked vector
        else:
            return input_shape  # T0 only, same shape
            
    def summary(self) -> Dict[str, Any]:
        """
        Return comprehensive summary of fitted preprocessor state.
        
        Returns:
            Dictionary with keys:
            - subject: Subject ID
            - n_voxels_total: Total number of voxels
            - n_voxels_kept: Number of voxels retained after masking
            - voxel_retention_rate: Fraction of voxels retained (0-1)
            - reliability_threshold: Reliability threshold used for masking
            - pca_fitted: Whether PCA is fitted
            - pca_components: Number of PCA components (if fitted)
            - explained_variance_ratio: Cumulative explained variance (if fitted)
            - roi_fitted: Whether ROI pooling is fitted
            - n_rois: Number of ROI regions (if fitted)
            - roi_names: List of ROI names (if fitted)
        """
        info = {
            "subject": self.subject,
            "n_voxels_total": int(self.mask_.size) if self.is_fitted_ else 0,
            "n_voxels_kept": int(self.mask_.sum()) if self.is_fitted_ else 0,
            "voxel_retention_rate": float(self.mask_.mean()) if self.is_fitted_ else 0.0,
            "reliability_threshold": self.meta_.get("reliability_threshold", 0.0) if self.is_fitted_ else 0.0,
            "pca_fitted": self.pca_fitted_,
            "roi_fitted": self.roi_fitted_,
        }
        
        if self.pca_fitted_ and self.pca_ is not None:
            # Use pca_info_ for logging (no sklearn access)
            if self.pca_info_:
                info["pca_components"] = self.pca_info_.get("k_eff", self.pca_.n_components_)
                info["explained_variance_ratio"] = self.pca_info_.get("explained_variance_ratio", 0.0)
            else:
                # Fallback for sklearn IncrementalPCA during fitting
                info["pca_components"] = int(self.pca_.n_components_)
                if hasattr(self.pca_, 'explained_variance_ratio_'):
                    info["explained_variance_ratio"] = float(self.pca_.explained_variance_ratio_.sum())
                elif "explained_variance_ratio" in self.meta_:
                    info["explained_variance_ratio"] = float(self.meta_["explained_variance_ratio"])
                else:
                    info["explained_variance_ratio"] = 0.0
        else:
            info["pca_components"] = 0
            info["explained_variance_ratio"] = 0.0
            
        if self.roi_fitted_ and self.roi_pooler_ is not None:
            info["n_rois"] = len(self.roi_pooler_.rois)
            info["roi_names"] = self.roi_pooler_.names() if hasattr(self.roi_pooler_, 'names') else []
        else:
            info["n_rois"] = 0
            info["roi_names"] = []
            
        return info