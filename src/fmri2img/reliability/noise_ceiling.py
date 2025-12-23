"""
Noise Ceiling and Reliability Analysis for fMRI Reconstruction
=============================================================

Implements reliability-aware evaluation using noise ceiling normalization.
Essential for neuroscience-correct reporting of reconstruction performance.

Scientific Context:
- Noise ceiling: Maximum possible performance given measurement noise
- NSD provides NCSNR (noise ceiling signal-to-noise ratio) per voxel
- Ceiling-normalized scores enable fair comparison across ROIs and subjects
- Repeat reliability quantifies decoder consistency across fMRI repetitions

Key Concepts:
1. **Noise Ceiling**: Upper bound on performance due to measurement noise
   - fMRI has limited SNR due to physiological noise, scanner noise
   - Can't expect perfect reconstruction beyond this limit
   
2. **NCSNR**: Noise ceiling SNR from NSD (split-half reliability)
   - Computed from 3 repetitions of Shared 1000
   - Higher NCSNR = more reliable voxel responses
   
3. **Ceiling Normalization**: raw_score / ceiling_score
   - Puts performance in context of achievable performance
   - Enables fair comparison across different ROIs

References:
- Allen et al. (2022). "A massive 7T fMRI dataset to bridge cognitive neuroscience and AI"
- Schoppe et al. (2016). "Measuring the performance of neural models"
- Naselaris et al. (2011). "Encoding and decoding in fMRI"
"""

import logging
from pathlib import Path
from typing import Union, Optional, Tuple, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_ncsnr(
    subject: str,
    roi: Optional[str] = None,
    data_root: Union[str, Path] = "data",
    variant: str = "nsdgeneral"
) -> np.ndarray:
    """
    Load NSD noise ceiling SNR (NCSNR) for a subject.
    
    NSD provides pre-computed noise ceiling estimates based on split-half
    reliability across the 3 repetitions of Shared 1000 stimuli.
    
    Args:
        subject: Subject ID (e.g., "subj01")
        roi: ROI name (e.g., "nsdgeneral", "V1", "V2")
             If None, loads full brain NCSNR
        data_root: Root directory for NSD data
        variant: NCSNR variant ("nsdgeneral" or "wholebrain")
    
    Returns:
        ncsnr: Array of noise ceiling SNR values, shape (n_voxels,)
               Values typically in range [0, 100+]
               Higher = more reliable voxel
    
    Example:
        >>> ncsnr = load_ncsnr("subj01", roi="nsdgeneral")
        >>> print(f"Mean NCSNR: {ncsnr.mean():.2f}")
        >>> print(f"Voxels with NCSNR > 50: {(ncsnr > 50).sum()}")
        
    Notes:
    - NCSNR is computed from split-half reliability of Shared 1000
    - Files should be at: data/nsd/ppdata/subj01/func/ncsnr_{variant}.npy
    - If files missing, returns None and evaluation should skip ceiling normalization
    """
    data_root = Path(data_root)
    
    # Try multiple possible file locations
    possible_paths = [
        data_root / "nsd" / "ppdata" / subject / "func" / f"ncsnr_{variant}.npy",
        data_root / subject / f"ncsnr_{variant}.npy",
        data_root / f"{subject}_ncsnr_{variant}.npy",
    ]
    
    if roi:
        # Also try ROI-specific files
        possible_paths.extend([
            data_root / "nsd" / "ppdata" / subject / "func" / f"ncsnr_{roi}.npy",
            data_root / subject / f"ncsnr_{roi}.npy",
        ])
    
    for path in possible_paths:
        if path.exists():
            logger.info(f"Loading NCSNR from {path}")
            ncsnr = np.load(path)
            logger.info(f"Loaded NCSNR: {ncsnr.shape[0]} voxels, mean={ncsnr.mean():.2f}")
            return ncsnr
    
    logger.warning(
        f"NCSNR not found for {subject} (roi={roi}, variant={variant}). "
        f"Tried: {[str(p) for p in possible_paths]}"
    )
    return None


def compute_voxel_noise_ceiling_from_ncsnr(
    ncsnr: np.ndarray,
    method: str = "standard",
    clip_range: Tuple[float, float] = (0.0, 1.0)
) -> np.ndarray:
    """
    Convert NCSNR to noise ceiling values in [0, 1].
    
    The noise ceiling represents the maximum achievable correlation for each
    voxel given measurement noise. Multiple conversion formulas available.
    
    Args:
        ncsnr: Noise ceiling SNR values, shape (n_voxels,)
        method: Conversion method:
            - "standard": ceiling = sqrt(ncsnr / (1 + ncsnr))
            - "correlation": ceiling = ncsnr / sqrt(1 + ncsnr^2)
            - "linear": ceiling = min(ncsnr / 100, 1.0)
        clip_range: Clip output to this range (default: [0, 1])
    
    Returns:
        ceiling: Noise ceiling per voxel, shape (n_voxels,)
                Values in [0, 1] after clipping
    
    Example:
        >>> ncsnr = np.array([0.5, 1.0, 2.0, 10.0, 100.0])
        >>> ceiling = compute_voxel_noise_ceiling_from_ncsnr(ncsnr)
        >>> print(ceiling)
        [0.50 0.71 0.89 0.95 0.995]
        
    Scientific Context:
    - Standard formula from Schoppe et al. (2016)
    - Ceiling = 1 means perfect reliability (no noise)
    - Ceiling = 0 means no signal (pure noise)
    - Typical visual cortex voxels: ceiling ~ 0.3-0.8
    """
    if ncsnr is None:
        return None
    
    if method == "standard":
        # Standard formula: r_ceiling = sqrt(SNR / (1 + SNR))
        ceiling = np.sqrt(ncsnr / (1.0 + ncsnr))
    
    elif method == "correlation":
        # Alternative: r_ceiling = SNR / sqrt(1 + SNR^2)
        ceiling = ncsnr / np.sqrt(1.0 + ncsnr**2)
    
    elif method == "linear":
        # Simple linear scaling (less theoretically motivated)
        ceiling = ncsnr / 100.0
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Clip to valid range
    ceiling = np.clip(ceiling, clip_range[0], clip_range[1])
    
    return ceiling


def aggregate_roi_ceiling(
    ceiling_map: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    aggregation: str = "mean"
) -> float:
    """
    Aggregate noise ceiling across an ROI.
    
    Args:
        ceiling_map: Noise ceiling per voxel, shape (n_voxels,)
        roi_mask: Boolean mask for ROI, shape (n_voxels,)
                 If None, uses all voxels
        aggregation: How to aggregate:
            - "mean": Average ceiling across ROI
            - "median": Median ceiling
            - "rms": Root mean square
    
    Returns:
        Scalar ceiling value for the ROI
    
    Example:
        >>> ceiling = compute_voxel_noise_ceiling_from_ncsnr(ncsnr)
        >>> roi_ceiling = aggregate_roi_ceiling(ceiling, roi_mask)
        >>> print(f"ROI ceiling: {roi_ceiling:.3f}")
        ROI ceiling: 0.653
        
    Scientific Context:
    - Mean ceiling: Average best-case performance across ROI
    - Used to normalize reconstruction performance
    - Higher ROI ceiling = easier reconstruction task
    """
    if ceiling_map is None:
        return None
    
    if roi_mask is not None:
        ceiling_map = ceiling_map[roi_mask]
    
    if aggregation == "mean":
        return float(np.mean(ceiling_map))
    elif aggregation == "median":
        return float(np.median(ceiling_map))
    elif aggregation == "rms":
        return float(np.sqrt(np.mean(ceiling_map**2)))
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


def compute_ceiling_normalized_score(
    raw_score: float,
    ceiling: float,
    eps: float = 1e-8
) -> float:
    """
    Normalize a score by the noise ceiling.
    
    Ceiling-normalized score indicates what fraction of the maximally
    achievable performance was obtained.
    
    Args:
        raw_score: Raw performance metric (e.g., correlation, R²)
        ceiling: Noise ceiling (maximum achievable performance)
        eps: Small constant to avoid division by zero
    
    Returns:
        Normalized score = raw_score / ceiling
        Values > 1 indicate performance exceeding noise ceiling
        (possible due to estimation error or different test set)
    
    Example:
        >>> raw_corr = 0.45
        >>> ceiling = 0.65
        >>> norm_score = compute_ceiling_normalized_score(raw_corr, ceiling)
        >>> print(f"Raw: {raw_corr:.3f}, Ceiling: {ceiling:.3f}, Normalized: {norm_score:.3f}")
        Raw: 0.450, Ceiling: 0.650, Normalized: 0.692
        >>> print(f"Achieved {norm_score:.1%} of noise ceiling")
        Achieved 69.2% of noise ceiling
        
    Scientific Context:
    - Normalized score = 1.0 means hitting noise ceiling (best possible)
    - Normalized score = 0.5 means halfway to ceiling
    - Enables fair comparison across ROIs with different SNR
    """
    if ceiling is None or raw_score is None:
        return None
    
    if ceiling < eps:
        logger.warning(f"Ceiling very low ({ceiling:.6f}), returning None")
        return None
    
    return raw_score / ceiling


def compute_repeat_consistency(
    predictions_per_rep: List[np.ndarray],
    metric: str = "cosine"
) -> dict:
    """
    Compute consistency of predictions across fMRI repetitions.
    
    When evaluating NSD Shared 1000 with rep-mode="all", each stimulus has
    3 independent fMRI repetitions. This function quantifies how consistent
    the decoder predictions are across these repetitions.
    
    High repeat consistency indicates:
    - Decoder is robust to fMRI noise
    - Predictions are reliable within same stimulus
    - Effective SNR of decoded representations
    
    Args:
        predictions_per_rep: List of prediction arrays, one per repetition
                            Each array has shape (n_samples, n_features)
                            E.g., [pred_rep0, pred_rep1, pred_rep2]
                            where each is (1000, 512) CLIP embeddings
        metric: Consistency metric:
            - "cosine": Mean pairwise cosine similarity
            - "correlation": Mean pairwise correlation
            - "l2": Mean pairwise L2 distance (inverted)
    
    Returns:
        Dictionary with:
        - mean: Mean consistency across all pairs
        - std: Std of pairwise consistencies
        - pairwise: All pairwise consistency values
        - n_reps: Number of repetitions
    
    Example:
        >>> # Predictions from 3 reps (1000 samples x 512 dims each)
        >>> pred_rep0 = model.predict(fmri_rep0)  # (1000, 512)
        >>> pred_rep1 = model.predict(fmri_rep1)
        >>> pred_rep2 = model.predict(fmri_rep2)
        >>> 
        >>> consistency = compute_repeat_consistency(
        ...     [pred_rep0, pred_rep1, pred_rep2],
        ...     metric="cosine"
        ... )
        >>> print(f"Repeat consistency: {consistency['mean']:.3f} ± {consistency['std']:.3f}")
        Repeat consistency: 0.823 ± 0.045
        
    Scientific Context:
    - Novel metric not typically reported in fMRI decoding papers
    - Quantifies decoder reliability independent of ground truth
    - High consistency + low accuracy = decoder is consistent but inaccurate
    - Low consistency = decoder predictions are noisy
    """
    if len(predictions_per_rep) < 2:
        logger.warning("Need at least 2 repetitions for consistency")
        return None
    
    n_reps = len(predictions_per_rep)
    
    # Compute all pairwise consistencies
    pairwise_scores = []
    
    for i in range(n_reps):
        for j in range(i + 1, n_reps):
            pred_i = predictions_per_rep[i]
            pred_j = predictions_per_rep[j]
            
            if metric == "cosine":
                # Per-sample cosine similarity
                # Normalize each prediction
                pred_i_norm = pred_i / (np.linalg.norm(pred_i, axis=1, keepdims=True) + 1e-8)
                pred_j_norm = pred_j / (np.linalg.norm(pred_j, axis=1, keepdims=True) + 1e-8)
                
                # Compute per-sample cosine
                per_sample = np.sum(pred_i_norm * pred_j_norm, axis=1)
                pairwise_scores.extend(per_sample)
            
            elif metric == "correlation":
                # Per-sample Pearson correlation
                per_sample = []
                for k in range(len(pred_i)):
                    corr = np.corrcoef(pred_i[k], pred_j[k])[0, 1]
                    per_sample.append(corr)
                pairwise_scores.extend(per_sample)
            
            elif metric == "l2":
                # Per-sample L2 distance (inverted to be consistency)
                per_sample = -np.linalg.norm(pred_i - pred_j, axis=1)
                pairwise_scores.extend(per_sample)
            
            else:
                raise ValueError(f"Unknown metric: {metric}")
    
    pairwise_scores = np.array(pairwise_scores)
    
    return {
        "mean": float(np.mean(pairwise_scores)),
        "std": float(np.std(pairwise_scores)),
        "median": float(np.median(pairwise_scores)),
        "min": float(np.min(pairwise_scores)),
        "max": float(np.max(pairwise_scores)),
        "n_reps": n_reps,
        "n_pairs": n_reps * (n_reps - 1) // 2,
        "n_samples": len(predictions_per_rep[0])
    }
