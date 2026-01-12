"""
Split-half reliability estimation for voxel selection.

Implements repeat-aware voxel reliability estimation using NSD stimulus repeats.
Uses split-half correlation to measure test-retest reliability per voxel.

Key features:
- Train-only estimation (no leakage)
- Balanced random splits of repeated trials
- Per-voxel Pearson correlation
- Comprehensive provenance metadata
- Fixed random seed for reproducibility

Scientific context:
    Split-half reliability is a standard psychometric measure of test-retest
    consistency. For fMRI, voxels with high split-half r show stable responses
    to repeated stimuli, indicating reliable signal over noise.
    
    Reference: Allen et al. (2022) NSD - massive fMRI with 3 repeats per stimulus
"""

import numpy as np
from typing import Tuple, Dict, List
from collections import defaultdict


def compute_split_half_reliability(
    X: np.ndarray,
    nsd_ids: np.ndarray,
    seed: int = 42,
    min_repeats: int = 2,
) -> Tuple[np.ndarray, Dict]:
    """
    Compute per-voxel split-half reliability using repeated stimulus presentations.
    
    Method:
        1. Identify nsdIds with ≥ min_repeats presentations
        2. For each repeated ID, randomly split trials into two halves
        3. Average fMRI per half → two vectors per ID
        4. Accumulate across all repeated IDs → matrices A, B
        5. Compute per-voxel Pearson r between A and B
    
    Args:
        X: fMRI data, shape (n_trials, n_voxels), dtype float32
        nsd_ids: Stimulus IDs, shape (n_trials,), dtype int
        seed: Random seed for reproducibility
        min_repeats: Minimum presentations required (default: 2)
    
    Returns:
        r: Per-voxel split-half correlation, shape (n_voxels,)
           Values in [-1, 1], with NaN set to 0
        meta: Dictionary with:
            - n_repeatable_trials: Total trials with repeated IDs
            - n_ids_with_repeats: Number of unique IDs with ≥ min_repeats
            - ids_used: List of repeated IDs used
            - per_voxel_n: Array of sample sizes per voxel (typically same)
            - seed: Random seed used
            - min_repeats: Minimum repeats threshold
    
    Example:
        >>> X = np.random.randn(100, 1000).astype(np.float32)
        >>> nsd_ids = np.repeat([1, 2, 3, 4, 5], 20)  # 20 repeats each
        >>> r, meta = compute_split_half_reliability(X, nsd_ids)
        >>> print(f"Mean reliability: {np.mean(r):.3f}")
        >>> print(f"Used {meta['n_ids_with_repeats']} repeated IDs")
    
    Notes:
        - Uses fixed RNG for reproducibility
        - Handles odd number of repeats with balanced split
        - Sets r=0 for voxels with zero variance in either half
        - Efficient for large datasets (float32, vectorized ops)
    """
    if X.shape[0] != len(nsd_ids):
        raise ValueError(f"X has {X.shape[0]} trials but nsd_ids has {len(nsd_ids)}")
    
    n_trials, n_voxels = X.shape
    
    # Ensure float32 for memory efficiency
    if X.dtype != np.float32:
        X = X.astype(np.float32)
    
    # Initialize RNG
    rng = np.random.default_rng(seed)
    
    # Group trials by nsdId
    id_to_trials = defaultdict(list)
    for trial_idx, nsd_id in enumerate(nsd_ids):
        id_to_trials[nsd_id].append(trial_idx)
    
    # Filter to IDs with enough repeats
    repeated_ids = [
        nsd_id for nsd_id, trials in id_to_trials.items()
        if len(trials) >= min_repeats
    ]
    
    if len(repeated_ids) == 0:
        # No repeated IDs - return zeros
        return (
            np.zeros(n_voxels, dtype=np.float32),
            {
                "n_repeatable_trials": 0,
                "n_ids_with_repeats": 0,
                "ids_used": [],
                "per_voxel_n": np.zeros(n_voxels, dtype=np.int32),
                "seed": seed,
                "min_repeats": min_repeats,
            }
        )
    
    # Sort for reproducibility
    repeated_ids.sort()
    
    # Accumulate split-half averages
    A_list = []  # First half averages
    B_list = []  # Second half averages
    
    for nsd_id in repeated_ids:
        trials = id_to_trials[nsd_id]
        n_trials_id = len(trials)
        
        # Shuffle trials for this ID
        shuffled_trials = rng.permutation(trials)
        
        # Split into two halves
        mid = n_trials_id // 2
        half_a = shuffled_trials[:mid]
        half_b = shuffled_trials[mid:]
        
        # Handle odd number of trials - ensure balanced split
        # If odd, both halves get at least floor(n/2) trials
        if len(half_a) == 0 or len(half_b) == 0:
            # Edge case: only 1 trial (shouldn't happen with min_repeats=2)
            continue
        
        # Average each half
        avg_a = X[half_a].mean(axis=0)  # shape: (n_voxels,)
        avg_b = X[half_b].mean(axis=0)
        
        A_list.append(avg_a)
        B_list.append(avg_b)
    
    if len(A_list) == 0:
        # No valid splits
        return (
            np.zeros(n_voxels, dtype=np.float32),
            {
                "n_repeatable_trials": sum(len(id_to_trials[i]) for i in repeated_ids),
                "n_ids_with_repeats": len(repeated_ids),
                "ids_used": repeated_ids,
                "per_voxel_n": np.zeros(n_voxels, dtype=np.int32),
                "seed": seed,
                "min_repeats": min_repeats,
            }
        )
    
    # Stack into matrices
    A = np.stack(A_list, axis=0)  # shape: (n_repeated_ids, n_voxels)
    B = np.stack(B_list, axis=0)
    
    # Compute per-voxel Pearson correlation
    # r_v = corr(A[:, v], B[:, v]) for each voxel v
    
    # Demean
    A_mean = A.mean(axis=0, keepdims=True)
    B_mean = B.mean(axis=0, keepdims=True)
    A_centered = A - A_mean
    B_centered = B - B_mean
    
    # Compute norms
    A_norm = np.sqrt((A_centered ** 2).sum(axis=0))  # shape: (n_voxels,)
    B_norm = np.sqrt((B_centered ** 2).sum(axis=0))
    
    # Compute dot products
    dot_products = (A_centered * B_centered).sum(axis=0)
    
    # Compute correlation
    # Avoid division by zero
    denom = A_norm * B_norm
    
    r = np.zeros(n_voxels, dtype=np.float32)
    valid_mask = denom > 0
    r[valid_mask] = dot_products[valid_mask] / denom[valid_mask]
    
    # Clip to [-1, 1] to handle numerical errors
    r = np.clip(r, -1.0, 1.0)
    
    # Replace any remaining NaNs with 0
    r = np.nan_to_num(r, nan=0.0)
    
    # Count repeatable trials
    n_repeatable_trials = sum(len(id_to_trials[i]) for i in repeated_ids)
    
    # Per-voxel sample size (typically same for all voxels)
    per_voxel_n = np.full(n_voxels, len(A_list), dtype=np.int32)
    
    # Trials per ID statistics
    trials_per_id = [len(id_to_trials[i]) for i in repeated_ids]
    mean_trials_per_id = np.mean(trials_per_id) if trials_per_id else 0.0
    median_trials_per_id = np.median(trials_per_id) if trials_per_id else 0.0
    
    meta = {
        "n_repeatable_trials": int(n_repeatable_trials),
        "n_ids_with_repeats": len(repeated_ids),
        "ids_used": [int(i) for i in repeated_ids],  # Convert to Python int
        "per_voxel_n": per_voxel_n,
        "mean_trials_per_id": float(mean_trials_per_id),
        "median_trials_per_id": float(median_trials_per_id),
        "seed": seed,
        "min_repeats": min_repeats,
    }
    
    return r, meta


def filter_voxels_by_reliability(
    r: np.ndarray,
    voxel_variance: np.ndarray,
    reliability_thr: float = 0.1,
    min_var: float = 1e-6,
) -> Tuple[np.ndarray, Dict]:
    """
    Create voxel mask combining reliability and variance thresholds.
    
    Args:
        r: Per-voxel split-half reliability, shape (n_voxels,)
        voxel_variance: Per-voxel variance, shape (n_voxels,)
        reliability_thr: Minimum reliability threshold (default: 0.1)
        min_var: Minimum variance threshold (default: 1e-6)
    
    Returns:
        mask: Boolean mask, shape (n_voxels,), True = keep voxel
        stats: Dictionary with:
            - n_voxels_total: Total voxels
            - n_voxels_reliable: Passing reliability threshold
            - n_voxels_variance: Passing variance threshold
            - n_voxels_retained: Passing both thresholds
            - mean_r_retained: Mean reliability of retained voxels
            - mean_r_rejected: Mean reliability of rejected voxels
            - median_r_retained: Median reliability of retained voxels
            - median_r_rejected: Median reliability of rejected voxels
    
    Example:
        >>> r = np.array([0.8, 0.5, 0.05, 0.3, -0.1])
        >>> var = np.array([1.0, 1.0, 1.0, 1e-8, 1.0])
        >>> mask, stats = filter_voxels_by_reliability(r, var, reliability_thr=0.1)
        >>> print(f"Retained {mask.sum()}/{len(mask)} voxels")
    """
    n_voxels = len(r)
    
    # Individual masks
    reliable_mask = r >= reliability_thr
    variance_mask = voxel_variance >= min_var
    
    # Combined mask
    mask = reliable_mask & variance_mask
    
    # Statistics
    n_reliable = reliable_mask.sum()
    n_variance = variance_mask.sum()
    n_retained = mask.sum()
    n_rejected = n_voxels - n_retained
    
    # Mean/median reliability
    if n_retained > 0:
        mean_r_retained = float(r[mask].mean())
        median_r_retained = float(np.median(r[mask]))
    else:
        mean_r_retained = np.nan
        median_r_retained = np.nan
    
    if n_rejected > 0:
        mean_r_rejected = float(r[~mask].mean())
        median_r_rejected = float(np.median(r[~mask]))
    else:
        mean_r_rejected = np.nan
        median_r_rejected = np.nan
    
    stats = {
        "n_voxels_total": int(n_voxels),
        "n_voxels_reliable": int(n_reliable),
        "n_voxels_variance": int(n_variance),
        "n_voxels_retained": int(n_retained),
        "n_retained": int(n_retained),  # Alias for backwards compatibility
        "n_rejected": int(n_rejected),
        "retention_rate": float(n_retained / n_voxels) if n_voxels > 0 else 0.0,
        "reliability_thr": float(reliability_thr),
        "min_var": float(min_var),
        "mean_r_retained": mean_r_retained,
        "mean_r_rejected": mean_r_rejected,
        "median_r_retained": median_r_retained,
        "median_r_rejected": median_r_rejected,
    }
    
    return mask, stats


def compute_soft_reliability_weights(
    r: np.ndarray,
    voxel_variance: np.ndarray,
    mode: str = "hard_threshold",
    reliability_thr: float = 0.1,
    min_var: float = 1e-6,
    curve: str = "sigmoid",
    temperature: float = 0.1,
) -> Tuple[np.ndarray, Dict]:
    """
    Compute continuous reliability-based voxel weights instead of binary mask.
    
    Novel contribution for paper: Soft reliability weighting enables model to learn
    optimal voxel weighting rather than hard thresholding, potentially improving
    generalization by retaining more voxels with appropriate downweighting.
    
    Three modes supported:
    - "hard_threshold": Binary mask (backward compatible, default)
    - "soft_weight": Continuous weights via sigmoid or linear curve
    - "none": All voxels weighted equally (weight=1.0)
    
    Args:
        r: Per-voxel split-half reliability, shape (n_voxels,)
        voxel_variance: Per-voxel variance, shape (n_voxels,)
        mode: Weighting mode, one of ["hard_threshold", "soft_weight", "none"]
        reliability_thr: Threshold/midpoint for weighting (default: 0.1)
        min_var: Minimum variance threshold (default: 1e-6)
        curve: Soft weighting curve, one of ["sigmoid", "linear"] (only for mode="soft_weight")
        temperature: Temperature for sigmoid curve (default: 0.1)
            Smaller = sharper transition, larger = smoother
    
    Returns:
        weights: Per-voxel weights, shape (n_voxels,), values in [0, 1]
        stats: Dictionary with:
            - mode: Weighting mode used
            - curve: Curve type (if soft_weight)
            - temperature: Temperature parameter (if sigmoid)
            - n_voxels_total: Total voxels
            - n_nonzero_weights: Number of voxels with weight > 0
            - mean_weight: Mean weight across all voxels
            - median_weight: Median weight
            - weight_percentiles: 10th, 25th, 50th, 75th, 90th percentiles
            - effective_voxels: Sum of weights (continuous version of n_retained)
    
    Example:
        >>> r = np.array([0.8, 0.5, 0.15, 0.05, -0.1])
        >>> var = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        >>> 
        >>> # Hard threshold (existing behavior)
        >>> weights, stats = compute_soft_reliability_weights(r, var, mode="hard_threshold")
        >>> print(weights)  # [1.0, 1.0, 1.0, 0.0, 0.0]
        >>> 
        >>> # Soft weighting with sigmoid
        >>> weights, stats = compute_soft_reliability_weights(
        ...     r, var, mode="soft_weight", curve="sigmoid", temperature=0.05
        ... )
        >>> print(weights)  # Smooth transition around threshold
        >>> 
        >>> # No weighting (all voxels equal)
        >>> weights, stats = compute_soft_reliability_weights(r, var, mode="none")
        >>> print(weights)  # [1.0, 1.0, 1.0, 1.0, 1.0]
    
    Notes:
        - Variance filtering always applied (weight=0 if var < min_var)
        - Sigmoid: weight = 1 / (1 + exp(-(r - threshold) / temperature))
        - Linear: weight = clip((r - threshold) / (1 - threshold), 0, 1)
        - For mode="hard_threshold", returns binary mask as float array
        - Designed for ablation studies comparing hard vs soft weighting
    """
    n_voxels = len(r)
    
    # Validate mode
    if mode not in ["hard_threshold", "soft_weight", "none"]:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: hard_threshold, soft_weight, none")
    
    # Validate curve
    if mode == "soft_weight" and curve not in ["sigmoid", "linear"]:
        raise ValueError(f"Invalid curve '{curve}'. Must be one of: sigmoid, linear")
    
    # Initialize weights
    weights = np.ones(n_voxels, dtype=np.float32)
    
    # Always filter by variance first
    variance_mask = voxel_variance >= min_var
    weights[~variance_mask] = 0.0
    
    # Apply reliability weighting based on mode
    if mode == "hard_threshold":
        # Binary mask (existing behavior)
        reliable_mask = r >= reliability_thr
        weights[~reliable_mask] = 0.0
        
    elif mode == "soft_weight":
        if curve == "sigmoid":
            # Smooth sigmoid transition
            # At r = threshold: weight = 0.5
            # At r = threshold + temperature: weight ≈ 0.73
            # At r = threshold - temperature: weight ≈ 0.27
            exponent = -(r - reliability_thr) / temperature
            # Clip exponent to avoid overflow
            exponent = np.clip(exponent, -50, 50)
            weights *= 1.0 / (1.0 + np.exp(exponent))
            
        elif curve == "linear":
            # Linear ramp from threshold to 1.0
            # Below threshold: weight = 0
            # At threshold: weight = 0
            # Above threshold: linear increase to weight = 1 at r = 1.0
            if reliability_thr >= 1.0:
                # Edge case: threshold at max, all weights become binary
                weights[r < reliability_thr] = 0.0
            else:
                slope = 1.0 / (1.0 - reliability_thr)
                linear_weights = slope * (r - reliability_thr)
                linear_weights = np.clip(linear_weights, 0.0, 1.0)
                weights *= linear_weights
                
    elif mode == "none":
        # All voxels weighted equally (just variance filter)
        pass
    
    # Ensure float32
    weights = weights.astype(np.float32)
    
    # Compute statistics
    nonzero_mask = weights > 0
    n_nonzero = nonzero_mask.sum()
    
    if n_nonzero > 0:
        mean_weight = float(weights[nonzero_mask].mean())
        median_weight = float(np.median(weights[nonzero_mask]))
        percentiles = np.percentile(weights[nonzero_mask], [10, 25, 50, 75, 90])
    else:
        mean_weight = 0.0
        median_weight = 0.0
        percentiles = np.zeros(5)
    
    # Effective number of voxels (sum of weights)
    effective_voxels = float(weights.sum())
    
    stats = {
        "mode": mode,
        "curve": curve if mode == "soft_weight" else None,
        "temperature": float(temperature) if mode == "soft_weight" and curve == "sigmoid" else None,
        "reliability_thr": float(reliability_thr),
        "min_var": float(min_var),
        "n_voxels_total": int(n_voxels),
        "n_nonzero_weights": int(n_nonzero),
        "mean_weight": mean_weight,
        "median_weight": median_weight,
        "weight_percentiles": {
            "p10": float(percentiles[0]),
            "p25": float(percentiles[1]),
            "p50": float(percentiles[2]),
            "p75": float(percentiles[3]),
            "p90": float(percentiles[4]),
        },
        "effective_voxels": effective_voxels,
        "retention_rate": float(n_nonzero / n_voxels) if n_voxels > 0 else 0.0,
    }
    
    return weights, stats
