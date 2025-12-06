"""
Utilities for aggregating and comparing evaluation results.

Provides helper functions for:
- Loading evaluation JSONs
- Extracting run metadata from paths
- Bootstrap confidence intervals
- Formatting metrics with CIs
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional


def load_eval_json(path: Path) -> Dict:
    """
    Load evaluation JSON from path.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Dictionary with evaluation results
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    
    with open(path) as f:
        data = json.load(f)
    
    return data


def guess_run_name(path: Path) -> str:
    """
    Guess a human-readable run name from file path.
    
    Heuristics:
    - If parent directory contains 'adapter', include that
    - If parent directory contains 'mlp' or 'ridge', include encoder
    - If parent directory contains '512' or '1024', include dimension
    
    Examples:
        outputs/reports/subj01/auto_no_adapter/recon_eval.json 
            → "no_adapter"
        outputs/reports/subj01/auto_with_adapter/recon_eval_1024.json 
            → "with_adapter_1024"
        outputs/reports/subj01/mlp_baseline/recon_eval.json
            → "mlp_baseline"
    
    Args:
        path: Path to JSON file
        
    Returns:
        Simplified run name
    """
    # Get parent directory names (up to 2 levels)
    parts = path.parts
    parent_dirs = []
    
    # Look at last 3 parts (excluding filename)
    for part in parts[-4:-1]:
        parent_dirs.append(part)
    
    # Build name from relevant parts
    name_parts = []
    
    for part in parent_dirs:
        part_lower = part.lower()
        
        # Skip common directory names
        if part_lower in ['outputs', 'reports', 'recon', 'eval']:
            continue
        
        # Keep informative parts
        if any(keyword in part_lower for keyword in [
            'adapter', 'mlp', 'ridge', 'auto', 'baseline', 
            '512', '768', '1024', 'no_adapter', 'with_adapter'
        ]):
            name_parts.append(part)
    
    # If we didn't find anything, use filename stem
    if not name_parts:
        name_parts.append(path.stem)
    
    return "_".join(name_parts)


def bootstrap_ci(
    values: np.ndarray,
    boots: int = 1000,
    alpha: float = 0.05,
    seed: int = 42
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for mean.
    
    Uses nonparametric bootstrap with replacement.
    
    Args:
        values: Array of per-sample values
        boots: Number of bootstrap resamples (default: 1000)
        alpha: Significance level (default: 0.05 for 95% CI)
        seed: Random seed for reproducibility (default: 42)
        
    Returns:
        Tuple of (lower_bound, upper_bound) for (1-alpha) CI
        
    Example:
        >>> values = np.array([0.5, 0.6, 0.7, 0.8])
        >>> low, high = bootstrap_ci(values, boots=1000)
        >>> print(f"95% CI: [{low:.3f}, {high:.3f}]")
    """
    if len(values) == 0:
        return (np.nan, np.nan)
    
    if len(values) == 1:
        # Can't bootstrap single value
        return (values[0], values[0])
    
    # Set seed for reproducibility
    rng = np.random.RandomState(seed)
    
    # Generate bootstrap samples
    n = len(values)
    boot_means = np.zeros(boots)
    
    for i in range(boots):
        # Resample with replacement
        indices = rng.choice(n, size=n, replace=True)
        boot_sample = values[indices]
        boot_means[i] = np.mean(boot_sample)
    
    # Compute percentile-based CI
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    low = np.percentile(boot_means, lower_percentile)
    high = np.percentile(boot_means, upper_percentile)
    
    return (low, high)


def format_mean_ci(
    mean: float,
    low: float,
    high: float,
    decimals: int = 3
) -> str:
    """
    Format mean with symmetric confidence interval.
    
    Computes half-width as max(mean-low, high-mean) and formats as:
        "mean ± half_width"
    
    Args:
        mean: Point estimate
        low: Lower CI bound
        high: Upper CI bound
        decimals: Number of decimal places (default: 3)
        
    Returns:
        Formatted string "mean ± half_width"
        
    Examples:
        >>> format_mean_ci(0.612, 0.571, 0.653)
        "0.612 ± 0.041"
        
        >>> format_mean_ci(0.543, 0.502, 0.584, decimals=2)
        "0.54 ± 0.04"
    """
    if np.isnan(mean) or np.isnan(low) or np.isnan(high):
        return "NA"
    
    # Compute symmetric half-width (conservative)
    half_width = max(abs(mean - low), abs(high - mean))
    
    # Format with specified decimals
    fmt = f"{{:.{decimals}f}}"
    mean_str = fmt.format(mean)
    hw_str = fmt.format(half_width)
    
    return f"{mean_str} ± {hw_str}"


def format_mean_ci_range(
    mean: float,
    low: float,
    high: float,
    decimals: int = 3
) -> str:
    """
    Format mean with confidence interval range.
    
    Formats as: "mean [low, high]"
    
    Args:
        mean: Point estimate
        low: Lower CI bound
        high: Upper CI bound
        decimals: Number of decimal places (default: 3)
        
    Returns:
        Formatted string "mean [low, high]"
        
    Example:
        >>> format_mean_ci_range(0.612, 0.571, 0.653)
        "0.612 [0.571, 0.653]"
    """
    if np.isnan(mean) or np.isnan(low) or np.isnan(high):
        return "NA"
    
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(mean)} [{fmt.format(low)}, {fmt.format(high)}]"
