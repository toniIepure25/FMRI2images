"""
Statistical Inference Utilities for Paper-Grade Evaluations
==========================================================

Rigorous statistical methods for comparing fMRI reconstruction strategies:
1. Bootstrap confidence intervals for uncertainty quantification
2. Paired permutation tests for significance testing
3. Effect size computation (Cohen's d)
4. Multiple comparison correction (Holm-Bonferroni)

Scientific Context:
- Bootstrap CI: Non-parametric confidence intervals (Efron & Tibshirani, 1993)
- Permutation tests: Exact p-values without distributional assumptions
- Cohen's d: Standardized effect size for comparing methods
- Holm-Bonferroni: Family-wise error rate control for multiple tests

References:
- Efron & Tibshirani (1993). "An Introduction to the Bootstrap"
- Good (2005). "Permutation, Parametric, and Bootstrap Tests of Hypotheses"
- Cohen (1988). "Statistical Power Analysis for the Behavioral Sciences"
- Holm (1979). "A Simple Sequentially Rejective Multiple Test Procedure"
"""

import logging
from typing import Union, Tuple, Dict, List

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
    stat_fn=None
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for a statistic.
    
    Uses percentile method to compute non-parametric confidence intervals.
    
    Args:
        values: Array of values to bootstrap, shape (n_samples,)
        n_boot: Number of bootstrap samples (default: 2000)
        alpha: Significance level (default: 0.05 for 95% CI)
        seed: Random seed for reproducibility
        stat_fn: Statistic function (default: np.mean)
                Takes array and returns scalar
    
    Returns:
        (lower_bound, upper_bound): Confidence interval
        
    Example:
        >>> scores = np.array([0.65, 0.68, 0.72, 0.70, 0.69])
        >>> lower, upper = bootstrap_ci(scores, n_boot=2000, alpha=0.05)
        >>> print(f"Mean: {scores.mean():.3f} [{lower:.3f}, {upper:.3f}]")
        Mean: 0.688 [0.660, 0.715]
        
    Scientific Context:
    - Bootstrap provides non-parametric CIs without normality assumptions
    - 2000 iterations is standard for stable percentile CIs
    - Percentile method: [B_{α/2}, B_{1-α/2}] where B are bootstrap estimates
    """
    if stat_fn is None:
        stat_fn = np.mean
    
    rng = np.random.RandomState(seed)
    
    n = len(values)
    boot_stats = np.zeros(n_boot)
    
    for i in range(n_boot):
        # Sample with replacement
        boot_sample = rng.choice(values, size=n, replace=True)
        boot_stats[i] = stat_fn(boot_sample)
    
    # Percentile method
    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    
    return lower, upper


def paired_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int = 10000,
    seed: int = 42,
    alternative: str = "two-sided"
) -> float:
    """
    Paired permutation test for comparing two methods on same samples.
    
    Tests H0: x and y have same distribution vs H1: they differ.
    Exact test that doesn't assume normality.
    
    Args:
        x: Method 1 scores, shape (n_samples,)
        y: Method 2 scores, shape (n_samples,)
        n_perm: Number of permutations (default: 10000)
        seed: Random seed
        alternative: "two-sided", "greater", or "less"
            - "two-sided": x != y
            - "greater": x > y
            - "less": x < y
    
    Returns:
        p_value: Probability of observing difference as extreme under H0
        
    Example:
        >>> # Compare CLIPScore for two strategies on same test set
        >>> strategy1_scores = np.array([0.65, 0.68, 0.72, 0.70])
        >>> strategy2_scores = np.array([0.70, 0.73, 0.75, 0.74])
        >>> p_value = paired_permutation_test(
        ...     strategy1_scores,
        ...     strategy2_scores,
        ...     n_perm=10000
        ... )
        >>> print(f"p-value: {p_value:.4f}")
        >>> if p_value < 0.05:
        ...     print("Significant difference (p < 0.05)")
        
    Scientific Context:
    - Paired test: same samples evaluated by both methods
    - Permutation test: exact p-value by randomly flipping pair labels
    - No distributional assumptions (unlike t-test)
    - 10000 permutations gives p-value precision of ~0.0001
    """
    if len(x) != len(y):
        raise ValueError(f"Arrays must have same length: {len(x)} vs {len(y)}")
    
    if alternative not in ["two-sided", "greater", "less"]:
        raise ValueError(f"Invalid alternative: {alternative}")
    
    rng = np.random.RandomState(seed)
    
    # Observed difference
    diff_obs = np.mean(x) - np.mean(y)
    
    # Permutation distribution
    n = len(x)
    diff_perm = np.zeros(n_perm)
    
    for i in range(n_perm):
        # For each pair, randomly swap x and y values
        swap_mask = rng.randint(0, 2, size=n).astype(bool)
        
        x_perm = np.where(swap_mask, y, x)
        y_perm = np.where(swap_mask, x, y)
        
        diff_perm[i] = np.mean(x_perm) - np.mean(y_perm)
    
    # Compute p-value
    if alternative == "two-sided":
        p_value = np.mean(np.abs(diff_perm) >= np.abs(diff_obs))
    elif alternative == "greater":
        p_value = np.mean(diff_perm >= diff_obs)
    else:  # "less"
        p_value = np.mean(diff_perm <= diff_obs)
    
    return p_value


def cohens_d_paired(
    x: np.ndarray,
    y: np.ndarray
) -> float:
    """
    Compute Cohen's d effect size for paired samples.
    
    Standardized mean difference: d = mean(x - y) / std(x - y)
    
    Interpretation (Cohen, 1988):
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large
    
    Args:
        x: Method 1 scores, shape (n_samples,)
        y: Method 2 scores, shape (n_samples,)
    
    Returns:
        Effect size (Cohen's d)
        
    Example:
        >>> strategy1 = np.array([0.65, 0.68, 0.72, 0.70])
        >>> strategy2 = np.array([0.70, 0.73, 0.75, 0.74])
        >>> d = cohens_d_paired(strategy1, strategy2)
        >>> print(f"Cohen's d: {d:.3f}")
        >>> if abs(d) >= 0.8:
        ...     print("Large effect size")
        
    Scientific Context:
    - Effect size quantifies magnitude of difference (not just significance)
    - Paired version uses std of differences (not pooled std)
    - Essential for power analysis and practical significance
    """
    if len(x) != len(y):
        raise ValueError(f"Arrays must have same length: {len(x)} vs {len(y)}")
    
    diff = x - y
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    
    if std_diff == 0:
        logger.warning("Zero std, returning inf effect size")
        return np.inf if mean_diff > 0 else -np.inf
    
    return mean_diff / std_diff


def holm_bonferroni_correction(
    p_values: Union[Dict[str, float], List[float]],
    alpha: float = 0.05
) -> Union[Dict[str, Tuple[float, bool]], List[Tuple[float, bool]]]:
    """
    Apply Holm-Bonferroni correction for multiple hypothesis testing.
    
    Controls family-wise error rate (FWER) - probability of any false positive.
    More powerful than Bonferroni correction (fewer false negatives).
    
    Procedure:
    1. Sort p-values in ascending order
    2. For rank k, compare p_k to α/(m - k + 1)
    3. Reject H0 while p_k <= α/(m - k + 1), stop at first non-rejection
    
    Args:
        p_values: Dictionary {test_name: p_value} or list of p-values
        alpha: Family-wise error rate (default: 0.05)
    
    Returns:
        If dict input: {test_name: (adjusted_p, is_significant)}
        If list input: [(adjusted_p, is_significant), ...]
        
    Example:
        >>> # Compare 3 strategies, 3 pairwise tests
        >>> p_values = {
        ...     "single_vs_best8": 0.012,
        ...     "single_vs_boi": 0.004,
        ...     "best8_vs_boi": 0.087
        ... }
        >>> results = holm_bonferroni_correction(p_values, alpha=0.05)
        >>> for test, (adj_p, sig) in results.items():
        ...     print(f"{test}: p={adj_p:.4f}, sig={sig}")
        single_vs_boi: p=0.012, sig=True    # Most significant
        single_vs_best8: p=0.024, sig=True  # Second
        best8_vs_boi: p=0.087, sig=False    # Not significant
        
    Scientific Context:
    - Holm (1979): Sequential rejective method
    - Controls FWER at level α
    - More powerful than Bonferroni (α/m for all tests)
    - Widely used in neuroimaging (Nichols & Hayasaka, 2003)
    """
    is_dict = isinstance(p_values, dict)
    
    if is_dict:
        test_names = list(p_values.keys())
        p_array = np.array([p_values[name] for name in test_names])
    else:
        test_names = None
        p_array = np.array(p_values)
    
    m = len(p_array)
    
    # Sort p-values and track original indices
    sorted_indices = np.argsort(p_array)
    sorted_p = p_array[sorted_indices]
    
    # Holm-Bonferroni adjusted p-values
    adjusted_p = np.zeros(m)
    is_significant = np.zeros(m, dtype=bool)
    
    for k, idx in enumerate(sorted_indices):
        # Adjusted threshold for rank k (0-indexed)
        threshold = alpha / (m - k)
        
        # Adjusted p-value (use maximum with previous to maintain monotonicity)
        adjusted_p[idx] = min(sorted_p[k] * (m - k), 1.0)
        if k > 0:
            prev_idx = sorted_indices[k-1]
            adjusted_p[idx] = max(adjusted_p[idx], adjusted_p[prev_idx])
        
        # Significant if below threshold and all previous were significant
        if k == 0:
            is_significant[idx] = sorted_p[k] <= threshold
        else:
            prev_idx = sorted_indices[k-1]
            is_significant[idx] = (sorted_p[k] <= threshold and is_significant[prev_idx])
    
    # Return in same format as input
    if is_dict:
        return {
            name: (adjusted_p[i], bool(is_significant[i]))
            for i, name in enumerate(test_names)
        }
    else:
        return list(zip(adjusted_p, is_significant.astype(bool)))


def aggregate_across_seeds(
    values_per_seed: List[np.ndarray],
    alpha: float = 0.05,
    seed: int = 42
) -> Dict[str, float]:
    """
    Aggregate metric values across multiple random seeds.
    
    Computes mean and bootstrap confidence interval across seeds.
    
    Args:
        values_per_seed: List of arrays, one per seed
                        Each array has shape (n_samples,)
        alpha: CI significance level
        seed: Bootstrap random seed
    
    Returns:
        Dictionary with:
        - mean: Mean across seeds
        - std: Std across seeds
        - ci_lower: Bootstrap CI lower bound
        - ci_upper: Bootstrap CI upper bound
        - n_seeds: Number of seeds
        
    Example:
        >>> # CLIPScores from 3 different seeds
        >>> seed0 = np.array([0.65, 0.68, 0.72])
        >>> seed1 = np.array([0.66, 0.69, 0.71])
        >>> seed2 = np.array([0.64, 0.67, 0.73])
        >>> agg = aggregate_across_seeds([seed0, seed1, seed2])
        >>> print(f"Mean: {agg['mean']:.3f} [{agg['ci_lower']:.3f}, {agg['ci_upper']:.3f}]")
    """
    # Concatenate all values
    all_values = np.concatenate(values_per_seed)
    
    # Per-seed means
    seed_means = np.array([np.mean(vals) for vals in values_per_seed])
    
    # Bootstrap CI on seed means
    ci_lower, ci_upper = bootstrap_ci(seed_means, n_boot=2000, alpha=alpha, seed=seed)
    
    return {
        "mean": float(np.mean(all_values)),
        "std": float(np.std(all_values)),
        "sem": float(np.std(all_values) / np.sqrt(len(all_values))),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n_seeds": len(values_per_seed),
        "n_samples_per_seed": [len(v) for v in values_per_seed]
    }
