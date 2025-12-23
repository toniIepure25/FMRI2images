"""Tests for statistical inference utilities."""
import numpy as np
import pytest

from fmri2img.stats.inference import (
    bootstrap_ci,
    paired_permutation_test,
    cohens_d_paired,
    holm_bonferroni_correction,
    aggregate_across_seeds
)


def test_bootstrap_ci_basic():
    """Test basic bootstrap CI computation."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    lower, upper = bootstrap_ci(values, n_boot=1000, alpha=0.05, seed=42)
    
    # CI should contain the mean
    mean_val = np.mean(values)
    assert lower < mean_val < upper
    
    # Lower should be less than upper
    assert lower < upper


def test_bootstrap_ci_reproducible():
    """Test that bootstrap CI is reproducible with same seed."""
    values = np.random.randn(100)
    
    ci1 = bootstrap_ci(values, n_boot=1000, seed=42)
    ci2 = bootstrap_ci(values, n_boot=1000, seed=42)
    
    assert ci1 == ci2


def test_bootstrap_ci_custom_statistic():
    """Test bootstrap CI with custom statistic function."""
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    # Median instead of mean
    lower, upper = bootstrap_ci(values, n_boot=1000, stat_fn=np.median, seed=42)
    
    median_val = np.median(values)
    assert lower <= median_val <= upper


def test_paired_permutation_test_identical():
    """Test permutation test with identical arrays (should give high p-value)."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = x.copy()
    
    p_value = paired_permutation_test(x, y, n_perm=1000, seed=42)
    
    # Identical arrays should give p-value close to 1
    assert p_value > 0.9


def test_paired_permutation_test_different():
    """Test permutation test with clearly different arrays."""
    rng = np.random.RandomState(42)
    
    x = rng.randn(100)
    y = x + 2.0  # Shift by 2
    
    p_value = paired_permutation_test(x, y, n_perm=1000, seed=42, alternative="two-sided")
    
    # Should detect significant difference
    assert p_value < 0.05


def test_paired_permutation_test_alternatives():
    """Test permutation test with different alternatives."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 3.0, 4.0, 5.0, 6.0])  # y > x
    
    p_less = paired_permutation_test(x, y, n_perm=1000, seed=42, alternative="less")
    p_greater = paired_permutation_test(x, y, n_perm=1000, seed=42, alternative="greater")
    
    # x < y, so "less" should be significant
    assert p_less < 0.05
    # x < y, so "greater" should not be significant
    assert p_greater > 0.95


def test_paired_permutation_test_length_mismatch():
    """Test that permutation test raises error for mismatched lengths."""
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3, 4])
    
    with pytest.raises(ValueError, match="same length"):
        paired_permutation_test(x, y)


def test_cohens_d_paired_basic():
    """Test Cohen's d computation."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    
    d = cohens_d_paired(x, y)
    
    # y > x, so d should be negative
    assert d < 0
    
    # With constant difference and moderate std, should be medium effect
    assert abs(d) > 0.5


def test_cohens_d_paired_identical():
    """Test Cohen's d with identical arrays."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = x.copy()
    
    d = cohens_d_paired(x, y)
    
    # Identical arrays should give d = inf (or close to 0)
    # Actually will be inf due to zero std
    assert np.isinf(d) or abs(d) < 1e-10


def test_cohens_d_paired_large_effect():
    """Test Cohen's d with large effect size."""
    rng = np.random.RandomState(42)
    x = rng.randn(100)
    y = x + 1.5  # Large shift relative to std ~1
    
    d = cohens_d_paired(x, y)
    
    # Should have large effect size
    assert abs(d) > 0.8


def test_holm_bonferroni_dict():
    """Test Holm-Bonferroni correction with dictionary input."""
    p_values = {
        "test1": 0.001,
        "test2": 0.01,
        "test3": 0.03,
        "test4": 0.08
    }
    
    results = holm_bonferroni_correction(p_values, alpha=0.05)
    
    # test1 should be significant (most significant)
    assert results["test1"][1] == True
    
    # test2 should be significant
    assert results["test2"][1] == True
    
    # test3 might or might not be significant depending on Holm procedure
    # test4 should not be significant (largest p-value)
    assert results["test4"][1] == False


def test_holm_bonferroni_list():
    """Test Holm-Bonferroni correction with list input."""
    p_values = [0.001, 0.01, 0.03, 0.08]
    
    results = holm_bonferroni_correction(p_values, alpha=0.05)
    
    # Should return list of tuples
    assert len(results) == 4
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
    
    # First test should be significant
    assert results[0][1] == True


def test_holm_bonferroni_all_significant():
    """Test Holm-Bonferroni with all very small p-values."""
    p_values = {"test1": 0.001, "test2": 0.002, "test3": 0.003}
    
    results = holm_bonferroni_correction(p_values, alpha=0.05)
    
    # All should be significant
    assert all(sig for _, sig in results.values())


def test_holm_bonferroni_none_significant():
    """Test Holm-Bonferroni with all large p-values."""
    p_values = {"test1": 0.1, "test2": 0.2, "test3": 0.3}
    
    results = holm_bonferroni_correction(p_values, alpha=0.05)
    
    # None should be significant
    assert all(not sig for _, sig in results.values())


def test_aggregate_across_seeds():
    """Test aggregation across multiple seeds."""
    rng = np.random.RandomState(42)
    
    # Generate data from 3 seeds
    seed0 = rng.randn(50) + 1.0
    seed1 = rng.randn(50) + 1.0
    seed2 = rng.randn(50) + 1.0
    
    agg = aggregate_across_seeds([seed0, seed1, seed2], alpha=0.05, seed=42)
    
    # Check keys
    assert "mean" in agg
    assert "std" in agg
    assert "ci_lower" in agg
    assert "ci_upper" in agg
    assert "n_seeds" in agg
    
    # Check values make sense
    assert agg["n_seeds"] == 3
    assert agg["ci_lower"] < agg["mean"] < agg["ci_upper"]
    
    # Mean should be close to 1.0
    assert 0.8 < agg["mean"] < 1.2


def test_aggregate_across_seeds_single_seed():
    """Test aggregation with single seed."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    agg = aggregate_across_seeds([values], alpha=0.05, seed=42)
    
    assert agg["n_seeds"] == 1
    assert agg["mean"] == np.mean(values)
