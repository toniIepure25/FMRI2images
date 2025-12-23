"""Tests for noise ceiling and reliability utilities."""
import numpy as np
import pytest
from pathlib import Path

from fmri2img.reliability.noise_ceiling import (
    compute_voxel_noise_ceiling_from_ncsnr,
    aggregate_roi_ceiling,
    compute_ceiling_normalized_score,
    compute_repeat_consistency
)


def test_compute_voxel_noise_ceiling_standard():
    """Test standard noise ceiling computation."""
    ncsnr = np.array([0.0, 0.5, 1.0, 2.0, 10.0, 100.0])
    
    ceiling = compute_voxel_noise_ceiling_from_ncsnr(ncsnr, method="standard")
    
    # Check shape
    assert ceiling.shape == ncsnr.shape
    
    # Check range [0, 1]
    assert np.all(ceiling >= 0.0)
    assert np.all(ceiling <= 1.0)
    
    # Check monotonicity (higher NCSNR -> higher ceiling)
    assert np.all(np.diff(ceiling) > 0)
    
    # Check specific values
    assert ceiling[0] == 0.0  # NCSNR=0 -> ceiling=0
    assert ceiling[-1] > 0.99  # NCSNR=100 -> ceiling≈1


def test_compute_voxel_noise_ceiling_methods():
    """Test different ceiling computation methods."""
    ncsnr = np.array([1.0, 2.0, 5.0, 10.0])
    
    ceiling_standard = compute_voxel_noise_ceiling_from_ncsnr(ncsnr, method="standard")
    ceiling_corr = compute_voxel_noise_ceiling_from_ncsnr(ncsnr, method="correlation")
    ceiling_linear = compute_voxel_noise_ceiling_from_ncsnr(ncsnr, method="linear")
    
    # All should be in [0, 1]
    for ceiling in [ceiling_standard, ceiling_corr, ceiling_linear]:
        assert np.all(ceiling >= 0.0)
        assert np.all(ceiling <= 1.0)
    
    # Different methods should give different values
    assert not np.allclose(ceiling_standard, ceiling_corr)
    assert not np.allclose(ceiling_standard, ceiling_linear)


def test_compute_voxel_noise_ceiling_none():
    """Test ceiling computation with None input."""
    ceiling = compute_voxel_noise_ceiling_from_ncsnr(None)
    assert ceiling is None


def test_aggregate_roi_ceiling_mean():
    """Test ROI ceiling aggregation with mean."""
    ceiling_map = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    
    roi_ceiling = aggregate_roi_ceiling(ceiling_map, aggregation="mean")
    
    assert roi_ceiling == pytest.approx(0.7)


def test_aggregate_roi_ceiling_with_mask():
    """Test ROI ceiling aggregation with mask."""
    ceiling_map = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    roi_mask = np.array([True, True, True, False, False])
    
    roi_ceiling = aggregate_roi_ceiling(ceiling_map, roi_mask=roi_mask, aggregation="mean")
    
    # Should only average first 3 values
    assert roi_ceiling == pytest.approx((0.5 + 0.6 + 0.7) / 3)


def test_aggregate_roi_ceiling_median():
    """Test median aggregation."""
    ceiling_map = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    
    roi_ceiling = aggregate_roi_ceiling(ceiling_map, aggregation="median")
    
    assert roi_ceiling == 0.7


def test_aggregate_roi_ceiling_none():
    """Test aggregation with None input."""
    roi_ceiling = aggregate_roi_ceiling(None)
    assert roi_ceiling is None


def test_compute_ceiling_normalized_score():
    """Test ceiling normalization."""
    raw_score = 0.45
    ceiling = 0.65
    
    norm_score = compute_ceiling_normalized_score(raw_score, ceiling)
    
    assert norm_score == pytest.approx(0.45 / 0.65)
    assert 0 < norm_score < 1


def test_compute_ceiling_normalized_score_exceeds_ceiling():
    """Test normalization when raw score exceeds ceiling."""
    raw_score = 0.75
    ceiling = 0.65
    
    norm_score = compute_ceiling_normalized_score(raw_score, ceiling)
    
    # Normalized score > 1 is allowed (can happen with estimation error)
    assert norm_score > 1.0


def test_compute_ceiling_normalized_score_none_inputs():
    """Test normalization with None inputs."""
    assert compute_ceiling_normalized_score(None, 0.5) is None
    assert compute_ceiling_normalized_score(0.5, None) is None
    assert compute_ceiling_normalized_score(None, None) is None


def test_compute_ceiling_normalized_score_zero_ceiling():
    """Test normalization with very low ceiling."""
    raw_score = 0.1
    ceiling = 1e-10
    
    # Should return None to avoid division issues
    norm_score = compute_ceiling_normalized_score(raw_score, ceiling)
    assert norm_score is None


def test_compute_repeat_consistency_cosine():
    """Test repeat consistency with cosine similarity."""
    rng = np.random.RandomState(42)
    
    # Generate 3 reps of predictions (1000 samples x 512 dims)
    base_pred = rng.randn(100, 512)
    
    # Add small noise to create correlated repetitions
    pred_rep0 = base_pred + rng.randn(100, 512) * 0.1
    pred_rep1 = base_pred + rng.randn(100, 512) * 0.1
    pred_rep2 = base_pred + rng.randn(100, 512) * 0.1
    
    consistency = compute_repeat_consistency(
        [pred_rep0, pred_rep1, pred_rep2],
        metric="cosine"
    )
    
    # Check keys
    assert "mean" in consistency
    assert "std" in consistency
    assert "n_reps" in consistency
    assert "n_pairs" in consistency
    
    # Check values
    assert consistency["n_reps"] == 3
    assert consistency["n_pairs"] == 3  # C(3,2) = 3 pairs
    
    # With small noise, consistency should be high
    assert consistency["mean"] > 0.8


def test_compute_repeat_consistency_identical():
    """Test consistency with identical predictions."""
    pred = np.random.randn(50, 512)
    
    # All reps identical
    consistency = compute_repeat_consistency(
        [pred, pred, pred],
        metric="cosine"
    )
    
    # Should have perfect consistency
    assert consistency["mean"] == pytest.approx(1.0, abs=1e-6)
    assert consistency["std"] == pytest.approx(0.0, abs=1e-6)


def test_compute_repeat_consistency_uncorrelated():
    """Test consistency with uncorrelated predictions."""
    rng = np.random.RandomState(42)
    
    # Completely independent predictions
    pred_rep0 = rng.randn(100, 512)
    pred_rep1 = rng.randn(100, 512)
    pred_rep2 = rng.randn(100, 512)
    
    consistency = compute_repeat_consistency(
        [pred_rep0, pred_rep1, pred_rep2],
        metric="cosine"
    )
    
    # With random noise, consistency should be near 0
    assert -0.2 < consistency["mean"] < 0.2


def test_compute_repeat_consistency_two_reps():
    """Test consistency with minimum 2 reps."""
    pred_rep0 = np.random.randn(50, 512)
    pred_rep1 = pred_rep0 + np.random.randn(50, 512) * 0.1
    
    consistency = compute_repeat_consistency(
        [pred_rep0, pred_rep1],
        metric="cosine"
    )
    
    assert consistency["n_reps"] == 2
    assert consistency["n_pairs"] == 1


def test_compute_repeat_consistency_single_rep():
    """Test consistency with only 1 rep (should return None)."""
    pred = np.random.randn(50, 512)
    
    consistency = compute_repeat_consistency([pred], metric="cosine")
    
    # Can't compute consistency with < 2 reps
    assert consistency is None


def test_compute_repeat_consistency_correlation():
    """Test consistency with correlation metric."""
    rng = np.random.RandomState(42)
    base_pred = rng.randn(50, 512)
    
    pred_rep0 = base_pred + rng.randn(50, 512) * 0.1
    pred_rep1 = base_pred + rng.randn(50, 512) * 0.1
    
    consistency = compute_repeat_consistency(
        [pred_rep0, pred_rep1],
        metric="correlation"
    )
    
    # Should have high correlation
    assert consistency["mean"] > 0.5


def test_ncsnr_to_ceiling_realistic_values():
    """Test ceiling computation with realistic NCSNR values."""
    # Typical visual cortex NCSNR values
    ncsnr_low = np.array([0.1, 0.2, 0.3])  # Low SNR voxels
    ncsnr_med = np.array([1.0, 2.0, 3.0])  # Medium SNR
    ncsnr_high = np.array([10.0, 20.0, 50.0])  # High SNR
    
    ceiling_low = compute_voxel_noise_ceiling_from_ncsnr(ncsnr_low)
    ceiling_med = compute_voxel_noise_ceiling_from_ncsnr(ncsnr_med)
    ceiling_high = compute_voxel_noise_ceiling_from_ncsnr(ncsnr_high)
    
    # Low SNR -> low ceiling
    assert np.all(ceiling_low < 0.5)
    
    # Medium SNR -> medium ceiling
    assert np.all((ceiling_med > 0.5) & (ceiling_med < 0.9))
    
    # High SNR -> high ceiling
    assert np.all(ceiling_high > 0.9)
