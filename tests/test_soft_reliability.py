"""
Tests for soft reliability weighting (Novel Contribution A).

Tests the compute_soft_reliability_weights function and its integration
into the NSDPreprocessor pipeline.
"""

import pytest
import numpy as np
from fmri2img.data.reliability import compute_soft_reliability_weights


class TestComputeSoftReliabilityWeights:
    """Test compute_soft_reliability_weights function."""
    
    def test_hard_threshold_mode(self):
        """Test hard_threshold mode produces binary weights."""
        r = np.array([0.8, 0.5, 0.15, 0.05, -0.1])
        var = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        
        weights, stats = compute_soft_reliability_weights(
            r, var, mode="hard_threshold", reliability_thr=0.1
        )
        
        # Should be binary (0 or 1)
        assert np.all((weights == 0) | (weights == 1))
        
        # Expected: [1, 1, 1, 0, 0]
        expected = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
        np.testing.assert_array_equal(weights, expected)
        
        # Check stats
        assert stats["mode"] == "hard_threshold"
        assert stats["n_nonzero_weights"] == 3
        assert stats["effective_voxels"] == 3.0
    
    def test_soft_weight_sigmoid_mode(self):
        """Test soft_weight with sigmoid curve produces smooth transition."""
        r = np.array([0.2, 0.15, 0.1, 0.05, 0.0])
        var = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        
        weights, stats = compute_soft_reliability_weights(
            r, var, mode="soft_weight", curve="sigmoid", 
            reliability_thr=0.1, temperature=0.05
        )
        
        # All weights should be in [0, 1]
        assert np.all(weights >= 0)
        assert np.all(weights <= 1)
        
        # Should be monotonically increasing with reliability
        assert weights[0] > weights[1] > weights[2] > weights[3] > weights[4]
        
        # At threshold (r=0.1), weight should be ~0.5
        assert abs(weights[2] - 0.5) < 0.05
        
        # Check stats
        assert stats["mode"] == "soft_weight"
        assert stats["curve"] == "sigmoid"
        assert stats["temperature"] == 0.05
        assert stats["n_nonzero_weights"] == 5  # All have some weight
        assert stats["effective_voxels"] < 5.0  # But less than 5 effective voxels
    
    def test_soft_weight_linear_mode(self):
        """Test soft_weight with linear curve produces linear ramp."""
        r = np.array([1.0, 0.55, 0.1, 0.05, 0.0])
        var = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        
        weights, stats = compute_soft_reliability_weights(
            r, var, mode="soft_weight", curve="linear", reliability_thr=0.1
        )
        
        # All weights should be in [0, 1]
        assert np.all(weights >= 0)
        assert np.all(weights <= 1)
        
        # At r=1.0, weight should be 1.0
        assert abs(weights[0] - 1.0) < 0.01
        
        # At r=0.55 (halfway between 0.1 and 1.0), weight should be 0.5
        assert abs(weights[1] - 0.5) < 0.01
        
        # Below threshold, weight should be 0
        assert weights[2] == 0.0
        assert weights[3] == 0.0
        assert weights[4] == 0.0
        
        # Check stats
        assert stats["mode"] == "soft_weight"
        assert stats["curve"] == "linear"
        assert stats["n_nonzero_weights"] == 2
    
    def test_none_mode(self):
        """Test none mode gives all voxels equal weight."""
        r = np.array([0.8, 0.5, 0.15, 0.05, -0.1])
        var = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        
        weights, stats = compute_soft_reliability_weights(
            r, var, mode="none", reliability_thr=0.1
        )
        
        # All weights should be 1.0
        expected = np.ones(5, dtype=np.float32)
        np.testing.assert_array_equal(weights, expected)
        
        # Check stats
        assert stats["mode"] == "none"
        assert stats["n_nonzero_weights"] == 5
        assert stats["effective_voxels"] == 5.0
    
    def test_variance_filtering(self):
        """Test variance threshold is applied in all modes."""
        r = np.array([0.8, 0.5, 0.15, 0.05, -0.1])
        var = np.array([1.0, 1.0, 1e-8, 1.0, 1.0])  # Third voxel has low variance
        
        # Test hard threshold mode
        weights_hard, _ = compute_soft_reliability_weights(
            r, var, mode="hard_threshold", reliability_thr=0.1, min_var=1e-6
        )
        assert weights_hard[2] == 0.0  # Low variance voxel excluded
        
        # Test soft weight mode
        weights_soft, _ = compute_soft_reliability_weights(
            r, var, mode="soft_weight", curve="sigmoid", 
            reliability_thr=0.1, min_var=1e-6
        )
        assert weights_soft[2] == 0.0  # Low variance voxel excluded
        
        # Test none mode
        weights_none, _ = compute_soft_reliability_weights(
            r, var, mode="none", reliability_thr=0.1, min_var=1e-6
        )
        assert weights_none[2] == 0.0  # Low variance voxel excluded
    
    def test_sigmoid_temperature_effect(self):
        """Test that temperature controls sigmoid sharpness."""
        r = np.array([0.2, 0.15, 0.1, 0.05, 0.0])
        var = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        
        # Low temperature (sharp transition)
        weights_sharp, _ = compute_soft_reliability_weights(
            r, var, mode="soft_weight", curve="sigmoid",
            reliability_thr=0.1, temperature=0.01
        )
        
        # High temperature (smooth transition)
        weights_smooth, _ = compute_soft_reliability_weights(
            r, var, mode="soft_weight", curve="sigmoid",
            reliability_thr=0.1, temperature=0.2
        )
        
        # Sharp transition should have more extreme values
        # (closer to 0 or 1) compared to smooth transition
        sharp_var = np.var(weights_sharp)
        smooth_var = np.var(weights_smooth)
        assert sharp_var > smooth_var
    
    def test_effective_voxels_calculation(self):
        """Test that effective_voxels equals sum of weights."""
        r = np.array([0.8, 0.5, 0.15, 0.1, 0.05])
        var = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        
        weights, stats = compute_soft_reliability_weights(
            r, var, mode="soft_weight", curve="sigmoid",
            reliability_thr=0.1, temperature=0.05
        )
        
        # Effective voxels should equal sum of weights
        assert abs(stats["effective_voxels"] - weights.sum()) < 1e-5
    
    def test_weight_percentiles(self):
        """Test that weight percentiles are computed correctly."""
        # Create array with known distribution
        r = np.linspace(0, 1, 100)
        var = np.ones(100)
        
        weights, stats = compute_soft_reliability_weights(
            r, var, mode="soft_weight", curve="linear",
            reliability_thr=0.1, temperature=0.1
        )
        
        # Check percentiles exist
        assert "weight_percentiles" in stats
        assert "p10" in stats["weight_percentiles"]
        assert "p25" in stats["weight_percentiles"]
        assert "p50" in stats["weight_percentiles"]
        assert "p75" in stats["weight_percentiles"]
        assert "p90" in stats["weight_percentiles"]
        
        # Percentiles should be monotonically increasing
        p = stats["weight_percentiles"]
        assert p["p10"] <= p["p25"] <= p["p50"] <= p["p75"] <= p["p90"]
    
    def test_invalid_mode(self):
        """Test that invalid mode raises ValueError."""
        r = np.array([0.8, 0.5, 0.15])
        var = np.array([1.0, 1.0, 1.0])
        
        with pytest.raises(ValueError, match="Invalid mode"):
            compute_soft_reliability_weights(r, var, mode="invalid")
    
    def test_invalid_curve(self):
        """Test that invalid curve raises ValueError."""
        r = np.array([0.8, 0.5, 0.15])
        var = np.array([1.0, 1.0, 1.0])
        
        with pytest.raises(ValueError, match="Invalid curve"):
            compute_soft_reliability_weights(
                r, var, mode="soft_weight", curve="invalid"
            )
    
    def test_float32_dtype(self):
        """Test that output weights are float32."""
        r = np.array([0.8, 0.5, 0.15])
        var = np.array([1.0, 1.0, 1.0])
        
        weights, _ = compute_soft_reliability_weights(
            r, var, mode="soft_weight", curve="sigmoid"
        )
        
        assert weights.dtype == np.float32
    
    def test_backward_compatibility_with_hard_threshold(self):
        """Test that hard_threshold mode produces same results as old filter_voxels_by_reliability."""
        from fmri2img.data.reliability import filter_voxels_by_reliability
        
        r = np.array([0.8, 0.5, 0.15, 0.05, -0.1])
        var = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        threshold = 0.1
        
        # Old method
        mask_old, stats_old = filter_voxels_by_reliability(
            r, var, reliability_thr=threshold
        )
        
        # New method with hard_threshold mode
        weights_new, stats_new = compute_soft_reliability_weights(
            r, var, mode="hard_threshold", reliability_thr=threshold
        )
        
        # Weights should equal mask (as float)
        np.testing.assert_array_equal(weights_new, mask_old.astype(np.float32))
        
        # Number of retained voxels should match
        assert stats_new["n_nonzero_weights"] == stats_old["n_retained"]


class TestSoftReliabilityIntegration:
    """Test integration of soft reliability weighting into preprocessing pipeline."""
    
    def test_preprocessor_saves_weights(self, tmp_path):
        """Test that NSDPreprocessor saves reliability weights to disk."""
        # This is a placeholder for integration testing
        # Actual test would require mocking NSDPreprocessor.fit()
        # and checking that reliability_weights.npy is created
        pass
    
    def test_transform_applies_sqrt_weights(self):
        """Test that transform_T1 applies sqrt(weights) correctly."""
        # This is a placeholder for integration testing
        # Actual test would create a mock preprocessor with known weights
        # and verify that output has correct scaling
        pass
    
    def test_ablation_study_compatibility(self):
        """Test that different reliability modes produce compatible outputs."""
        # This is a placeholder for ablation testing
        # Would verify that models can be trained with different modes
        # and results can be fairly compared
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
