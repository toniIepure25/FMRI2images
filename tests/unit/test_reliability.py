"""
Unit Tests for Split-Half Reliability Module
=============================================

Tests the robust split-half reliability estimator used for voxel selection.
"""

import numpy as np
import pytest
from fmri2img.data.reliability import (
    compute_split_half_reliability,
    filter_voxels_by_reliability
)


class TestComputeSplitHalfReliability:
    """Test suite for compute_split_half_reliability function."""
    
    def test_basic_functionality_with_repeats(self):
        """Test basic split-half computation with repeated stimuli."""
        np.random.seed(42)
        
        # Create synthetic data: 10 stimuli, 3 repeats each, 100 voxels
        n_stim = 10
        n_repeats = 3
        n_voxels = 100
        
        # Generate signal voxels (50) and noise voxels (50)
        X_list = []
        nsd_ids_list = []
        
        for stim_id in range(n_stim):
            # Signal for this stimulus (consistent across repeats)
            base_signal = np.random.randn(50)
            
            for rep in range(n_repeats):
                # Signal voxels: base + small noise
                signal_voxels = base_signal + np.random.randn(50) * 0.1
                
                # Noise voxels: pure random
                noise_voxels = np.random.randn(50)
                
                trial = np.concatenate([signal_voxels, noise_voxels])
                X_list.append(trial)
                nsd_ids_list.append(stim_id)
        
        X = np.array(X_list, dtype=np.float32)  # (30, 100)
        nsd_ids = np.array(nsd_ids_list, dtype=np.int32)
        
        # Compute split-half reliability
        r, meta = compute_split_half_reliability(X, nsd_ids, seed=42, min_repeats=2)
        
        # Check output shape
        assert r.shape == (n_voxels,), f"Expected shape ({n_voxels},), got {r.shape}"
        
        # Check metadata
        assert meta["n_ids_with_repeats"] == n_stim
        assert meta["n_repeatable_trials"] == 30
        assert len(meta["ids_used"]) == n_stim
        
        # Signal voxels should have higher reliability than noise voxels
        signal_r = r[:50]
        noise_r = r[50:]
        
        assert np.mean(signal_r) > np.mean(noise_r), \
            f"Signal voxels (mean r={np.mean(signal_r):.3f}) should have higher r than noise voxels (mean r={np.mean(noise_r):.3f})"
        
        # Most signal voxels should have positive r
        assert np.sum(signal_r > 0) > 40, f"Expected >40 signal voxels with r>0, got {np.sum(signal_r > 0)}"
    
    def test_handles_no_repeats(self):
        """Test that function handles case with no repeated stimuli."""
        X = np.random.randn(10, 50).astype(np.float32)
        nsd_ids = np.arange(10, dtype=np.int32)  # All unique IDs
        
        r, meta = compute_split_half_reliability(X, nsd_ids, seed=42, min_repeats=2)
        
        # Should return zeros when no repeats
        assert r.shape == (50,)
        assert np.all(r == 0), "Expected all zeros when no repeats"
        assert meta["n_ids_with_repeats"] == 0
        assert meta["n_repeatable_trials"] == 0
    
    def test_handles_insufficient_repeats(self):
        """Test that function requires min_repeats presentations."""
        X = np.random.randn(20, 50).astype(np.float32)
        
        # 10 stimuli with 2 repeats each
        nsd_ids = np.repeat(np.arange(10), 2).astype(np.int32)
        
        # min_repeats=3 should return zeros
        r, meta = compute_split_half_reliability(X, nsd_ids, seed=42, min_repeats=3)
        
        assert np.all(r == 0), "Expected zeros when repeats < min_repeats"
        assert meta["n_ids_with_repeats"] == 0
    
    def test_handles_odd_trial_counts(self):
        """Test that function handles odd number of trials per stimulus."""
        X_list = []
        nsd_ids_list = []
        
        # Stimulus 0: 5 trials (odd)
        base_signal = np.array([1.0, 2.0, 3.0])
        for _ in range(5):
            trial = base_signal + np.random.randn(3) * 0.1
            X_list.append(trial)
            nsd_ids_list.append(0)
        
        X = np.array(X_list, dtype=np.float32)
        nsd_ids = np.array(nsd_ids_list, dtype=np.int32)
        
        r, meta = compute_split_half_reliability(X, nsd_ids, seed=42, min_repeats=2)
        
        assert r.shape == (3,)
        assert meta["n_ids_with_repeats"] == 1
        # Should handle 2 vs 3 split (or 3 vs 2, depending on shuffle)
    
    def test_perfect_signal_high_correlation(self):
        """Test that perfect signal (no noise) gives r ≈ 1.0."""
        X_list = []
        nsd_ids_list = []
        
        # Perfect signal: each stimulus has its own unique response,
        # and that response is perfectly consistent across repeats
        for stim_id in range(5):
            # Each stimulus has a unique signal
            base_signal = np.arange(5, dtype=np.float32) + stim_id * 10.0
            
            for _ in range(4):  # 4 repeats
                # No noise added - perfectly consistent
                X_list.append(base_signal.copy())
                nsd_ids_list.append(stim_id)
        
        X = np.array(X_list, dtype=np.float32)
        nsd_ids = np.array(nsd_ids_list, dtype=np.int32)
        
        r, meta = compute_split_half_reliability(X, nsd_ids, seed=42, min_repeats=2)
        
        # All voxels should have r close to 1.0
        # (might not be exactly 1.0 due to the nature of split-half correlation)
        assert np.all(r > 0.90), f"Expected r > 0.90 for perfect signal, got min r={r.min():.3f}, mean r={r.mean():.3f}"
    
    def test_zero_variance_voxels(self):
        """Test handling of voxels with zero variance."""
        X = np.zeros((20, 50), dtype=np.float32)
        
        # 10 stimuli, 2 repeats each
        nsd_ids = np.repeat(np.arange(10), 2).astype(np.int32)
        
        r, meta = compute_split_half_reliability(X, nsd_ids, seed=42, min_repeats=2)
        
        # Zero variance should give r=0
        assert np.all(r == 0), "Expected r=0 for zero variance voxels"
    
    def test_reproducibility_with_seed(self):
        """Test that same seed gives identical results."""
        np.random.seed(123)
        X = np.random.randn(30, 50).astype(np.float32)
        nsd_ids = np.repeat(np.arange(10), 3).astype(np.int32)
        
        r1, meta1 = compute_split_half_reliability(X, nsd_ids, seed=42, min_repeats=2)
        r2, meta2 = compute_split_half_reliability(X, nsd_ids, seed=42, min_repeats=2)
        
        np.testing.assert_array_equal(r1, r2, err_msg="Same seed should give identical results")
        assert meta1["ids_used"] == meta2["ids_used"]
    
    def test_different_seeds_give_different_results(self):
        """Test that different seeds give different (but valid) results."""
        np.random.seed(456)
        X = np.random.randn(30, 50).astype(np.float32)
        nsd_ids = np.repeat(np.arange(10), 3).astype(np.int32)
        
        r1, _ = compute_split_half_reliability(X, nsd_ids, seed=42, min_repeats=2)
        r2, _ = compute_split_half_reliability(X, nsd_ids, seed=999, min_repeats=2)
        
        # Should be different due to random splits
        assert not np.allclose(r1, r2), "Different seeds should give different results"
        
        # But both should have valid ranges
        assert np.all((r1 >= -1) & (r1 <= 1))
        assert np.all((r2 >= -1) & (r2 <= 1))


class TestFilterVoxelsByReliability:
    """Test suite for filter_voxels_by_reliability function."""
    
    def test_basic_filtering(self):
        """Test basic filtering with reliability and variance thresholds."""
        n_voxels = 100
        
        # Create reliability values: half high, half low
        r = np.concatenate([
            np.random.uniform(0.3, 0.9, 50),  # High reliability
            np.random.uniform(-0.5, 0.05, 50)  # Low reliability
        ])
        
        # Create variance: all above threshold
        voxel_variance = np.ones(n_voxels) * 1e-3
        
        mask, stats = filter_voxels_by_reliability(
            r=r,
            voxel_variance=voxel_variance,
            reliability_thr=0.1,
            min_var=1e-6
        )
        
        assert mask.shape == (n_voxels,)
        assert mask.dtype == bool
        
        # Should retain approximately first 50 voxels (high r)
        assert stats["n_retained"] > 40, f"Expected ~50 retained, got {stats['n_retained']}"
        assert stats["n_retained"] < 60
        
        # Mean r of retained should be higher
        assert stats["mean_r_retained"] > stats["mean_r_rejected"]
    
    def test_variance_threshold_filtering(self):
        """Test that low variance voxels are rejected regardless of reliability."""
        n_voxels = 50
        
        # All voxels have high reliability
        r = np.ones(n_voxels) * 0.8
        
        # Half have low variance
        voxel_variance = np.concatenate([
            np.ones(25) * 1e-3,  # High variance
            np.ones(25) * 1e-9   # Low variance (below threshold)
        ])
        
        mask, stats = filter_voxels_by_reliability(
            r=r,
            voxel_variance=voxel_variance,
            reliability_thr=0.1,
            min_var=1e-6
        )
        
        # Should only retain first 25 (high variance)
        assert stats["n_retained"] == 25, f"Expected 25 retained, got {stats['n_retained']}"
        assert np.all(mask[:25])  # First 25 should be True
        assert not np.any(mask[25:])  # Last 25 should be False
    
    def test_combined_thresholding(self):
        """Test combined reliability AND variance thresholding."""
        # 4 groups: high r & high var, high r & low var, low r & high var, low r & low var
        r = np.array([0.5, 0.5, 0.05, 0.05])
        voxel_variance = np.array([1e-3, 1e-9, 1e-3, 1e-9])
        
        mask, stats = filter_voxels_by_reliability(
            r=r,
            voxel_variance=voxel_variance,
            reliability_thr=0.1,
            min_var=1e-6
        )
        
        # Only first voxel should pass (high r AND high var)
        expected_mask = np.array([True, False, False, False])
        np.testing.assert_array_equal(mask, expected_mask)
        assert stats["n_retained"] == 1
    
    def test_all_voxels_pass(self):
        """Test case where all voxels pass thresholds."""
        n_voxels = 30
        r = np.ones(n_voxels) * 0.8
        voxel_variance = np.ones(n_voxels) * 1e-3
        
        mask, stats = filter_voxels_by_reliability(
            r=r,
            voxel_variance=voxel_variance,
            reliability_thr=0.1,
            min_var=1e-6
        )
        
        assert stats["n_retained"] == n_voxels
        assert stats["retention_rate"] == 1.0
        assert np.all(mask)
    
    def test_no_voxels_pass(self):
        """Test case where no voxels pass thresholds."""
        n_voxels = 30
        r = np.ones(n_voxels) * 0.05  # All below threshold
        voxel_variance = np.ones(n_voxels) * 1e-3
        
        mask, stats = filter_voxels_by_reliability(
            r=r,
            voxel_variance=voxel_variance,
            reliability_thr=0.1,
            min_var=1e-6
        )
        
        assert stats["n_retained"] == 0
        assert stats["retention_rate"] == 0.0
        assert not np.any(mask)
        
        # mean_r_retained should be NaN when nothing retained
        assert np.isnan(stats["mean_r_retained"])
    
    def test_statistics_correctness(self):
        """Test that returned statistics are computed correctly."""
        r = np.array([0.8, 0.6, 0.4, 0.2, 0.0])
        voxel_variance = np.ones(5) * 1e-3
        
        mask, stats = filter_voxels_by_reliability(
            r=r,
            voxel_variance=voxel_variance,
            reliability_thr=0.5,
            min_var=1e-6
        )
        
        # Should retain first 2 voxels (r >= 0.5)
        assert stats["n_retained"] == 2
        assert stats["n_rejected"] == 3
        assert stats["retention_rate"] == 0.4
        
        # Check mean r values
        assert stats["mean_r_retained"] == pytest.approx(0.7)  # (0.8 + 0.6) / 2
        assert stats["mean_r_rejected"] == pytest.approx(0.2)  # (0.4 + 0.2 + 0.0) / 3
        
        # Check median r values
        assert stats["median_r_retained"] == pytest.approx(0.7)  # median of [0.8, 0.6]
        assert stats["median_r_rejected"] == pytest.approx(0.2)  # median of [0.4, 0.2, 0.0]


class TestIntegrationScenarios:
    """Integration tests simulating real-world scenarios."""
    
    def test_typical_nsd_scenario(self):
        """Simulate typical NSD scenario with 3 repeats per stimulus."""
        np.random.seed(789)
        
        # 50 stimuli, 3 repeats each = 150 trials
        # 1000 voxels: 700 noisy, 300 signal
        n_stim = 50
        n_repeats = 3
        n_signal = 300
        n_noise = 700
        n_voxels = n_signal + n_noise
        
        X_list = []
        nsd_ids_list = []
        
        for stim_id in range(n_stim):
            base_signal = np.random.randn(n_signal)
            
            for _ in range(n_repeats):
                # Signal voxels: consistent signal + small noise
                signal = base_signal + np.random.randn(n_signal) * 0.2
                
                # Noise voxels: pure noise
                noise = np.random.randn(n_noise)
                
                trial = np.concatenate([signal, noise])
                X_list.append(trial)
                nsd_ids_list.append(stim_id)
        
        X = np.array(X_list, dtype=np.float32)
        nsd_ids = np.array(nsd_ids_list, dtype=np.int32)
        
        # Compute reliability
        r, meta = compute_split_half_reliability(X, nsd_ids, seed=42, min_repeats=2)
        
        # Check that we used all stimuli
        assert meta["n_ids_with_repeats"] == n_stim
        
        # Filter with typical threshold
        voxel_variance = np.var(X, axis=0)
        mask, stats = filter_voxels_by_reliability(
            r=r,
            voxel_variance=voxel_variance,
            reliability_thr=0.1,
            min_var=1e-6
        )
        
        # Should retain more signal voxels than noise voxels
        signal_retained = np.sum(mask[:n_signal])
        noise_retained = np.sum(mask[n_signal:])
        
        assert signal_retained > noise_retained, \
            f"Signal voxels retained ({signal_retained}) should exceed noise voxels retained ({noise_retained})"
        
        # Retention rate should be reasonable (10-50%)
        assert 0.1 <= stats["retention_rate"] <= 0.5, \
            f"Retention rate {stats['retention_rate']:.2%} seems unreasonable"
    
    def test_fallback_scenario_insufficient_repeats(self):
        """Test that system gracefully handles insufficient repeats."""
        # Only 5 stimuli with repeats (below typical min_repeat_ids=20)
        X = np.random.randn(10, 100).astype(np.float32)
        nsd_ids = np.repeat(np.arange(5), 2).astype(np.int32)
        
        r, meta = compute_split_half_reliability(X, nsd_ids, seed=42, min_repeats=2)
        
        # Should still compute reliability for the 5 stimuli
        assert meta["n_ids_with_repeats"] == 5
        
        # But in actual preprocessing, this would trigger variance fallback
        # (that check happens in preprocess.py, not in the reliability module)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
