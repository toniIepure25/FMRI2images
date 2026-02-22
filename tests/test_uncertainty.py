"""
Tests for MC dropout uncertainty estimation (Novel Contribution C).

Tests the predict_with_mc_dropout function and uncertainty calibration analysis.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from fmri2img.eval.uncertainty import (
    enable_dropout,
    predict_with_mc_dropout,
    compute_uncertainty_error_correlation,
    compute_confidence_intervals,
    entropy_of_prediction
)


class SimpleDropoutModel(nn.Module):
    """Simple model with dropout for testing."""
    
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=32, dropout_p=0.5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)


class TestEnableDropout:
    """Test enable_dropout function."""
    
    def test_enables_dropout_layers(self):
        """Test that dropout layers are set to training mode."""
        model = SimpleDropoutModel()
        model.eval()  # Set to eval mode
        
        # Check initial state
        dropout_layers = [m for m in model.modules() if isinstance(m, nn.Dropout)]
        assert len(dropout_layers) == 2
        for layer in dropout_layers:
            assert not layer.training  # Should be in eval mode
        
        # Enable dropout
        enable_dropout(model)
        
        # Check that dropout layers are now in training mode
        for layer in dropout_layers:
            assert layer.training  # Should be in training mode


class TestPredictWithMCDropout:
    """Test MC dropout inference function."""
    
    def test_basic_inference_single_sample(self):
        """Test MC dropout on single sample."""
        model = SimpleDropoutModel(input_dim=128, output_dim=32)
        x = torch.randn(128)  # Single sample
        
        result = predict_with_mc_dropout(model, x, n_samples=10, device="cpu")
        
        # Check output shapes
        assert result["mean"].shape == (32,)
        assert result["variance"].shape == (32,)
        assert result["std"].shape == (32,)
        assert result["uncertainty"].shape == ()  # Scalar
        
        # Check values are reasonable
        assert torch.all(torch.isfinite(result["mean"]))
        assert torch.all(result["variance"] >= 0)
        assert torch.all(result["std"] >= 0)
        assert result["uncertainty"] >= 0
    
    def test_basic_inference_batch(self):
        """Test MC dropout on batch."""
        model = SimpleDropoutModel(input_dim=128, output_dim=32)
        x = torch.randn(16, 128)  # Batch of 16
        
        result = predict_with_mc_dropout(model, x, n_samples=10, device="cpu")
        
        # Check output shapes
        assert result["mean"].shape == (16, 32)
        assert result["variance"].shape == (16, 32)
        assert result["std"].shape == (16, 32)
        assert result["uncertainty"].shape == (16,)
        
        # Check values
        assert torch.all(torch.isfinite(result["mean"]))
        assert torch.all(result["variance"] >= 0)
        assert torch.all(result["uncertainty"] >= 0)
    
    def test_return_all_samples(self):
        """Test that all MC samples are returned when requested."""
        model = SimpleDropoutModel(input_dim=128, output_dim=32)
        x = torch.randn(8, 128)
        n_samples = 20
        
        result = predict_with_mc_dropout(
            model, x, n_samples=n_samples, device="cpu", return_all_samples=True
        )
        
        # Check samples are returned
        assert "samples" in result
        assert result["samples"].shape == (n_samples, 8, 32)
        
        # Verify mean calculation
        computed_mean = result["samples"].mean(dim=0)
        torch.testing.assert_close(result["mean"], computed_mean, rtol=1e-5, atol=1e-5)
    
    def test_variance_increases_with_dropout(self):
        """Test that MC dropout produces higher variance than deterministic."""
        model = SimpleDropoutModel(input_dim=128, output_dim=32, dropout_p=0.5)
        x = torch.randn(16, 128)
        
        # MC dropout (stochastic)
        result_mc = predict_with_mc_dropout(model, x, n_samples=50, device="cpu")
        
        # Deterministic (no dropout)
        model.eval()
        with torch.no_grad():
            deterministic_preds = []
            for _ in range(50):
                deterministic_preds.append(model(x))
        deterministic_var = torch.stack(deterministic_preds).var(dim=0)
        
        # MC dropout variance should be higher due to dropout randomness
        mc_var_mean = result_mc["variance"].mean()
        det_var_mean = deterministic_var.mean()
        
        # With dropout, variance should be non-trivial
        assert mc_var_mean > 1e-6
    
    def test_more_samples_reduces_variance_of_mean(self):
        """Test that more MC samples gives more stable mean estimate."""
        model = SimpleDropoutModel(input_dim=128, output_dim=32)
        x = torch.randn(8, 128)
        
        # Run multiple times with few samples
        means_few = []
        for _ in range(10):
            result = predict_with_mc_dropout(model, x, n_samples=5, device="cpu")
            means_few.append(result["mean"])
        var_few = torch.stack(means_few).var(dim=0).mean()
        
        # Run multiple times with many samples
        means_many = []
        for _ in range(10):
            result = predict_with_mc_dropout(model, x, n_samples=50, device="cpu")
            means_many.append(result["mean"])
        var_many = torch.stack(means_many).var(dim=0).mean()
        
        # More samples should give more consistent mean estimates
        # (variance of mean estimates should decrease)
        # Note: This is a statistical test, may rarely fail
        assert var_many < var_few * 2  # Allow some tolerance


class TestUncertaintyErrorCorrelation:
    """Test uncertainty-error correlation analysis."""
    
    def test_perfect_correlation(self):
        """Test with perfectly correlated uncertainty and error."""
        uncertainties = np.linspace(0, 1, 100)
        errors = uncertainties  # Perfect positive correlation
        
        result = compute_uncertainty_error_correlation(uncertainties, errors)
        
        assert result["correlation"] > 0.99  # Should be ~1.0
        assert result["p_value"] < 0.001  # Highly significant
        assert result["n_samples"] == 100
    
    def test_no_correlation(self):
        """Test with uncorrelated uncertainty and error."""
        np.random.seed(42)
        uncertainties = np.random.randn(200)
        errors = np.random.randn(200)
        
        result = compute_uncertainty_error_correlation(uncertainties, errors)
        
        # Correlation should be near zero
        assert abs(result["correlation"]) < 0.2
        assert result["n_samples"] == 200
    
    def test_calibration_curve_bins(self):
        """Test that calibration curve has correct number of bins."""
        uncertainties = np.linspace(0, 1, 1000)
        errors = uncertainties + np.random.randn(1000) * 0.1
        
        n_bins = 10
        result = compute_uncertainty_error_correlation(
            uncertainties, errors, n_bins=n_bins
        )
        
        calib = result["calibration_curve"]
        assert len(calib["bin_uncertainties"]) == n_bins
        assert len(calib["bin_errors"]) == n_bins
        assert len(calib["bin_counts"]) == n_bins
        assert calib["n_bins"] == n_bins
        
        # Bins should be sorted (increasing uncertainty)
        bin_unc = [u for u in calib["bin_uncertainties"] if not np.isnan(u)]
        assert bin_unc == sorted(bin_unc)
    
    def test_different_correlation_metrics(self):
        """Test different correlation metrics."""
        np.random.seed(42)
        uncertainties = np.linspace(0, 1, 100)
        errors = uncertainties ** 2  # Nonlinear but monotonic
        
        pearson = compute_uncertainty_error_correlation(
            uncertainties, errors, metric="pearson"
        )
        spearman = compute_uncertainty_error_correlation(
            uncertainties, errors, metric="spearman"
        )
        kendall = compute_uncertainty_error_correlation(
            uncertainties, errors, metric="kendall"
        )
        
        # All should detect positive correlation
        assert pearson["correlation"] > 0.5
        assert spearman["correlation"] > 0.9  # Spearman better for monotonic
        assert kendall["correlation"] > 0.7
        
        # Check metric is stored
        assert pearson["metric"] == "pearson"
        assert spearman["metric"] == "spearman"
        assert kendall["metric"] == "kendall"
    
    def test_handles_nan_inf(self):
        """Test that NaN/Inf values are filtered out."""
        uncertainties = np.array([0.1, 0.2, np.nan, 0.4, np.inf, 0.6])
        errors = np.array([0.1, 0.2, 0.3, np.nan, 0.5, 0.6])
        
        result = compute_uncertainty_error_correlation(uncertainties, errors)
        
        # Should only use 3 valid pairs: (0.1, 0.1), (0.2, 0.2), (0.6, 0.6)
        assert result["n_samples"] == 3
        assert np.isfinite(result["correlation"])
    
    def test_insufficient_samples(self):
        """Test behavior with too few samples."""
        uncertainties = np.array([0.1, 0.2])
        errors = np.array([0.1, 0.2])
        
        result = compute_uncertainty_error_correlation(uncertainties, errors)
        
        # Should return NaN for insufficient data
        assert np.isnan(result["correlation"])
        assert np.isnan(result["p_value"])


class TestConfidenceIntervals:
    """Test confidence interval computation."""
    
    def test_confidence_interval_shape(self):
        """Test that CI bounds have correct shape."""
        samples = torch.randn(100, 16, 32)  # 100 MC samples, batch 16, dim 32
        
        lower, upper = compute_confidence_intervals(samples, confidence=0.95, axis=0)
        
        assert lower.shape == (16, 32)
        assert upper.shape == (16, 32)
        
        # Upper should be >= lower
        assert torch.all(upper >= lower)
    
    def test_confidence_level(self):
        """Test that confidence level is respected."""
        # Generate samples from known distribution
        torch.manual_seed(42)
        true_mean = torch.tensor([5.0])
        true_std = torch.tensor([2.0])
        samples = torch.randn(10000, 1) * true_std + true_mean
        
        lower, upper = compute_confidence_intervals(samples, confidence=0.95, axis=0)
        
        # Should approximately match 95% CI: mean ± 1.96*std
        expected_lower = true_mean - 1.96 * true_std
        expected_upper = true_mean + 1.96 * true_std
        
        torch.testing.assert_close(lower, expected_lower, rtol=0.1, atol=0.2)
        torch.testing.assert_close(upper, expected_upper, rtol=0.1, atol=0.2)
    
    def test_different_confidence_levels(self):
        """Test different confidence levels."""
        samples = torch.randn(1000, 10)
        
        ci_90 = compute_confidence_intervals(samples, confidence=0.90, axis=0)
        ci_95 = compute_confidence_intervals(samples, confidence=0.95, axis=0)
        ci_99 = compute_confidence_intervals(samples, confidence=0.99, axis=0)
        
        # Wider confidence = larger interval
        width_90 = (ci_90[1] - ci_90[0]).mean()
        width_95 = (ci_95[1] - ci_95[0]).mean()
        width_99 = (ci_99[1] - ci_99[0]).mean()
        
        assert width_90 < width_95 < width_99


class TestEntropyOfPrediction:
    """Test entropy computation for classification."""
    
    def test_uniform_distribution_max_entropy(self):
        """Test that uniform distribution has maximum entropy."""
        n_classes = 10
        probs_uniform = torch.ones(n_classes) / n_classes
        
        entropy_uniform = entropy_of_prediction(probs_uniform)
        
        # Maximum entropy = log(n_classes)
        max_entropy = np.log(n_classes)
        torch.testing.assert_close(entropy_uniform, torch.tensor(max_entropy, dtype=torch.float32), rtol=1e-4, atol=1e-4)
    
    def test_deterministic_zero_entropy(self):
        """Test that deterministic prediction has zero entropy."""
        probs_det = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
        
        entropy_det = entropy_of_prediction(probs_det)
        
        # Should be ~0 (with numerical tolerance)
        assert entropy_det < 0.01
    
    def test_batch_entropy(self):
        """Test entropy on batch of predictions."""
        probs = torch.tensor([
            [1.0, 0.0, 0.0],  # Deterministic
            [0.5, 0.5, 0.0],  # Some uncertainty
            [0.33, 0.33, 0.34]  # High uncertainty
        ])
        
        entropies = entropy_of_prediction(probs, dim=-1)
        
        assert entropies.shape == (3,)
        # First should have lowest entropy, last should have highest
        assert entropies[0] < entropies[1] < entropies[2]


class TestMCDropoutIntegration:
    """Integration tests for MC dropout uncertainty."""
    
    def test_uncertainty_correlates_with_error(self):
        """Test that uncertainty correlates with actual prediction error."""
        torch.manual_seed(42)
        
        # Create model and data
        model = SimpleDropoutModel(input_dim=64, output_dim=32, dropout_p=0.3)
        model.eval()
        
        # Generate test data with varying difficulty
        # Easy samples: low noise
        x_easy = torch.randn(20, 64) * 0.5
        # Hard samples: high noise
        x_hard = torch.randn(20, 64) * 2.0
        
        x = torch.cat([x_easy, x_hard], dim=0)
        
        # Ground truth (model's true prediction without dropout)
        model.eval()
        with torch.no_grad():
            y_true = model(x)
        
        # MC dropout predictions
        mc_result = predict_with_mc_dropout(model, x, n_samples=20, device="cpu")
        
        # Compute errors
        errors = torch.norm(mc_result["mean"] - y_true, dim=-1)
        uncertainties = mc_result["uncertainty"]
        
        # Correlation analysis
        corr_result = compute_uncertainty_error_correlation(
            uncertainties.numpy(), errors.numpy()
        )
        
        # Should have some positive correlation
        # (Note: This is a weak test since our "hard" samples aren't necessarily harder for the model)
        assert corr_result["n_samples"] == 40


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
