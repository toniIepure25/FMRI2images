"""
Sanity Tests for Evaluation Suite
==================================

Critical tests to ensure evaluation metrics are correct and catch common bugs.

Run with: pytest tests/test_evaluation_sanity.py -v

Author: Research-grade evaluation suite
Date: January 2026
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fmri2img.eval.embedding_eval import (
    normalize_embeddings,
    compute_retrieval_metrics,
    compute_two_afc,
    compute_separability_auc,
    compute_rsa,
    compute_collapse_diagnostics,
)

from fmri2img.eval.probabilistic_eval import (
    gaussian_nll,
    energy_score,
    bayesian_retrieval,
    calibration_analysis,
)


class TestEmbeddingEvalSanity:
    """Sanity tests for standard embedding evaluation."""
    
    def test_oracle_retrieval(self):
        """
        CRITICAL: Oracle retrieval sanity check.
        
        Using ground truth embeddings as predictions should give:
        - Top-1 accuracy = 100%
        - Mean rank = 1.0
        
        If this fails, there's a bug in indexing/matching logic.
        """
        N, D = 100, 768
        np.random.seed(42)
        
        # Ground truth embeddings
        gt_embeddings = np.random.randn(N, D)
        gt_embeddings = normalize_embeddings(gt_embeddings)
        
        # Use GT as predictions (oracle)
        predictions = gt_embeddings.copy()
        
        # Compute metrics
        metrics = compute_retrieval_metrics(predictions, gt_embeddings, normalize=False)
        
        # Assertions
        assert metrics['top1_accuracy'] == 1.0, \
            f"Oracle Top-1 should be 1.0, got {metrics['top1_accuracy']}"
        assert metrics['top5_accuracy'] == 1.0
        assert metrics['top10_accuracy'] == 1.0
        assert abs(metrics['mean_rank'] - 1.0) < 0.01, \
            f"Oracle mean rank should be 1.0, got {metrics['mean_rank']}"
        assert abs(metrics['median_rank'] - 1.0) < 0.01
        
        print("✓ Oracle retrieval sanity check passed")
    
    def test_random_retrieval_near_chance(self):
        """
        Random predictions should give near-chance retrieval.
        
        For N=1000:
        - Top-1 accuracy ≈ 0.001 (1/N)
        - Mean rank ≈ N/2 = 500
        """
        N, D = 1000, 768
        np.random.seed(42)
        
        predictions = np.random.randn(N, D)
        ground_truth = np.random.randn(N, D)
        
        predictions = normalize_embeddings(predictions)
        ground_truth = normalize_embeddings(ground_truth)
        
        metrics = compute_retrieval_metrics(predictions, ground_truth, normalize=False)
        
        # Should be near chance
        assert metrics['top1_accuracy'] < 0.01, \
            f"Random Top-1 should be near 0.001, got {metrics['top1_accuracy']}"
        
        # Mean rank should be roughly N/2
        expected_rank = N / 2
        assert abs(metrics['mean_rank'] - expected_rank) < N * 0.2, \
            f"Random mean rank should be ~{expected_rank}, got {metrics['mean_rank']}"
        
        print("✓ Random retrieval near-chance check passed")
    
    def test_two_afc_oracle(self):
        """Oracle 2AFC should give 100% accuracy."""
        N, D = 100, 768
        np.random.seed(42)
        
        gt = np.random.randn(N, D)
        gt = normalize_embeddings(gt)
        
        predictions = gt.copy()
        
        metrics = compute_two_afc(predictions, gt, n_trials=500, normalize=False, seed=42)
        
        assert metrics['accuracy'] == 1.0, \
            f"Oracle 2AFC should be 1.0, got {metrics['accuracy']}"
        
        print("✓ Oracle 2AFC sanity check passed")
    
    def test_two_afc_random_near_chance(self):
        """Random 2AFC should be near 50% (chance)."""
        N, D = 100, 768
        np.random.seed(42)
        
        predictions = np.random.randn(N, D)
        ground_truth = np.random.randn(N, D)
        
        predictions = normalize_embeddings(predictions)
        ground_truth = normalize_embeddings(ground_truth)
        
        metrics = compute_two_afc(predictions, ground_truth, n_trials=5000, normalize=False, seed=42)
        
        # Should be close to 0.5 (within 99% CI for 5000 trials: ±4%)
        # With more trials, we get tighter bounds
        assert 0.40 < metrics['accuracy'] < 0.60, \
            f"Random 2AFC should be ~0.5, got {metrics['accuracy']}"
        
        print("✓ Random 2AFC near-chance check passed")
    
    def test_separability_auc_oracle(self):
        """Oracle separability should give AUC = 1.0."""
        N, D = 100, 768
        np.random.seed(42)
        
        gt = np.random.randn(N, D)
        gt = normalize_embeddings(gt)
        
        predictions = gt.copy()
        
        metrics = compute_separability_auc(predictions, gt, normalize=False, seed=42)
        
        assert metrics['auc'] > 0.99, \
            f"Oracle AUC should be ~1.0, got {metrics['auc']}"
        
        print("✓ Oracle separability sanity check passed")
    
    def test_collapse_detection(self):
        """
        Collapsed predictions (all same) should be detected.
        """
        N, D = 100, 768
        np.random.seed(42)
        
        ground_truth = np.random.randn(N, D)
        ground_truth = normalize_embeddings(ground_truth)
        
        # Collapsed predictions: all vectors are the same
        collapsed = np.ones((N, D))
        collapsed = normalize_embeddings(collapsed)
        
        metrics = compute_collapse_diagnostics(collapsed, ground_truth, normalize=False)
        
        # Should detect collapse
        assert metrics['collapse_ratio'] < 0.1, \
            f"Collapsed predictions should have low ratio, got {metrics['collapse_ratio']}"
        
        # High pairwise similarity (all vectors identical)
        assert metrics['avg_pairwise_sim'] > 0.99, \
            f"Collapsed predictions should have high pairwise sim, got {metrics['avg_pairwise_sim']}"
        
        print("✓ Collapse detection check passed")
    
    def test_normalization_invariance(self):
        """
        Cosine-based metrics should be invariant to L2 normalization.
        """
        N, D = 50, 768
        np.random.seed(42)
        
        predictions = np.random.randn(N, D)
        ground_truth = np.random.randn(N, D)
        
        # Compute with and without normalization
        metrics_with_norm = compute_retrieval_metrics(predictions, ground_truth, normalize=True)
        
        # Pre-normalize
        pred_norm = normalize_embeddings(predictions)
        gt_norm = normalize_embeddings(ground_truth)
        metrics_without_norm = compute_retrieval_metrics(pred_norm, gt_norm, normalize=False)
        
        # Should be identical
        assert abs(metrics_with_norm['top1_accuracy'] - metrics_without_norm['top1_accuracy']) < 1e-6
        assert abs(metrics_with_norm['mean_rank'] - metrics_without_norm['mean_rank']) < 1e-6
        
        print("✓ Normalization invariance check passed")


class TestProbabilisticEvalSanity:
    """Sanity tests for probabilistic evaluation."""
    
    def test_gaussian_nll_correctness(self):
        """
        Test NLL computation for known case.
        
        For z ~ N(μ, σ²), with z = μ and σ = 1:
        NLL = 0.5 * [D * log(2π) + D * log(1) + 0] = 0.5 * D * log(2π)
        """
        D = 768
        N = 10
        
        mu = np.random.randn(N, D)
        z_gt = mu.copy()  # z = μ (perfect prediction)
        logvar = np.zeros((N, D))  # σ² = 1 → log(σ²) = 0
        
        nll = gaussian_nll(z_gt, mu, logvar)
        
        expected_nll = 0.5 * D * np.log(2 * np.pi)
        
        assert np.allclose(nll, expected_nll), \
            f"NLL should be {expected_nll:.2f}, got {nll[0]:.2f}"
        
        print("✓ Gaussian NLL correctness check passed")
    
    def test_bayesian_retrieval_oracle(self):
        """
        Oracle Bayesian retrieval: μ = z_gt, small variance.
        Should give Top-1 = 100%.
        """
        N, D = 50, 768
        np.random.seed(42)
        
        z_gt = np.random.randn(N, D)
        z_gt = normalize_embeddings(z_gt)
        
        # Oracle: μ = z_gt, small variance
        mu = z_gt.copy()
        logvar = np.full((N, D), -10.0)  # Very small variance
        
        metrics = bayesian_retrieval(mu, logvar, z_gt)
        
        assert metrics['top1_accuracy'] > 0.95, \
            f"Oracle Bayesian Top-1 should be ~1.0, got {metrics['top1_accuracy']}"
        
        print("✓ Oracle Bayesian retrieval sanity check passed")
    
    def test_calibration_oracle(self):
        """
        Oracle calibration: samples drawn from N(μ, σ²) should be well-calibrated.
        """
        N, D = 1000, 768
        np.random.seed(42)
        
        mu = np.random.randn(N, D)
        logvar = np.random.uniform(-2, 0, (N, D))  # Reasonable variances
        
        # Sample z_gt from the model distribution (oracle scenario)
        std = np.sqrt(np.exp(logvar))
        z_gt = mu + std * np.random.randn(N, D)
        
        # Calibration should be near-perfect
        calib = calibration_analysis(z_gt, mu, logvar)
        
        # Check empirical coverage is close to nominal
        for nominal, empirical in calib['coverage'].items():
            deviation = abs(empirical - nominal)
            assert deviation < 0.05, \
                f"Calibration deviation for {nominal:.0%} coverage should be <5%, got {deviation:.1%}"
        
        print("✓ Oracle calibration check passed")
    
    def test_energy_score_zero_for_deterministic(self):
        """
        When all samples are identical (deterministic), energy score should be E||z - z_gt||.
        """
        N, D, S = 10, 768, 20
        np.random.seed(42)
        
        z_gt = np.random.randn(N, D)
        z_gt = normalize_embeddings(z_gt)
        
        # All samples are the same (deterministic)
        samples = np.tile(z_gt[:, None, :], (1, S, 1))  # (N, S, D)
        
        es = energy_score(z_gt, samples)
        
        # Energy score should be 0 (perfect match, no diversity penalty)
        # E||z - z_gt|| = 0, E||z - z'|| = 0 → ES = 0
        assert np.allclose(es, 0.0, atol=1e-6), \
            f"Energy score for deterministic should be ~0, got {es[0]:.3f}"
        
        print("✓ Energy score deterministic check passed")


class TestReproducibility:
    """Test reproducibility with fixed seeds."""
    
    def test_retrieval_reproducibility(self):
        """Same seed should give same results."""
        N, D = 100, 768
        
        predictions = np.random.randn(N, D)
        ground_truth = np.random.randn(N, D)
        
        metrics1 = compute_two_afc(predictions, ground_truth, n_trials=100, seed=42)
        metrics2 = compute_two_afc(predictions, ground_truth, n_trials=100, seed=42)
        
        assert metrics1['accuracy'] == metrics2['accuracy'], \
            "Results should be reproducible with fixed seed"
        
        print("✓ Reproducibility check passed")


def run_all_sanity_tests():
    """Run all sanity tests and report results."""
    print("\n" + "="*60)
    print("RUNNING EVALUATION SANITY TESTS")
    print("="*60 + "\n")
    
    test_embedding = TestEmbeddingEvalSanity()
    test_embedding.test_oracle_retrieval()
    test_embedding.test_random_retrieval_near_chance()
    test_embedding.test_two_afc_oracle()
    test_embedding.test_two_afc_random_near_chance()
    test_embedding.test_separability_auc_oracle()
    test_embedding.test_collapse_detection()
    test_embedding.test_normalization_invariance()
    
    test_prob = TestProbabilisticEvalSanity()
    test_prob.test_gaussian_nll_correctness()
    test_prob.test_bayesian_retrieval_oracle()
    test_prob.test_calibration_oracle()
    test_prob.test_energy_score_zero_for_deterministic()
    
    test_repro = TestReproducibility()
    test_repro.test_retrieval_reproducibility()
    
    print("\n" + "="*60)
    print("✅ ALL SANITY TESTS PASSED!")
    print("="*60 + "\n")


if __name__ == '__main__':
    # Can run directly or with pytest
    run_all_sanity_tests()
