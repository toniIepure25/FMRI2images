"""Tests for vMF-compatible risk-coverage evaluation."""

import pytest
import numpy as np

from fmri2img.eval.vmf_risk_coverage import (
    compute_vmf_risk_coverage,
    compute_dual_risk_coverage,
    compute_hierarchical_selective,
    compare_uncertainty_orderings,
    RiskCoverageResult,
)


@pytest.fixture
def synthetic_data():
    """Well-calibrated synthetic data: higher kappa -> lower error."""
    np.random.seed(42)
    N = 200
    kappas = np.random.uniform(1, 200, N)
    errors = 1.0 / (1.0 + kappas / 50) + np.random.normal(0, 0.05, N)
    errors = np.clip(errors, 0, 2)
    deltas = np.random.uniform(0, 1, N)
    return errors, kappas, deltas


class TestVMFRiskCoverage:
    def test_output_shape(self, synthetic_data):
        errors, kappas, _ = synthetic_data
        result = compute_vmf_risk_coverage(errors, kappas, n_points=50)
        assert isinstance(result, RiskCoverageResult)
        assert len(result.coverages) == 50
        assert len(result.risks) == 50

    def test_risk_nondecreasing_with_coverage(self, synthetic_data):
        errors, kappas, _ = synthetic_data
        result = compute_vmf_risk_coverage(errors, kappas)
        # Risk should generally increase with coverage for well-calibrated
        # uncertainty (we're adding noisier samples)
        assert result.risks[-1] >= result.risks[1]

    def test_aurc_positive(self, synthetic_data):
        errors, kappas, _ = synthetic_data
        result = compute_vmf_risk_coverage(errors, kappas)
        assert result.aurc > 0

    def test_oracle_aurc_leq_aurc(self, synthetic_data):
        errors, kappas, _ = synthetic_data
        result = compute_vmf_risk_coverage(errors, kappas)
        assert result.oracle_aurc <= result.aurc + 1e-6

    def test_excess_aurc_nonnegative(self, synthetic_data):
        errors, kappas, _ = synthetic_data
        result = compute_vmf_risk_coverage(errors, kappas)
        assert result.e_aurc >= -1e-6

    def test_risk_at_coverages_valid(self, synthetic_data):
        errors, kappas, _ = synthetic_data
        result = compute_vmf_risk_coverage(errors, kappas)
        assert 0 <= result.r_at_80
        assert 0 <= result.r_at_90
        assert 0 <= result.r_at_95

    def test_perfect_ordering_low_excess(self):
        N = 100
        errors = np.arange(N, dtype=float) / N
        kappas = np.arange(N, 0, -1, dtype=float)  # Perfect reverse order
        result = compute_vmf_risk_coverage(errors, kappas)
        assert result.e_aurc < 0.01


class TestDualRiskCoverage:
    def test_output_shape(self, synthetic_data):
        errors, kappas, deltas = synthetic_data
        result = compute_dual_risk_coverage(errors, kappas, deltas)
        assert isinstance(result, RiskCoverageResult)
        assert result.aurc > 0

    def test_dual_vs_kappa_only(self, synthetic_data):
        errors, kappas, deltas = synthetic_data
        kappa_only = compute_vmf_risk_coverage(errors, kappas)
        dual = compute_dual_risk_coverage(errors, kappas, deltas)
        # Both should produce valid results
        assert kappa_only.aurc > 0
        assert dual.aurc > 0


class TestHierarchicalSelective:
    def test_output_keys(self, synthetic_data):
        errors, kappas, deltas = synthetic_data
        cat_errors = errors * 0.5  # Category is easier
        result = compute_hierarchical_selective(
            errors, cat_errors, kappas, kappa_threshold=0.5
        )
        assert "aurc_instance" in result
        assert "aurc_hierarchical" in result
        assert "instance_fraction" in result

    def test_hierarchical_leq_instance(self, synthetic_data):
        errors, kappas, _ = synthetic_data
        cat_errors = errors * 0.3  # Category much easier
        result = compute_hierarchical_selective(
            errors, cat_errors, kappas, kappa_threshold=0.5
        )
        assert result["aurc_hierarchical"] <= result["aurc_instance"] + 1e-6


class TestCompareOrderings:
    def test_multiple_orderings(self, synthetic_data):
        errors, kappas, deltas = synthetic_data
        orderings = {
            "kappa": kappas,
            "random": np.random.randn(len(errors)),
            "inverse_delta": 1.0 - deltas,
        }
        results = compare_uncertainty_orderings(errors, orderings)
        assert "kappa" in results
        assert "random" in results
        # Kappa ordering should generally be better than random
        assert results["kappa"].aurc <= results["random"].aurc + 0.1
