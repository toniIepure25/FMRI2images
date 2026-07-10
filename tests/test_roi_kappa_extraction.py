"""Tests for per-ROI kappa extraction and analysis module."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from fmri2img.eval.roi_kappa_extraction import (
    PerROIKappaResult,
    compute_category_conditional_kappa,
    compute_kappa_ncsnr_dissociation,
    compute_per_roi_calibration,
    compute_roi_kappa_topography,
    DEFAULT_ROI_NAMES,
    ROI_TIERS,
)


@pytest.fixture
def synthetic_result():
    """Create a synthetic PerROIKappaResult for testing."""
    np.random.seed(42)
    n_trials = 200
    n_rois = 17
    d_embed = 768

    # Simulate per-ROI kappas with some structure:
    # FFA regions (indices 9, 10) have higher kappa overall
    base_kappa = np.random.exponential(scale=20.0, size=(n_trials, n_rois))
    base_kappa[:, 9] += 30   # FFA1 boost
    base_kappa[:, 10] += 25  # FFA2 boost
    base_kappa[:, 11] += 20  # PPA boost
    base_kappa = np.clip(base_kappa, 1.0, 500.0)

    consensus = base_kappa.mean(axis=1) * 0.7
    delta = np.random.uniform(0.01, 0.3, size=(n_trials,))
    alphas = np.random.dirichlet(np.ones(n_rois), size=n_trials)

    per_roi_mus = np.random.randn(n_trials, n_rois, d_embed)
    per_roi_mus /= np.linalg.norm(per_roi_mus, axis=-1, keepdims=True)

    mu_fused = np.random.randn(n_trials, d_embed)
    mu_fused /= np.linalg.norm(mu_fused, axis=-1, keepdims=True)

    return PerROIKappaResult(
        per_roi_kappas=base_kappa.astype(np.float32),
        consensus_kappa=consensus.astype(np.float32),
        delta=delta.astype(np.float32),
        alphas=alphas.astype(np.float32),
        per_roi_mus=per_roi_mus.astype(np.float32),
        mu_fused=mu_fused.astype(np.float32),
        roi_names=list(DEFAULT_ROI_NAMES),
    )


class TestPerROIKappaResult:
    def test_post_init(self, synthetic_result):
        assert synthetic_result.n_trials == 200
        assert synthetic_result.n_rois == 17

    def test_shapes(self, synthetic_result):
        assert synthetic_result.per_roi_kappas.shape == (200, 17)
        assert synthetic_result.consensus_kappa.shape == (200,)
        assert synthetic_result.delta.shape == (200,)
        assert synthetic_result.alphas.shape == (200, 17)
        assert synthetic_result.per_roi_mus.shape == (200, 17, 768)


class TestTopography:
    def test_basic(self, synthetic_result):
        report = compute_roi_kappa_topography(synthetic_result)
        assert "roi_stats" in report
        assert "tier_stats" in report
        assert "ranked_rois" in report
        assert len(report["roi_stats"]) == 17

    def test_ffa_highest(self, synthetic_result):
        report = compute_roi_kappa_topography(synthetic_result)
        top_roi = report["ranked_rois"][0][0]
        assert top_roi in ("FFA1", "FFA2"), f"Expected FFA at top, got {top_roi}"

    def test_tier_stats(self, synthetic_result):
        report = compute_roi_kappa_topography(synthetic_result)
        assert "face_selective" in report["tier_stats"]
        face_kappa = report["tier_stats"]["face_selective"]["mean_kappa"]
        early_kappa = report["tier_stats"]["early_visual"]["mean_kappa"]
        assert face_kappa > early_kappa


class TestCategoryConditional:
    def test_basic(self, synthetic_result):
        categories = np.random.randint(0, 5, size=200)
        cat_names = ["face", "scene", "object", "animal", "food"]
        report = compute_category_conditional_kappa(
            synthetic_result, categories, cat_names
        )
        assert report["n_categories"] == 5
        assert report["n_rois"] == 17
        assert len(report["kappa_matrix"]) == 5
        assert len(report["kappa_matrix"][0]) == 17

    def test_preferred_category(self, synthetic_result):
        # Create categories where 0=face trials have higher kappa in FFA
        categories = np.random.randint(0, 5, size=200)
        # Boost FFA kappa for face trials
        face_mask = categories == 0
        synthetic_result.per_roi_kappas[face_mask, 9] += 50  # FFA1
        synthetic_result.per_roi_kappas[face_mask, 10] += 50  # FFA2

        cat_names = ["face", "scene", "object", "animal", "food"]
        report = compute_category_conditional_kappa(
            synthetic_result, categories, cat_names
        )
        assert report["preferred_category"]["FFA1"]["category"] == "face"


class TestPerROICalibration:
    def test_basic(self, synthetic_result):
        gt = np.random.randn(200, 768).astype(np.float32)
        gt /= np.linalg.norm(gt, axis=-1, keepdims=True)

        report = compute_per_roi_calibration(synthetic_result, gt)
        assert len(report) == 18  # 17 ROIs + _fused
        assert "_fused" in report
        for roi_name in DEFAULT_ROI_NAMES:
            assert roi_name in report
            assert "spearman_rho" in report[roi_name]

    def test_requires_mus(self, synthetic_result):
        synthetic_result.per_roi_mus = None
        gt = np.random.randn(200, 768).astype(np.float32)
        with pytest.raises(ValueError, match="per_roi_mus not stored"):
            compute_per_roi_calibration(synthetic_result, gt)


class TestNCSnrDissociation:
    def test_basic(self, synthetic_result):
        ncsnr = np.random.uniform(0.5, 3.0, size=17).astype(np.float32)
        report = compute_kappa_ncsnr_dissociation(synthetic_result, ncsnr)
        assert "pearson_kappa_ncsnr" in report
        assert "residual_variance_fraction" in report
        assert "interpretation" in report

    def test_correlated_case(self, synthetic_result):
        # Make NCSNR proportional to mean kappa per ROI
        mean_kappa = synthetic_result.per_roi_kappas.mean(axis=0)
        ncsnr = mean_kappa * 0.1 + np.random.normal(0, 1, size=17)
        report = compute_kappa_ncsnr_dissociation(synthetic_result, ncsnr)
        assert report["pearson_kappa_ncsnr"]["r"] > 0


class TestROITiers:
    def test_all_rois_covered(self):
        all_tier_rois = set()
        for rois in ROI_TIERS.values():
            all_tier_rois.update(rois)
        for roi in DEFAULT_ROI_NAMES:
            assert roi in all_tier_rois, f"{roi} not in any tier"

    def test_no_overlap(self):
        seen = set()
        for tier_name, rois in ROI_TIERS.items():
            for roi in rois:
                assert roi not in seen, f"{roi} in multiple tiers"
                seen.add(roi)
