"""Tests for fmri2img.eval.conformal_prediction."""

import numpy as np
import pytest

from fmri2img.eval.conformal_prediction import (
    kappa_nonconformity,
    margin_nonconformity,
    agreement_nonconformity,
    rank_nonconformity,
    composite_nonconformity,
    calibrate_threshold,
    retrieval_conformal_sets,
    score_based_conformal,
    conformal_sweep,
    cross_subject_conformal,
    weighted_conformal_transfer,
)


class TestNonconformityScores:
    def test_kappa_nonconformity_invert(self):
        kappas = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        scores = kappa_nonconformity(kappas)
        assert scores.shape == (5,)
        assert np.all(np.diff(scores) < 0), "Higher kappa should give lower nonconformity"

    def test_kappa_nonconformity_negate(self):
        kappas = np.array([10.0, 20.0, 30.0])
        scores = kappa_nonconformity(kappas, invert=False)
        np.testing.assert_array_equal(scores, -kappas)

    def test_kappa_handles_zero(self):
        kappas = np.array([0.0, 1.0, 10.0])
        scores = kappa_nonconformity(kappas)
        assert np.isfinite(scores).all()

    def test_margin_nonconformity(self):
        rng = np.random.RandomState(42)
        d = 64
        n, m = 20, 50
        preds = rng.randn(n, d)
        preds /= np.linalg.norm(preds, axis=1, keepdims=True)
        gallery = rng.randn(m, d)
        gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
        gt = rng.randint(0, m, size=n)

        scores = margin_nonconformity(preds, gallery, gt)
        assert scores.shape == (n,)
        assert np.all(scores >= 0) and np.all(scores <= 2)

    def test_agreement_nonconformity(self):
        correct = np.array([
            [True, True, True],
            [True, False, False],
            [False, False, False],
        ])
        scores = agreement_nonconformity(correct)
        np.testing.assert_allclose(scores, [0.0, 2/3, 1.0])

    def test_rank_nonconformity(self):
        rng = np.random.RandomState(42)
        d = 32
        n, m = 10, 30
        preds = rng.randn(n, d)
        preds /= np.linalg.norm(preds, axis=1, keepdims=True)
        gallery = rng.randn(m, d)
        gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
        gt = np.arange(n) % m

        scores = rank_nonconformity(preds, gallery, gt)
        assert scores.shape == (n,)
        assert np.all(scores >= 0)
        assert np.all(scores < m)

    def test_composite_nonconformity(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        scores = composite_nonconformity({"a": a, "b": b})
        assert scores.shape == (5,)


class TestCalibration:
    def test_calibrate_threshold_basic(self):
        scores = np.arange(100, dtype=np.float64)
        tau = calibrate_threshold(scores, alpha=0.10)
        assert tau >= 89

    def test_calibrate_threshold_coverage(self):
        rng = np.random.RandomState(42)
        scores = rng.randn(1000)
        for alpha in [0.05, 0.10, 0.20]:
            tau = calibrate_threshold(scores, alpha)
            coverage = (scores <= tau).mean()
            assert coverage >= 1 - alpha - 0.01


class TestRetrievalConformal:
    def test_retrieval_conformal_sets_coverage(self):
        rng = np.random.RandomState(42)
        d = 128
        n, m = 200, 200
        preds = rng.randn(n, d).astype(np.float32)
        preds /= np.linalg.norm(preds, axis=1, keepdims=True)
        gallery = rng.randn(m, d).astype(np.float32)
        gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
        gt = np.arange(n) % m
        kappas = rng.uniform(20, 40, size=n)

        result = retrieval_conformal_sets(
            preds, gallery, gt, kappas, alpha=0.10, seed=42,
        )
        assert result.n_calibration + result.n_test == n
        assert result.guaranteed_coverage == 0.90


class TestScoreConformal:
    def test_score_based_conformal_runs(self):
        rng = np.random.RandomState(42)
        n = 500
        scores = rng.exponential(1.0, size=n)
        is_correct = rng.random(n) > 0.3

        result = score_based_conformal(
            scores, is_correct, alpha=0.10, n_bootstrap=50,
        )
        assert result.n_total == n
        assert 0 <= result.mean_acceptance_rate <= 1
        assert 0 <= result.mean_accepted_accuracy <= 1


class TestConformalSweep:
    def test_sweep_returns_all_alphas(self):
        rng = np.random.RandomState(42)
        n = 300
        scores = rng.exponential(1.0, size=n)
        is_correct = rng.random(n) > 0.4
        alphas = [0.05, 0.10, 0.20]

        results = conformal_sweep(scores, is_correct, alphas=alphas)
        assert len(results) == len(alphas)
        for r, a in zip(results, alphas):
            assert r["alpha"] == a
            assert r["target_coverage"] == 1 - a


class TestCrossSubject:
    def test_cross_subject_conformal(self):
        rng = np.random.RandomState(42)
        cal_scores = rng.exponential(1.0, size=200)

        test_by_subj = {
            "subj02": rng.exponential(1.0, size=100),
            "subj05": rng.exponential(1.5, size=100),
        }
        correct_by_subj = {
            "subj02": rng.random(100) > 0.3,
            "subj05": rng.random(100) > 0.5,
        }

        results = cross_subject_conformal(
            cal_scores, test_by_subj, correct_by_subj, alpha=0.10,
        )
        assert "subj02" in results
        assert "subj05" in results
        assert "coverage_gap" in results["subj02"]

    def test_weighted_conformal(self):
        rng = np.random.RandomState(42)
        cal_scores = rng.exponential(1.0, size=200)
        cal_weights = rng.uniform(0.5, 1.5, size=200)
        test_scores = rng.exponential(1.0, size=100)

        tau = weighted_conformal_transfer(
            cal_scores, cal_weights, test_scores, alpha=0.10,
        )
        assert isinstance(tau, float)
        assert tau > 0
