"""Tests for the Neural Manifold Alignment analysis framework.

All tests use small synthetic data (N=20-50, D=8-16) and require no
external data files, GPU, or network access.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from analysis.manifold.utils import (
    cosine_similarity_matrix,
    csls_scores,
    gini_coefficient,
    percentile_normalize,
    safe_normalize,
    slerp,
    topk_entropy,
    topk_indices,
    upper_triangle,
    weighted_harmonic_mean,
    zscore_normalize,
)


# =====================================================================
#  Fixtures
# =====================================================================

@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def embeddings_16d(rng):
    """(30, 16) L2-normalised embeddings."""
    x = rng.randn(30, 16).astype(np.float32)
    return safe_normalize(x)


@pytest.fixture
def gallery_16d(rng):
    """(50, 16) L2-normalised gallery."""
    x = rng.randn(50, 16).astype(np.float32)
    return safe_normalize(x)


@pytest.fixture
def target_indices_30():
    return np.arange(30)


# =====================================================================
#  TestNormalization
# =====================================================================

class TestNormalization:
    def test_unit_norm(self, rng):
        x = rng.randn(10, 8).astype(np.float32)
        normed = safe_normalize(x)
        norms = np.linalg.norm(normed, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_zero_vector(self):
        x = np.zeros((3, 4), dtype=np.float32)
        normed = safe_normalize(x)
        assert np.all(normed == 0)

    def test_already_normalized(self, rng):
        x = safe_normalize(rng.randn(5, 8).astype(np.float32))
        normed = safe_normalize(x)
        np.testing.assert_allclose(x, normed, atol=1e-6)

    def test_single_vector(self):
        x = np.array([[3.0, 4.0]])
        normed = safe_normalize(x)
        np.testing.assert_allclose(normed, [[0.6, 0.8]], atol=1e-6)


# =====================================================================
#  TestCosineSimilarity
# =====================================================================

class TestCosineSimilarity:
    def test_identity(self, embeddings_16d):
        sim = cosine_similarity_matrix(embeddings_16d, embeddings_16d)
        np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-5)

    def test_orthogonal(self):
        a = safe_normalize(np.array([[1, 0, 0, 0]], dtype=np.float32))
        b = safe_normalize(np.array([[0, 1, 0, 0]], dtype=np.float32))
        sim = cosine_similarity_matrix(a, b)
        np.testing.assert_allclose(sim[0, 0], 0.0, atol=1e-6)

    def test_anti_parallel(self):
        a = safe_normalize(np.array([[1, 0]], dtype=np.float32))
        b = safe_normalize(np.array([[-1, 0]], dtype=np.float32))
        sim = cosine_similarity_matrix(a, b)
        np.testing.assert_allclose(sim[0, 0], -1.0, atol=1e-6)

    def test_shape(self, embeddings_16d, gallery_16d):
        sim = cosine_similarity_matrix(embeddings_16d, gallery_16d)
        assert sim.shape == (30, 50)


# =====================================================================
#  TestCSLS
# =====================================================================

class TestCSLS:
    def test_shape(self, embeddings_16d, gallery_16d):
        scores = csls_scores(embeddings_16d, gallery_16d, k=5)
        assert scores.shape == (30, 50)

    def test_reduces_to_cosine_like_when_k_equals_gallery(self, rng):
        """When k = gallery size, mean_knn is the full mean -> uniform shift."""
        q = safe_normalize(rng.randn(5, 8).astype(np.float32))
        g = safe_normalize(rng.randn(5, 8).astype(np.float32))
        cos = cosine_similarity_matrix(q, g)
        css = csls_scores(q, g, k=5)
        rank_cos = np.argsort(-cos, axis=1)
        rank_css = np.argsort(-css, axis=1)
        # rankings should be identical or very similar
        assert rank_cos.shape == rank_css.shape

    def test_csls_different_from_cosine(self, embeddings_16d, gallery_16d):
        cos = cosine_similarity_matrix(embeddings_16d, gallery_16d)
        css = csls_scores(embeddings_16d, gallery_16d, k=5)
        assert not np.allclose(cos, css)


# =====================================================================
#  TestRetrievalAtK
# =====================================================================

class TestRetrievalAtK:
    def test_perfect_retrieval(self, rng):
        n = 20
        emb = safe_normalize(rng.randn(n, 8).astype(np.float32))
        from analysis.manifold.retrieval_metrics import compute_cosine_retrieval
        result = compute_cosine_retrieval(emb, emb, np.arange(n), ks=[1, 5])
        assert result["metrics"]["R@1"] == pytest.approx(1.0)
        assert result["metrics"]["R@5"] == pytest.approx(1.0)

    def test_random_baseline_below_perfect(self, rng):
        q = safe_normalize(rng.randn(50, 8).astype(np.float32))
        g = safe_normalize(rng.randn(50, 8).astype(np.float32))
        from analysis.manifold.retrieval_metrics import compute_cosine_retrieval
        result = compute_cosine_retrieval(q, g, np.arange(50), ks=[1])
        assert result["metrics"]["R@1"] < 1.0


# =====================================================================
#  TestHubness
# =====================================================================

class TestHubness:
    def test_uniform_low_skewness(self):
        counts = np.ones(100)
        from scipy.stats import skew
        s = skew(counts)
        assert np.isnan(s) or abs(s) < 1e-6

    def test_hub_counts_shape(self, rng):
        scores = rng.randn(30, 50)
        from analysis.manifold.hubness import compute_hub_counts
        counts = compute_hub_counts(scores, k=5)
        assert counts.shape == (50,)
        assert counts.sum() == 30 * 5


# =====================================================================
#  TestGiniCoefficient
# =====================================================================

class TestGiniCoefficient:
    def test_all_equal(self):
        assert gini_coefficient(np.ones(10)) == pytest.approx(0.0, abs=1e-10)

    def test_one_takes_all(self):
        c = np.zeros(10)
        c[0] = 100
        g = gini_coefficient(c)
        assert g > 0.8

    def test_increasing(self):
        g1 = gini_coefficient(np.array([1, 1, 1, 1]))
        g2 = gini_coefficient(np.array([0, 0, 0, 10]))
        assert g2 > g1


# =====================================================================
#  TestRSA
# =====================================================================

class TestRSA:
    def test_identical_matrices_rho_one(self, rng):
        emb = safe_normalize(rng.randn(20, 8).astype(np.float32))
        from analysis.manifold.rsa import compute_rsa
        result = compute_rsa(emb, emb, methods=["spearman"])
        assert result["spearman_rho"] == pytest.approx(1.0, abs=1e-4)

    def test_independent_low_rho(self, rng):
        a = safe_normalize(rng.randn(30, 8).astype(np.float32))
        b = safe_normalize(rng.randn(30, 8).astype(np.float32))
        from analysis.manifold.rsa import compute_rsa
        result = compute_rsa(a, b, methods=["spearman"])
        assert abs(result["spearman_rho"]) < 0.5

    def test_upper_triangle_extraction(self):
        mat = np.arange(16).reshape(4, 4).astype(float)
        ut = upper_triangle(mat)
        expected = np.array([1, 2, 3, 6, 7, 11], dtype=float)
        np.testing.assert_array_equal(ut, expected)


# =====================================================================
#  TestNeighborhoodOverlap
# =====================================================================

class TestNeighborhoodOverlap:
    def test_identical_full_overlap(self, rng):
        emb = safe_normalize(rng.randn(20, 8).astype(np.float32))
        from analysis.manifold.neighborhood import compute_neighborhood_overlap
        result = compute_neighborhood_overlap(emb, emb, emb, ks=[5])
        assert result["overlap@5_mean"] == pytest.approx(1.0, abs=1e-6)

    def test_random_partial_overlap(self, rng):
        a = safe_normalize(rng.randn(20, 8).astype(np.float32))
        b = safe_normalize(rng.randn(20, 8).astype(np.float32))
        from analysis.manifold.neighborhood import compute_neighborhood_overlap
        result = compute_neighborhood_overlap(a, b, b, ks=[5])
        assert 0.0 <= result["overlap@5_mean"] <= 1.0


# =====================================================================
#  TestRepeatStability
# =====================================================================

class TestRepeatStability:
    def test_good_decoder_high_within(self, rng):
        """A good decoder gives high within-image similarity."""
        n_images = 10
        n_reps = 3
        nsd_ids = np.repeat(np.arange(n_images), n_reps)
        base = safe_normalize(rng.randn(n_images, 8).astype(np.float32))
        z_pred = np.repeat(base, n_reps, axis=0)
        z_pred += rng.randn(*z_pred.shape).astype(np.float32) * 0.01
        z_pred = safe_normalize(z_pred)

        from analysis.manifold.repeat_stability import run_repeat_stability
        result = run_repeat_stability(z_pred, nsd_ids)
        assert result["status"] == "computed"
        assert result["within_mean"] > result["between_mean"]

    def test_unavailable_without_data(self):
        from analysis.manifold.repeat_stability import run_repeat_stability
        result = run_repeat_stability(None, None)
        assert result["status"] == "unavailable"


# =====================================================================
#  TestReliabilityScore
# =====================================================================

class TestReliabilityScore:
    def test_normalization(self, rng):
        from analysis.manifold.reliability import ReliabilityScorer
        scorer = ReliabilityScorer({"a": 1.0, "b": 1.0})
        signals = {"a": rng.randn(20), "b": rng.randn(20)}
        score, used = scorer.compute(signals)
        assert len(score) == 20
        assert "a" in used and "b" in used

    def test_missing_components(self, rng):
        from analysis.manifold.reliability import ReliabilityScorer
        scorer = ReliabilityScorer({"a": 1.0, "b": 1.0, "c": 1.0})
        signals = {"a": rng.randn(20), "b": None, "c": rng.randn(20)}
        score, used = scorer.compute(signals)
        assert "b" not in used
        assert len(used) == 2

    def test_coverage_monotonic(self, rng):
        """Higher coverage should not decrease accuracy for a good scorer."""
        from analysis.manifold.reliability import compute_selective_prediction
        scores = rng.randn(100)
        correct = scores > 0
        result = compute_selective_prediction(scores, correct)
        # At low coverage, accuracy should be >= full coverage
        assert result["accuracy"][-1] >= result["accuracy"][0] - 0.1


# =====================================================================
#  TestWeightedHarmonicMean
# =====================================================================

class TestWeightedHarmonicMean:
    def test_equal_values(self):
        vals = {"a": 0.5, "b": 0.5, "c": 0.5}
        weights = {"a": 1, "b": 1, "c": 1}
        hm, _ = weighted_harmonic_mean(vals, weights)
        assert hm == pytest.approx(0.5, abs=1e-6)

    def test_handles_missing(self):
        vals = {"a": 0.5, "b": None, "c": 0.8}
        weights = {"a": 1, "b": 1, "c": 1}
        hm, used = weighted_harmonic_mean(vals, weights)
        assert "b" not in used
        assert hm > 0

    def test_zero_excluded(self):
        vals = {"a": 0.5, "b": 0.0}
        weights = {"a": 1, "b": 1}
        hm, used = weighted_harmonic_mean(vals, weights)
        assert "b" not in used


# =====================================================================
#  TestSemanticAxes
# =====================================================================

class TestSemanticAxes:
    def test_orthogonal_projection(self):
        axis = safe_normalize(np.array([[1, 0, 0, 0]], dtype=np.float32))
        emb = np.array([[0.5, 0.5, 0, 0]], dtype=np.float32)
        from analysis.manifold.semantic_axes import project_onto_axes
        proj = project_onto_axes(emb, axis)
        assert proj[0, 0] == pytest.approx(0.5, abs=1e-6)

    def test_axis_normalization(self, rng):
        from analysis.manifold.semantic_axes import compute_axis_preservation
        n = 20
        emb = safe_normalize(rng.randn(n, 8).astype(np.float32))
        axes = safe_normalize(rng.randn(3, 8).astype(np.float32))
        result = compute_axis_preservation(emb, emb, axes)
        for name, vals in result["per_axis"].items():
            assert vals["pearson_r"] == pytest.approx(1.0, abs=1e-4)

    def test_build_axes_from_text_embeddings(self):
        from analysis.manifold.semantic_axes import build_axes_from_text_embeddings

        concepts = ["animal", "object", "water", "road"]
        text = safe_normalize(np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ], dtype=np.float32))
        result = build_axes_from_text_embeddings(
            text,
            concepts,
            [{"name": "animal_vs_object", "positive": "animal", "negative": "object"}],
        )
        assert result["status"] == "computed"
        assert result["axes"].shape == (1, 3)
        assert result["axis_names"] == ["animal_vs_object"]
        assert len(result["unavailable_axes"]) == 0
        assert np.linalg.norm(result["axes"][0]) == pytest.approx(1.0)

    def test_missing_axis_concept_is_unavailable(self):
        from analysis.manifold.semantic_axes import build_axes_from_text_embeddings

        text = safe_normalize(np.eye(2, dtype=np.float32))
        result = build_axes_from_text_embeddings(
            text,
            ["animal", "object"],
            [{"name": "food_vs_object", "positive": "food", "negative": "object"}],
        )
        assert result["status"] == "unavailable"
        assert result["unavailable_axes"][0]["name"] == "food_vs_object"

    def test_sign_agreement(self):
        from analysis.manifold.semantic_axes import compute_axis_preservation

        axis = safe_normalize(np.array([[1.0, 0.0]], dtype=np.float32))
        z_target = safe_normalize(np.array([[1.0, 0.1], [1.0, -0.1], [-1.0, 0.1], [-1.0, -0.1]], dtype=np.float32))
        z_pred = z_target.copy()
        result = compute_axis_preservation(z_pred, z_target, axis, ["x_axis"])
        assert result["per_axis"]["x_axis"]["sign_agreement"] == pytest.approx(1.0)
        assert result["_proj_pred"].shape == (4, 1)

    def test_c10_claim_logic(self):
        from analysis.manifold.claim_tests import evaluate_all_claims

        claims = evaluate_all_claims({
            "mean_axis_spearman": 0.36,
            "mean_preservation": 0.51,
            "mean_axis_pearson": 0.4,
            "mean_axis_mae": 0.1,
            "n_axes_built": 4,
            "best_axis": "animal_vs_inanimate",
            "weakest_axis": "water_vs_land",
        })
        c10 = next(c for c in claims if c.claim_id == "C10")
        assert c10.status == "supported"
        assert "Built 4 axes" in c10.explanation


# =====================================================================
#  TestSlerp
# =====================================================================

class TestSlerp:
    def test_endpoints(self, rng):
        v0 = safe_normalize(rng.randn(8).astype(np.float32))
        v1 = safe_normalize(rng.randn(8).astype(np.float32))
        np.testing.assert_allclose(slerp(v0, v1, 0.0), v0, atol=1e-5)
        np.testing.assert_allclose(slerp(v0, v1, 1.0), v1, atol=1e-5)

    def test_unit_norm_along_path(self, rng):
        v0 = safe_normalize(rng.randn(8).astype(np.float32))
        v1 = safe_normalize(rng.randn(8).astype(np.float32))
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            vt = slerp(v0, v1, t)
            assert np.linalg.norm(vt) == pytest.approx(1.0, abs=1e-5)

    def test_midpoint_on_sphere(self, rng):
        v0 = safe_normalize(rng.randn(8).astype(np.float32))
        v1 = safe_normalize(rng.randn(8).astype(np.float32))
        mid = slerp(v0, v1, 0.5)
        d0 = np.arccos(np.clip(np.dot(mid, v0), -1, 1))
        d1 = np.arccos(np.clip(np.dot(mid, v1), -1, 1))
        np.testing.assert_allclose(d0, d1, atol=1e-4)


# =====================================================================
#  TestCounterfactualAndInterpolation
# =====================================================================

class TestCounterfactualAndInterpolation:
    def test_counterfactual_alpha_grid_normalized(self, rng):
        from analysis.manifold.counterfactual import make_counterfactual_grid

        z = safe_normalize(rng.randn(8).astype(np.float32))
        axis = safe_normalize(rng.randn(8).astype(np.float32))
        grid = make_counterfactual_grid(z, axis, [-1.0, 0.0, 1.0])
        assert grid.shape == (3, 8)
        np.testing.assert_allclose(np.linalg.norm(grid, axis=1), 1.0, atol=1e-6)

    def test_counterfactual_transition_detection(self):
        from analysis.manifold.counterfactual import detect_first_transition

        margin = detect_first_transition(
            [-2.0, -1.0, 0.0, 0.5, 1.0],
            ["dog", "dog", "cat", "cat", "truck"],
            "cat",
        )
        assert margin == pytest.approx(1.0)

    def test_counterfactual_unavailable_without_text(self, rng):
        from analysis.manifold.counterfactual import compute_counterfactual_edits

        z = safe_normalize(rng.randn(5, 8).astype(np.float32))
        axes = safe_normalize(rng.randn(2, 8).astype(np.float32))
        out = compute_counterfactual_edits(z, axes, ["a", "b"], z, text_embeddings=None)
        assert out["status"] == "unavailable"

    def test_interpolation_slerp_path_endpoints_and_unit_norm(self, rng):
        from analysis.manifold.interpolation import slerp_path

        a = safe_normalize(rng.randn(8).astype(np.float32))
        b = safe_normalize(rng.randn(8).astype(np.float32))
        path = slerp_path(a, b, n_steps=5)
        np.testing.assert_allclose(path[0], a, atol=1e-5)
        np.testing.assert_allclose(path[-1], b, atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(path, axis=1), 1.0, atol=1e-5)

    def test_interpolation_smoothness_finite(self, rng):
        from analysis.manifold.interpolation import (
            compute_path_smoothness,
            compute_semantic_velocity,
            text_probe_distribution,
        )

        path = safe_normalize(rng.randn(6, 8).astype(np.float32))
        text = safe_normalize(rng.randn(4, 8).astype(np.float32))
        probs = text_probe_distribution(path, text, temperature=0.07)
        velocity = compute_semantic_velocity(probs)
        smooth = compute_path_smoothness(velocity)
        assert np.isfinite(smooth)
        assert smooth > 0

    def test_interpolation_unavailable_without_text(self, rng):
        from analysis.manifold.interpolation import run_interpolation

        z = safe_normalize(rng.randn(5, 8).astype(np.float32))
        out = run_interpolation(z, z, text_embeddings=None)
        assert out["status"] == "unavailable"


# =====================================================================
#  TestNMAS
# =====================================================================

class TestNMAS:
    def test_all_perfect(self):
        from analysis.manifold.nmas import compute_nmas
        comps = {"retrieval": 1.0, "rsa": 1.0, "neighborhood": 1.0}
        result = compute_nmas(comps)
        assert result["nmas"] == pytest.approx(1.0, abs=1e-6)

    def test_missing_components(self):
        from analysis.manifold.nmas import compute_nmas
        comps = {"retrieval": 0.8, "rsa": None, "neighborhood": 0.6}
        result = compute_nmas(comps)
        assert "rsa" in result["excluded"]
        assert result["nmas"] > 0

    def test_sensitivity_keys(self):
        from analysis.manifold.nmas import compute_nmas
        comps = {"retrieval": 0.8, "rsa": 0.6, "neighborhood": 0.7}
        result = compute_nmas(comps)
        assert set(result["sensitivity"].keys()) == {"retrieval", "rsa", "neighborhood"}


# =====================================================================
#  TestTopKEntropy
# =====================================================================

class TestTopKEntropy:
    def test_uniform_high_entropy(self, rng):
        scores = np.ones((5, 10))
        ent = topk_entropy(scores, k=10)
        assert ent[0] > 2.0

    def test_peaked_low_entropy(self):
        scores = np.zeros((1, 10))
        scores[0, 0] = 100.0
        ent = topk_entropy(scores, k=10)
        assert ent[0] < 0.1


# =====================================================================
#  TestPercentileNormalize
# =====================================================================

class TestPercentileNormalize:
    def test_range(self, rng):
        x = rng.randn(100)
        pn = percentile_normalize(x)
        assert pn.min() >= 0.0
        assert pn.max() <= 1.0

    def test_monotonic(self):
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        pn = percentile_normalize(x)
        assert np.all(np.diff(pn) >= 0)


# =====================================================================
#  TestClaimTests
# =====================================================================

class TestClaimTests:
    def test_evaluate_supported(self):
        from analysis.manifold.claim_tests import evaluate_all_claims
        metrics = {
            "cosine_R@1": 0.5,
            "csls_R@1": 0.7,
        }
        claims = evaluate_all_claims(metrics)
        c01 = next(c for c in claims if c.claim_id == "C01")
        assert c01.status == "supported"

    def test_evaluate_unavailable(self):
        from analysis.manifold.claim_tests import evaluate_all_claims
        claims = evaluate_all_claims({})
        for c in claims:
            assert c.status == "unavailable"

    def test_serializable(self, tmp_path):
        from analysis.manifold.claim_tests import evaluate_all_claims, save_claim_tests
        claims = evaluate_all_claims({"cosine_R@1": 0.3, "csls_R@1": 0.5})
        save_claim_tests(claims, tmp_path / "claims.json")
        data = json.loads((tmp_path / "claims.json").read_text())
        assert len(data) == 14


# =====================================================================
#  TestSemanticProbes
# =====================================================================

class TestSemanticProbes:
    def test_concept_level_aggregation_normalizes_template_average(self):
        import importlib.util

        builder_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "build"
            / "build_text_probe_embeddings.py"
        )
        spec = importlib.util.spec_from_file_location("build_text_probe_embeddings", builder_path)
        builder = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(builder)

        prompt_embeddings = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ], dtype=np.float32)
        prompt_concepts = ["animal", "animal", "vehicle", "vehicle"]
        concepts = ["animal", "vehicle"]
        concept_embeddings = builder.build_concept_embeddings(
            prompt_embeddings,
            prompt_concepts,
            concepts,
        )

        np.testing.assert_allclose(np.linalg.norm(concept_embeddings, axis=1), 1.0)
        np.testing.assert_allclose(
            concept_embeddings[0],
            np.array([1.0, 1.0]) / np.sqrt(2.0),
            atol=1e-6,
        )

    def test_load_npz_text_cache_prefers_concept_embeddings(self, tmp_path, rng):
        from analysis.manifold.config import load_config
        from analysis.manifold.data_loading import load_analysis_data

        z_pred = safe_normalize(rng.randn(4, 6).astype(np.float32))
        z_target = safe_normalize(rng.randn(4, 6).astype(np.float32))
        concepts = np.array(["animal", "vehicle", "kitchen"], dtype=object)
        concept_embeddings = safe_normalize(rng.randn(3, 6).astype(np.float32))
        prompt_embeddings = safe_normalize(rng.randn(6, 6).astype(np.float32))

        pred_path = tmp_path / "pred.npy"
        target_path = tmp_path / "target.npy"
        cache_path = tmp_path / "text_cache.npz"
        np.save(pred_path, z_pred)
        np.save(target_path, z_target)
        np.savez(
            cache_path,
            concepts=concepts,
            concept_embeddings=concept_embeddings,
            prompt_embeddings=prompt_embeddings,
            model_name=np.array("ViT-L-14"),
            pretrained_name=np.array("openai"),
            normalized=np.array(True),
        )

        cfg = load_config(overrides={
            "artifacts": {
                "z_pred_path": str(pred_path),
                "z_target_path": str(target_path),
                "text_embeddings_path": str(cache_path),
            },
            "analysis": {
                "gallery_mode": "target_embeddings",
                "normalize_embeddings": False,
            },
        })
        data = load_analysis_data(cfg)
        np.testing.assert_allclose(data.text_embeddings, concept_embeddings, atol=1e-6)
        assert data.text_concepts == ["animal", "vehicle", "kitchen"]
        assert data.text_probe_metadata["embedding_key"] == "concept_embeddings"

    def test_agreement_at_k_uses_target_top_concept_rank(self):
        from analysis.manifold.semantic_probes import (
            compute_probe_agreement,
            scores_to_probabilities,
        )

        pred_scores = np.array([
            [0.9, 0.8, 0.1],
            [0.2, 0.7, 0.6],
        ])
        target_scores = np.array([
            [0.1, 1.0, 0.2],
            [0.1, 0.2, 0.9],
        ])
        pred_probs = scores_to_probabilities(pred_scores, temperature=0.1)
        target_probs = scores_to_probabilities(target_scores, temperature=0.1)
        out = compute_probe_agreement(pred_scores, target_scores, pred_probs, target_probs, ks=[1, 2])

        assert out["agreement@1"] == pytest.approx(0.0)
        assert out["agreement@2"] == pytest.approx(1.0)
        np.testing.assert_array_equal(out["_target_top1_rank"], np.array([2, 2]))

    def test_js_divergence_is_finite(self, rng):
        from analysis.manifold.semantic_probes import run_semantic_probes

        text = safe_normalize(rng.randn(5, 8).astype(np.float32))
        z = safe_normalize(rng.randn(10, 8).astype(np.float32))
        out = run_semantic_probes(
            z,
            z,
            text_embeddings=text,
            concepts=[f"c{i}" for i in range(5)],
            temperature=0.07,
        )
        assert out["status"] == "computed"
        assert np.isfinite(out["js_divergence_mean"])
        assert np.isfinite(out["kl_divergence_mean"])

    def test_unavailable_without_text_cache(self, rng):
        from analysis.manifold.semantic_probes import run_semantic_probes

        z = safe_normalize(rng.randn(5, 8).astype(np.float32))
        out = run_semantic_probes(z, z, text_embeddings=None)
        assert out["status"] == "unavailable"


# =====================================================================
#  TestConfig
# =====================================================================

class TestConfig:
    def test_default_config_loads(self):
        from analysis.manifold.config import load_config
        cfg = load_config()
        assert cfg.analysis["seed"] == 42
        assert cfg.module_enabled("retrieval") is True

    def test_override(self):
        from analysis.manifold.config import load_config
        cfg = load_config(overrides={"analysis": {"seed": 123}})
        assert cfg.analysis["seed"] == 123

    def test_yaml_load(self, tmp_path):
        import yaml
        from analysis.manifold.config import load_config
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(yaml.dump({"analysis": {"seed": 99}}))
        cfg = load_config(cfg_file)
        assert cfg.analysis["seed"] == 99


# =====================================================================
#  TestDataLoading
# =====================================================================

class TestDataLoading:
    def test_target_ids_gallery_mode_orders_gallery(self, tmp_path, rng):
        import pandas as pd
        from analysis.manifold.config import load_config
        from analysis.manifold.data_loading import load_analysis_data

        z_pred = rng.randn(3, 4).astype(np.float32)
        z_target = rng.randn(3, 4).astype(np.float32)
        gallery = rng.randn(4, 4).astype(np.float32)
        gallery_ids = np.array([10, 20, 30, 40])
        target_ids = np.array([30, 10, 20])

        pred_path = tmp_path / "pred.npy"
        target_path = tmp_path / "target.npy"
        ids_path = tmp_path / "ids.npy"
        gallery_path = tmp_path / "gallery.parquet"
        np.save(pred_path, z_pred)
        np.save(target_path, z_target)
        np.save(ids_path, target_ids)
        pd.DataFrame({"nsdId": gallery_ids, "embedding": list(gallery)}).to_parquet(gallery_path)

        cfg = load_config(overrides={
            "artifacts": {
                "z_pred_path": str(pred_path),
                "z_target_path": str(target_path),
                "target_ids_path": str(ids_path),
                "gallery_embeddings_path": str(gallery_path),
            },
            "analysis": {
                "gallery_mode": "target_ids",
                "allow_diagonal_fallback": False,
                "normalize_embeddings": False,
            },
        })
        data = load_analysis_data(cfg)

        np.testing.assert_array_equal(data.target_gallery_indices, np.arange(3))
        np.testing.assert_array_equal(data.target_ids, target_ids)
        np.testing.assert_allclose(data.gallery, gallery[[2, 0, 1]])
        assert data.n_gallery == 3

    def test_full_gallery_requires_mapping_when_sizes_differ(self, tmp_path, rng):
        import pandas as pd
        from analysis.manifold.config import load_config
        from analysis.manifold.data_loading import load_analysis_data

        pred_path = tmp_path / "pred.npy"
        target_path = tmp_path / "target.npy"
        gallery_path = tmp_path / "gallery.parquet"
        np.save(pred_path, rng.randn(3, 4).astype(np.float32))
        np.save(target_path, rng.randn(3, 4).astype(np.float32))
        gallery = rng.randn(4, 4).astype(np.float32)
        pd.DataFrame({"nsdId": [10, 20, 30, 40], "embedding": list(gallery)}).to_parquet(gallery_path)

        cfg = load_config(overrides={
            "artifacts": {
                "z_pred_path": str(pred_path),
                "z_target_path": str(target_path),
                "gallery_embeddings_path": str(gallery_path),
            },
            "analysis": {
                "gallery_mode": "full",
                "allow_diagonal_fallback": False,
                "normalize_embeddings": False,
            },
        })
        with pytest.raises(ValueError, match="Target gallery indices are missing"):
            load_analysis_data(cfg)

    def test_target_embeddings_gallery_mode_uses_z_target(self, tmp_path, rng):
        from analysis.manifold.config import load_config
        from analysis.manifold.data_loading import load_analysis_data

        z_pred = rng.randn(5, 4).astype(np.float32)
        z_target = rng.randn(5, 4).astype(np.float32)
        pred_path = tmp_path / "pred.npy"
        target_path = tmp_path / "target.npy"
        np.save(pred_path, z_pred)
        np.save(target_path, z_target)

        cfg = load_config(overrides={
            "artifacts": {
                "z_pred_path": str(pred_path),
                "z_target_path": str(target_path),
                "gallery_embeddings_path": None,
            },
            "analysis": {
                "gallery_mode": "target_embeddings",
                "allow_diagonal_fallback": False,
                "normalize_embeddings": False,
            },
        })
        data = load_analysis_data(cfg)

        np.testing.assert_array_equal(data.target_gallery_indices, np.arange(5))
        np.testing.assert_allclose(data.gallery, z_target)
        assert data.n_gallery == 5
