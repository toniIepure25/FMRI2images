"""
Tests for Ridge Encoder and Retrieval Metrics
============================================

Unit tests for Ridge baseline model and evaluation utilities.
"""

import numpy as np
import pytest
import logging
from pathlib import Path
import tempfile

from fmri2img.models.ridge import RidgeEncoder, evaluate_predictions
from fmri2img.eval.retrieval import cosine_sim, retrieval_at_k, compute_ranking_metrics

log = logging.getLogger(__name__)


def test_ridge_encoder_fit_predict():
    """Test Ridge encoder basic fit and predict."""
    np.random.seed(42)
    
    # Mock data: 100 samples, 50 features → 512D
    n_samples = 100
    n_features = 50
    
    X_train = np.random.randn(n_samples, n_features).astype(np.float32)
    Y_train = np.random.randn(n_samples, 512).astype(np.float32)
    
    # L2-normalize targets (standard for CLIP)
    Y_train = Y_train / np.linalg.norm(Y_train, axis=1, keepdims=True)
    
    # Fit model
    model = RidgeEncoder(alpha=1.0)
    model.fit(X_train, Y_train)
    
    assert model.input_dim == n_features
    assert model.output_dim == 512
    assert model.model is not None
    
    # Predict on train (sanity check)
    Y_pred = model.predict(X_train, normalize=True)
    
    assert Y_pred.shape == (n_samples, 512)
    assert Y_pred.dtype == np.float32
    
    # Check normalization
    norms = np.linalg.norm(Y_pred, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    
    # Check reasonable correlation
    metrics = evaluate_predictions(Y_train, Y_pred, normalize=True)
    assert "cosine" in metrics
    assert "mse" in metrics
    
    log.info(f"✅ Ridge fit/predict test passed: cosine={metrics['cosine']:.4f}")


def test_ridge_encoder_save_load():
    """Test Ridge encoder save/load."""
    np.random.seed(42)
    
    X_train = np.random.randn(50, 30).astype(np.float32)
    Y_train = np.random.randn(50, 512).astype(np.float32)
    Y_train = Y_train / np.linalg.norm(Y_train, axis=1, keepdims=True)
    
    # Fit and save
    model1 = RidgeEncoder(alpha=10.0)
    model1.fit(X_train, Y_train)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_ridge.pkl"
        model1.save(path)
        
        # Load
        model2 = RidgeEncoder.load(path)
        
        assert model2.alpha == 10.0
        assert model2.input_dim == 30
        assert model2.output_dim == 512
        
        # Test predictions match
        X_test = np.random.randn(10, 30).astype(np.float32)
        Y_pred1 = model1.predict(X_test, normalize=True)
        Y_pred2 = model2.predict(X_test, normalize=True)
        
        assert np.allclose(Y_pred1, Y_pred2, atol=1e-5)
    
    log.info("✅ Ridge save/load test passed")


def test_cosine_sim():
    """Test cosine similarity computation."""
    np.random.seed(42)
    
    # Create normalized vectors
    query = np.random.randn(10, 128).astype(np.float32)
    query = query / np.linalg.norm(query, axis=1, keepdims=True)
    
    gallery = np.random.randn(50, 128).astype(np.float32)
    gallery = gallery / np.linalg.norm(gallery, axis=1, keepdims=True)
    
    sim = cosine_sim(query, gallery)
    
    assert sim.shape == (10, 50)
    assert np.all(sim >= -1.0) and np.all(sim <= 1.0)
    
    # Test self-similarity
    self_sim = cosine_sim(query, query)
    diag = np.diag(self_sim)
    assert np.allclose(diag, 1.0, atol=1e-5)
    
    log.info("✅ Cosine similarity test passed")


def test_retrieval_at_k():
    """Test retrieval@K metric."""
    np.random.seed(42)
    
    n_queries = 20
    n_gallery = 100
    
    # Create normalized embeddings
    query = np.random.randn(n_queries, 512).astype(np.float32)
    query = query / np.linalg.norm(query, axis=1, keepdims=True)
    
    gallery = np.random.randn(n_gallery, 512).astype(np.float32)
    gallery = gallery / np.linalg.norm(gallery, axis=1, keepdims=True)
    
    # Ground truth: each query matches a specific gallery index
    gt_index = np.random.choice(n_gallery, size=n_queries, replace=False)
    
    # Compute retrieval@K
    metrics = retrieval_at_k(query, gallery, gt_index, ks=(1, 5, 10))
    
    assert "R@1" in metrics
    assert "R@5" in metrics
    assert "R@10" in metrics
    
    assert 0.0 <= metrics["R@1"] <= 1.0
    assert 0.0 <= metrics["R@5"] <= 1.0
    assert 0.0 <= metrics["R@10"] <= 1.0
    
    # R@10 should be >= R@5 >= R@1
    assert metrics["R@10"] >= metrics["R@5"]
    assert metrics["R@5"] >= metrics["R@1"]
    
    log.info(f"✅ Retrieval@K test passed: R@1={metrics['R@1']:.2%}, R@5={metrics['R@5']:.2%}, R@10={metrics['R@10']:.2%}")


def test_retrieval_perfect_match():
    """Test retrieval with perfect predictions (should get 100%)."""
    np.random.seed(42)
    
    n_samples = 50
    
    # Same embeddings for query and gallery
    embeddings = np.random.randn(n_samples, 512).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Each query matches itself
    gt_index = np.arange(n_samples)
    
    metrics = retrieval_at_k(embeddings, embeddings, gt_index, ks=(1, 5, 10))
    
    # Perfect match: should be 100% for all K
    assert np.isclose(metrics["R@1"], 1.0)
    assert np.isclose(metrics["R@5"], 1.0)
    assert np.isclose(metrics["R@10"], 1.0)
    
    log.info("✅ Perfect retrieval test passed")


def test_ranking_metrics():
    """Test ranking metrics computation."""
    np.random.seed(42)
    
    n_queries = 30
    n_gallery = 100
    
    query = np.random.randn(n_queries, 512).astype(np.float32)
    query = query / np.linalg.norm(query, axis=1, keepdims=True)
    
    gallery = np.random.randn(n_gallery, 512).astype(np.float32)
    gallery = gallery / np.linalg.norm(gallery, axis=1, keepdims=True)
    
    gt_index = np.random.choice(n_gallery, size=n_queries, replace=False)
    
    metrics = compute_ranking_metrics(query, gallery, gt_index)
    
    assert "mean_rank" in metrics
    assert "median_rank" in metrics
    assert "mrr" in metrics
    
    assert 1 <= metrics["mean_rank"] <= n_gallery
    assert 1 <= metrics["median_rank"] <= n_gallery
    assert 0.0 <= metrics["mrr"] <= 1.0
    
    log.info(f"✅ Ranking metrics test passed: mean_rank={metrics['mean_rank']:.2f}, MRR={metrics['mrr']:.4f}")
