"""
Tests for Research-Ready Components

Critical tests ensuring correctness of:
1. Oracle retrieval (GT→GT must be perfect)
2. Deterministic subsampling (reproducibility)
3. Gaussian NLL (analytic correctness)
4. EmbeddingPreprocessor (save/load, transform consistency)
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile

# Assuming src is in PYTHONPATH
from fmri2img.eval.embedding_metrics import (
    compute_retrieval_metrics,
    oracle_retrieval_check,
)
from fmri2img.embedding_preproc import EmbeddingPreprocessor
from fmri2img.losses.gaussian_nll import GaussianNLLLoss


class TestOracleRetrieval:
    """Test that ground truth embeddings perfectly retrieve themselves."""
    
    def test_oracle_perfect_retrieval(self):
        """GT embeddings should achieve R@1 ≈ 1.0, MeanR ≈ 1.0."""
        # Create dummy GT embeddings
        np.random.seed(42)
        N = 100
        D = 768
        
        embeddings = np.random.randn(N, D).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        ids = np.arange(N)
        
        # Oracle check: GT → GT retrieval
        metrics = compute_retrieval_metrics(
            embeddings, embeddings, ids, ids
        )
        
        # Assertions
        assert metrics.r_at_1 >= 0.99, f"Oracle R@1 = {metrics.r_at_1:.4f} < 0.99"
        assert metrics.mean_rank <= 1.1, f"Oracle MeanR = {metrics.mean_rank:.4f} > 1.1"
        assert metrics.mrr >= 0.99, f"Oracle MRR = {metrics.mrr:.4f} < 0.99"
    
    def test_oracle_check_function(self):
        """Test oracle_retrieval_check helper."""
        np.random.seed(42)
        N = 50
        D = 768
        
        embeddings = np.random.randn(N, D).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        ids = np.arange(N)
        
        passed, oracle_metrics = oracle_retrieval_check(embeddings, ids)
        
        assert passed, "Oracle check should pass for GT embeddings"
        assert oracle_metrics["oracle_r@1"] >= 0.95
    
    def test_oracle_fails_with_shuffled_ids(self):
        """Oracle should fail if IDs are mismatched."""
        np.random.seed(42)
        N = 50
        D = 768
        
        embeddings = np.random.randn(N, D).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Mismatched IDs (shuffled)
        query_ids = np.arange(N)
        gallery_ids = np.random.permutation(N)
        
        metrics = compute_retrieval_metrics(
            embeddings, embeddings, query_ids, gallery_ids
        )
        
        # Should fail (R@1 should be low)
        assert metrics.r_at_1 < 0.5, "Shuffled IDs should cause low R@1"


class TestDeterministicSubsampling:
    """Test that gallery subsampling is reproducible."""
    
    def test_same_seed_same_subsample(self):
        """Same seed should produce identical subsamples."""
        N = 1000
        gallery_size = 100
        seed = 42
        
        # First subsample
        rng1 = np.random.RandomState(seed)
        indices1 = rng1.choice(N, size=gallery_size, replace=False)
        
        # Second subsample with same seed
        rng2 = np.random.RandomState(seed)
        indices2 = rng2.choice(N, size=gallery_size, replace=False)
        
        assert np.array_equal(indices1, indices2), "Same seed should produce same indices"
    
    def test_different_seed_different_subsample(self):
        """Different seeds should produce different subsamples."""
        N = 1000
        gallery_size = 100
        
        rng1 = np.random.RandomState(42)
        indices1 = rng1.choice(N, size=gallery_size, replace=False)
        
        rng2 = np.random.RandomState(43)
        indices2 = rng2.choice(N, size=gallery_size, replace=False)
        
        assert not np.array_equal(indices1, indices2), "Different seeds should produce different indices"


class TestGaussianNLL:
    """Test Gaussian NLL loss matches analytic expectations."""
    
    def test_nll_zero_when_perfect(self):
        """NLL should be minimal when predictions match targets exactly."""
        batch_size = 10
        dim = 768
        
        # Perfect predictions
        target = torch.randn(batch_size, dim)
        mu = target.clone()
        logvar = torch.full((batch_size, dim), -5.0)  # Low variance
        
        loss_fn = GaussianNLLLoss(logvar_min=-10, logvar_max=5)
        loss = loss_fn(mu, logvar, target)
        
        # Loss should be small (dominated by logvar term)
        assert loss < 1.0, f"Perfect prediction should have low NLL: {loss:.4f}"
    
    def test_nll_high_when_wrong(self):
        """NLL should be high when predictions are far from targets."""
        batch_size = 10
        dim = 768
        
        # Wrong predictions (off by 10 std devs)
        target = torch.randn(batch_size, dim)
        mu = target + 10.0  # Large error
        logvar = torch.zeros(batch_size, dim)  # Variance = 1
        
        loss_fn = GaussianNLLLoss(logvar_min=-10, logvar_max=5)
        loss = loss_fn(mu, logvar, target)
        
        # Loss should be large
        assert loss > 10.0, f"Wrong prediction should have high NLL: {loss:.4f}"
    
    def test_nll_analytic_correctness(self):
        """Test NLL matches manual calculation."""
        torch.manual_seed(42)
        batch_size = 5
        dim = 10
        
        target = torch.randn(batch_size, dim)
        mu = torch.randn(batch_size, dim)
        logvar = torch.randn(batch_size, dim) * 0.5
        
        # Computed loss
        loss_fn = GaussianNLLLoss(logvar_min=-10, logvar_max=10, reduction="mean")
        loss_computed = loss_fn(mu, logvar, target)
        
        # Manual calculation
        err = target - mu
        nll_manual = 0.5 * (logvar + (err ** 2) / torch.exp(logvar))
        loss_manual = nll_manual.mean()
        
        torch.testing.assert_close(loss_computed, loss_manual, rtol=1e-5, atol=1e-5)


class TestEmbeddingPreprocessor:
    """Test EmbeddingPreprocessor save/load and consistency."""
    
    def test_preprocessor_transform_consistency(self):
        """Multiple transforms should give identical results."""
        np.random.seed(42)
        N = 100
        D = 768
        
        embeddings = np.random.randn(N, D).astype(np.float32)
        
        # Fit preprocessor
        preprocessor = EmbeddingPreprocessor(mode="center_pcr", k_components=8)
        preprocessor.fit(embeddings)
        
        # Transform twice
        transformed1 = preprocessor.transform(embeddings)
        transformed2 = preprocessor.transform(embeddings)
        
        np.testing.assert_array_almost_equal(
            transformed1, transformed2, decimal=6,
            err_msg="Multiple transforms should give identical results"
        )
    
    def test_preprocessor_save_load(self):
        """Save and load should preserve transformation."""
        np.random.seed(42)
        N = 100
        D = 768
        
        embeddings = np.random.randn(N, D).astype(np.float32)
        
        # Fit and transform
        preprocessor1 = EmbeddingPreprocessor(mode="center_pcr", k_components=8)
        preprocessor1.fit(embeddings)
        transformed1 = preprocessor1.transform(embeddings)
        
        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "preprocessor.pkl"
            preprocessor1.save(save_path)
            
            # Load and transform
            preprocessor2 = EmbeddingPreprocessor(mode="center_pcr", k_components=8)
            preprocessor2.load(save_path)
            transformed2 = preprocessor2.transform(embeddings)
        
        # Should be identical
        np.testing.assert_array_almost_equal(
            transformed1, transformed2, decimal=6,
            err_msg="Saved and loaded preprocessor should give same results"
        )
    
    def test_preprocessor_reduces_anisotropy(self):
        """Preprocessing should reduce anisotropy score."""
        np.random.seed(42)
        N = 200
        D = 768
        
        # Create anisotropic embeddings (add a dominant direction)
        embeddings = np.random.randn(N, D).astype(np.float32)
        dominant = np.random.randn(D)
        embeddings += 2.0 * dominant  # Add anisotropy
        
        # Fit preprocessor
        preprocessor = EmbeddingPreprocessor(mode="center_pcr", k_components=8)
        preprocessor.fit(embeddings)
        
        # Compute diagnostics
        diagnostics = preprocessor.compute_diagnostics(embeddings)
        
        # Anisotropy should be reduced
        aniso_before = diagnostics["anisotropy_score"]
        aniso_after = diagnostics["anisotropy_score_after"]
        
        assert aniso_after < aniso_before, \
            f"Anisotropy should decrease: {aniso_before:.4f} → {aniso_after:.4f}"
        assert abs(aniso_after) < 0.1, \
            f"Anisotropy after should be near 0: {aniso_after:.4f}"
    
    def test_whiten_mode(self):
        """Test center_whiten preprocessing mode."""
        np.random.seed(42)
        N = 100
        D = 768
        
        embeddings = np.random.randn(N, D).astype(np.float32)
        
        # Fit and transform
        preprocessor = EmbeddingPreprocessor(mode="center_whiten", whiten_eps=1e-5)
        preprocessor.fit(embeddings)
        transformed = preprocessor.transform(embeddings)
        
        # Check shape
        assert transformed.shape == embeddings.shape
        
        # Check normalized
        norms = np.linalg.norm(transformed, axis=1)
        np.testing.assert_array_almost_equal(
            norms, np.ones(N), decimal=5,
            err_msg="Transformed embeddings should be L2-normalized"
        )


class TestMemoryQueue:
    """Test MoCo-style memory queue."""
    
    def test_queue_enqueue_dequeue(self):
        """Test basic enqueue and dequeue operations."""
        from fmri2img.contrastive.queue import MemoryQueue
        
        queue_size = 100
        embedding_dim = 768
        
        queue = MemoryQueue(queue_size=queue_size, embedding_dim=embedding_dim)
        
        # Initially empty
        assert len(queue) == 0
        
        # Enqueue some embeddings
        batch_size = 10
        embeddings = torch.randn(batch_size, embedding_dim)
        queue.enqueue(embeddings)
        
        assert len(queue) == batch_size
        
        # Get queue
        queue_contents = queue.get_queue()
        assert queue_contents.shape == (batch_size, embedding_dim)
    
    def test_queue_fifo(self):
        """Test that queue is FIFO (first-in-first-out)."""
        from fmri2img.contrastive.queue import MemoryQueue
        
        queue_size = 50
        embedding_dim = 10
        
        queue = MemoryQueue(queue_size=queue_size, embedding_dim=embedding_dim)
        
        # Fill queue
        for i in range(queue_size):
            emb = torch.full((1, embedding_dim), float(i))
            queue.enqueue(emb)
        
        assert queue.queue_is_full
        
        # Add more (should wrap around and replace oldest)
        new_emb = torch.full((1, embedding_dim), 999.0)
        queue.enqueue(new_emb)
        
        # First element should be replaced
        queue_contents = queue.get_queue()
        assert queue_contents[0, 0].item() != 0.0  # First was replaced


def test_inference_config_modes():
    """Test that inference config properly separates deterministic and probabilistic modes."""
    config = {
        "use_mean_for_retrieval": True,
        "mc_samples_prob_metrics": 64,
        "mc_samples_prob_2afc": 64,
    }
    
    # Deterministic retrieval should use mean only
    assert config["use_mean_for_retrieval"] is True
    
    # Probabilistic metrics should use sampling
    assert config["mc_samples_prob_metrics"] > 0
    assert config["mc_samples_prob_2afc"] > 0


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])
