"""
Test Reconstruction Evaluation
==============================

Smoke tests for eval_reconstruction.py and related functions.
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
from PIL import Image

from fmri2img.eval import clip_score, retrieval_at_k, compute_ranking_metrics


def test_clip_score_basic():
    """Test CLIPScore computation with normalized embeddings."""
    # Create normalized embeddings
    gen_emb = np.random.randn(10, 512).astype(np.float32)
    gen_emb = gen_emb / np.linalg.norm(gen_emb, axis=1, keepdims=True)
    
    gt_emb = np.random.randn(10, 512).astype(np.float32)
    gt_emb = gt_emb / np.linalg.norm(gt_emb, axis=1, keepdims=True)
    
    # Compute scores
    scores = clip_score(gen_emb, gt_emb)
    
    # Check output shape and range
    assert scores.shape == (10,)
    assert scores.dtype == np.float32
    assert np.all(scores >= -1.0) and np.all(scores <= 1.0)
    
    print(f"✅ CLIPScore basic test passed: mean={scores.mean():.3f}")


def test_clip_score_perfect():
    """Test CLIPScore with identical embeddings (should be 1.0)."""
    emb = np.random.randn(5, 512).astype(np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    
    scores = clip_score(emb, emb)
    
    # Should be exactly 1.0 for identical embeddings
    assert np.allclose(scores, 1.0, atol=1e-5)
    
    print(f"✅ CLIPScore perfect match test passed: all scores ≈ 1.0")


def test_retrieval_metrics():
    """Test retrieval metrics with known rankings."""
    # Create query and gallery (normalized)
    query = np.random.randn(20, 512).astype(np.float32)
    query = query / np.linalg.norm(query, axis=1, keepdims=True)
    
    gallery = np.random.randn(100, 512).astype(np.float32)
    gallery = gallery / np.linalg.norm(gallery, axis=1, keepdims=True)
    
    # GT indices (each query should retrieve corresponding gallery item)
    gt_indices = np.arange(20)
    
    # Compute metrics
    retrieval = retrieval_at_k(query, gallery, gt_indices, ks=(1, 5, 10))
    ranking = compute_ranking_metrics(query, gallery, gt_indices)
    
    # Check keys
    assert "R@1" in retrieval
    assert "R@5" in retrieval
    assert "R@10" in retrieval
    assert "mean_rank" in ranking
    assert "median_rank" in ranking
    assert "mrr" in ranking
    
    # Check ranges
    assert 0.0 <= retrieval["R@1"] <= 1.0
    assert 0.0 <= retrieval["R@5"] <= 1.0
    assert 0.0 <= retrieval["R@10"] <= 1.0
    assert ranking["mean_rank"] >= 1.0
    assert ranking["median_rank"] >= 1.0
    assert 0.0 <= ranking["mrr"] <= 1.0
    
    print(f"✅ Retrieval metrics test passed: R@1={retrieval['R@1']:.3f}, mean_rank={ranking['mean_rank']:.2f}")


def test_evaluation_pipeline_mock():
    """Test full evaluation pipeline with mock data (no actual model loading)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create dummy reconstructed images
        recon_dir = tmpdir / "recon"
        recon_dir.mkdir()
        
        nsd_ids = [12345, 12346, 12347, 12348]
        
        for nsd_id in nsd_ids:
            # Create dummy image
            img = Image.new("RGB", (256, 256), color=(128, 128, 128))
            img.save(recon_dir / f"gen_nsd{nsd_id}.png")
        
        # Create dummy embeddings
        gen_emb = np.random.randn(len(nsd_ids), 512).astype(np.float32)
        gen_emb = gen_emb / np.linalg.norm(gen_emb, axis=1, keepdims=True)
        
        gt_emb = np.random.randn(len(nsd_ids), 512).astype(np.float32)
        gt_emb = gt_emb / np.linalg.norm(gt_emb, axis=1, keepdims=True)
        
        # Compute metrics
        scores = clip_score(gen_emb, gt_emb)
        gt_indices = np.arange(len(nsd_ids))
        retrieval = retrieval_at_k(gen_emb, gt_emb, gt_indices, ks=(1, 5, 10))
        ranking = compute_ranking_metrics(gen_emb, gt_emb, gt_indices)
        
        # Save results
        results = {
            "n_samples": len(nsd_ids),
            "clipscore": {
                "mean": float(scores.mean()),
                "std": float(scores.std()),
            },
            "retrieval": retrieval,
            "ranking": ranking,
        }
        
        output_json = tmpdir / "results.json"
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        
        # Verify output
        assert output_json.exists()
        
        with open(output_json, "r") as f:
            loaded = json.load(f)
        
        assert loaded["n_samples"] == len(nsd_ids)
        assert "clipscore" in loaded
        assert "retrieval" in loaded
        assert "ranking" in loaded
        
        print(f"✅ Mock evaluation pipeline test passed")
        print(f"   CLIPScore: {results['clipscore']['mean']:.3f}")
        print(f"   R@1: {results['retrieval']['R@1']:.3f}")


def test_filename_pattern_matching():
    """Test filename pattern matching for NSD IDs."""
    import re
    
    patterns = [
        r"nsd_?(\d+)",  # nsd12345 or nsd_12345
        r"_(\d{5,})(?:_|\.)",  # _12345_ or _12345.
    ]
    
    test_cases = [
        ("gen_nsd12345.png", 12345),
        ("output_nsd_00123.jpg", 123),
        ("recon_54321_final.png", 54321),
        ("test_12345.png", 12345),
    ]
    
    for filename, expected_id in test_cases:
        found = False
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                nsd_id = int(match.group(1))
                assert nsd_id == expected_id, f"Expected {expected_id}, got {nsd_id} for {filename}"
                found = True
                break
        assert found, f"No pattern matched for {filename}"
    
    print(f"✅ Filename pattern matching test passed ({len(test_cases)} patterns)")


if __name__ == "__main__":
    # Run tests
    print("Running reconstruction evaluation smoke tests...\n")
    
    test_clip_score_basic()
    test_clip_score_perfect()
    test_retrieval_metrics()
    test_filename_pattern_matching()
    test_evaluation_pipeline_mock()
    
    print("\n" + "=" * 80)
    print("✅ All smoke tests passed!")
    print("=" * 80)
