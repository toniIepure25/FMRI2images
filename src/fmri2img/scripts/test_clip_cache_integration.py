#!/usr/bin/env python3
"""
Integration tests for CLIP cache ergonomics and dataset integration.
Tests fluent API and string path support.
"""

import tempfile
import numpy as np
import pandas as pd
from pathlib import Path


def test_clip_cache_fluent_api():
    """Test that CLIPCache.load() returns self for fluent chaining."""
    from fmri2img.data.clip_cache import CLIPCache
    
    tmpdir = tempfile.mkdtemp()
    cache_path = Path(tmpdir) / "test_clip.parquet"
    
    try:
        # Test fluent API
        cache = CLIPCache(str(cache_path)).load()
        assert cache is not None, "load() should return self"
        assert cache.is_loaded, "Cache should be marked as loaded"
        
        # Test that we can chain operations
        result = CLIPCache(str(cache_path)).load().stats()
        assert "cache_size" in result
        print("✓ Fluent API test passed")
        
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_clip_cache_l2_normalization():
    """Test that get() returns L2-normalized embeddings."""
    from fmri2img.data.clip_cache import CLIPCache
    
    tmpdir = tempfile.mkdtemp()
    cache_path = Path(tmpdir) / "test_clip.parquet"
    
    try:
        cache = CLIPCache(str(cache_path)).load()
        
        # Add unnormalized embeddings
        emb1 = np.random.randn(512).astype(np.float32) * 10  # Not normalized
        emb2 = np.random.randn(512).astype(np.float32) * 5
        
        rows = pd.DataFrame({
            "nsdId": [1, 2],
            "clip512": [emb1.tolist(), emb2.tolist()]
        })
        cache.save_rows(rows)
        
        # Retrieve and check normalization
        result = cache.get([1, 2])
        for nsd_id, emb in result.items():
            norm = np.linalg.norm(emb)
            assert np.isclose(norm, 1.0, atol=1e-5), f"Expected norm=1.0, got {norm}"
            assert emb.dtype == np.float32, f"Expected float32, got {emb.dtype}"
        
        print("✓ L2 normalization test passed")
        
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_dataset_with_clip_cache_fluent():
    """Test NSDIterableDataset with fluent CLIPCache."""
    from fmri2img.data.clip_cache import CLIPCache
    from fmri2img.data.torch_dataset import NSDIterableDataset
    
    tmpdir = tempfile.mkdtemp()
    cache_path = Path(tmpdir) / "test_clip.parquet"
    
    try:
        # Create and populate cache
        cache = CLIPCache(str(cache_path)).load()
        rows = pd.DataFrame({
            "nsdId": [0, 1, 2],
            "clip512": [np.random.randn(512).astype(np.float32).tolist() for _ in range(3)]
        })
        cache.save_rows(rows)
        
        # Test fluent API: pass CLIPCache(...).load() directly
        ds = NSDIterableDataset(
            "data/indices/nsd_index",
            subject="subj01",
            clip_cache=CLIPCache(str(cache_path)).load(),  # Fluent!
            limit=1,
            shuffle=False
        )
        
        assert ds.clip_cache is not None, "Cache should be attached"
        assert ds.clip_cache.is_loaded, "Cache should be loaded"
        print("✓ Dataset with fluent CLIPCache test passed")
        
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_dataset_with_clip_cache_string():
    """Test NSDIterableDataset with string path to cache."""
    from fmri2img.data.clip_cache import CLIPCache
    from fmri2img.data.torch_dataset import NSDIterableDataset
    
    tmpdir = tempfile.mkdtemp()
    cache_path = Path(tmpdir) / "test_clip.parquet"
    
    try:
        # Create and populate cache
        cache = CLIPCache(str(cache_path)).load()
        rows = pd.DataFrame({
            "nsdId": [0, 1, 2],
            "clip512": [np.random.randn(512).astype(np.float32).tolist() for _ in range(3)]
        })
        cache.save_rows(rows)
        
        # Test string path: pass path directly
        ds = NSDIterableDataset(
            "data/indices/nsd_index",
            subject="subj01",
            clip_cache=str(cache_path),  # String path!
            limit=1,
            shuffle=False
        )
        
        assert ds.clip_cache is not None, "Cache should be attached"
        assert ds.clip_cache.is_loaded, "Cache should be loaded"
        print("✓ Dataset with string path test passed")
        
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    print("=" * 60)
    print("CLIP Cache Integration Tests")
    print("=" * 60)
    
    test_clip_cache_fluent_api()
    test_clip_cache_l2_normalization()
    test_dataset_with_clip_cache_fluent()
    test_dataset_with_clip_cache_string()
    
    print("=" * 60)
    print("✅ All integration tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
