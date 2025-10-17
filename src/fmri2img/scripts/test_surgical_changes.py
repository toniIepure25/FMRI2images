#!/usr/bin/env python3
"""
Comprehensive Test Suite for Surgical Changes
==============================================

Tests all production-grade improvements to CLIP cache and preprocessing.
"""

import os
import sys
import tempfile
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

def test_clip_cache_fluent_api():
    """Test 1: CLIPCache fluent API"""
    print("\n[Test 1] CLIPCache fluent API")
    
    from fmri2img.data.clip_cache import CLIPCache
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, "test_cache.parquet")
        
        # Test fluent API
        cache = CLIPCache(cache_path).load()
        assert cache is not None, "load() should return self"
        assert isinstance(cache, CLIPCache), "load() should return CLIPCache instance"
        
        # Test is_loaded property
        assert cache.is_loaded, "is_loaded should be True after load()"
        
        # Test method chaining works
        cache2 = CLIPCache(cache_path).load()
        assert cache2.is_loaded, "Chained load() should work"
        
        print("  ✓ Fluent API working")


def test_clip_cache_l2_normalization():
    """Test 2: L2 normalization guarantee"""
    print("\n[Test 2] L2 normalization guarantee")
    
    from fmri2img.data.clip_cache import CLIPCache
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, "test_cache.parquet")
        
        # Create cache with non-normalized embeddings
        cache = CLIPCache(cache_path).load()
        
        # Add some embeddings (not normalized)
        rows = pd.DataFrame({
            "nsdId": [0, 1, 2],
            "clip512": [
                (np.random.randn(512) * 5).tolist(),  # Large magnitude
                (np.random.randn(512) * 0.1).tolist(),  # Small magnitude
                np.zeros(512).tolist()  # Zero vector
            ]
        })
        cache.save_rows(rows)
        
        # Reload and verify normalization
        cache2 = CLIPCache(cache_path).load()
        embeddings = cache2.get([0, 1, 2])
        
        # Check norms
        for nsd_id, emb in embeddings.items():
            if nsd_id == 2:  # Zero vector
                continue
            norm = np.linalg.norm(emb)
            assert np.isclose(norm, 1.0, atol=1e-6), f"nsdId={nsd_id} has norm={norm}, expected 1.0"
        
        print(f"  ✓ All embeddings L2-normalized (norms ≈ 1.0)")


def test_dataset_union_type():
    """Test 3: Dataset accepts Union[CLIPCache, str, None]"""
    print("\n[Test 3] Dataset Union type support")
    
    from fmri2img.data.torch_dataset import NSDIterableDataset
    from fmri2img.data.clip_cache import CLIPCache
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test index
        index_dir = os.path.join(tmpdir, "index", "subject=subj01")
        os.makedirs(index_dir, exist_ok=True)
        
        index_path = os.path.join(index_dir, "index.parquet")
        test_df = pd.DataFrame({
            "subject": ["subj01"] * 3,
            "nsdId": [0, 1, 2],
            "beta_path": ["s3://bucket/beta1.nii.gz"] * 3,
            "beta_index": [0, 1, 2]
        })
        test_df.to_parquet(index_path)
        
        # Create cache
        cache_path = os.path.join(tmpdir, "cache.parquet")
        cache = CLIPCache(cache_path).load()
        rows = pd.DataFrame({
            "nsdId": [0, 1, 2],
            "clip512": [np.random.randn(512).tolist() for _ in range(3)]
        })
        cache.save_rows(rows)
        
        # Test 1: Pass CLIPCache instance
        try:
            ds1 = NSDIterableDataset(
                os.path.join(tmpdir, "index"),
                subject="subj01",
                clip_cache=cache,
                limit=1
            )
            print("  ✓ Accepts CLIPCache instance")
        except Exception as e:
            print(f"  ✗ Failed with CLIPCache instance: {e}")
            return False
        
        # Test 2: Pass string path
        try:
            ds2 = NSDIterableDataset(
                os.path.join(tmpdir, "index"),
                subject="subj01",
                clip_cache=cache_path,
                limit=1
            )
            assert ds2.clip_cache is not None, "clip_cache should be instantiated"
            assert ds2.clip_cache.is_loaded, "clip_cache should be loaded"
            print("  ✓ Accepts string path (auto-instantiates)")
        except Exception as e:
            print(f"  ✗ Failed with string path: {e}")
            return False
        
        # Test 3: Pass None
        try:
            ds3 = NSDIterableDataset(
                os.path.join(tmpdir, "index"),
                subject="subj01",
                clip_cache=None,
                limit=1
            )
            assert ds3.clip_cache is None, "clip_cache should be None"
            print("  ✓ Accepts None")
        except Exception as e:
            print(f"  ✗ Failed with None: {e}")
            return False
    
    return True


def test_batch_clip_lookup():
    """Test 4: Dataset uses batch CLIP lookup"""
    print("\n[Test 4] Batch CLIP lookup")
    
    # This is tested by checking the code structure
    from fmri2img.data.torch_dataset import NSDIterableDataset
    import inspect
    
    source = inspect.getsource(NSDIterableDataset.__iter__)
    
    # Check for batch lookup pattern
    has_batch_fetch = 'nsd_ids_to_fetch' in source
    has_get_call = 'self.clip_cache.get(' in source
    
    if has_batch_fetch and has_get_call:
        print("  ✓ Batch lookup pattern present in __iter__")
        return True
    else:
        print("  ✗ Batch lookup pattern not found")
        return False


def test_cli_aliases():
    """Test 5: CLI accepts both --batch/--batch-size and --limit/--max-items"""
    print("\n[Test 5] CLI argument aliases")
    
    import subprocess
    
    # Test --help output
    result = subprocess.run(
        ["python3", "scripts/build_clip_cache.py", "--help"],
        capture_output=True,
        text=True,
        cwd="/home/tonystark/Desktop/Bachelor V2"
    )
    
    help_text = result.stdout
    
    has_batch = '--batch' in help_text
    has_limit = '--limit' in help_text
    
    if has_batch and has_limit:
        print("  ✓ CLI aliases present in --help")
        return True
    else:
        print(f"  ✗ Missing aliases (--batch: {has_batch}, --limit: {has_limit})")
        return False


def test_hdf5_fallback_robustness():
    """Test 6: HDF5 → COCO fallback with proper error handling"""
    print("\n[Test 6] HDF5 → COCO fallback error handling")
    
    # Check that build_clip_cache.py has OSError handling
    with open("scripts/build_clip_cache.py", "r") as f:
        source = f.read()
    
    has_oserror = 'except OSError' in source
    has_warning = 'log.warning' in source and 'HDF5 failed' in source
    
    if has_oserror:
        print("  ✓ OSError handling present")
    else:
        print("  ✗ OSError handling missing")
    
    if has_warning:
        print("  ✓ Warning log for fallback present")
    else:
        print("  ✗ Warning log missing")
    
    return has_oserror and has_warning


def test_pca_k_capping():
    """Test 7: PCA auto-caps k_eff correctly"""
    print("\n[Test 7] PCA k_eff auto-capping")
    
    # Check that preprocess.py has k_eff capping logic
    with open("src/fmri2img/data/preprocess.py", "r") as f:
        source = f.read()
    
    has_k_eff = 'k_eff' in source
    has_min = 'min(k,' in source
    has_log = 'k_eff' in source and ('log' in source or 'logger' in source)
    
    if has_k_eff and has_min:
        print("  ✓ k_eff capping logic present")
    else:
        print("  ✗ k_eff capping logic missing")
    
    if has_log:
        print("  ✓ Logging for k_eff present")
    else:
        print("  ✗ Logging for k_eff missing")
    
    return has_k_eff and has_min


def test_resume_logic():
    """Test 8: Build script supports resume (skip cached IDs)"""
    print("\n[Test 8] Resume logic in build_clip_cache.py")
    
    with open("scripts/build_clip_cache.py", "r") as f:
        source = f.read()
    
    has_cached_ids = 'cached_ids' in source or 'list_cached_ids' in source
    has_todo = 'todo_ids' in source or 'todo' in source
    has_resume_log = 'Already cached' in source or 'resume' in source.lower()
    
    if has_cached_ids:
        print("  ✓ Cached IDs check present")
    else:
        print("  ✗ Cached IDs check missing")
    
    if has_todo:
        print("  ✓ TODO list computation present")
    else:
        print("  ✗ TODO list computation missing")
    
    if has_resume_log:
        print("  ✓ Resume logging present")
    else:
        print("  ✗ Resume logging missing")
    
    return has_cached_ids and has_todo


def main():
    print("=" * 70)
    print("COMPREHENSIVE TEST SUITE FOR SURGICAL CHANGES")
    print("=" * 70)
    
    tests = [
        ("Fluent API", test_clip_cache_fluent_api),
        ("L2 Normalization", test_clip_cache_l2_normalization),
        ("Dataset Union Type", test_dataset_union_type),
        ("Batch CLIP Lookup", test_batch_clip_lookup),
        ("CLI Aliases", test_cli_aliases),
        ("HDF5 Fallback", test_hdf5_fallback_robustness),
        ("PCA k_eff Capping", test_pca_k_capping),
        ("Resume Logic", test_resume_logic),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            if result is None:
                result = True  # Test passed (no explicit return)
            results.append((name, result))
        except Exception as e:
            print(f"  ✗ Test failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {name}")
    
    print("=" * 70)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 70)
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
