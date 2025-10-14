#!/usr/bin/env python3
"""
Integration Test - CLIP Cache End-to-End
========================================

Tests the complete CLIP cache workflow:
1. Create cache
2. Build embeddings (mock)
3. Load in dataset
4. Verify batch output
"""

import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from fmri2img.data.clip_cache import CLIPCache


def test_clip_cache_workflow():
    """Test complete CLIP cache workflow."""
    print("Testing CLIP Cache End-to-End Workflow")
    print("=" * 50)
    
    # Step 1: Create cache
    print("\n[1] Creating CLIPCache...")
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_clip.parquet"
        cache = CLIPCache(cache_path=str(cache_path))
        cache.load()
        print(f"✓ Cache initialized at {cache_path}")
        
        # Step 2: Add mock embeddings
        print("\n[2] Adding mock embeddings...")
        mock_embeddings = []
        for nsd_id in [1, 2, 3, 4, 5]:
            emb = np.random.randn(512).astype(np.float32)
            emb = emb / np.linalg.norm(emb)  # L2 normalize
            mock_embeddings.append({
                "nsdId": nsd_id,
                "clip512": emb.tolist()
            })
        
        df = pd.DataFrame(mock_embeddings)
        cache.save_rows(df)
        print(f"✓ Saved {len(mock_embeddings)} embeddings")
        
        # Step 3: Verify persistence
        print("\n[3] Verifying persistence...")
        cache2 = CLIPCache(cache_path=str(cache_path))
        cache2.load()
        stats = cache2.stats()
        print(f"✓ Reloaded cache: {stats['cache_size']} items")
        
        # Step 4: Test lookup
        print("\n[4] Testing lookup...")
        embeddings = cache2.get([1, 3, 5])
        print(f"✓ Retrieved {len(embeddings)} embeddings")
        for nsd_id, emb in embeddings.items():
            print(f"  - nsdId={nsd_id}: shape={emb.shape}, dtype={emb.dtype}")
            assert emb.shape == (512,), f"Expected (512,), got {emb.shape}"
            assert emb.dtype == np.float32, f"Expected float32, got {emb.dtype}"
        
        # Step 5: Test resume (deduplication)
        print("\n[5] Testing resume/deduplication...")
        # Add overlapping data
        new_embeddings = []
        for nsd_id in [3, 4, 5, 6, 7]:  # 3,4,5 already exist
            emb = np.random.randn(512).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            new_embeddings.append({
                "nsdId": nsd_id,
                "clip512": emb.tolist()
            })
        
        df_new = pd.DataFrame(new_embeddings)
        cache2.save_rows(df_new)
        final_stats = cache2.stats()
        print(f"✓ After adding 5 (3 overlap): {final_stats['cache_size']} total")
        assert final_stats['cache_size'] == 7, f"Expected 7 unique, got {final_stats['cache_size']}"
        
        # Step 6: Test contains
        print("\n[6] Testing contains...")
        for nsd_id in [1, 3, 5, 7]:
            assert cache2.contains(nsd_id), f"nsdId={nsd_id} should be cached"
        assert not cache2.contains(999), "nsdId=999 should not be cached"
        print("✓ Contains checks pass")
        
        # Step 7: Test list_cached_ids
        print("\n[7] Testing list_cached_ids...")
        cached_ids = cache2.list_cached_ids()
        print(f"✓ Cached IDs: {sorted(cached_ids)}")
        assert len(cached_ids) == 7, f"Expected 7, got {len(cached_ids)}"
        assert set(cached_ids) == {1, 2, 3, 4, 5, 6, 7}
    
    print("\n" + "=" * 50)
    print("✅ All CLIP cache tests passed!")


if __name__ == "__main__":
    test_clip_cache_workflow()
