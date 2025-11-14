#!/usr/bin/env python3
"""
Check Target CLIP Cache Dimensions
==================================

Verify that all embeddings in the target cache have consistent dimensions.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

def check_cache_dimensions(cache_path: str):
    """Check dimensions of all embeddings in cache."""
    print(f"Checking cache: {cache_path}")
    print("=" * 80)
    
    if not Path(cache_path).exists():
        print(f"❌ Cache file not found: {cache_path}")
        return 1
    
    # Load cache
    df = pd.read_parquet(cache_path)
    print(f"Total entries: {len(df)}")
    
    # Check dimensions
    dimensions = []
    for idx, row in df.iterrows():
        emb = np.array(row["embedding"])
        dimensions.append(len(emb))
    
    # Count dimension distribution
    dim_counts = Counter(dimensions)
    
    print(f"\nDimension distribution:")
    for dim, count in sorted(dim_counts.items()):
        print(f"  {dim}-D: {count} embeddings ({100*count/len(df):.1f}%)")
    
    if len(dim_counts) == 1:
        print(f"\n✅ All embeddings have consistent dimension: {list(dim_counts.keys())[0]}")
        return 0
    else:
        print(f"\n❌ INCONSISTENT DIMENSIONS DETECTED!")
        print(f"   This will cause adapter training to fail.")
        print(f"   Expected: 1024-D for SD-2.1 (OpenCLIP ViT-H/14)")
        print(f"\n   To fix:")
        print(f"   1. Delete the corrupted cache:")
        print(f"      rm -f {cache_path}")
        print(f"   2. Rebuild with fixed script:")
        print(f"      python scripts/build_target_clip_cache_robust.py \\")
        print(f"          --subject subj01 \\")
        print(f"          --index-root data/indices/nsd_index \\")
        print(f"          --model-id stabilityai/stable-diffusion-2-1 \\")
        print(f"          --output {cache_path} \\")
        print(f"          --batch-size 200 \\")
        print(f"          --inference-batch-size 32 \\")
        print(f"          --device cuda")
        return 1

if __name__ == "__main__":
    cache_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/clip_cache/target_clip_stabilityai_stable_diffusion_2_1.parquet"
    sys.exit(check_cache_dimensions(cache_path))
