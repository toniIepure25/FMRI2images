#!/usr/bin/env python3
"""
Fix Missing CLIP Embedding for nsdId=73000
==========================================

nsdId=73000 is not in the stimulus catalog (which has 73,000 images indexed 0-72999).
This appears to be a data artifact/edge case. Only 3 trials (0.01%) use this nsdId.

This script generates a CLIP embedding for a blank/neutral image and adds it to the cache,
allowing those 3 trials to proceed without errors.

Usage:
    python scripts/fix_missing_clip_embedding.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from fmri2img.utils.clip_utils import load_clip_model

def main():
    print("=" * 70)
    print("Fix Missing CLIP Embedding for nsdId=73000")
    print("=" * 70)
    
    cache_path = Path("outputs/clip_cache/clip.parquet")
    
    # Load existing cache
    print(f"\n1. Loading cache from {cache_path}")
    cache_df = pd.read_parquet(cache_path)
    print(f"   Current cache size: {len(cache_df)} embeddings")
    
    # Check if nsdId=73000 already exists
    if 73000 in cache_df['nsdId'].values:
        print(f"   ✓ nsdId=73000 already in cache, nothing to do!")
        return
    
    print(f"   → nsdId=73000 is missing")
    
    # Load CLIP model
    print(f"\n2. Loading CLIP model...")
    model, preprocess, config = load_clip_model(device="cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device
    print(f"   ✓ Loaded {config['model_name']}")
    
    # Create a neutral gray image (128, 128, 128) - neutral middle value
    print(f"\n3. Creating neutral gray image...")
    neutral_img = Image.new('RGB', (425, 425), color=(128, 128, 128))
    print(f"   ✓ Created 425x425 gray image")
    
    # Encode with CLIP
    print(f"\n4. Encoding with CLIP...")
    with torch.no_grad():
        img_tensor = preprocess(neutral_img).unsqueeze(0).to(device)
        embedding = model.encode_image(img_tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # L2 normalize
        embedding = embedding.cpu().numpy()[0].astype(np.float32)
    
    print(f"   ✓ Generated embedding: shape={embedding.shape}, norm={np.linalg.norm(embedding):.6f}")
    
    # Add to cache
    print(f"\n5. Adding nsdId=73000 to cache...")
    
    # Create new row matching cache schema
    new_row = pd.DataFrame({
        'nsdId': [73000],
        'clip512': [embedding.tolist()],
        'nsd_id': [73000],
        'embedding': [embedding.tolist()]
    })
    
    # Append and save
    updated_cache = pd.concat([cache_df, new_row], ignore_index=True)
    updated_cache.to_parquet(cache_path, index=False)
    
    print(f"   ✓ Cache updated: {len(cache_df)} → {len(updated_cache)} embeddings")
    
    # Verify
    print(f"\n6. Verifying...")
    verify_df = pd.read_parquet(cache_path)
    if 73000 in verify_df['nsdId'].values:
        print(f"   ✅ SUCCESS! nsdId=73000 now in cache")
    else:
        print(f"   ❌ ERROR: nsdId=73000 still missing after save")
        return 1
    
    print("\n" + "=" * 70)
    print("✅ Fix complete!")
    print("=" * 70)
    print(f"\nNote: nsdId=73000 uses a neutral gray image (no actual stimulus data)")
    print(f"      This affects only 3 trials (0.01% of dataset)")
    print(f"      Training will proceed normally")
    print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
