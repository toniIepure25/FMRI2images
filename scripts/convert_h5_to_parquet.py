#!/usr/bin/env python3
"""
Convert H5 CLIP cache to parquet format with train/test split.
"""

import h5py
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    # Load H5 cache
    h5_path = Path("cache/nsd_clip_sd21_full.h5")
    print(f"Loading {h5_path}...")
    
    df = pd.read_parquet(h5_path)
    print(f"Loaded {len(df)} embeddings")
    
    # Load index to get train/test split info
    index_path = Path("data/indices/nsd_index/subject=subj01/index.parquet")
    index_df = pd.read_parquet(index_path)
    print(f"Loaded index with {len(index_df)} trials")
    
    # Map nsdId to split (train = first 30 sessions, test = last 10)
    # Sessions 1-30 are training (22770 trials)
    # Sessions 31-40 are testing (6571 trials)
    nsd_to_split = {}
    for _, row in index_df.iterrows():
        nsdId = row["nsdId"]
        session = row["session"]
        split = "train" if session <= 30 else "test"
        nsd_to_split[nsdId] = split
    
    # Add split column to embeddings
    df["split"] = df["nsdId"].map(nsd_to_split)
    
    # Filter out any missing splits
    df = df.dropna(subset=["split"])
    
    print(f"Train embeddings: {(df['split'] == 'train').sum()}")
    print(f"Test embeddings: {(df['split'] == 'test').sum()}")
    
    # Save to cache/clip_embeddings/
    output_dir = Path("cache/clip_embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "nsd_clipvitl14.parquet"
    df.to_parquet(output_path, index=False)
    print(f"✅ Saved to {output_path}")

if __name__ == "__main__":
    main()
