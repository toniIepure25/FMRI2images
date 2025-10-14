#!/usr/bin/env python3
"""
Build CLIP Embedding Cache with Resume Support
==============================================

Populates clip_cache.parquet with embeddings for all images in NSD index.
Loads images from nsd_stimuli.hdf5 via nsdId, with COCO HTTP fallback.
Supports batching, GPU, and automatic resume from existing cache.

Usage:
    # From single index file
    python scripts/build_clip_cache.py \
        --index-file data/indices/nsd_index/subject=subj01/index.parquet \
        --cache outputs/clip_cache/clip.parquet \
        --batch 128 --device cuda
    
    # From partitioned index root
    python scripts/build_clip_cache.py \
        --index-root data/indices/nsd_index \
        --subject subj01 \
        --cache outputs/clip_cache/clip.parquet \
        --batch 64 --device cuda --limit 256
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple
from glob import glob
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# Import NSD data loading
from fmri2img.data.clip_cache import CLIPCache
from fmri2img.io.s3 import HDF5Loader
from fmri2img.io.nsd_layout import NSDLayout

# CLIP imports
try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False

# Optional requests for COCO fallback
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


def load_index(
    index_root: Optional[str] = None,
    index_file: Optional[str] = None,
    subject: Optional[str] = None
) -> pd.DataFrame:
    """
    Load NSD index from either partitioned root or single file.
    
    Args:
        index_root: Directory with partitioned Parquets (subject=subjXX/)
        index_file: Single parquet file
        subject: Subject filter (e.g., 'subj01')
        
    Returns:
        DataFrame with at least nsdId column, plus cocoId/cocoSplit if present
    """
    if index_file:
        log.info(f"Loading index from file: {index_file}")
        df = pd.read_parquet(index_file)
    elif index_root:
        log.info(f"Loading index from partitioned root: {index_root}")
        root_path = Path(index_root)
        
        # Try subject-specific partition first if subject is provided
        if subject:
            subject_partition = root_path / f"subject={subject}" / "index.parquet"
            if subject_partition.exists():
                log.info(f"Loading subject partition: {subject_partition}")
                df = pd.read_parquet(subject_partition)
            else:
                # Fall back to globbing
                log.info(f"Subject partition not found, globbing all parquets under {index_root}")
                parquet_files = glob(str(root_path / "**/*.parquet"), recursive=True)
                if not parquet_files:
                    raise FileNotFoundError(f"No parquet files found under {index_root}")
                dfs = [pd.read_parquet(pf) for pf in parquet_files]
                df = pd.concat(dfs, ignore_index=True)
        else:
            # Glob all parquets
            parquet_files = glob(str(root_path / "**/*.parquet"), recursive=True)
            if not parquet_files:
                raise FileNotFoundError(f"No parquet files found under {index_root}")
            log.info(f"Found {len(parquet_files)} parquet files, concatenating...")
            dfs = [pd.read_parquet(pf) for pf in parquet_files]
            df = pd.concat(dfs, ignore_index=True)
    else:
        raise ValueError("Must provide either --index-root or --index-file")
    
    # Normalize column names (handle both snake_case and camelCase)
    column_mapping = {
        "nsd_id": "nsdId",
        "coco_id": "cocoId",
        "coco_split": "cocoSplit"
    }
    df = df.rename(columns=column_mapping)
    
    # Check for required nsdId column
    if "nsdId" not in df.columns:
        raise ValueError("Index must contain 'nsdId' or 'nsd_id' column")
    
    # Drop duplicates on nsdId
    initial_count = len(df)
    df = df.drop_duplicates(subset=["nsdId"]).reset_index(drop=True)
    if len(df) < initial_count:
        log.info(f"Dropped {initial_count - len(df)} duplicate nsdIds")
    
    # Filter by subject if requested and column exists
    if subject and "subject" in df.columns:
        df = df[df["subject"] == subject].reset_index(drop=True)
        log.info(f"Filtered to subject={subject}: {len(df)} rows")
    
    log.info(f"Loaded index with {len(df)} rows")
    return df


def load_image_from_hdf5(
    hdf5_loader: HDF5Loader,
    hdf5_path: str,
    nsd_id: int
) -> Optional[Image.Image]:
    """
    Load image from nsd_stimuli.hdf5 by nsdId.
    
    Args:
        hdf5_loader: HDF5Loader instance
        hdf5_path: S3 path to nsd_stimuli.hdf5
        nsd_id: NSD stimulus ID (0-indexed into imgBrick)
        
    Returns:
        PIL Image or None if failed
    """
    try:
        with hdf5_loader.open(hdf5_path) as hf:
            if "imgBrick" not in hf:
                log.debug(f"'imgBrick' dataset not found in HDF5")
                return None
            
            # Load single image slice
            img_arr = hf["imgBrick"][nsd_id]  # Should be (H, W, 3) or (H, W)
            
            # Convert to PIL Image
            if img_arr.ndim == 2:
                img = Image.fromarray(img_arr.astype(np.uint8), mode='L').convert('RGB')
            elif img_arr.ndim == 3:
                img = Image.fromarray(img_arr.astype(np.uint8), mode='RGB')
            else:
                log.debug(f"Unexpected image shape for nsdId={nsd_id}: {img_arr.shape}")
                return None
            
            return img
    except OSError as e:
        # Truncated file or other HDF5 error - log at debug level
        log.debug(f"HDF5 OSError for nsdId={nsd_id}: {e}")
        return None
    except Exception as e:
        log.debug(f"HDF5 load failed for nsdId={nsd_id}: {e}")
        return None


def load_image_from_coco(
    layout: NSDLayout,
    coco_id: int,
    coco_split: str = "train2017"
) -> Optional[Image.Image]:
    """
    Load image from COCO HTTP as fallback.
    
    Args:
        layout: NSDLayout instance
        coco_id: COCO image ID
        coco_split: COCO dataset split
        
    Returns:
        PIL Image or None if failed
    """
    if not REQUESTS_AVAILABLE:
        return None
    
    try:
        url = layout.coco_http_url(coco_id, coco_split)
        log.debug(f"Fetching COCO image from {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        from io import BytesIO
        img = Image.open(BytesIO(response.content)).convert('RGB')
        return img
    except Exception as e:
        log.debug(f"COCO HTTP load failed for cocoId={coco_id}: {e}")
        return None


def load_image(
    hdf5_loader: HDF5Loader,
    hdf5_path: str,
    layout: NSDLayout,
    row: pd.Series
) -> Tuple[Optional[Image.Image], int]:
    """
    Load image for a given index row (nsdId required, cocoId optional).
    Tries HDF5 first, falls back to COCO HTTP immediately on any error.
    
    Args:
        hdf5_loader: HDF5Loader instance
        hdf5_path: S3 path to nsd_stimuli.hdf5
        layout: NSDLayout instance
        row: Index row with nsdId and optionally cocoId/cocoSplit
        
    Returns:
        (PIL Image or None, nsdId)
    """
    nsd_id = int(row["nsdId"])
    
    # Try HDF5 first
    img = load_image_from_hdf5(hdf5_loader, hdf5_path, nsd_id)
    if img is not None:
        return img, nsd_id
    
    # HDF5 failed - try COCO fallback if available
    if "cocoId" in row and pd.notna(row["cocoId"]):
        coco_id = int(row["cocoId"])
        coco_split = row.get("cocoSplit", "train2017")
        if pd.isna(coco_split):
            coco_split = "train2017"
        
        log.warning(f"HDF5 failed for nsdId={nsd_id}, falling back to COCO HTTP")
        img = load_image_from_coco(layout, coco_id, coco_split)
        if img is not None:
            log.debug(f"Successfully loaded nsdId={nsd_id} via COCO fallback")
            return img, nsd_id
    
    return None, nsd_id


def load_clip_model(device: str = "cuda"):
    """Load OpenCLIP ViT-B/32 model and preprocessor."""
    if not OPEN_CLIP_AVAILABLE:
        raise ImportError("open_clip_torch required. Install with: pip install open-clip-torch")
    
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model = model.to(device).eval()
    log.info(f"Loaded CLIP ViT-B/32 model on {device}")
    return model, preprocess


def autocast_ctx(device: str):
    """
    Get appropriate autocast context for device.
    
    Args:
        device: Device string ("cuda" or "cpu")
        
    Returns:
        Context manager for autocast or nullcontext
    """
    if device == "cuda" and torch.cuda.is_available():
        return torch.amp.autocast("cuda")
    return nullcontext()


def compute_embeddings_batch(
    model,
    preprocess,
    images: List[Image.Image],
    device: str = "cuda"
) -> np.ndarray:
    """
    Compute CLIP embeddings for batch of PIL images.
    
    Args:
        model: CLIP model
        preprocess: CLIP preprocessing transform
        images: List of PIL Images
        device: Device for computation
    
    Returns:
        (N, 512) float32 array, L2 normalized
    """
    # Preprocess images
    imgs_tensor = torch.stack([preprocess(img) for img in images]).to(device)
    
    # Extract embeddings with autocast
    with torch.no_grad(), autocast_ctx(device):
        features = model.encode_image(imgs_tensor)
        # L2 normalize
        features = features / features.norm(dim=-1, keepdim=True)
    
    return features.cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Build CLIP embedding cache for NSD dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From single index file
  python scripts/build_clip_cache.py \\
      --index-file data/indices/nsd_index/subject=subj01/index.parquet \\
      --cache outputs/clip_cache/clip.parquet \\
      --batch 64 --device cuda --limit 256
  
  # From partitioned index root
  python scripts/build_clip_cache.py \\
      --index-root data/indices/nsd_index \\
      --subject subj01 \\
      --cache outputs/clip_cache/clip.parquet \\
      --batch 128 --device cuda
        """
    )
    
    # Index source (mutually exclusive)
    index_group = parser.add_mutually_exclusive_group()
    index_group.add_argument("--index-root", type=str, default=None,
                             help="Directory with partitioned Parquets (subject=subjXX/)")
    index_group.add_argument("--index-file", type=str, default=None,
                             help="Single parquet index file")
    
    # Legacy aliases (for backward compatibility)
    parser.add_argument("--index", type=str, default=None,
                        help="(Deprecated) Alias for --index-file")
    
    # Filtering and processing
    parser.add_argument("--subject", type=str, default=None,
                        help="Subject filter (e.g., 'subj01')")
    parser.add_argument("--cache", type=str, default="outputs/clip_cache/clip.parquet",
                        help="Path to CLIP cache parquet file")
    parser.add_argument("--batch-size", "--batch", type=int, default=128, dest="batch_size",
                        help="Batch size for CLIP inference")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for CLIP model (cuda/cpu)")
    parser.add_argument("--max-items", "--limit", type=int, default=None, dest="max_items",
                        help="Max items to process (for testing)")
    
    # Legacy flags (no-ops, for backward compatibility)
    parser.add_argument("--use-hdf5", action="store_true",
                        help="(Deprecated, no-op) HDF5 is now default")
    
    args = parser.parse_args()
    
    # Handle legacy --index flag
    if args.index:
        log.warning("⚠️  --index is deprecated. Use --index-file instead.")
        if not args.index_file:
            args.index_file = args.index
    
    # Handle legacy --use-hdf5 flag
    if args.use_hdf5:
        log.warning("⚠️  --use-hdf5 is deprecated (HDF5 is now the default path)")
    
    # Validate index source
    if not args.index_file and not args.index_root:
        # Try default path
        default_path = "data/indices/nsd_index/subject=subj01/index.parquet"
        if Path(default_path).exists():
            log.info(f"No index specified, using default: {default_path}")
            args.index_file = default_path
        else:
            parser.print_help()
            print("\n❌ Error: Must provide either --index-root or --index-file")
            print(f"   (Default path {default_path} not found)")
            sys.exit(1)
    
    # Load index
    try:
        df = load_index(
            index_root=args.index_root,
            index_file=args.index_file,
            subject=args.subject
        )
    except Exception as e:
        log.error(f"Failed to load index: {e}")
        sys.exit(1)
    
    # Get unique nsdIds
    all_nsd_ids = df["nsdId"].unique().tolist()
    log.info(f"Found {len(all_nsd_ids)} unique nsdIds in index")
    
    # Initialize CLIP cache
    log.info(f"Loading CLIP cache from {args.cache}")
    clip_cache = CLIPCache(cache_path=args.cache)
    clip_cache.load()
    
    # Compute todo list (resume logic)
    cached_ids = set(clip_cache.list_cached_ids())
    log.info(f"Already cached: {len(cached_ids)} nsdIds")
    
    todo_ids = [nid for nid in all_nsd_ids if nid not in cached_ids]
    if args.max_items:
        todo_ids = todo_ids[:args.max_items]
    
    log.info(f"Need to compute: {len(todo_ids)} nsdIds")
    
    if len(todo_ids) == 0:
        log.info("✓ All embeddings already cached!")
        return
    
    # Load CLIP model
    model, preprocess = load_clip_model(device=args.device)
    
    # Initialize loaders
    hdf5_loader = HDF5Loader()
    layout = NSDLayout()
    hdf5_path = layout.stim_hdf5_path(full_url=True)
    log.info(f"Will load images from: {hdf5_path}")
    
    # Create lookup for rows by nsdId (handle multiple rows per nsdId)
    nsd_to_row = {}
    for _, row in df.iterrows():
        nsd_id = int(row["nsdId"])
        if nsd_id not in nsd_to_row:
            nsd_to_row[nsd_id] = row
    
    # Process in batches
    batch_size = args.batch_size
    num_batches = (len(todo_ids) + batch_size - 1) // batch_size
    
    log.info(f"Processing {len(todo_ids)} images in {num_batches} batches of size {batch_size}")
    
    total_processed = 0
    total_failed = 0
    
    for batch_idx in tqdm(range(num_batches), desc="Building CLIP cache"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(todo_ids))
        batch_nsd_ids = todo_ids[start_idx:end_idx]
        
        # Load images
        images = []
        valid_nsd_ids = []
        
        for nsd_id in batch_nsd_ids:
            try:
                if nsd_id not in nsd_to_row:
                    log.warning(f"nsdId={nsd_id} not found in index")
                    total_failed += 1
                    continue
                
                row = nsd_to_row[nsd_id]
                img, _ = load_image(hdf5_loader, hdf5_path, layout, row)
                
                if img is not None:
                    images.append(img)
                    valid_nsd_ids.append(nsd_id)
                else:
                    log.warning(f"Failed to load image for nsdId={nsd_id}")
                    total_failed += 1
            except Exception as e:
                log.warning(f"Error loading nsdId={nsd_id}: {e}")
                total_failed += 1
                continue
        
        if len(images) == 0:
            continue
        
        # Compute embeddings
        try:
            embeddings = compute_embeddings_batch(model, preprocess, images, device=args.device)
            
            # Save to cache
            rows = pd.DataFrame({
                "nsdId": valid_nsd_ids,
                "clip512": [emb.tolist() for emb in embeddings]
            })
            clip_cache.save_rows(rows)
            
            total_processed += len(valid_nsd_ids)
            log.debug(f"Batch {batch_idx+1}/{num_batches}: Processed {len(valid_nsd_ids)} images")
        except Exception as e:
            log.error(f"Failed to process batch {batch_idx}: {e}")
            continue
    
    # Final stats
    stats = clip_cache.stats()
    log.info("=" * 60)
    log.info(f"✓ CLIP cache build complete!")
    log.info(f"  Total in cache: {stats['cache_size']} embeddings")
    log.info(f"  Newly processed: {total_processed} images")
    log.info(f"  Failed: {total_failed} images")
    log.info(f"  Cache location: {stats['path']}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
