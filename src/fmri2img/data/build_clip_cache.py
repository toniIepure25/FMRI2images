#!/usr/bin/env python3
"""
Build CLIP Embedding Cache with Resume Support
==============================================

Populates clip_cache.parquet with embeddings for all images in NSD index.
Loads images from nsd_stimuli.hdf5 via nsdId, with COCO HTTP fallback.
Supports batching, GPU, and automatic resume from existing cache.

CLIP model configuration is loaded from configs/clip.yaml (single source of truth).

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
import os
from pathlib import Path
from typing import List, Optional, Tuple
from glob import glob
from contextlib import nullcontext
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# Import NSD data loading
from fmri2img.data.clip_cache import CLIPCache
from fmri2img.io.s3 import HDF5Loader
from fmri2img.io.nsd_layout import NSDLayout
from fmri2img.io.image_loader import RobustImageLoader
from fmri2img.utils.clip_utils import load_clip_model, load_clip_config, verify_embedding_dimension

# Optional requests for COCO fallback
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# Setup logging early (before any log.info() calls)
log = logging.getLogger("build_clip_cache")
log.setLevel(logging.INFO)
if not log.handlers:
    _sh = logging.StreamHandler(sys.stdout)
    _sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(_sh)


def configure_file_logging(log_file: Optional[str] = None) -> None:
    """
    Configure optional file logging.
    
    Args:
        log_file: Optional path to log file. If None, log to stdout only.
    """
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        log.addHandler(file_handler)
        log.info(f"Log file: {log_file}")
    else:
        log.info("Log file: none (stdout only)")


def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging with stdout always, and optional file handler.
    
    DEPRECATED: Use configure_file_logging() instead.
    This is kept for backward compatibility.
    
    Args:
        log_file: Optional path to log file. If None, log to stdout only.
        
    Returns:
        Configured logger instance
    """
    configure_file_logging(log_file)
    return log


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
    
    Robust handling of S3 HDF5 fragility:
    - Catches OSError for truncated files
    - Returns None on any error (caller handles fallback)
    
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
            
            log.debug(f"✓ Loaded nsdId={nsd_id} from HDF5")
            return img
    except OSError as e:
        # Truncated file or other HDF5 error (common with S3)
        log.debug(f"HDF5 OSError for nsdId={nsd_id}: {e}")
        return None
    except KeyError as e:
        # Missing key in HDF5
        log.debug(f"HDF5 KeyError for nsdId={nsd_id}: {e}")
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
        log.debug(f"✓ Loaded cocoId={coco_id} from COCO HTTP")
        return img
    except Exception as e:
        log.debug(f"COCO HTTP load failed for cocoId={coco_id}: {e}")
        return None


def load_image(
    hdf5_loader: HDF5Loader,
    hdf5_path: str,
    layout: NSDLayout,
    row: pd.Series,
    load_stats: dict
) -> Tuple[Optional[Image.Image], int]:
    """
    Load image for a given index row (nsdId required, cocoId optional).
    
    Tries HDF5 first, falls back to COCO HTTP immediately on any error.
    Logs which path was used (HDF5 vs JPEG) via load_stats.
    
    Args:
        hdf5_loader: HDF5Loader instance
        hdf5_path: S3 path to nsd_stimuli.hdf5
        layout: NSDLayout instance
        row: Index row with nsdId and optionally cocoId/cocoSplit
        load_stats: Dictionary to track loading statistics
        
    Returns:
        (PIL Image or None, nsdId)
    """
    nsd_id = int(row["nsdId"])
    
    # Try HDF5 first
    img = load_image_from_hdf5(hdf5_loader, hdf5_path, nsd_id)
    if img is not None:
        load_stats['hdf5'] = load_stats.get('hdf5', 0) + 1
        return img, nsd_id
    
    # HDF5 failed - try COCO fallback if available
    if "cocoId" in row and pd.notna(row["cocoId"]):
        coco_id = int(row["cocoId"])
        coco_split = row.get("cocoSplit", "train2017")
        if pd.isna(coco_split):
            coco_split = "train2017"
        
        # Single WARNING per nsdId
        log.warning(f"HDF5 failed for nsdId={nsd_id}, falling back to COCO HTTP (cocoId={coco_id})")
        img = load_image_from_coco(layout, coco_id, coco_split)
        if img is not None:
            load_stats['coco_http'] = load_stats.get('coco_http', 0) + 1
            return img, nsd_id
    
    # Both failed
    load_stats['failed'] = load_stats.get('failed', 0) + 1
    return None, nsd_id


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
    Compute CLIP embeddings for a batch of images.
    
    Args:
        model: CLIP model
        preprocess: CLIP preprocessing function
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
                        help="Path to CLIP cache parquet file (canonical output flag)")
    parser.add_argument("--out", type=str, default=None,
                        help="(Alias for --cache) Output path, for backward compatibility")
    parser.add_argument("--batch-size", "--batch", type=int, default=128, dest="batch_size",
                        help="Batch size for CLIP inference")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for CLIP model (cuda/cpu)")
    parser.add_argument("--max-items", "--limit", type=int, default=None, dest="max_items",
                        help="Max items to process (for testing)")
    parser.add_argument("--include-ids", action="store_true", default=True,
                        help="Include nsd_id column in output (default: True)")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Optional log file path (if not set, logs to stdout only)")
    
    # Legacy flags (no-ops, for backward compatibility)
    parser.add_argument("--use-hdf5", action="store_true",
                        help="(Deprecated, no-op) HDF5 is now default")
    
    args = parser.parse_args()
    
    # Handle --out as alias for --cache
    if args.out:
        if args.cache != "outputs/clip_cache/clip.parquet":  # Non-default cache was provided
            # Both provided, --cache wins
            cache_path = args.cache
        else:
            # Only --out provided
            cache_path = args.out
    else:
        cache_path = args.cache
    
    # Configure file logging first (before any other log.info calls)
    configure_file_logging(log_file=args.log_file)
    
    # Log --out alias usage
    if args.out:
        if args.cache != "outputs/clip_cache/clip.parquet":
            log.info(f"Note: Both --out and --cache provided; using --cache={cache_path}")
        else:
            log.info(f"Note: --out is an alias for --cache; writing to {cache_path}")
    
    # Handle legacy --index flag
    if args.index:
        log.warning("⚠️  --index is deprecated. Use --index-file instead.")
        if not args.index_file:
            args.index_file = args.index
    
    # Handle legacy --use-hdf5 flag
    if args.use_hdf5:
        log.warning("⚠️  --use-hdf5 is deprecated (HDF5 is now the default path)")
    
    # Resolve index source with improved default handling
    if not args.index_file and not args.index_root:
        # Compute default based on subject
        subject = args.subject or "subj01"
        default_index = Path("data/indices/nsd_index") / f"subject={subject}" / "index.parquet"
        
        if default_index.exists():
            log.info(f"No index specified, using default: {default_index}")
            args.index_file = str(default_index)
        else:
            log.error(f"NSD index not found at: {default_index}")
            log.error(f"Hint: pass --index-file <.../index.parquet> or --index-root <data/indices/nsd_index>,")
            log.error(f"      or generate the index first (e.g., make nsd-index SUBJECT={subject}).")
            sys.exit(1)
    
    # Log configuration
    log.info("=" * 60)
    log.info("CLIP Cache Build Configuration")
    log.info("=" * 60)
    log.info(f"Subject:     {args.subject or 'all'}")
    log.info(f"Device:      {args.device}")
    log.info(f"Cache path:  {cache_path}")
    log.info(f"Batch size:  {args.batch_size}")
    log.info(f"Limit:       {args.max_items or 'none'}")
    log.info(f"Include IDs: {args.include_ids}")
    log.info("=" * 60)
    
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
    
    # Check if index is empty
    if len(df) == 0:
        log.warning("Index is empty after filtering. Nothing to process.")
        sys.exit(1)
    
    # Get unique nsdIds
    all_nsd_ids = df["nsdId"].unique().tolist()
    log.info(f"Found {len(all_nsd_ids)} unique nsdIds in index")
    
    # Initialize CLIP cache
    log.info(f"Loading CLIP cache from {cache_path}")
    clip_cache = CLIPCache(cache_path=cache_path)
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
    
    # Load CLIP model from config
    log.info("Loading CLIP model from configs/clip.yaml")
    model, preprocess, clip_config = load_clip_model(device=args.device)
    log.info(f"CLIP model: {clip_config['model_name']} → {clip_config['embedding_dim']}-dim embeddings")
    
    # Initialize robust image loader with fallback chain
    layout = NSDLayout()
    local_hdf5 = os.getenv('NSD_HDF5', 'cache/nsd_hdf5/nsd_stimuli.hdf5')
    s3_hdf5 = layout.stim_hdf5_path(full_url=True)
    
    image_loader = RobustImageLoader(
        local_hdf5_path=local_hdf5 if Path(local_hdf5).exists() else None,
        s3_hdf5_path=s3_hdf5,
        coco_cache_dir=".cache/coco",
        enable_warnings=True
    )
    
    log.info(f"Image load order: Local HDF5 → S3 HDF5 → COCO HTTP (with caching)")
    
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
                img = image_loader.load(row)
                
                if img is not None:
                    images.append(img)
                    valid_nsd_ids.append(nsd_id)
                else:
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
            
            # Verify dimension matches config
            verify_embedding_dimension(embeddings, config_path="configs/clip.yaml")
            
            # Build cache rows with proper schema
            if args.include_ids:
                # Include nsd_id as int column + embedding as single list column
                rows = pd.DataFrame({
                    "nsd_id": [int(nid) for nid in valid_nsd_ids],
                    "embedding": [emb.astype(np.float32).tolist() for emb in embeddings]
                })
            else:
                # Only embedding column (backward compatibility)
                rows = pd.DataFrame({
                    "embedding": [emb.astype(np.float32).tolist() for emb in embeddings]
                })
            
            # Also keep legacy "clip512" column name for CLIPCache compatibility
            rows["clip512"] = rows.get("embedding", [emb.astype(np.float32).tolist() for emb in embeddings])
            if args.include_ids and "nsd_id" in rows.columns:
                rows["nsdId"] = rows["nsd_id"]  # Legacy column name
            
            # Save to cache
            clip_cache.save_rows(rows)
            
            total_processed += len(valid_nsd_ids)
            log.debug(f"Batch {batch_idx+1}/{num_batches}: Processed {len(valid_nsd_ids)} images")
        except Exception as e:
            log.error(f"Failed to process batch {batch_idx}: {e}")
            continue
    
    # Get final loading stats
    load_stats = image_loader.get_stats()
    
    # Final stats
    stats = clip_cache.stats()
    log.info("=" * 60)
    log.info(f"✓ CLIP cache build complete!")
    log.info(f"  Total in cache: {stats['cache_size']} embeddings")
    log.info(f"  Newly processed: {total_processed} images")
    log.info(f"  Failed: {total_failed} images")
    log.info(f"  Image loading sources:")
    log.info(f"    - Local HDF5: {load_stats.get('local_hdf5', 0)} images")
    log.info(f"    - S3 HDF5: {load_stats.get('s3_hdf5', 0)} images")
    log.info(f"    - COCO (cached): {load_stats.get('coco_cached', 0)} images")
    log.info(f"    - COCO (HTTP): {load_stats.get('coco_http', 0)} images")
    log.info(f"    - Failed: {load_stats.get('failed', 0)} images")
    log.info(f"  Cache location: {stats['path']}")
    log.info("=" * 60)
    
    # Assert cache is not empty
    if stats['cache_size'] == 0 and len(todo_ids) > 0:
        raise RuntimeError(
            "CLIP cache is empty after processing! "
            "Check that images are accessible and CLIP model is working."
        )
    
    # Validate final schema
    final_df = pd.read_parquet(cache_path)
    log.info(f"Validating final schema at {cache_path}")
    
    # Ensure nsd_id exists (create alias from image_id if needed)
    if "nsd_id" not in final_df.columns:
        if "nsdId" in final_df.columns:
            final_df["nsd_id"] = final_df["nsdId"]
        elif "image_id" in final_df.columns:
            log.info("Creating nsd_id alias from image_id column")
            final_df["nsd_id"] = final_df["image_id"]
        else:
            log.warning("⚠️  Cache missing nsd_id column (compatibility issue)")
    
    # Ensure embedding exists
    if "embedding" not in final_df.columns:
        if "clip512" in final_df.columns:
            log.info("Creating embedding alias from clip512 column")
            final_df["embedding"] = final_df["clip512"]
        else:
            log.warning("⚠️  Cache missing embedding column")
    
    # Save if we added aliases
    if "nsd_id" in final_df.columns or "embedding" in final_df.columns:
        final_df.to_parquet(cache_path, index=False)
    
    log.info(f"✓ Wrote {len(final_df)} rows to {cache_path}")
    if "nsd_id" in final_df.columns and "embedding" in final_df.columns:
        log.info(f"  Schema: nsd_id (int), embedding (512-D float32 list)")
    else:
        log.info(f"  Columns: {list(final_df.columns)}")


if __name__ == "__main__":
    main()
