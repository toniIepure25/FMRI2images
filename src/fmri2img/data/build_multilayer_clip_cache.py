#!/usr/bin/env python3
"""
Multi-Layer CLIP Embedding Cache Builder
=========================================

Builds a cache of CLIP features from multiple ViT layers for multi-level
supervision training. Extracts features from layers 4, 8, 12 plus final output.

Usage:
    # Build multi-layer cache for subj01
    python scripts/build_multilayer_clip_cache.py \\
        --index-root data/indices/nsd_index \\
        --subject subj01 \\
        --cache outputs/clip_cache/clip_multilayer.parquet \\
        --layers 4 8 12 \\
        --batch-size 128 \\
        --device cuda
        
    # Resume from existing cache
    python scripts/build_multilayer_clip_cache.py \\
        --index-root data/indices/nsd_index \\
        --subject subj01 \\
        --cache outputs/clip_cache/clip_multilayer.parquet \\
        --resume

Output Schema:
    The cache will have columns:
    - nsdId (int): NSD stimulus ID
    - layer_4 (list[float]): 768-D features from layer 4
    - layer_8 (list[float]): 768-D features from layer 8
    - layer_12 (list[float]): 768-D features from layer 12
    - final (list[float]): 512-D features from final projection
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fmri2img.utils.clip_utils import load_clip_model, encode_images_multilayer
from fmri2img.data.nsd_index_reader import read_subject_index
from fmri2img.io.nsd_layout import NSDLayout

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import h5py
try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    logger.warning("h5py not available - will skip HDF5 loading")

# Try to import requests
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not available - will skip COCO HTTP fallback")


def load_image_from_hdf5(
    hdf5_file,  # Open h5py.File object
    nsd_id: int
) -> Optional[Image.Image]:
    """Load image from open NSD HDF5 file."""
    if hdf5_file is None:
        return None
    
    try:
        if "imgBrick" not in hdf5_file:
            return None
        
        img_arr = hdf5_file["imgBrick"][nsd_id]
        
        if img_arr.ndim == 2:
            img = Image.fromarray(img_arr.astype(np.uint8), mode='L').convert('RGB')
        elif img_arr.ndim == 3:
            img = Image.fromarray(img_arr.astype(np.uint8), mode='RGB')
        else:
            return None
        
        return img
    except Exception:
        return None


def load_image_from_coco(
    coco_id: int,
    coco_split: str = "train2017"
) -> Optional[Image.Image]:
    """Load image from COCO HTTP as fallback."""
    if not REQUESTS_AVAILABLE:
        return None
    
    try:
        url = f"http://images.cocodataset.org/{coco_split}/{coco_id:012d}.jpg"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            from io import BytesIO
            img = Image.open(BytesIO(response.content)).convert('RGB')
            return img
    except Exception:
        pass
    
    return None


def load_nsd_image(
    nsd_id: int,
    coco_id: int,
    coco_split: str,
    hdf5_file  # Open h5py.File object
) -> Optional[Image.Image]:
    """
    Load NSD stimulus image with HDF5 + COCO HTTP fallback.
    
    Args:
        nsd_id: NSD stimulus ID (0-72999)
        coco_id: COCO image ID  
        coco_split: COCO split (train2017/val2017)
        hdf5_file: Open h5py.File object
        
    Returns:
        PIL Image or None if failed
    """
    # Try HDF5 first
    img = load_image_from_hdf5(hdf5_file, nsd_id)
    if img is not None:
        return img
    
    # Fallback to COCO HTTP
    if coco_id > 0 and coco_split:
        img = load_image_from_coco(coco_id, coco_split)
        if img is not None:
            return img
    
    return None


def build_multilayer_cache(
    index_df: pd.DataFrame,
    model,
    preprocess,
    hdf5_path: str,
    layers: List[int],
    batch_size: int = 128,
    device: str = "cuda",
    existing_cache: Optional[pd.DataFrame] = None,
    num_workers: int = 8
) -> pd.DataFrame:
    """
    Build multi-layer CLIP cache with parallel image loading.
    
    Args:
        index_df: NSD index DataFrame
        model: CLIP model
        preprocess: CLIP preprocessing function
        hdf5_path: Path to nsd_stimuli.hdf5
        layers: List of layer indices to extract (e.g., [4, 8, 12])
        batch_size: Batch size for encoding
        device: Device for computation
        existing_cache: Existing cache to skip already processed
        num_workers: Number of parallel image loading threads
        
    Returns:
        DataFrame with columns: nsdId, layer_4, layer_8, layer_12, final
    """
    # Get unique nsdIds
    unique_nsdids = index_df['nsdId'].unique()
    logger.info(f"Total unique nsdIds: {len(unique_nsdids)}")
    
    # Skip already processed if resuming
    if existing_cache is not None:
        already_processed = set(existing_cache['nsdId'].values)
        unique_nsdids = [nid for nid in unique_nsdids if nid not in already_processed]
        logger.info(f"Resuming: {len(already_processed)} already cached, {len(unique_nsdids)} remaining")
    
    # Open HDF5 file once for entire process
    hdf5_file = None
    if H5PY_AVAILABLE and hdf5_path and Path(hdf5_path).exists():
        try:
            hdf5_file = h5py.File(hdf5_path, 'r')
            logger.info(f"✓ Opened HDF5 file: {hdf5_path}")
        except Exception as e:
            logger.warning(f"Failed to open HDF5: {e}")
    
    try:
        # Prepare metadata lookup
        metadata_dict = {}
        for nsd_id in unique_nsdids:
            row = index_df[index_df['nsdId'] == nsd_id].iloc[0]
            metadata_dict[nsd_id] = {
                'coco_id': int(row.get('cocoId', 0)),
                'coco_split': row.get('cocoSplit', '')
            }
        
        # Process in chunks
        results = []
        total_batches = (len(unique_nsdids) + batch_size - 1) // batch_size
        
        with tqdm(total=len(unique_nsdids), desc="Encoding images") as pbar:
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(unique_nsdids))
                batch_nsdids = unique_nsdids[start_idx:end_idx]
                
                # Load images in parallel
                batch_images = []
                batch_valid_nsdids = []
                
                def load_image_task(nsd_id):
                    """Load single image for parallel execution."""
                    meta = metadata_dict[nsd_id]
                    img = load_nsd_image(nsd_id, meta['coco_id'], meta['coco_split'], hdf5_file)
                    return nsd_id, img
                
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = {executor.submit(load_image_task, nid): nid for nid in batch_nsdids}
                    
                    for future in as_completed(futures):
                        nsd_id, img = future.result()
                        if img is not None:
                            batch_images.append(img)
                            batch_valid_nsdids.append(nsd_id)
                
                # Encode batch with CLIP
                if batch_images:
                    features_dict = encode_images_multilayer(
                        model, preprocess, batch_images,
                        layers=layers, device=device, normalize=True
                    )
                    
                    # Store results
                    for i, nid in enumerate(batch_valid_nsdids):
                        row_data = {'nsdId': int(nid)}
                        for layer_name, features in features_dict.items():
                            # Features are already numpy arrays from encode_images_multilayer
                            feat = features[i]
                            if torch.is_tensor(feat):
                                feat = feat.cpu().numpy()
                            row_data[layer_name] = feat.tolist()
                        results.append(row_data)
                
                pbar.update(len(batch_nsdids))
                pbar.set_postfix({'cached': len(results), 'failed': len(batch_nsdids) - len(batch_images)})
        
        # Create DataFrame
        cache_df = pd.DataFrame(results)
        
        # Combine with existing if resuming
        if existing_cache is not None:
            cache_df = pd.concat([existing_cache, cache_df], ignore_index=True)
        
        return cache_df
    
    finally:
        # Close HDF5 file
        if hdf5_file is not None:
            hdf5_file.close()
            logger.info("✓ Closed HDF5 file")


def main():
    parser = argparse.ArgumentParser(
        description="Build multi-layer CLIP embedding cache",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Data arguments
    parser.add_argument("--index-root", default="data/indices/nsd_index",
                       help="Root directory containing subject indices")
    parser.add_argument("--subject", default="subj01",
                       help="Subject ID (e.g., subj01)")
    parser.add_argument("--hdf5-path", default="cache/nsd_hdf5/nsd_stimuli.hdf5",
                       help="Path to NSD HDF5 stimulus file")
    
    # Output arguments
    parser.add_argument("--cache", default="outputs/clip_cache/clip_multilayer.parquet",
                       help="Output cache file (Parquet format)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from existing cache")
    
    # Model arguments
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 8, 12],
                       help="ViT layer indices to extract (e.g., 4 8 12)")
    parser.add_argument("--batch-size", type=int, default=128,
                       help="Batch size for encoding")
    parser.add_argument("--num-workers", type=int, default=8,
                       help="Number of parallel image loading threads")
    parser.add_argument("--device", default="cuda",
                       help="Device for computation (cuda/cpu)")
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("Multi-Layer CLIP Cache Builder")
    logger.info("=" * 70)
    logger.info(f"Subject: {args.subject}")
    logger.info(f"Layers: {args.layers}")
    logger.info(f"Output: {args.cache}")
    logger.info(f"Device: {args.device}")
    
    # Create output directory
    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing cache if resuming
    existing_cache = None
    if args.resume and cache_path.exists():
        logger.info(f"\nResuming from existing cache: {cache_path}")
        existing_cache = pd.read_parquet(cache_path)
        logger.info(f"  Existing entries: {len(existing_cache)}")
    
    # Load CLIP model
    logger.info("\n1. Loading CLIP model...")
    model, preprocess, config = load_clip_model(device=args.device)
    logger.info(f"   Model: {config['model_name']}")
    
    # Load subject index
    logger.info("\n2. Loading subject index...")
    index_df = read_subject_index(args.index_root, args.subject)
    logger.info(f"   Trials: {len(index_df)}")
    logger.info(f"   Unique stimuli: {index_df['nsdId'].nunique()}")
    
    # Build cache
    logger.info("\n3. Building multi-layer cache...")
    cache_df = build_multilayer_cache(
        index_df=index_df,
        model=model,
        preprocess=preprocess,
        hdf5_path=args.hdf5_path,
        layers=args.layers,
        batch_size=args.batch_size,
        device=args.device,
        existing_cache=existing_cache,
        num_workers=args.num_workers
    )
    
    # Save cache
    logger.info(f"\n4. Saving cache to {cache_path}...")
    cache_df.to_parquet(cache_path, index=False)
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("✅ Multi-Layer CLIP Cache Complete!")
    logger.info("=" * 70)
    logger.info(f"Total entries: {len(cache_df)}")
    logger.info(f"Columns: {list(cache_df.columns)}")
    logger.info(f"File size: {cache_path.stat().st_size / 1024**2:.1f} MB")
    logger.info(f"\nLayer dimensions:")
    for col in cache_df.columns:
        if col != 'nsdId':
            dim = len(cache_df[col].iloc[0])
            logger.info(f"  {col}: {dim}-D")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
