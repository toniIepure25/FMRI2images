#!/usr/bin/env python3
"""
CLIP Embedding Cache Builder (DEPRECATED - use build_clip_cache.py instead)
============================================================================

⚠️  DEPRECATION NOTICE:
    This script is deprecated in favor of scripts/build_clip_cache.py which:
    - Uses centralized CLIP config from configs/clip.yaml
    - Has better HDF5 + COCO HTTP fallback handling
    - Supports resume from partially built caches
    - Provides comprehensive logging to outputs/logs/

    Please use:
        python scripts/build_clip_cache.py --index-file <path> --cache <output>

LEGACY USAGE (maintained for backward compatibility):
    python scripts/nsd_build_clip_cache.py --stim-info cache/nsd_stim_info_merged.csv --limit 1000
    python scripts/nsd_build_clip_cache.py --from-index data/indices/nsd_index/subject=subj01/
"""

import argparse
import logging
import sys
import pandas as pd
from pathlib import Path
from typing import Optional

# Import our modules
try:
    from fmri2img.data.clip_cache import CLIPCache
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Show deprecation warning on import
logger.warning("=" * 80)
logger.warning("⚠️  DEPRECATION WARNING")
logger.warning("This script (nsd_build_clip_cache.py) is deprecated.")
logger.warning("Please use: scripts/build_clip_cache.py instead")
logger.warning("=" * 80)


def load_stimulus_info(stim_info_path: str, limit: Optional[int] = None) -> pd.DataFrame:
    """Load stimulus information CSV."""
    logger.info(f"Loading stimulus info from {stim_info_path}")
    df = pd.read_csv(stim_info_path)
    
    if limit is not None:
        df = df.head(limit)
        logger.info(f"Limited to {limit} stimuli")
    
    logger.info(f"Loaded {len(df)} stimuli")
    return df


def load_from_index(index_path: str, limit: Optional[int] = None) -> pd.DataFrame:
    """Load stimulus info from NSD index files."""
    index_path = Path(index_path)
    
    if index_path.is_file() and index_path.suffix == '.parquet':
        # Single parquet file
        df = pd.read_parquet(index_path)
    elif index_path.is_dir():
        # Directory with parquet files
        parquet_files = list(index_path.glob("*.parquet"))
        if not parquet_files:
            raise ValueError(f"No parquet files found in {index_path}")
        
        dfs = []
        for pf in parquet_files:
            dfs.append(pd.read_parquet(pf))
        df = pd.concat(dfs, ignore_index=True)
    else:
        raise ValueError(f"Invalid index path: {index_path}")
    
    # Extract unique stimulus info
    if 'nsdId' in df.columns:
        stim_df = df[['nsdId']].drop_duplicates()
        stim_df = stim_df.rename(columns={'nsdId': 'nsd_id'})  # Normalize column name
        
        # Add dummy columns if needed for CLIP processing
        if 'cocoId' in df.columns:
            # Get cocoId mapping for each nsdId
            coco_mapping = df[['nsdId', 'cocoId']].drop_duplicates().set_index('nsdId')['cocoId']
            stim_df['cocoId'] = stim_df['nsd_id'].map(coco_mapping)
        else:
            stim_df['cocoId'] = stim_df['nsd_id']  # fallback
            
        if 'image_url' not in stim_df.columns:
            # Generate COCO URL pattern (this is a placeholder)
            stim_df['image_url'] = stim_df['cocoId'].apply(
                lambda x: f"http://images.cocodataset.org/train2017/{x:012d}.jpg"
            )
    elif 'nsd_id' in df.columns:
        stim_df = df[['nsd_id']].drop_duplicates()
        
        # Add dummy columns if needed for CLIP processing
        if 'cocoId' not in stim_df.columns:
            stim_df['cocoId'] = stim_df['nsd_id']  # fallback
        if 'image_url' not in stim_df.columns:
            # Generate COCO URL pattern (this is a placeholder)
            stim_df['image_url'] = stim_df['cocoId'].apply(
                lambda x: f"http://images.cocodataset.org/train2017/{x:012d}.jpg"
            )
    else:
        raise ValueError("Index does not contain 'nsdId' or 'nsd_id' column")
    
    if limit is not None:
        stim_df = stim_df.head(limit)
        logger.info(f"Limited to {limit} stimuli")
    
    logger.info(f"Loaded {len(stim_df)} unique stimuli from index")
    return stim_df


def main():
    parser = argparse.ArgumentParser(description="Build CLIP embeddings cache for NSD stimuli")
    
    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--stim-info", help="Path to stimulus info CSV file")
    input_group.add_argument("--from-index", help="Path to NSD index file or directory")
    
    # Processing options
    parser.add_argument("--limit", type=int, help="Limit number of stimuli to process")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--save-interval", type=int, default=100, 
                       help="Save cache every N processed items")
    
    # CLIP model options
    parser.add_argument("--model", default="ViT-B-32", help="CLIP model name")
    parser.add_argument("--pretrained", default="openai", help="Pretrained weights")
    
    # Output options
    parser.add_argument("--cache-dir", default="cache/clip_embeddings", 
                       help="Directory for CLIP cache")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be processed without computing embeddings")
    
    args = parser.parse_args()
    
    if not CLIP_AVAILABLE:
        logger.error("CLIP functionality not available. Install dependencies: "
                    "torch, open_clip_torch, pillow, requests")
        return 1
    
    try:
        # Load stimulus data
        if args.stim_info:
            stim_df = load_stimulus_info(args.stim_info, args.limit)
        else:
            stim_df = load_from_index(args.from_index, args.limit)
        
        # Validate required columns
        if 'nsd_id' not in stim_df.columns:
            logger.error("Stimulus data must contain 'nsd_id' column")
            return 1
        
        # Use image_url if available, otherwise construct from cocoId
        if 'image_url' in stim_df.columns:
            image_sources = stim_df['image_url'].tolist()
        elif 'cocoId' in stim_df.columns:
            # Generate COCO URLs (placeholder pattern)
            image_sources = [
                f"http://images.cocodataset.org/train2017/{coco_id:012d}.jpg"
                for coco_id in stim_df['cocoId']
            ]
        else:
            logger.error("Stimulus data must contain 'image_url' or 'cocoId' column")
            return 1
        
        nsd_ids = stim_df['nsd_id'].tolist()
        
        if args.dry_run:
            logger.info(f"DRY RUN: Would process {len(nsd_ids)} stimuli")
            logger.info(f"Sample NSD IDs: {nsd_ids[:5]}")
            logger.info(f"Sample image sources: {image_sources[:5]}")
            logger.info(f"Model: {args.model} ({args.pretrained})")
            logger.info(f"Cache directory: {args.cache_dir}")
            return 0
        
        # Initialize CLIP cache
        logger.info(f"Initializing CLIP cache with model {args.model}")
        clip_cache = CLIPCache(
            cache_dir=args.cache_dir,
            model_name=args.model,
            pretrained=args.pretrained
        )
        
        # Show current cache stats
        stats = clip_cache.cache_stats()
        logger.info(f"Current cache: {stats['cache_size']} embeddings")
        
        # Filter out already cached items
        cached_ids = set(clip_cache.list_cached_ids())
        to_process = [(nsd_id, img_src) for nsd_id, img_src in zip(nsd_ids, image_sources) 
                     if nsd_id not in cached_ids]
        
        if not to_process:
            logger.info("All requested stimuli are already cached!")
            return 0
        
        logger.info(f"Processing {len(to_process)} new embeddings...")
        
        # Extract lists for batch processing
        process_ids, process_sources = zip(*to_process)
        
        # Compute embeddings in batches
        results = clip_cache.batch_compute(
            nsd_ids=list(process_ids),
            image_sources=list(process_sources),
            batch_size=args.batch_size,
            save_interval=args.save_interval
        )
        
        # Final stats
        final_stats = clip_cache.cache_stats()
        logger.info(f"Processing complete!")
        logger.info(f"Cache now contains {final_stats['cache_size']} embeddings")
        logger.info(f"Added {len(results)} new embeddings")
        logger.info(f"Cache file: {final_stats['cache_file']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to build CLIP cache: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())