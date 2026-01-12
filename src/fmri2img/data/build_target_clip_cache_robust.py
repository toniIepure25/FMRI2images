#!/usr/bin/env python3
"""
Robust Target CLIP Cache Builder
================================

Build target CLIP embeddings incrementally with proper error handling
and resumability. Handles large datasets by processing in batches.

Usage:
    python scripts/build_target_clip_cache_robust.py \\
        --subject subj01 \\
        --index-root data/indices/nsd_index \\
        --model-id stabilityai/stable-diffusion-2-1 \\
        --output outputs/clip_cache/target_clip_sd21.parquet \\
        --batch-size 100 \\
        --source individual
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_clip_encoder(model_id: str, device: str):
    """Load CLIP vision encoder from diffusion model."""
    from transformers import CLIPImageProcessor, CLIPModel
    
    logger.info(f"Loading CLIP encoder for {model_id}...")
    
    # SD 2.1 uses OpenCLIP ViT-H/14 with 1024-D embeddings
    if "2-1" in model_id or "2.1" in model_id or "v2-1" in model_id:
        logger.info("Detected SD 2.1 → using OpenCLIP ViT-H/14")
        
        # Load the FULL CLIP model (not just vision_model) to get projection layer
        clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        processor = CLIPImageProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        target_dim = 1024
        
        logger.info(f"✓ Loaded ViT-H/14 with projection layer")
        logger.info(f"  Hidden size: {clip_model.vision_model.config.hidden_size}")
        logger.info(f"  Projection dim: {clip_model.vision_model.config.projection_dim}")
        logger.info(f"  Output embedding dim: {target_dim}")
    else:
        logger.info("Detected SD 1.x → using OpenAI CLIP ViT-L/14")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        target_dim = 768
        
        logger.info(f"✓ Loaded ViT-L/14 with projection layer")
        logger.info(f"  Hidden size: {clip_model.vision_model.config.hidden_size}")
        logger.info(f"  Projection dim: {clip_model.vision_model.config.projection_dim}")
        logger.info(f"  Output embedding dim: {target_dim}")
    
    clip_model = clip_model.to(device).eval()
    
    # Return full model, not just vision_model, so we can use projection
    return clip_model, processor, target_dim


def load_nsd_images_individual(nsd_ids: List[int], s3_fs, max_retries: int = 3) -> Dict[int, Image.Image]:
    """
    Load NSD images individually from S3 (avoids HDF5 issues).
    
    Args:
        nsd_ids: List of NSD IDs (0-72999)
        s3_fs: S3 filesystem
        max_retries: Maximum retry attempts per image
    
    Returns:
        Dictionary mapping nsd_id → PIL Image
    """
    images = {}
    
    for nsd_id in tqdm(nsd_ids, desc="Loading images"):
        # Convert to 73k ID format (5-digit zero-padded)
        img_path = f"nsddata_stimuli/stimuli/nsd/nsd_stimuli_{nsd_id:05d}.png"
        s3_path = f"s3://natural-scenes-dataset/{img_path}"
        
        success = False
        for attempt in range(max_retries):
            try:
                with s3_fs.open(s3_path, "rb") as f:
                    img = Image.open(f).convert("RGB")
                    images[nsd_id] = img
                    success = True
                    break
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.warning(f"Failed to load nsdId={nsd_id} after {max_retries} attempts: {e}")
                else:
                    logger.debug(f"Retry {attempt+1}/{max_retries} for nsdId={nsd_id}")
        
    return images


def compute_embeddings_batch(
    images: Dict[int, Image.Image],
    clip_model,
    processor,
    device: str,
    batch_size: int = 32
) -> Dict[int, np.ndarray]:
    """Compute CLIP embeddings for a batch of images."""
    
    embeddings = {}
    nsd_ids = list(images.keys())
    
    for i in tqdm(range(0, len(nsd_ids), batch_size), desc="Computing embeddings"):
        batch_ids = nsd_ids[i:i+batch_size]
        batch_images = [images[nsd_id] for nsd_id in batch_ids]
        
        # Process batch
        inputs = processor(images=batch_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Encode using full CLIP model to get projected embeddings
        with torch.no_grad():
            # Use get_image_features which applies vision encoder + projection
            batch_embeddings = clip_model.get_image_features(**inputs).cpu().numpy()
            
            # L2 normalize
            norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
            batch_embeddings = batch_embeddings / (norms + 1e-8)
        
        # Store
        for nsd_id, emb in zip(batch_ids, batch_embeddings):
            embeddings[nsd_id] = emb
    
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Build target CLIP cache robustly")
    parser.add_argument("--subject", default="subj01", help="Subject ID")
    parser.add_argument("--index-root", default="data/indices/nsd_index",
                       help="Index directory")
    parser.add_argument("--model-id", default="stabilityai/stable-diffusion-2-1",
                       help="Diffusion model ID")
    parser.add_argument("--output", required=True,
                       help="Output parquet file")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Batch size for image loading")
    parser.add_argument("--inference-batch-size", type=int, default=32,
                       help="Batch size for CLIP inference")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, help="Limit number of images (for testing)")
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("ROBUST TARGET CLIP CACHE BUILDER")
    logger.info("="*80)
    logger.info(f"Subject: {args.subject}")
    logger.info(f"Model: {args.model_id}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Device: {args.device}")
    
    # Load index
    from fmri2img.data.nsd_index_reader import read_subject_index
    
    index_path = Path(args.index_root) / f"subject={args.subject}" / "index.parquet"
    if not index_path.exists():
        logger.error(f"Index not found: {index_path}")
        return 1
    
    logger.info(f"Loading index from {index_path}")
    df = pd.read_parquet(index_path)
    
    if args.limit:
        df = df.head(args.limit)
    
    nsd_ids = df["nsdId"].values
    logger.info(f"Processing {len(nsd_ids)} images")
    
    # Check existing cache
    output_path = Path(args.output)
    existing_ids = set()
    
    if output_path.exists():
        logger.info(f"Loading existing cache from {output_path}")
        df_cache = pd.read_parquet(output_path)
        existing_ids = set(df_cache["nsdId"].values)
        logger.info(f"Found {len(existing_ids)} existing embeddings")
        
        # Load existing data
        existing_data = {
            row["nsdId"]: np.array(row["embedding"])
            for _, row in df_cache.iterrows()
        }
    else:
        existing_data = {}
    
    # Filter to only missing IDs
    missing_ids = [nsd_id for nsd_id in nsd_ids if nsd_id not in existing_ids]
    logger.info(f"Need to compute {len(missing_ids)} new embeddings")
    
    if not missing_ids:
        logger.info("✅ Cache is complete!")
        return 0
    
    # Load CLIP encoder (returns full model with projection)
    clip_model, processor, target_dim = load_clip_encoder(args.model_id, args.device)
    
    # Setup S3
    from fmri2img.io.s3 import get_s3_filesystem
    s3_fs = get_s3_filesystem()
    
    # Process in batches
    all_embeddings = existing_data.copy()
    
    for i in range(0, len(missing_ids), args.batch_size):
        batch_ids = missing_ids[i:i+args.batch_size]
        logger.info(f"Processing batch {i//args.batch_size + 1}/{(len(missing_ids)-1)//args.batch_size + 1}")
        
        # Load images using HDF5 (individual PNGs don't exist in S3)
        from fmri2img.io.nsd_images import load_nsd_images
        
        try:
            images = load_nsd_images(batch_ids, s3_fs=s3_fs, prefer="hdf5")
        except Exception as e:
            logger.warning(f"HDF5 loading failed for batch: {e}")
            logger.info("Trying HTTP fallback...")
            images = load_nsd_images(batch_ids, s3_fs=s3_fs, prefer="http")
        
        if not images:
            logger.warning(f"No images loaded for batch starting at {i}")
            continue
        
        # Compute embeddings
        batch_embeddings = compute_embeddings_batch(
            images, clip_model, processor, args.device, args.inference_batch_size
        )
        
        all_embeddings.update(batch_embeddings)
        
        # Save incrementally
        df_save = pd.DataFrame([
            {"nsdId": nsd_id, "embedding": emb.tolist()}
            for nsd_id, emb in all_embeddings.items()
        ])
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_save.to_parquet(output_path, index=False)
        logger.info(f"💾 Saved {len(all_embeddings)} embeddings to {output_path}")
    
    logger.info("="*80)
    logger.info(f"✅ Complete! Saved {len(all_embeddings)} embeddings")
    logger.info(f"   Output: {output_path}")
    logger.info(f"   Dimension: {target_dim}")
    logger.info("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
