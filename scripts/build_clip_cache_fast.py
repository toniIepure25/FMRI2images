#!/usr/bin/env python3
"""
Fast CLIP Cache Builder - Uses Local Cached Images
==================================================

Much faster than downloading from S3/HDF5. Uses your cached stimuli directly.

Usage:
    python scripts/build_clip_cache_fast.py \\
        --subject subj01 \\
        --index-path data/indices/nsd_index/subject=subj01/index.parquet \\
        --stimuli-dir cache/stimuli \\
        --output cache/clip_embeddings/nsd_clipvitl14.parquet \\
        --model-id runwayml/stable-diffusion-v1-5
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

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
    
    if "2-1" in model_id or "2.1" in model_id or "v2-1" in model_id:
        logger.info("Detected SD 2.1 → using OpenCLIP ViT-H/14")
        clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        processor = CLIPImageProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        target_dim = 1024
    else:
        logger.info("Detected SD 1.x → using OpenAI CLIP ViT-L/14")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        target_dim = 768
    
    clip_model = clip_model.to(device).eval()
    
    logger.info(f"✓ Loaded CLIP model (output dim: {target_dim})")
    return clip_model, processor, target_dim


def load_local_images(
    nsd_ids: list,
    coco_ids: list,
    stimuli_dir: Path
) -> Dict[int, Image.Image]:
    """Load images from local cache using COCO IDs."""
    
    images = {}
    missing = []
    
    for nsd_id, coco_id in tqdm(zip(nsd_ids, coco_ids), total=len(nsd_ids), desc="Loading images"):
        # Try both train and val splits
        for split in ["train2017", "val2017"]:
            img_path = stimuli_dir / f"{coco_id:06d}_{split}.jpg"
            if img_path.exists():
                try:
                    img = Image.open(img_path).convert("RGB")
                    images[nsd_id] = img
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {img_path}: {e}")
        
        if nsd_id not in images:
            missing.append(nsd_id)
    
    if missing:
        logger.warning(f"Missing {len(missing)}/{len(nsd_ids)} images: {missing[:5]}...")
    else:
        logger.info(f"✓ Loaded all {len(images)} images from cache")
    
    return images


def compute_embeddings_batch(
    images: Dict[int, Image.Image],
    clip_model,
    processor,
    device: str,
    batch_size: int = 64
) -> Dict[int, np.ndarray]:
    """Compute CLIP embeddings for images."""
    
    embeddings = {}
    nsd_ids = list(images.keys())
    
    for i in tqdm(range(0, len(nsd_ids), batch_size), desc="Computing embeddings"):
        batch_ids = nsd_ids[i:i+batch_size]
        batch_images = [images[nsd_id] for nsd_id in batch_ids]
        
        # Process batch
        inputs = processor(images=batch_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Encode with projection
        with torch.no_grad():
            batch_embeddings = clip_model.get_image_features(**inputs).cpu().numpy()
            
            # L2 normalize
            norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
            batch_embeddings = batch_embeddings / (norms + 1e-8)
        
        # Store
        for nsd_id, emb in zip(batch_ids, batch_embeddings):
            embeddings[nsd_id] = emb
    
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Fast CLIP cache builder using local images")
    parser.add_argument("--subject", default="subj01", help="Subject ID")
    parser.add_argument("--index-path", required=True, help="Path to index parquet file")
    parser.add_argument("--stimuli-dir", default="cache/stimuli", help="Local stimuli directory")
    parser.add_argument("--output", required=True, help="Output parquet file")
    parser.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5", help="Model ID")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("FAST CLIP CACHE BUILDER (using local images)")
    logger.info("="*80)
    logger.info(f"Subject: {args.subject}")
    logger.info(f"Model: {args.model_id}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Device: {args.device}")
    
    # Load index
    index_path = Path(args.index_path)
    if not index_path.exists():
        logger.error(f"Index not found: {index_path}")
        return 1
    
    logger.info(f"Loading index from {index_path}")
    df = pd.read_parquet(index_path)
    
    # Check if cocoId column exists
    if "cocoId" not in df.columns:
        logger.error("Index missing 'cocoId' column. Cannot map to cached images.")
        return 1
    
    nsd_ids = df["nsdId"].tolist()
    coco_ids = df["cocoId"].tolist()
    logger.info(f"Processing {len(nsd_ids)} images")
    
    # Load CLIP encoder
    clip_model, processor, target_dim = load_clip_encoder(args.model_id, args.device)
    
    # Load images from local cache
    stimuli_dir = Path(args.stimuli_dir)
    if not stimuli_dir.exists():
        logger.error(f"Stimuli directory not found: {stimuli_dir}")
        return 1
    
    images = load_local_images(nsd_ids, coco_ids, stimuli_dir)
    
    if not images:
        logger.error("No images loaded! Check stimuli directory.")
        return 1
    
    # Compute embeddings
    embeddings = compute_embeddings_batch(
        images, clip_model, processor, args.device, args.batch_size
    )
    
    # Save to parquet
    df_save = pd.DataFrame([
        {"nsdId": nsd_id, "embedding": emb.tolist()}
        for nsd_id, emb in embeddings.items()
    ])
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_save.to_parquet(output_path, index=False)
    
    logger.info("="*80)
    logger.info(f"✅ Complete! Saved {len(embeddings)} embeddings")
    logger.info(f"   Output: {output_path}")
    logger.info(f"   Dimension: {target_dim}")
    logger.info("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
