#!/usr/bin/env python3
"""
Ultra-Fast CLIP Cache Builder - Uses Local HDF5
===============================================

Uses your cached nsd_stimuli.hdf5 file directly. Fastest option!

Usage:
    python scripts/build_clip_cache_hdf5.py \\
        --subject subj01 \\
        --index-path data/indices/nsd_index/subject=subj01/index.parquet \\
        --hdf5-path cache/nsd_hdf5/nsd_stimuli.hdf5 \\
        --stim-info cache/nsd_stim_info_merged.csv \\
        --output cache/clip_embeddings/nsd_clipvitl14.parquet \\
        --model-id runwayml/stable-diffusion-v1-5
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

import h5py
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


def load_images_from_hdf5(
    nsd_ids: list,
    hdf5_path: Path,
    stim_info: pd.DataFrame
) -> Dict[int, Image.Image]:
    """Load images from local HDF5 file."""
    
    # Build mapping: nsdId → row index in HDF5
    row_by_nsd = {int(row.nsdId): i for i, row in stim_info.iterrows()}
    
    images = {}
    missing = []
    
    logger.info(f"Loading images from {hdf5_path}...")
    
    with h5py.File(hdf5_path, "r") as hf:
        img_brick = hf["imgBrick"]
        
        for nsd_id in tqdm(nsd_ids, desc="Loading from HDF5"):
            if nsd_id not in row_by_nsd:
                missing.append(nsd_id)
                continue
            
            row_idx = row_by_nsd[nsd_id]
            
            try:
                # HDF5 stores images as (N, H, W, C) uint8
                img_array = img_brick[row_idx]  # Shape: (425, 425, 3)
                
                # Convert to PIL Image
                img = Image.fromarray(img_array, mode="RGB")
                images[nsd_id] = img
                
            except Exception as e:
                logger.warning(f"Failed to load nsdId={nsd_id} from HDF5: {e}")
                missing.append(nsd_id)
    
    if missing:
        logger.warning(f"Missing {len(missing)}/{len(nsd_ids)} images: {missing[:5]}...")
    else:
        logger.info(f"✓ Loaded all {len(images)} images from HDF5")
    
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
    parser = argparse.ArgumentParser(description="Ultra-fast CLIP cache builder using HDF5")
    parser.add_argument("--subject", default="subj01", help="Subject ID")
    parser.add_argument("--index-path", required=True, help="Path to index parquet file")
    parser.add_argument("--hdf5-path", default="cache/nsd_hdf5/nsd_stimuli.hdf5", help="HDF5 file path")
    parser.add_argument("--stim-info", default="cache/nsd_stim_info_merged.csv", help="Stimulus info CSV")
    parser.add_argument("--output", required=True, help="Output parquet file")
    parser.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5", help="Model ID")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("ULTRA-FAST CLIP CACHE BUILDER (using local HDF5)")
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
    nsd_ids = df["nsdId"].tolist()
    logger.info(f"Processing {len(nsd_ids)} images")
    
    # Load stim_info
    stim_info_path = Path(args.stim_info)
    if not stim_info_path.exists():
        logger.error(f"Stimulus info not found: {stim_info_path}")
        return 1
    
    logger.info("Loading stimulus info...")
    stim_info = pd.read_csv(stim_info_path, index_col=0)
    
    # Check HDF5 file
    hdf5_path = Path(args.hdf5_path)
    if not hdf5_path.exists():
        logger.error(f"HDF5 file not found: {hdf5_path}")
        return 1
    
    # Load CLIP encoder
    clip_model, processor, target_dim = load_clip_encoder(args.model_id, args.device)
    
    # Load images from HDF5
    images = load_images_from_hdf5(nsd_ids, hdf5_path, stim_info)
    
    if not images:
        logger.error("No images loaded!")
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
