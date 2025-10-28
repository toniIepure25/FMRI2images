#!/usr/bin/env python3
"""
Reconstruction Evaluation Script
=================================

Evaluates reconstructed images using CLIPScore and retrieval metrics.

Metrics:
- CLIPScore: Per-sample cosine similarity between generated and GT image embeddings
- Retrieval@K: How often generated image retrieves correct GT from gallery
- Ranking stats: Mean/median rank, MRR

Supports both 512-D (ViT-B/32) and target-D (768/1024 for diffusion models) evaluation
using the --use-adapter flag.

Scientific Context:
- CLIPScore measures semantic similarity without pixel-level matching (Hessel et al. 2021)
- Retrieval metrics evaluate how well generated images capture semantic content
- Standard evaluation for image generation quality in neural decoding

Usage:
    # Evaluate in 512-D space (ViT-B/32)
    python scripts/eval_reconstruction.py \\
        --subject subj01 \\
        --recon-dir outputs/recon/subj01/run_001 \\
        --clip-cache outputs/clip_cache/clip.parquet \\
        --out-csv outputs/reports/subj01/recon_eval.csv \\
        --out-fig outputs/reports/subj01/recon_grid.png
    
    # Evaluate in 1024-D space (SD 2.1 target CLIP)
    python scripts/eval_reconstruction.py \\
        --subject subj01 \\
        --recon-dir outputs/recon/subj01/run_001 \\
        --clip-cache outputs/clip_cache/clip.parquet \\
        --use-adapter \\
        --model-id stabilityai/stable-diffusion-2-1 \\
        --out-csv outputs/reports/subj01/recon_eval_1024.csv \\
        --out-fig outputs/reports/subj01/recon_grid_1024.png
"""

import argparse
import json
import logging
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import project modules
from fmri2img.data.nsd_index_reader import read_subject_index
from fmri2img.data.clip_cache import CLIPCache
from fmri2img.io.s3 import get_s3_filesystem
from fmri2img.models.train_utils import train_val_test_split, torch_seed_all
from fmri2img.eval import clip_score, retrieval_at_k, compute_ranking_metrics
from fmri2img.utils.clip_utils import load_clip_model, encode_images


def load_target_clip_encoder(model_id: str, device: str):
    """
    Load the CLIP image encoder from a diffusion model.
    
    Reuses logic from train_clip_adapter.py for consistency.
    
    Args:
        model_id: HuggingFace model ID
        device: Device to load on
    
    Returns:
        (vision_model, processor, target_dim)
    """
    from transformers import CLIPVisionModel, CLIPImageProcessor
    
    logger.info(f"Loading target CLIP encoder from {model_id}...")
    
    try:
        # Try loading from subfolder
        vision_model = CLIPVisionModel.from_pretrained(
            model_id,
            subfolder="image_encoder" if "stable-diffusion" in model_id else None
        )
        processor = CLIPImageProcessor.from_pretrained(
            model_id,
            subfolder="image_encoder" if "stable-diffusion" in model_id else None
        )
        target_dim = vision_model.config.hidden_size
        
    except Exception as e:
        logger.warning(f"Could not load image_encoder subfolder: {e}")
        logger.info("Inferring target CLIP from model ID...")
        
        # Infer from model_id
        if "2-1" in model_id or "2.1" in model_id:
            target_dim = 1024
            logger.info("Detected SD 2.1 → OpenCLIP ViT-H/14 (1024-D)")
            vision_model = CLIPVisionModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
            processor = CLIPImageProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        elif "1-5" in model_id or "1.5" in model_id:
            target_dim = 768
            logger.info("Detected SD 1.5 → CLIP ViT-L/14 (768-D)")
            vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
            processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        else:
            target_dim = 1024
            logger.warning("Unknown model, defaulting to 1024-D (OpenCLIP ViT-H/14)")
            vision_model = CLIPVisionModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
            processor = CLIPImageProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    
    vision_model = vision_model.to(device)
    vision_model.eval()
    
    logger.info(f"✅ Target CLIP encoder loaded: {target_dim}-D")
    
    return vision_model, processor, target_dim


def encode_images_target_clip(
    images: List[Image.Image],
    vision_model,
    processor,
    device: str,
    batch_size: int = 32
) -> np.ndarray:
    """
    Encode images using target CLIP model.
    
    Args:
        images: List of PIL images
        vision_model: CLIP vision model
        processor: CLIP image processor
        device: Device
        batch_size: Batch size for encoding
    
    Returns:
        Embeddings (n_images, target_dim), L2-normalized
    """
    all_embeddings = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        
        # Process batch
        inputs = processor(images=batch, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Encode
        with torch.no_grad():
            outputs = vision_model(**inputs)
            embeddings = outputs.pooler_output.cpu().numpy()
            
            # L2 normalize
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            all_embeddings.append(embeddings)
    
    return np.vstack(all_embeddings)


def find_reconstructed_images(
    recon_dir: Path,
    nsd_ids: np.ndarray,
    map_csv: Optional[Path] = None
) -> Dict[int, Path]:
    """
    Find reconstructed images for given NSD IDs.
    
    Supports two modes:
    1. Pattern matching: *_nsd{nsdId}.* or *_{nsdId}.*
    2. CSV mapping: columns [nsdId, path]
    
    Args:
        recon_dir: Directory containing reconstructed images
        nsd_ids: Array of NSD IDs to find
        map_csv: Optional CSV with nsdId→path mapping
    
    Returns:
        Dictionary: {nsdId: Path}
    """
    nsd_to_path = {}
    
    if map_csv:
        # Load from CSV
        logger.info(f"Loading image paths from {map_csv}")
        df = pd.read_csv(map_csv)
        
        if "nsdId" not in df.columns or "path" not in df.columns:
            raise ValueError("CSV must have 'nsdId' and 'path' columns")
        
        for _, row in df.iterrows():
            nsd_id = int(row["nsdId"])
            if nsd_id in nsd_ids:
                path = recon_dir / row["path"]
                if path.exists():
                    nsd_to_path[nsd_id] = path
                else:
                    logger.warning(f"Image not found: {path}")
    
    else:
        # Pattern matching
        logger.info(f"Searching for images in {recon_dir}")
        
        # Find all image files
        image_files = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]:
            image_files.extend(recon_dir.glob(ext))
        
        logger.info(f"Found {len(image_files)} image files")
        
        # Try to extract NSD ID from filename
        patterns = [
            r"nsd_?(\d+)",  # nsd12345 or nsd_12345
            r"_(\d{5,})(?:_|\.)",  # _12345_ or _12345.
        ]
        
        for img_path in image_files:
            filename = img_path.stem
            
            for pattern in patterns:
                match = re.search(pattern, filename)
                if match:
                    nsd_id = int(match.group(1))
                    if nsd_id in nsd_ids:
                        nsd_to_path[nsd_id] = img_path
                    break
    
    logger.info(f"Matched {len(nsd_to_path)}/{len(nsd_ids)} images")
    
    if len(nsd_to_path) == 0:
        logger.error("No images matched! Check filename pattern or provide --map-csv")
        logger.error("Expected patterns: *_nsd{ID}.* or *_{ID}.*")
        logger.error(f"Example files in {recon_dir}:")
        for i, f in enumerate(recon_dir.glob("*")):
            if i >= 5:
                break
            logger.error(f"  {f.name}")
    
    return nsd_to_path


def create_evaluation_grid(
    nsd_ids: List[int],
    gt_images: List[Image.Image],
    nn_images: List[Image.Image],
    gen_images: List[Image.Image],
    clip_scores: np.ndarray,
    nn_ranks: np.ndarray,
    output_path: Path,
    max_rows: int = 16
) -> None:
    """
    Create visualization grid: GT | NN | Generated
    
    Args:
        nsd_ids: List of NSD IDs
        gt_images: List of ground truth images
        nn_images: List of nearest neighbor images
        gen_images: List of generated images
        clip_scores: CLIPScore for each sample
        nn_ranks: Rank of GT in retrieval for each sample
        output_path: Output path for grid image
        max_rows: Maximum number of rows to show
    """
    n_samples = min(len(nsd_ids), max_rows)
    
    fig = plt.figure(figsize=(15, 4 * n_samples))
    gs = gridspec.GridSpec(n_samples, 3, figure=fig, hspace=0.3, wspace=0.1)
    
    for i in range(n_samples):
        # Ground truth
        ax_gt = fig.add_subplot(gs[i, 0])
        ax_gt.imshow(gt_images[i])
        ax_gt.axis('off')
        ax_gt.set_title(f"GT (nsd{nsd_ids[i]})", fontsize=10, pad=5)
        
        # Nearest neighbor
        ax_nn = fig.add_subplot(gs[i, 1])
        ax_nn.imshow(nn_images[i])
        ax_nn.axis('off')
        rank_str = f"Rank: {nn_ranks[i]}" if nn_ranks[i] > 0 else "Rank: 1 (Perfect)"
        ax_nn.set_title(f"NN Retrieval\n{rank_str}", fontsize=10, pad=5)
        
        # Generated
        ax_gen = fig.add_subplot(gs[i, 2])
        ax_gen.imshow(gen_images[i])
        ax_gen.axis('off')
        score_color = 'green' if clip_scores[i] > 0.5 else 'orange' if clip_scores[i] > 0.3 else 'red'
        ax_gen.set_title(f"Generated\nCLIPScore: {clip_scores[i]:.3f}", 
                        fontsize=10, pad=5, color=score_color)
    
    plt.suptitle("Reconstruction Evaluation: GT | Nearest Neighbor | Generated", 
                fontsize=14, y=0.995)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Evaluation grid saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate reconstructed images")
    
    # Data paths
    parser.add_argument("--index-root", default="data/indices/nsd_index",
                       help="NSD index root directory")
    parser.add_argument("--index-file", help="Path to single index file (overrides --index-root)")
    parser.add_argument("--subject", default="subj01", help="Subject ID")
    
    # Reconstruction
    parser.add_argument("--recon-dir", required=True,
                       help="Directory containing reconstructed images")
    parser.add_argument("--map-csv", help="Optional CSV mapping nsdId→path")
    
    # CLIP cache and model
    parser.add_argument("--clip-cache", default="outputs/clip_cache/clip.parquet",
                       help="Path to ViT-B/32 CLIP cache (512-D)")
    
    # Adapter mode
    parser.add_argument("--use-adapter", action="store_true",
                       help="Evaluate in target CLIP space (768/1024-D)")
    parser.add_argument("--model-id", default="stabilityai/stable-diffusion-2-1",
                       help="Diffusion model ID for target CLIP (if --use-adapter)")
    parser.add_argument("--target-cache-dir", default="outputs/clip_cache",
                       help="Directory for target CLIP embedding cache")
    
    # Output
    parser.add_argument("--out-csv", required=True,
                       help="Output CSV path for per-sample metrics")
    parser.add_argument("--out-fig", required=True,
                       help="Output image path for visualization grid")
    parser.add_argument("--out-json", help="Optional JSON path for aggregate metrics")
    
    # System
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device (cuda/cpu)")
    parser.add_argument("--limit", type=int, help="Limit number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--config", default="configs/data.yaml",
                       help="Data config file")
    
    args = parser.parse_args()
    
    # Set random seeds
    torch_seed_all(args.seed)
    np.random.seed(args.seed)
    
    try:
        logger.info("=" * 80)
        logger.info("RECONSTRUCTION EVALUATION")
        logger.info("=" * 80)
        logger.info(f"Subject: {args.subject}")
        logger.info(f"Recon dir: {args.recon_dir}")
        logger.info(f"Adapter mode: {'ENABLED' if args.use_adapter else 'DISABLED'}")
        if args.use_adapter:
            logger.info(f"Target model: {args.model_id}")
        logger.info(f"Device: {args.device}")
        
        # Load subject index
        if args.index_file:
            logger.info(f"Loading index from {args.index_file}")
            df = pd.read_parquet(args.index_file)
        else:
            logger.info(f"Loading index for {args.subject} from {args.index_root}")
            df = read_subject_index(args.index_root, args.subject)
        
        if args.limit:
            df = df.head(args.limit)
            logger.info(f"Limited to {len(df)} samples")
        
        # Split data (same as training)
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
        
        splits_config = config.get("preprocessing", {}).get("splits", {})
        
        _, _, test_df = train_val_test_split(
            df,
            train_ratio=splits_config.get("train_ratio", 0.8),
            val_ratio=splits_config.get("val_ratio", 0.1),
            test_ratio=splits_config.get("test_ratio", 0.1),
            random_seed=splits_config.get("random_seed", 42)
        )
        
        test_nsd_ids = test_df["nsdId"].values
        logger.info(f"Test set: {len(test_nsd_ids)} samples")
        
        # Find reconstructed images
        recon_dir = Path(args.recon_dir)
        map_csv = Path(args.map_csv) if args.map_csv else None
        
        nsd_to_path = find_reconstructed_images(recon_dir, test_nsd_ids, map_csv)
        
        if len(nsd_to_path) == 0:
            logger.error("No reconstructed images found!")
            return 1
        
        if len(nsd_to_path) < len(test_nsd_ids):
            logger.warning(f"Only found {len(nsd_to_path)}/{len(test_nsd_ids)} images")
            logger.warning("Evaluation will be partial")
        
        # Filter test set to matched images
        matched_nsd_ids = np.array(sorted(nsd_to_path.keys()))
        logger.info(f"Evaluating {len(matched_nsd_ids)} matched samples")
        
        # Setup CLIP model
        if args.use_adapter:
            # Load target CLIP
            vision_model, processor, target_dim = load_target_clip_encoder(
                args.model_id, args.device
            )
            clip_dim = target_dim
            clip_space = f"{target_dim}-D (target)"
        else:
            # Load ViT-B/32
            clip_model, preprocess, clip_dim = load_clip_model()
            clip_model = clip_model.to(args.device)
            clip_model.eval()
            clip_space = "512-D (ViT-B/32)"
        
        logger.info(f"CLIP space: {clip_space}")
        
        # Load ground truth CLIP embeddings
        logger.info(f"Loading ground truth embeddings from {args.clip_cache}")
        clip_cache = CLIPCache(args.clip_cache).load()
        
        if args.use_adapter:
            # Load or compute target CLIP embeddings for GT images
            target_cache_file = Path(args.target_cache_dir) / f"target_clip_{args.model_id.replace('/', '_')}.parquet"
            
            if target_cache_file.exists():
                logger.info(f"Loading target GT embeddings from {target_cache_file}")
                df_target = pd.read_parquet(target_cache_file)
                gt_embeddings_dict = {
                    int(row["nsdId"]): np.array(row["embedding"])
                    for _, row in df_target.iterrows()
                    if int(row["nsdId"]) in matched_nsd_ids
                }
            else:
                logger.error(f"Target CLIP cache not found: {target_cache_file}")
                logger.error("Please run train_clip_adapter.py first to generate target embeddings")
                return 1
        else:
            # Use 512-D cache
            gt_embeddings_dict = clip_cache.get(matched_nsd_ids)
        
        # Load and encode generated images
        logger.info("Loading and encoding generated images...")
        gen_images = []
        gen_embeddings_list = []
        valid_nsd_ids = []
        
        for nsd_id in matched_nsd_ids:
            try:
                # Load image
                img_path = nsd_to_path[nsd_id]
                img = Image.open(img_path).convert("RGB")
                gen_images.append(img)
                valid_nsd_ids.append(nsd_id)
                
            except Exception as e:
                logger.warning(f"Failed to load image for nsd{nsd_id}: {e}")
                continue
        
        logger.info(f"Loaded {len(gen_images)} images")
        
        # Encode generated images
        logger.info("Encoding generated images...")
        if args.use_adapter:
            gen_embeddings = encode_images_target_clip(
                gen_images, vision_model, processor, args.device
            )
        else:
            gen_embeddings = encode_images(
                images=gen_images,
                model=clip_model,
                preprocess=preprocess,
                device=args.device,
            )
        
        logger.info(f"✅ Generated embeddings: {gen_embeddings.shape}")
        
        # Get GT embeddings in order
        gt_embeddings_list = []
        for nsd_id in valid_nsd_ids:
            emb = gt_embeddings_dict.get(nsd_id)
            if emb is None:
                logger.warning(f"Missing GT embedding for nsd{nsd_id}")
                continue
            gt_embeddings_list.append(emb)
        
        gt_embeddings = np.vstack(gt_embeddings_list)
        logger.info(f"✅ GT embeddings: {gt_embeddings.shape}")
        
        # Ensure same length
        min_len = min(len(gen_embeddings), len(gt_embeddings), len(valid_nsd_ids))
        gen_embeddings = gen_embeddings[:min_len]
        gt_embeddings = gt_embeddings[:min_len]
        valid_nsd_ids = valid_nsd_ids[:min_len]
        gen_images = gen_images[:min_len]
        
        # Compute CLIPScore
        logger.info("Computing CLIPScore...")
        clip_scores = clip_score(gen_embeddings, gt_embeddings)
        
        logger.info(f"CLIPScore: {clip_scores.mean():.3f} ± {clip_scores.std():.3f}")
        logger.info(f"Min: {clip_scores.min():.3f}, Max: {clip_scores.max():.3f}")
        
        # Compute retrieval metrics
        logger.info("Computing retrieval metrics...")
        
        # Build gallery (all GT embeddings)
        gallery_embeddings = gt_embeddings
        gt_indices = np.arange(len(gallery_embeddings))  # Each gen image should retrieve its own GT
        
        retrieval_metrics = retrieval_at_k(
            gen_embeddings, gallery_embeddings, gt_indices, ks=(1, 5, 10)
        )
        
        ranking_metrics = compute_ranking_metrics(
            gen_embeddings, gallery_embeddings, gt_indices
        )
        
        for k, v in retrieval_metrics.items():
            logger.info(f"{k}: {v:.4f} ({v*100:.2f}%)")
        
        logger.info(f"Mean rank: {ranking_metrics['mean_rank']:.2f}")
        logger.info(f"Median rank: {ranking_metrics['median_rank']:.2f}")
        logger.info(f"MRR: {ranking_metrics['mrr']:.4f}")
        
        # Find NN ranks for visualization
        from fmri2img.eval import cosine_sim
        sim = cosine_sim(gen_embeddings, gallery_embeddings)
        ranks = np.argsort(-sim, axis=1)
        nn_ranks = []
        for i in range(len(gen_embeddings)):
            gt_pos = np.where(ranks[i] == gt_indices[i])[0][0]
            nn_ranks.append(gt_pos + 1)  # 1-based rank
        nn_ranks = np.array(nn_ranks)
        
        # Load images for visualization (GT and NN)
        logger.info("Loading images for visualization...")
        s3_fs = get_s3_filesystem()
        gt_images_vis = []
        nn_images_vis = []
        
        for i, nsd_id in enumerate(valid_nsd_ids[:16]):  # Limit to 16 for grid
            # Load GT image
            try:
                import io
                img_path = f"nsd-data/nsddata_stimuli/stimuli/nsd/nsd_{nsd_id:05d}.png"
                with s3_fs.open(img_path, "rb") as f:
                    gt_img = Image.open(io.BytesIO(f.read())).convert("RGB")
                gt_images_vis.append(gt_img)
            except Exception as e:
                logger.warning(f"Failed to load GT image for nsd{nsd_id}: {e}")
                gt_images_vis.append(Image.new("RGB", (256, 256), (128, 128, 128)))
            
            # Load NN image
            nn_idx = ranks[i, 0]  # Top-1 retrieval
            nn_nsd_id = valid_nsd_ids[nn_idx]
            try:
                img_path = f"nsd-data/nsddata_stimuli/stimuli/nsd/nsd_{nn_nsd_id:05d}.png"
                with s3_fs.open(img_path, "rb") as f:
                    nn_img = Image.open(io.BytesIO(f.read())).convert("RGB")
                nn_images_vis.append(nn_img)
            except Exception as e:
                logger.warning(f"Failed to load NN image for nsd{nn_nsd_id}: {e}")
                nn_images_vis.append(Image.new("RGB", (256, 256), (128, 128, 128)))
        
        # Create visualization grid
        logger.info("Creating visualization grid...")
        create_evaluation_grid(
            valid_nsd_ids[:16],
            gt_images_vis,
            nn_images_vis,
            gen_images[:16],
            clip_scores[:16],
            nn_ranks[:16],
            Path(args.out_fig)
        )
        
        # Save per-sample CSV
        logger.info("Saving per-sample metrics...")
        results_df = pd.DataFrame({
            "nsdId": valid_nsd_ids,
            "clipscore": clip_scores,
            "rank": nn_ranks,
            "r@1": [1 if r == 1 else 0 for r in nn_ranks],
            "r@5": [1 if r <= 5 else 0 for r in nn_ranks],
            "r@10": [1 if r <= 10 else 0 for r in nn_ranks],
        })
        
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(args.out_csv, index=False)
        logger.info(f"✅ Per-sample metrics saved to {args.out_csv}")
        
        # Save aggregate JSON
        aggregate_metrics = {
            "subject": args.subject,
            "recon_dir": str(args.recon_dir),
            "clip_space": clip_space,
            "clip_dim": clip_dim,
            "use_adapter": args.use_adapter,
            "model_id": args.model_id if args.use_adapter else "ViT-B/32",
            "n_samples": len(valid_nsd_ids),
            "n_test_total": len(test_nsd_ids),
            "clipscore": {
                "mean": float(clip_scores.mean()),
                "std": float(clip_scores.std()),
                "min": float(clip_scores.min()),
                "max": float(clip_scores.max()),
            },
            "retrieval": retrieval_metrics,
            "ranking": ranking_metrics,
        }
        
        if args.out_json:
            json_path = Path(args.out_json)
        else:
            json_path = Path(args.out_csv).parent / Path(args.out_csv).stem + ".json"
        
        with open(json_path, "w") as f:
            json.dump(aggregate_metrics, f, indent=2)
        
        logger.info(f"✅ Aggregate metrics saved to {json_path}")
        
        logger.info("=" * 80)
        logger.info("✅ Evaluation complete!")
        logger.info(f"CSV: {args.out_csv}")
        logger.info(f"JSON: {json_path}")
        logger.info(f"Grid: {args.out_fig}")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
