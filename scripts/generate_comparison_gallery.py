#!/usr/bin/env python3
"""
Generate Comparison Galleries for fMRI Reconstruction
====================================================

Creates side-by-side comparison grids showing:
- Ground Truth
- Single sample generation
- Best-of-N generation
- BOI-lite refined generation

Useful for:
- Visual quality assessment
- Paper figures
- Presentations
- Debugging

Usage:
    # Generate comparison gallery for 16 test samples
    python scripts/generate_comparison_gallery.py \\
        --subject subj01 \\
        --encoder-checkpoint checkpoints/two_stage/subj01/two_stage_best.pt \\
        --encoder-type two_stage \\
        --output-dir outputs/galleries/subj01 \\
        --num-samples 16 \\
        --strategies single best_of_8 boi_lite \\
        --grid-cols 4
    
    # Quick test with 4 samples
    python scripts/generate_comparison_gallery.py \\
        --subject subj01 \\
        --encoder-checkpoint checkpoints/mlp/subj01/mlp.pt \\
        --encoder-type mlp \\
        --output-dir outputs/galleries_test \\
        --num-samples 4 \\
        --strategies single best_of_4
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import project modules
from fmri2img.data.nsd_index_reader import read_subject_index
from fmri2img.data.preprocess import NSDPreprocessor
from fmri2img.data.clip_cache import CLIPCache
from fmri2img.io.s3 import get_s3_filesystem, NIfTILoader
from fmri2img.models.ridge import RidgeEncoder
from fmri2img.models.mlp import load_mlp
from fmri2img.models.encoders import load_two_stage_encoder
from fmri2img.models.encoding_model import load_encoding_model
from fmri2img.models.train_utils import train_val_test_split
from fmri2img.generation.diffusion_utils import (
    load_diffusion_pipeline,
    generate_from_clip_embedding,
    load_clip_model
)
from fmri2img.generation.advanced_diffusion import (
    generate_best_of_n,
    refine_with_boi_lite
)
from fmri2img.eval.image_metrics import clip_score


def add_text_to_image(
    image: Image.Image,
    text: str,
    font_size: int = 20,
    position: str = "top"
) -> Image.Image:
    """
    Add text label to image.
    
    Args:
        image: PIL Image
        text: Text to add
        font_size: Font size
        position: "top" or "bottom"
        
    Returns:
        labeled_image: Image with text
    """
    # Create a new image with extra space for text
    img_width, img_height = image.size
    text_height = font_size + 10
    
    if position == "top":
        new_image = Image.new("RGB", (img_width, img_height + text_height), "white")
        new_image.paste(image, (0, text_height))
        text_y = 5
    else:  # bottom
        new_image = Image.new("RGB", (img_width, img_height + text_height), "white")
        new_image.paste(image, (0, 0))
        text_y = img_height + 5
    
    # Draw text
    draw = ImageDraw.Draw(new_image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Center text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_x = (img_width - text_width) // 2
    
    draw.text((text_x, text_y), text, fill="black", font=font)
    
    return new_image


def create_comparison_grid(
    images_dict: Dict[str, List[Image.Image]],
    sample_indices: List[int],
    num_cols: int = 4,
    img_size: int = 256,
    add_labels: bool = True
) -> Image.Image:
    """
    Create a comparison grid showing multiple strategies.
    
    Args:
        images_dict: Dict mapping strategy name to list of images
        sample_indices: Indices of samples to show
        num_cols: Number of columns
        img_size: Size to resize images
        add_labels: Add strategy labels
        
    Returns:
        grid: Combined grid image
    """
    strategies = list(images_dict.keys())
    num_strategies = len(strategies)
    num_samples = len(sample_indices)
    num_rows = (num_samples + num_cols - 1) // num_cols
    
    # Create figure
    fig_width = num_cols * (num_strategies + 1) * 3  # +1 for GT
    fig_height = num_rows * 3
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(
        num_rows, num_cols,
        figure=fig,
        hspace=0.3,
        wspace=0.1
    )
    
    for idx, sample_idx in enumerate(sample_indices):
        row = idx // num_cols
        col = idx % num_cols
        
        # Create subplot for this sample
        ax = fig.add_subplot(gs[row, col])
        ax.axis("off")
        
        # Collect images for this sample (GT + all strategies)
        sample_images = []
        labels = ["Ground Truth"]
        
        # Add GT
        if "ground_truth" in images_dict:
            sample_images.append(images_dict["ground_truth"][sample_idx])
        
        # Add strategies
        for strategy in strategies:
            if strategy == "ground_truth":
                continue
            sample_images.append(images_dict[strategy][sample_idx])
            labels.append(strategy.replace("_", " ").title())
        
        # Create horizontal strip
        strip_width = len(sample_images) * img_size
        strip = Image.new("RGB", (strip_width, img_size))
        
        for i, img in enumerate(sample_images):
            # Resize
            img_resized = img.resize((img_size, img_size), Image.LANCZOS)
            strip.paste(img_resized, (i * img_size, 0))
        
        # Show in subplot
        ax.imshow(strip)
        
        if add_labels:
            # Add labels as title
            ax.set_title(" | ".join(labels), fontsize=10)
    
    plt.tight_layout()
    
    # Convert to PIL
    fig.canvas.draw()
    grid_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    grid_array = grid_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    grid_image = Image.fromarray(grid_array)
    
    plt.close(fig)
    
    return grid_image


def load_ground_truth_images(
    nsd_ids: np.ndarray,
    stimuli_dir: Path
) -> List[Image.Image]:
    """
    Load ground truth NSD images.
    
    Args:
        nsd_ids: NSD stimulus IDs
        stimuli_dir: Path to stimuli directory
        
    Returns:
        images: List of PIL Images
    """
    images = []
    
    for nsd_id in tqdm(nsd_ids, desc="Loading GT images"):
        # NSD images are stored as nsd{nsdId:05d}.png
        img_path = stimuli_dir / f"nsd{nsd_id:05d}.png"
        
        if not img_path.exists():
            logger.warning(f"Missing GT image: {img_path}")
            # Create placeholder
            img = Image.new("RGB", (512, 512), "gray")
        else:
            img = Image.open(img_path).convert("RGB")
        
        images.append(img)
    
    return images


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison galleries",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required
    parser.add_argument("--subject", type=str, required=True,
                        help="Subject ID (e.g., subj01)")
    parser.add_argument("--encoder-checkpoint", type=str, required=True,
                        help="Path to encoder checkpoint")
    parser.add_argument("--encoder-type", type=str, required=True,
                        choices=["ridge", "mlp", "two_stage"],
                        help="Encoder type")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory")
    
    # Data paths
    parser.add_argument("--data-root", type=str, default="s3://natural-scenes-dataset",
                        help="NSD data root")
    parser.add_argument("--cache-root", type=str, default="cache",
                        help="Local cache directory")
    parser.add_argument("--stimuli-dir", type=str, default="cache/stimuli",
                        help="Directory with NSD stimulus images")
    parser.add_argument("--clip-cache", type=str,
                        default="outputs/clip_cache/clip.parquet",
                        help="CLIP cache path")
    
    # Gallery options
    parser.add_argument("--num-samples", type=int, default=16,
                        help="Number of samples to show")
    parser.add_argument("--strategies", nargs="+",
                        default=["single", "best_of_8"],
                        choices=["single", "best_of_4", "best_of_8", "best_of_16", "boi_lite"],
                        help="Generation strategies")
    parser.add_argument("--grid-cols", type=int, default=4,
                        help="Number of columns in grid")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Data split to use")
    
    # Generation parameters
    parser.add_argument("--model-id", type=str,
                        default="stabilityai/stable-diffusion-2-1",
                        help="Diffusion model ID")
    parser.add_argument("--num-inference-steps", type=int, default=50,
                        help="Number of diffusion steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5,
                        help="Guidance scale")
    
    # Compute
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stimuli_dir = Path(args.stimuli_dir)
    
    logger.info("=" * 80)
    logger.info("Comparison Gallery Generation")
    logger.info("=" * 80)
    logger.info(f"Subject: {args.subject}")
    logger.info(f"Encoder: {args.encoder_type}")
    logger.info(f"Strategies: {args.strategies}")
    logger.info(f"Num samples: {args.num_samples}")
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load data
    logger.info("\nLoading data...")
    index_df = read_subject_index(args.subject, args.data_root, args.cache_root)
    train_indices, val_indices, test_indices = train_val_test_split(index_df)
    
    if args.split == "train":
        split_indices = train_indices
    elif args.split == "val":
        split_indices = val_indices
    else:
        split_indices = test_indices
    
    # Select samples
    sample_indices = split_indices[:args.num_samples]
    sample_df = index_df.iloc[sample_indices]
    nsd_ids = sample_df["nsd_id"].values
    
    logger.info(f"Selected {len(sample_indices)} samples from {args.split} split")
    
    # Load fMRI
    logger.info("\nLoading fMRI...")
    fs = get_s3_filesystem() if args.data_root.startswith("s3://") else None
    nifti_loader = NIfTILoader(fs)
    all_fmri = nifti_loader.load_all_trials(index_df, verbose=True)
    fmri_data = all_fmri[sample_indices]
    
    # Preprocess
    logger.info("Preprocessing fMRI...")
    preprocessor = NSDPreprocessor(args.subject, args.cache_root, pca_k=512)
    train_fmri = all_fmri[train_indices]
    preprocessor.fit(train_fmri)
    fmri_features = preprocessor.transform(fmri_data)
    
    # Load encoder
    logger.info("\nLoading encoder...")
    if args.encoder_type == "ridge":
        import pickle
        with open(args.encoder_checkpoint, "rb") as f:
            encoder = pickle.load(f)
        predictions = encoder.predict(fmri_features)
        predictions = predictions / np.linalg.norm(predictions, axis=1, keepdims=True)
        predictions = torch.from_numpy(predictions).float().to(args.device)
    else:
        if args.encoder_type == "mlp":
            encoder = load_mlp(args.encoder_checkpoint, device=args.device)
        else:
            encoder = load_two_stage_encoder(args.encoder_checkpoint, device=args.device)
        
        encoder.eval()
        with torch.no_grad():
            fmri_t = torch.from_numpy(fmri_features).float().to(args.device)
            predictions = encoder(fmri_t)
    
    logger.info(f"Predicted embeddings: {predictions.shape}")
    
    # Load diffusion pipeline
    logger.info("\nLoading diffusion pipeline...")
    pipe = load_diffusion_pipeline(args.model_id, args.device)
    
    # Load CLIP for best-of-N scoring
    clip_model = None
    if any("best_of" in s for s in args.strategies):
        logger.info("Loading CLIP model for best-of-N...")
        clip_model, _ = load_clip_model(args.device)
    
    # Load encoding model for BOI-lite
    encoding_model = None
    if "boi_lite" in args.strategies:
        logger.info("Loading encoding model for BOI-lite...")
        # Try to find encoding model checkpoint
        enc_model_path = Path("checkpoints/encoding_model") / args.subject / "encoding_model.pt"
        if enc_model_path.exists():
            encoding_model = load_encoding_model(str(enc_model_path), device=args.device)
        else:
            logger.warning(f"Encoding model not found at {enc_model_path}, skipping BOI-lite")
            args.strategies = [s for s in args.strategies if s != "boi_lite"]
    
    # Generate images with all strategies
    logger.info("\nGenerating images...")
    images_dict = {}
    
    # Ground truth
    logger.info("Loading ground truth images...")
    images_dict["ground_truth"] = load_ground_truth_images(nsd_ids, stimuli_dir)
    
    # Single sample
    if "single" in args.strategies:
        logger.info("Generating single samples...")
        single_images = []
        for i in tqdm(range(len(predictions))):
            img = generate_from_clip_embedding(
                pipe,
                predictions[i],
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed + i
            )
            single_images.append(img)
        images_dict["single"] = single_images
    
    # Best-of-N strategies
    for strategy in args.strategies:
        if strategy.startswith("best_of_"):
            n = int(strategy.split("_")[-1])
            logger.info(f"Generating best-of-{n}...")
            best_images = []
            for i in tqdm(range(len(predictions))):
                img = generate_best_of_n(
                    pipe,
                    predictions[i].unsqueeze(0),
                    clip_model,
                    n=n,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    seed=args.seed + i,
                    device=args.device
                )
                best_images.append(img)
            images_dict[strategy] = best_images
    
    # BOI-lite
    if "boi_lite" in args.strategies and encoding_model is not None:
        logger.info("Generating with BOI-lite...")
        boi_images = []
        
        # Need initial images
        if "single" in images_dict:
            initial_images = images_dict["single"]
        else:
            logger.info("Generating initial images for BOI-lite...")
            initial_images = []
            for i in tqdm(range(len(predictions))):
                img = generate_from_clip_embedding(
                    pipe,
                    predictions[i],
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    seed=args.seed + i
                )
                initial_images.append(img)
        
        # Refine
        for i in tqdm(range(len(predictions))):
            true_fmri = fmri_features[i:i+1]
            refined = refine_with_boi_lite(
                pipe,
                initial_images[i],
                true_fmri,
                encoding_model,
                num_steps=3,
                num_candidates=4,
                strength=0.3,
                seed=args.seed + i,
                device=args.device
            )
            boi_images.append(refined)
        images_dict["boi_lite"] = boi_images
    
    # Save individual images
    logger.info("\nSaving individual images...")
    for strategy, images in images_dict.items():
        if strategy == "ground_truth":
            continue
        strategy_dir = output_dir / strategy
        strategy_dir.mkdir(exist_ok=True)
        for i, img in enumerate(images):
            img.save(strategy_dir / f"sample_{i:03d}.png")
    
    # Create comparison grid
    logger.info("Creating comparison grid...")
    grid = create_comparison_grid(
        images_dict,
        list(range(len(sample_indices))),
        num_cols=args.grid_cols
    )
    grid.save(output_dir / "comparison_grid.png")
    logger.info(f"Saved comparison grid to {output_dir / 'comparison_grid.png'}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Gallery generation complete!")
    logger.info(f"Results saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
