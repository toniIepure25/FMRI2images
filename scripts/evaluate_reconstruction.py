#!/usr/bin/env python3
"""
Evaluate reconstruction quality by comparing generated images with ground truth.

Metrics:
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
- CLIP Similarity (in CLIP embedding space)
- Pixel MSE
- Inception Score (IS)

Outputs:
- Quantitative metrics (JSON, CSV)
- Comparison grid images
- Per-sample scores
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Import CLIP for similarity
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logger.warning("CLIP not available, will skip CLIP similarity")

# Import LPIPS
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    logger.warning("LPIPS not available, will skip perceptual similarity")

# Import SSIM
try:
    from skimage.metrics import structural_similarity as ssim
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False
    logger.warning("SSIM not available, will skip structural similarity")


def load_image_pairs(recon_dir: Path, stimuli_dir: Path, index_df: pd.DataFrame, 
                     limit: int = None) -> List[Tuple[Image.Image, Image.Image, str]]:
    """
    Load reconstruction-groundtruth image pairs.
    
    Returns:
        List of (reconstructed_image, ground_truth_image, filename) tuples
    """
    pairs = []
    recon_files = sorted(list(recon_dir.glob("sample_*.png")))
    
    if limit:
        recon_files = recon_files[:limit]
    
    logger.info(f"Loading {len(recon_files)} image pairs...")
    
    for recon_file in tqdm(recon_files, desc="Loading pairs"):
        try:
            # Load reconstructed image
            recon_img = Image.open(recon_file).convert("RGB")
            
            # Get corresponding ground truth from index
            idx = int(recon_file.stem.split("_")[1])
            if idx >= len(index_df):
                logger.warning(f"Index {idx} out of range, skipping")
                continue
            
            # Get original stimulus filename from index
            row = index_df.iloc[idx]
            stim_file = row.get("filename", row.get("stim_locator", ""))
            
            if not stim_file:
                logger.warning(f"No filename found for index {idx}, skipping")
                continue
            
            # Find stimulus in cache
            gt_file = stimuli_dir / stim_file
            
            if not gt_file.exists():
                logger.warning(f"Ground truth not found: {gt_file}, skipping")
                continue
            
            gt_img = Image.open(gt_file).convert("RGB")
            
            # Resize to same dimensions (resize GT to match reconstruction)
            if recon_img.size != gt_img.size:
                gt_img = gt_img.resize(recon_img.size, Image.Resampling.LANCZOS)
            
            pairs.append((recon_img, gt_img, stim_file))
            
        except Exception as e:
            logger.error(f"Failed to load pair for {recon_file}: {e}")
            continue
    
    logger.info(f"✓ Loaded {len(pairs)} valid pairs")
    return pairs


def compute_pixel_metrics(img1: np.ndarray, img2: np.ndarray) -> Dict[str, float]:
    """Compute pixel-level metrics (MSE)."""
    mse = mean_squared_error(img1.flatten(), img2.flatten())
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
    
    return {
        "mse": float(mse),
        "psnr": float(psnr)
    }


def compute_ssim(img1: Image.Image, img2: Image.Image) -> float:
    """Compute SSIM between two images."""
    if not SSIM_AVAILABLE:
        return -1.0
    
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    
    # Compute SSIM for each channel and average
    ssim_vals = []
    for i in range(3):
        s = ssim(arr1[:, :, i], arr2[:, :, i], data_range=255)
        ssim_vals.append(s)
    
    return float(np.mean(ssim_vals))


def compute_lpips(img1: Image.Image, img2: Image.Image, lpips_model) -> float:
    """Compute LPIPS (perceptual similarity)."""
    if not LPIPS_AVAILABLE or lpips_model is None:
        return -1.0
    
    # Convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img1_tensor = transform(img1).unsqueeze(0)
    img2_tensor = transform(img2).unsqueeze(0)
    
    if torch.cuda.is_available():
        img1_tensor = img1_tensor.cuda()
        img2_tensor = img2_tensor.cuda()
    
    with torch.no_grad():
        dist = lpips_model(img1_tensor, img2_tensor)
    
    return float(dist.item())


def compute_clip_similarity(img1: Image.Image, img2: Image.Image, 
                            clip_model, clip_preprocess, device) -> float:
    """Compute CLIP cosine similarity."""
    if not CLIP_AVAILABLE or clip_model is None:
        return -1.0
    
    img1_tensor = clip_preprocess(img1).unsqueeze(0).to(device)
    img2_tensor = clip_preprocess(img2).unsqueeze(0).to(device)
    
    with torch.no_grad():
        feat1 = clip_model.encode_image(img1_tensor)
        feat2 = clip_model.encode_image(img2_tensor)
        
        feat1 = feat1 / feat1.norm(dim=-1, keepdim=True)
        feat2 = feat2 / feat2.norm(dim=-1, keepdim=True)
        
        similarity = (feat1 * feat2).sum().item()
    
    return float(similarity)


def evaluate_reconstructions(pairs: List[Tuple[Image.Image, Image.Image, int]],
                            device: str = "cuda") -> pd.DataFrame:
    """Evaluate all reconstruction pairs."""
    
    # Load models
    lpips_model = None
    if LPIPS_AVAILABLE:
        logger.info("Loading LPIPS model...")
        lpips_model = lpips.LPIPS(net='alex')
        if torch.cuda.is_available():
            lpips_model = lpips_model.cuda()
        lpips_model.eval()
    
    clip_model = None
    clip_preprocess = None
    if CLIP_AVAILABLE:
        logger.info("Loading CLIP model...")
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        clip_model.eval()
    
    # Compute metrics for each pair
    results = []
    
    logger.info("Computing metrics...")
    for recon_img, gt_img, nsd_id in tqdm(pairs, desc="Evaluating"):
        metrics = {"nsd_id": nsd_id}
        
        # Convert to numpy for pixel metrics
        recon_arr = np.array(recon_img)
        gt_arr = np.array(gt_img)
        
        # Pixel metrics
        pixel_metrics = compute_pixel_metrics(recon_arr, gt_arr)
        metrics.update(pixel_metrics)
        
        # SSIM
        metrics["ssim"] = compute_ssim(recon_img, gt_img)
        
        # LPIPS
        metrics["lpips"] = compute_lpips(recon_img, gt_img, lpips_model)
        
        # CLIP similarity
        metrics["clip_sim"] = compute_clip_similarity(
            recon_img, gt_img, clip_model, clip_preprocess, device
        )
        
        results.append(metrics)
    
    return pd.DataFrame(results)


def create_comparison_grid(pairs: List[Tuple[Image.Image, Image.Image, int]], 
                          output_path: Path, n_samples: int = 8):
    """Create a visual comparison grid."""
    n_cols = 2  # Reconstructed, Ground Truth
    n_rows = min(n_samples, len(pairs))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 4 * n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_rows):
        recon_img, gt_img, filename = pairs[i]
        
        # Reconstructed
        axes[i, 0].imshow(recon_img)
        axes[i, 0].set_title(f"Reconstructed\n{filename[:20]}...")
        axes[i, 0].axis("off")
        
        # Ground truth
        axes[i, 1].imshow(gt_img)
        axes[i, 1].set_title(f"Ground Truth\n{filename[:20]}...")
        axes[i, 1].axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"✓ Saved comparison grid to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate reconstruction quality")
    parser.add_argument("--recon-dir", type=str, required=True, help="Reconstruction directory")
    parser.add_argument("--stimuli-dir", type=str, required=True, help="NSD stimuli cache directory")
    parser.add_argument("--subject", type=str, required=True, help="Subject ID (e.g., subj01)")
    parser.add_argument("--index-root", type=str, required=True, help="Index root directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("RECONSTRUCTION EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Reconstruction dir: {args.recon_dir}")
    logger.info(f"Stimuli dir: {args.stimuli_dir}")
    logger.info(f"Subject: {args.subject}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 80)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load index (same split logic as decode_two_stage.py)
    logger.info("Loading index...")
    from fmri2img.data.nsd_index_reader import read_subject_index
    index_data = read_subject_index(args.index_root, args.subject)
    index_df = pd.DataFrame(index_data)
    
    # Load stimulus info to get filenames
    stim_info = pd.read_csv("cache/nsd_stim_info_merged.csv")
    
    # Merge to add filenames
    index_df = index_df.merge(
        stim_info[['nsdId', 'cocoId', 'cocoSplit']],
        on='nsdId',
        how='left',
        suffixes=('', '_stim')
    )
    
    # Use cocoId from stim_info if not in index
    if 'cocoId_stim' in index_df.columns:
        index_df['cocoId'] = index_df['cocoId_stim'].fillna(index_df['cocoId'])
        index_df['cocoSplit'] = index_df['cocoSplit_stim'].fillna(index_df['cocoSplit'])
    
    # Construct filename from cocoId and cocoSplit
    index_df['filename'] = index_df['cocoId'].astype(int).astype(str) + '_' + index_df['cocoSplit'].astype(str) + '.jpg'
    
    # Apply same split (80/10/10) with seed 42
    n_total = len(index_df)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    
    index_shuffled = index_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = index_shuffled[n_train + n_val:].reset_index(drop=True)
    
    logger.info(f"Total samples: {n_total}, Test samples: {len(test_df)}")
    
    # Load image pairs
    pairs = load_image_pairs(
        Path(args.recon_dir),
        Path(args.stimuli_dir),
        test_df,
        args.limit
    )
    
    if not pairs:
        logger.error("No valid image pairs found!")
        return
    
    # Evaluate
    results_df = evaluate_reconstructions(pairs, args.device)
    
    # Compute summary statistics
    summary = {
        "n_samples": len(results_df),
        "mean_mse": results_df["mse"].mean(),
        "mean_psnr": results_df["psnr"].mean(),
        "mean_ssim": results_df["ssim"].mean() if results_df["ssim"].mean() > 0 else None,
        "mean_lpips": results_df["lpips"].mean() if results_df["lpips"].mean() > 0 else None,
        "mean_clip_sim": results_df["clip_sim"].mean() if results_df["clip_sim"].mean() > 0 else None,
    }
    
    # Log summary
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)
    for key, value in summary.items():
        if value is not None:
            logger.info(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    logger.info("=" * 80)
    
    # Save results
    results_df.to_csv(output_dir / "metrics_per_sample.csv", index=False)
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Create comparison grid
    create_comparison_grid(pairs, output_dir / "comparison_grid.png", n_samples=8)
    
    logger.info(f"✓ Results saved to {output_dir}")
    logger.info("=" * 80)
    logger.info("EVALUATION COMPLETE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
