#!/usr/bin/env python3
"""
Image reconstruction using TwoStageEncoder + Stable Diffusion.
Simplified script specifically for TwoStageEncoder architecture.
"""
import argparse
import logging
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fmri2img.models.encoders import TwoStageEncoder
from fmri2img.data.nsd_index_reader import read_subject_index
from fmri2img.data.preprocess import NSDPreprocessor
from fmri2img.io.s3 import get_s3_filesystem, NIfTILoader
from fmri2img.data.clip_cache import CLIPCache


def load_two_stage_encoder(ckpt_path: Path, device: str = "cuda") -> TwoStageEncoder:
    """Load TwoStageEncoder from checkpoint."""
    logger.info(f"Loading TwoStageEncoder from {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Get architecture from checkpoint
    config = checkpoint.get("config", checkpoint.get("architecture", {}))
    
    input_dim = config.get("input_dim", 512)
    latent_dim = config.get("latent_dim", 512)
    n_blocks = config.get("n_blocks", 4)
    dropout = config.get("dropout", 0.3)
    head_type = config.get("head_type", "linear")
    
    logger.info(f"Architecture: input_dim={input_dim}, latent_dim={latent_dim}, n_blocks={n_blocks}")
    
    # Create model
    model = TwoStageEncoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        n_blocks=n_blocks,
        dropout=dropout,
        head_type=head_type
    )
    
    # Load weights
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()
    
    logger.info("✓ Model loaded successfully")
    return model


def predict_clip_embeddings(model: TwoStageEncoder, fmri_data: torch.Tensor, 
                            device: str = "cuda", batch_size: int = 32) -> np.ndarray:
    """Predict CLIP embeddings from fMRI data."""
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for i in range(0, len(fmri_data), batch_size):
            batch = fmri_data[i:i + batch_size].to(device)
            pred = model(batch)
            # Normalize to unit length (standard CLIP space)
            pred = pred / pred.norm(dim=-1, keepdim=True)
            all_preds.append(pred.cpu().numpy())
    
    return np.vstack(all_preds)


def generate_images(pipe, clip_embeddings: np.ndarray, output_dir: Path,
                   guidance_scale: float = 7.5, num_steps: int = 50, seed: int = 42) -> list:
    """Generate images from CLIP embeddings using Stable Diffusion."""
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    generated_images = []
    
    logger.info(f"Generating {len(clip_embeddings)} images...")
    logger.info(f"Guidance scale: {guidance_scale}, Steps: {num_steps}")
    
    # Create a simple projection layer (512 → 1024 for SD 2.1)
    projection = torch.nn.Linear(512, 1024).to(pipe.device)
    torch.nn.init.xavier_uniform_(projection.weight)
    
    for idx, clip_emb in enumerate(tqdm(clip_embeddings, desc="Generating")):
        try:
            # Convert CLIP embedding to tensor
            clip_emb_tensor = torch.from_numpy(clip_emb).float().to(pipe.device)
            
            with torch.no_grad():
                # Project 512-D to 1024-D (SD 2.1 text encoder dimension)
                projected = projection(clip_emb_tensor)  # (1024,)
                
                # Repeat across sequence length (77 tokens for SD)
                # Shape: (1, 77, 1024)
                prompt_embeds = projected.unsqueeze(0).unsqueeze(0).repeat(1, 77, 1)
                
                # Generate image
                image = pipe(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=None,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_steps,
                    generator=generator
                ).images[0]
            
            # Save image
            img_path = output_dir / f"sample_{idx:04d}.png"
            image.save(img_path)
            generated_images.append(image)
            
        except Exception as e:
            logger.error(f"Failed to generate image {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.info(f"✓ Generated {len(generated_images)} images")
    return generated_images


def main():
    parser = argparse.ArgumentParser(description="Reconstruct images using TwoStageEncoder")
    parser.add_argument("--subject", type=str, required=True, help="Subject ID (e.g., subj01)")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to TwoStageEncoder checkpoint")
    parser.add_argument("--model-id", type=str, default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--preproc-dir", type=str, default="outputs/preproc")
    parser.add_argument("--index-root", type=str, default="data/indices/nsd_index")
    parser.add_argument("--limit", type=int, default=16, help="Number of test samples")
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("IMAGE RECONSTRUCTION - TwoStageEncoder")
    logger.info("=" * 80)
    logger.info(f"Subject: {args.subject}")
    logger.info(f"Checkpoint: {args.ckpt}")
    logger.info(f"Diffusion model: {args.model_id}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Limit: {args.limit} samples")
    logger.info("=" * 80)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load encoder
    encoder = load_two_stage_encoder(Path(args.ckpt), args.device)
    
    # Load preprocessing
    logger.info("Loading preprocessing...")
    preprocessor = NSDPreprocessor(args.subject, args.preproc_dir)
    if not preprocessor.load_artifacts():
        raise RuntimeError(f"Failed to load preprocessing artifacts from {args.preproc_dir}/{args.subject}")
    logger.info(f"✓ Preprocessing loaded: fitted={preprocessor.is_fitted_}, pca_fitted={preprocessor.pca_fitted_}")
    
    # Load index and get test set
    logger.info("Loading index...")
    index_df = pd.DataFrame(read_subject_index(args.index_root, args.subject))
    
    # Split data (same split as training)
    n_total = len(index_df)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    
    index_shuffled = index_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = index_shuffled[n_train + n_val:].reset_index(drop=True)
    
    # Limit samples
    test_df = test_df.head(args.limit)
    logger.info(f"Using {len(test_df)} test samples")
    
    # Load fMRI data
    logger.info("Loading fMRI data...")
    s3_fs = get_s3_filesystem()
    nifti_loader = NIfTILoader(s3_fs)
    
    fmri_data = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Loading fMRI"):
        beta_path = row.get("beta_path", row.get("beta_file"))
        beta_index = int(row.get("beta_index", row.get("volume_index", 0)))
        
        img = nifti_loader.load(beta_path)
        vol = img.slicer[..., beta_index].get_fdata().astype(np.float32)
        
        # Preprocess (full pipeline: z-score → scaler+mask → PCA)
        fmri_vec = preprocessor.transform(vol)  # Returns (512,) vector
        fmri_data.append(fmri_vec)
    
    fmri_data = torch.from_numpy(np.vstack(fmri_data)).float()  # Stack into (N, 512)
    logger.info(f"✓ Loaded fMRI data: {fmri_data.shape}")
    
    # Predict CLIP embeddings
    logger.info("Predicting CLIP embeddings...")
    clip_preds = predict_clip_embeddings(encoder, fmri_data, args.device)
    logger.info(f"✓ Predicted embeddings: {clip_preds.shape}")
    
    # Load diffusion model
    logger.info("Loading Stable Diffusion...")
    logger.info(f"Model ID: {args.model_id}")
    logger.info("This may take 1-2 minutes...")
    
    try:
        # Try loading from cache with FP16 for speed
        pipe = StableDiffusionPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True,  # Use cached version only
            low_cpu_mem_usage=True
        )
        logger.info("✓ Model loaded from cache (FP16)")
    except Exception as e:
        logger.warning(f"Cache load failed ({e}), trying full download...")
        pipe = StableDiffusionPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )
        logger.info("✓ Model loaded (FP16)")
    
    logger.info("Configuring scheduler...")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    logger.info("✓ Scheduler configured")
    
    logger.info(f"Moving model to {args.device}...")
    pipe = pipe.to(args.device)
    logger.info(f"✓ Model on {args.device}, ready to generate")
    
    # Generate images
    generated = generate_images(
        pipe, clip_preds, output_dir,
        guidance_scale=args.guidance,
        num_steps=args.steps,
        seed=args.seed
    )
    
    # Save metadata
    metadata = {
        "subject": args.subject,
        "checkpoint": args.ckpt,
        "model_id": args.model_id,
        "n_samples": len(generated),
        "guidance_scale": args.guidance,
        "num_steps": args.steps,
        "seed": args.seed
    }
    
    import json
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("=" * 80)
    logger.info("RECONSTRUCTION COMPLETE!")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Generated {len(generated)} images")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
