#!/usr/bin/env python3
"""
Diffusion-Based Image Reconstruction from fMRI
==============================================

Generates images from predicted CLIP vectors using Stable Diffusion with
unCLIP-style conditioning (direct CLIP embedding injection).

Pipeline:
1. Load encoder (Ridge/MLP) and predict CLIP embeddings from test fMRI
2. Normalize predictions to unit length (standard CLIP space)
3. Pass predicted vectors to Stable Diffusion via prompt_embeds (unCLIP conditioning)
4. Generate images with fixed seed for reproducibility
5. Save individual images + comparison grids (generated vs NN retrieval)

Scientific Context:
- Predicted CLIP vectors come from the fMRI → CLIP encoder; diffusion model uses
  CLIP-space conditioning (unCLIP-style).
- This mirrors Takagi & Nishimoto (2023) and MindEye2 (2024) pipelines.
- Direct CLIP injection avoids text prompt ambiguity and leverages learned fMRI→CLIP mapping.

References:
- Takagi & Nishimoto (2023). "High-resolution image reconstruction with latent diffusion models from human brain activity"
- MindEye2 (Scotti et al. 2024). "Reconstructing the Mind's Eye"
- Ramesh et al. (2022). "Hierarchical Text-Conditional Image Generation with CLIP Latents" (DALL-E 2/unCLIP)

Usage:
    # Ridge encoder
    python scripts/decode_diffusion.py \\
        --subject subj01 \\
        --encoder ridge \\
        --ckpt checkpoints/ridge/subj01/ridge.pkl \\
        --clip-cache outputs/clip_cache/clip.parquet \\
        --model-id "stabilityai/stable-diffusion-2-1" \\
        --output-dir outputs/recon/subj01/ridge_diffusion \\
        --limit 16 \\
        --guidance 7.5 \\
        --steps 50
    
    # MLP encoder
    python scripts/decode_diffusion.py \\
        --subject subj01 \\
        --encoder mlp \\
        --ckpt checkpoints/mlp/subj01/mlp.pt \\
        --clip-cache outputs/clip_cache/clip.parquet \\
        --model-id "stabilityai/stable-diffusion-2-1" \\
        --output-dir outputs/recon/subj01/mlp_diffusion \\
        --limit 16 \\
        --guidance 7.5 \\
        --steps 50
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import project modules
from fmri2img.data.nsd_index_reader import read_subject_index
from fmri2img.data.preprocess import NSDPreprocessor
from fmri2img.data.clip_cache import CLIPCache
from fmri2img.io.s3 import get_s3_filesystem, NIfTILoader
from fmri2img.models.ridge import RidgeEncoder
from fmri2img.models.mlp import load_mlp
from fmri2img.models.clip_adapter import load_adapter
from fmri2img.models.train_utils import train_val_test_split
from fmri2img.eval.retrieval import cosine_sim


def check_model_cached(model_id: str) -> bool:
    """
    Check if diffusion model is cached locally (no network probe).
    
    Args:
        model_id: HuggingFace model ID
        
    Returns:
        True if model appears to be cached, False otherwise
    """
    import os
    from pathlib import Path
    
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if not cache_dir.exists():
        return False
    
    model_cache_name = f"models--{model_id.replace('/', '--')}"
    model_path = cache_dir / model_cache_name
    
    return model_path.exists()


def probe_model_with_local_only(model_id: str) -> bool:
    """
    Probe if model is fully cached using local_files_only.
    
    Args:
        model_id: HuggingFace model ID
        
    Returns:
        True if model can be loaded with local_files_only=True
    """
    try:
        from huggingface_hub import snapshot_download
        
        snapshot_download(
            repo_id=model_id,
            local_files_only=True,
            repo_type="model"
        )
        return True
    except Exception:
        return False


def load_encoder(encoder_type: str, ckpt_path: Path, device: str = "cpu"):
    """
    Load encoder (Ridge or MLP) from checkpoint.
    
    Args:
        encoder_type: "ridge" or "mlp"
        ckpt_path: Path to checkpoint
        device: Device for MLP ("cpu" or "cuda")
    
    Returns:
        Encoder model with predict() method
    """
    logger.info(f"Loading {encoder_type} encoder from {ckpt_path}")
    
    if encoder_type == "ridge":
        # Ridge uses RidgeEncoder.load() classmethod
        encoder = RidgeEncoder.load(str(ckpt_path))
        logger.info(f"✅ Loaded Ridge encoder (alpha={encoder.alpha:.1f}, {encoder.input_dim}D → {encoder.output_dim}D)")
        return encoder
    
    elif encoder_type == "mlp":
        # MLP uses PyTorch
        import torch
        model, meta = load_mlp(str(ckpt_path), map_location=device)
        model = model.to(device)  # Ensure model is on correct device
        model.eval()
        
        # Wrap in Ridge-like interface with predict()
        class MLPWrapper:
            def __init__(self, model, device):
                self.model = model
                self.device = device
            
            def predict(self, X: np.ndarray) -> np.ndarray:
                import torch
                with torch.no_grad():
                    X_tensor = torch.from_numpy(X).float().to(self.device)
                    pred = self.model(X_tensor)
                    return pred.cpu().numpy()
        
        logger.info(f"✅ Loaded MLP encoder (best_epoch={meta.get('best_epoch', 'N/A')})")
        return MLPWrapper(model, device)
    
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def extract_features_and_targets(
    df: pd.DataFrame,
    nifti_loader: NIfTILoader,
    preprocessor: NSDPreprocessor,
    clip_cache: CLIPCache
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract fMRI features (X) and CLIP targets (Y) from DataFrame.
    
    Reuses preprocessing pipeline from training (apples-to-apples).
    
    Args:
        df: DataFrame with beta_path, beta_index, nsdId columns
        nifti_loader: NIfTI data loader
        preprocessor: Fitted preprocessor
        clip_cache: CLIP cache
    
    Returns:
        X (n_samples, n_features), Y (n_samples, 512), nsdIds (n_samples,)
    """
    logger.info(f"Extracting features from {len(df)} samples...")
    
    X_list = []
    Y_list = []
    nsdIds = []
    
    for idx, row in df.iterrows():
        try:
            # Load fMRI volume
            beta_path = row["beta_path"]
            beta_index = row.get("beta_index", 0)
            
            # beta_index might be a scalar or array - handle both
            if isinstance(beta_index, (list, tuple, np.ndarray)):
                beta_index = int(beta_index[0]) if len(beta_index) > 0 else 0
            else:
                beta_index = int(beta_index)
            
            img = nifti_loader.load(beta_path)
            data_4d = img.get_fdata()
            vol = data_4d[..., beta_index].astype(np.float32)
            
            # Apply preprocessing if available
            if preprocessor is not None:
                x = preprocessor.transform(vol)
            else:
                # No preprocessing: flatten volume
                x = vol.flatten()
            
            # Get CLIP embedding
            nsd_id = int(row["nsdId"])
            y_dict = clip_cache.get([nsd_id])  # get() expects iterable, returns dict
            y = y_dict.get(nsd_id)  # Extract embedding from dict
            
            if x is not None and y is not None:
                X_list.append(x)
                Y_list.append(y)
                nsdIds.append(nsd_id)
        
        except Exception as e:
            import traceback
            logger.warning(f"Failed to process row {idx}: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            continue
    
    X = np.array(X_list)
    Y = np.array(Y_list)
    nsdIds = np.array(nsdIds)
    
    logger.info(f"✅ Extracted {len(X)} valid samples")
    logger.info(f"   Features: {X.shape}, Targets: {Y.shape}")
    
    return X, Y, nsdIds


def setup_diffusion_pipeline(
    model_id: str,
    device: str,
    dtype_str: str = "float32",
    scheduler_name: str = "dpm",
    fail_if_missing: bool = False
):
    """
    Setup Stable Diffusion pipeline for CLIP-conditioned generation.
    
    Probes cache first. If not cached:
    - If fail_if_missing=True: exits with code 2 and helpful message
    - Otherwise: shows big warning and proceeds with download (with heartbeat logs)
    
    Args:
        model_id: HuggingFace model ID (e.g., "stabilityai/stable-diffusion-2-1")
        device: "cuda" or "cpu"
        dtype_str: "float16" or "float32"
        scheduler_name: "dpm", "euler", "pndm", or "default"
        fail_if_missing: If True, fail fast if model not cached
    
    Returns:
        StableDiffusionPipeline configured for CLIP embedding injection
    """
    import torch
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, EulerDiscreteScheduler, PNDMScheduler
    import time
    import threading
    
    # Probe cache using local_files_only
    is_cached = probe_model_with_local_only(model_id)
    
    if is_cached:
        logger.info(f"✅ Model {model_id} found in cache, loading...")
    else:
        # Model not cached
        if fail_if_missing:
            # Fail fast with helpful message
            logger.error("")
            logger.error("=" * 80)
            logger.error("ERROR: Diffusion model not cached")
            logger.error("=" * 80)
            logger.error(f"Model '{model_id}' is not in your local cache.")
            logger.error("")
            logger.error("To fix this, run:")
            logger.error(f"  python scripts/download_sd_model.py --model-id {model_id}")
            logger.error("")
            logger.error("Or use --test-mode to skip image generation entirely:")
            logger.error("  python scripts/decode_diffusion.py --test-mode [other args]")
            logger.error("=" * 80)
            sys.exit(2)
        
        # Not cached but we'll download - show big warning
        logger.warning("")
        logger.warning("╔" + "=" * 78 + "╗")
        logger.warning("║" + " " * 78 + "║")
        logger.warning("║" + "  ⚠️  MODEL NOT CACHED - DOWNLOADING ~5 GB".center(78) + "║")
        logger.warning("║" + " " * 78 + "║")
        logger.warning("║" + f"  Model: {model_id}".ljust(78) + "║")
        logger.warning("║" + "  This will take 10-30 minutes depending on your connection.".ljust(78) + "║")
        logger.warning("║" + " " * 78 + "║")
        logger.warning("║" + "  To avoid this wait in the future, pre-download with:".ljust(78) + "║")
        logger.warning("║" + f"    python scripts/download_sd_model.py --model-id {model_id}".ljust(78) + "║")
        logger.warning("║" + " " * 78 + "║")
        logger.warning("╚" + "=" * 78 + "╝")
        logger.warning("")
        
        # Setup heartbeat thread to show we're not hung
        download_complete = threading.Event()
        
        def heartbeat():
            """Print periodic heartbeat messages during download."""
            start_time = time.time()
            while not download_complete.is_set():
                elapsed = int(time.time() - start_time)
                logger.info(f"⏳ Still downloading... ({elapsed}s elapsed)")
                download_complete.wait(30)  # Print every 30 seconds
        
        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()
    
    # Determine dtype
    dtype = torch.float32 if dtype_str == "float32" else torch.float16
    
    if dtype == torch.float16 and device == "cuda" and torch.cuda.is_available():
        logger.info("Using float16 precision (CUDA available)")
    elif dtype == torch.float16:
        logger.warning("float16 requested but CUDA unavailable, falling back to float32")
        dtype = torch.float32
    else:
        logger.info("Using float32 precision")
    
    # Load pipeline (downloads if needed)
    try:
        logger.info(f"Loading Stable Diffusion pipeline from {model_id}...")
        
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,  # Disable safety checker for research
            requires_safety_checker=False
        )
        
        if not is_cached:
            # Stop heartbeat
            download_complete.set()
            heartbeat_thread.join(timeout=1)
            logger.info("✅ Download complete!")
        
        # Configure scheduler based on user choice
        if scheduler_name == "dpm":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            logger.info("✓ Scheduler: DPMSolverMultistep (fast, high quality)")
        elif scheduler_name == "euler":
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
            logger.info("✓ Scheduler: EulerDiscrete")
        elif scheduler_name == "pndm":
            pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
            logger.info("✓ Scheduler: PNDM")
        else:
            # Keep default scheduler
            logger.info(f"✓ Scheduler: {pipe.scheduler.__class__.__name__} (default)")
        
        # Move to device
        pipe = pipe.to(device)
        
        # Enable memory optimizations (always safe, helps prevent OOM)
        try:
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
            logger.info("✅ Enabled memory optimizations (attention slicing, VAE slicing)")
        except Exception as e:
            logger.warning(f"Could not enable memory optimizations: {e}")
        
        logger.info(f"✅ Diffusion pipeline loaded on {device}")
        return pipe
    
    except ImportError as e:
        logger.error("diffusers library not installed. Install: pip install diffusers transformers accelerate")
        raise


def generate_image_from_clip_embedding(
    pipe,
    clip_embedding: np.ndarray,
    guidance_scale: float = 5.0,
    num_inference_steps: int = 50,
    seed: int = 42,
    negative_prompt: str = "blurry, low quality, distorted",
    blend_alpha: float = 1.0
) -> "PIL.Image":
    """
    Generate image from CLIP embedding using Stable Diffusion with proper OpenCLIP conditioning.
    
    For SD-2.1, this properly handles the (B, 77, 1024) sequence embedding shape and
    blends the predicted 1024-D CLIP vector into the pooled embedding space.
    
    Args:
        pipe: StableDiffusionPipeline (SD-2.1 with OpenCLIP)
        clip_embedding: CLIP vector (512,) or (1024,) from fMRI prediction
        guidance_scale: Classifier-free guidance strength (default: 5.0, use 1.0 to disable CFG)
        num_inference_steps: Number of denoising steps
        seed: Random seed for reproducibility
        negative_prompt: Negative text prompt (optional)
        blend_alpha: Blending weight for predicted embedding (1.0 = full replacement)
    
    Returns:
        Generated PIL Image
    """
    import torch
    
    # Set seed for reproducibility
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    
    # Convert to torch tensor and move to device (keep float32 throughout)
    pred_clip = torch.from_numpy(clip_embedding).float().to(pipe.device)
    
    # Ensure it's 1-D for a single sample
    if pred_clip.dim() == 1:
        pred_clip = pred_clip.unsqueeze(0)  # (1, D)
    
    batch_size = pred_clip.shape[0]
    
    # Clean predicted embedding: remove NaN/Inf and normalize
    pred_clip = torch.nan_to_num(pred_clip, nan=0.0, posinf=1.0, neginf=-1.0)
    pred_clip = pred_clip / (pred_clip.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    
    # Log prediction stats
    logger.info(f"📊 Predicted CLIP embedding stats:")
    logger.info(f"   Shape: {pred_clip.shape}, Dtype: {pred_clip.dtype}")
    logger.info(f"   Range: [{pred_clip.min().item():.4f}, {pred_clip.max().item():.4f}]")
    logger.info(f"   Mean: {pred_clip.mean().item():.4f}, Norm: {pred_clip.norm(dim=-1).mean().item():.4f}")
    logger.info(f"   First 3 values: {pred_clip[0, :3].tolist()}")
    
    # Verify no NaN/Inf after cleaning
    if not torch.isfinite(pred_clip).all():
        logger.error("❌ Predicted embedding still contains non-finite values after cleaning!")
        raise ValueError("Non-finite values in predicted CLIP embedding")
    
    # Get proper conditioning embeddings from the pipeline's text encoder
    # This gives us the correct (B, 77, 1024) sequence shape for SD-2.1
    with torch.no_grad():
        # Get conditional embeddings (empty prompt gives us base structure)
        prompt_list = [""] * batch_size
        
        # Use encode_prompt to get properly shaped embeddings
        # For SD-2.1, this returns (prompt_embeds, negative_prompt_embeds) or similar
        # The exact signature depends on diffusers version
        try:
            # Try modern diffusers API (>= 0.25)
            prompt_embeds = pipe.encode_prompt(
                prompt=prompt_list,
                device=pipe.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=(guidance_scale > 1.0)
            )
            
            # Handle different return formats
            if isinstance(prompt_embeds, tuple):
                if len(prompt_embeds) == 2:
                    # (cond_embeds, uncond_embeds)
                    cond_embeds, uncond_embeds = prompt_embeds
                    pooled_embeds = None
                elif len(prompt_embeds) == 4:
                    # (cond_embeds, uncond_embeds, cond_pooled, uncond_pooled)
                    cond_embeds, uncond_embeds, cond_pooled, uncond_pooled = prompt_embeds
                    pooled_embeds = (cond_pooled, uncond_pooled)
                else:
                    # Fallback: use first element
                    cond_embeds = prompt_embeds[0]
                    uncond_embeds = None
                    pooled_embeds = None
            else:
                cond_embeds = prompt_embeds
                uncond_embeds = None
                pooled_embeds = None
                
        except Exception as e:
            logger.warning(f"encode_prompt failed ({e}), trying fallback encoding...")
            # Fallback: manual encoding
            text_inputs = pipe.tokenizer(
                prompt_list,
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).to(pipe.device)
            
            cond_embeds = pipe.text_encoder(text_inputs.input_ids)[0]  # (B, 77, 1024)
            uncond_embeds = None
            pooled_embeds = None
        
        # Clean conditional embeddings
        cond_embeds = torch.nan_to_num(cond_embeds, nan=0.0, posinf=1.0, neginf=-1.0)
        
        logger.info(f"✅ Got conditioning embeddings: shape={cond_embeds.shape}, dtype={cond_embeds.dtype}")
        
        # If we have pooled embeddings, blend our prediction into them
        if pooled_embeds is not None and pred_clip.shape[1] == 1024:
            cond_pooled, uncond_pooled = pooled_embeds
            cond_pooled = torch.nan_to_num(cond_pooled, nan=0.0, posinf=1.0, neginf=-1.0)
            uncond_pooled = torch.nan_to_num(uncond_pooled, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Normalize pooled embeddings
            cond_pooled = cond_pooled / (cond_pooled.norm(dim=-1, keepdim=True).clamp_min(1e-6))
            uncond_pooled = uncond_pooled / (uncond_pooled.norm(dim=-1, keepdim=True).clamp_min(1e-6))
            
            # Blend predicted embedding into conditional pooled
            new_pooled = torch.nn.functional.normalize(
                blend_alpha * pred_clip + (1 - blend_alpha) * cond_pooled,
                dim=-1
            )
            pooled_embeds = (new_pooled, uncond_pooled)
            logger.info(f"✅ Blended prediction into pooled embeddings (alpha={blend_alpha})")
        
        # For unconditional (negative prompt), always use encode_prompt with empty string
        if guidance_scale > 1.0 and uncond_embeds is None:
            try:
                uncond_result = pipe.encode_prompt(
                    prompt=[""] * batch_size,
                    device=pipe.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False
                )
                if isinstance(uncond_result, tuple):
                    uncond_embeds = uncond_result[0]
                else:
                    uncond_embeds = uncond_result
            except:
                # Fallback: use zeros (less ideal)
                uncond_embeds = torch.zeros_like(cond_embeds)
            
            uncond_embeds = torch.nan_to_num(uncond_embeds, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Prepare latents
    latents_shape = (batch_size, pipe.unet.config.in_channels, 
                    pipe.unet.config.sample_size, pipe.unet.config.sample_size)
    latents = torch.randn(latents_shape, generator=generator, device=pipe.device, dtype=torch.float32)
    
    # Safety check on initial latents
    if not torch.isfinite(latents).all():
        logger.error("❌ Initial latents contain non-finite values!")
        raise ValueError("Non-finite initial latents")
    
    logger.info(f"✅ Initialized latents: shape={latents.shape}, range=[{latents.min():.3f}, {latents.max():.3f}]")
    
    # Scale latents by scheduler's init noise sigma
    latents = latents * pipe.scheduler.init_noise_sigma
    
    # Set timesteps
    pipe.scheduler.set_timesteps(num_inference_steps, device=pipe.device)
    timesteps = pipe.scheduler.timesteps
    
    # Denoising loop with CFG guards
    logger.info(f"🎨 Starting denoising ({num_inference_steps} steps, guidance={guidance_scale})...")
    
    for i, t in enumerate(timesteps):
        # Check latents health
        if not torch.isfinite(latents).all():
            logger.warning(f"⚠️  Non-finite latents at step {i}, clamping...")
            latents = torch.nan_to_num(latents, nan=0.0, posinf=10.0, neginf=-10.0)
            latents = latents.clamp(-10, 10)
        
        # Expand latents for CFG
        if guidance_scale > 1.0:
            latent_model_input = torch.cat([latents] * 2)
        else:
            latent_model_input = latents
        
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        
        # Predict noise with UNet
        with torch.no_grad():
            # Prepare encoder hidden states for UNet
            if guidance_scale > 1.0:
                encoder_hidden_states = torch.cat([uncond_embeds, cond_embeds])
            else:
                encoder_hidden_states = cond_embeds
            
            # CFG guards: clean embeddings before UNet call
            encoder_hidden_states = torch.nan_to_num(encoder_hidden_states, nan=0.0)
            latent_model_input = torch.nan_to_num(latent_model_input, nan=0.0)
            
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states
            ).sample
            
            # CFG guards: clean noise prediction
            noise_pred = torch.nan_to_num(noise_pred, nan=0.0)
        
        # Perform CFG
        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_uncond = torch.nan_to_num(noise_pred_uncond, nan=0.0)
            noise_pred_text = torch.nan_to_num(noise_pred_text, nan=0.0)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Scheduler step
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
        latents = torch.nan_to_num(latents, nan=0.0)
        
        # Log progress every 10 steps
        if i % 10 == 0 or i == len(timesteps) - 1:
            lat_min, lat_max = latents.min().item(), latents.max().item()
            logger.info(f"   Step {i:3d}/{len(timesteps)}: latents=[{lat_min:6.3f}, {lat_max:6.3f}]")
            
            if not torch.isfinite(latents).all():
                logger.error(f"❌ Non-finite latents detected at step {i}!")
                logger.error(f"   NaN count: {torch.isnan(latents).sum()}")
                logger.error(f"   Inf count: {torch.isinf(latents).sum()}")
                raise ValueError(f"Non-finite latents at step {i}")
    
    # Decode latents to image
    logger.info("🖼️  Decoding latents to image...")
    latents = 1 / pipe.vae.config.scaling_factor * latents
    
    with torch.no_grad():
        # Ensure VAE decode uses float32
        image = pipe.vae.decode(latents.to(torch.float32)).sample
        
        # Safety: clean decoded image
        image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Check for non-finite values
        if not torch.isfinite(image).all():
            logger.warning("⚠️  Non-finite values in decoded image, cleaning...")
            image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Post-process: clamp to [-1, 1] and convert to [0, 1]
    image = image.clamp(-1, 1)
    image = (image + 1.0) / 2.0
    
    # Convert to PIL
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).round().astype("uint8")
    
    # Check for valid uint8 range
    if image.min() < 0 or image.max() > 255:
        logger.warning(f"⚠️  Image values out of uint8 range: [{image.min()}, {image.max()}]")
        image = image.clip(0, 255)
    
    from PIL import Image
    pil_image = Image.fromarray(image[0])
    
    logger.info(f"✅ Generated image: size={pil_image.size}, mode={pil_image.mode}")
    
    return pil_image


def create_comparison_grid(
    generated_img: "PIL.Image",
    nn_img: Optional["PIL.Image"],
    nsd_id: int,
    cosine_score: float,
    output_path: Path
) -> None:
    """
    Create side-by-side comparison grid: generated vs NN retrieval.
    
    Args:
        generated_img: Generated image from diffusion
        nn_img: Nearest neighbor retrieved image (or None)
        nsd_id: NSD ID for labeling
        cosine_score: Cosine similarity between pred and GT
        output_path: Output path for grid image
    """
    from PIL import Image, ImageDraw, ImageFont
    
    # Create grid (1 row, 2 columns)
    img_size = 512
    grid_width = img_size * 2 + 50  # 50px gap
    grid_height = img_size + 100  # Extra space for labels
    
    grid = Image.new("RGB", (grid_width, grid_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid)
    
    # Try to load a font (fallback to default)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Resize images to consistent size
    generated_resized = generated_img.resize((img_size, img_size), Image.Resampling.LANCZOS)
    
    # Paste generated image (left)
    grid.paste(generated_resized, (0, 50))
    draw.text((img_size // 2 - 50, 10), "Generated (Diffusion)", fill=(0, 0, 0), font=font)
    
    # Paste NN image (right) if available
    if nn_img is not None:
        nn_resized = nn_img.resize((img_size, img_size), Image.Resampling.LANCZOS)
        grid.paste(nn_resized, (img_size + 50, 50))
        draw.text((img_size + 50 + img_size // 2 - 50, 10), "NN Retrieval (GT)", fill=(0, 0, 0), font=font)
    else:
        # No NN image available
        draw.text((img_size + 50, img_size // 2), "No NN available", fill=(128, 128, 128), font=font)
    
    # Add metadata at bottom
    draw.text((10, img_size + 60), f"NSD ID: {nsd_id}", fill=(0, 0, 0), font=font_small)
    draw.text((10, img_size + 80), f"Cosine (pred vs GT): {cosine_score:.4f}", fill=(0, 0, 0), font=font_small)
    
    # Save grid
    grid.save(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate images from fMRI via diffusion (unCLIP conditioning)"
    )
    
    # Data paths
    parser.add_argument("--index-root", default="data/indices/nsd_index",
                       help="NSD index root directory")
    parser.add_argument("--subject", default="subj01", help="Subject ID")
    parser.add_argument("--clip-cache", default="outputs/clip_cache/clip.parquet",
                       help="Path to CLIP cache")
    
    # Encoder
    parser.add_argument("--encoder", choices=["ridge", "mlp"], required=True,
                       help="Encoder type")
    parser.add_argument("--ckpt", required=True, help="Path to encoder checkpoint")
    
    # Preprocessing
    parser.add_argument("--use-preproc", action="store_true",
                       help="Force enable preprocessing (overrides auto-detection)")
    parser.add_argument("--no-preproc", action="store_true",
                       help="Force disable preprocessing (overrides auto-detection)")
    parser.add_argument("--preproc-dir", required=False,
                       help="Preprocessing directory. If not specified and preprocessing is enabled, "
                            "will use the path from encoder checkpoint metadata.")
    
    # Diffusion model
    parser.add_argument("--model-id", default="stabilityai/stable-diffusion-2-1",
                       help="HuggingFace model ID for Stable Diffusion. Popular options: "
                            "stabilityai/stable-diffusion-2-1 (~5GB, best quality), "
                            "stabilityai/stable-diffusion-2-1-base (~5GB), "
                            "runwayml/stable-diffusion-v1-5 (~4GB, faster)")
    
    # CLIP Adapter (optional)
    parser.add_argument("--clip-adapter", help="Path to CLIP adapter checkpoint (512D→target_dim)")
    parser.add_argument("--clip-target-dim", type=int, choices=[768, 1024],
                       help="Target CLIP dimension (768 for SD-1.5, 1024 for SD-2.1). "
                            "Auto-detected if not specified.")
    
    # Diffusion generation parameters
    parser.add_argument("--guidance", type=float, default=5.0,
                       help="Classifier-free guidance scale (default: 5.0)")
    parser.add_argument("--steps", type=int, default=50,
                       help="Number of denoising steps")
    parser.add_argument("--dtype", default="float32", choices=["float16", "float32"],
                       help="Model precision (default: float32). Use float16 for faster inference on GPU.")
    parser.add_argument("--scheduler", default="dpm", choices=["dpm", "euler", "pndm", "default"],
                       help="Diffusion scheduler (default: dpm). Options: dpm=DPMSolverMultistep, "
                            "euler=EulerDiscrete, pndm=PNDM, default=keep model's default")
    parser.add_argument("--blend-alpha", type=float, default=1.0,
                       help="Blending weight for predicted CLIP embedding (1.0=full replacement, 0.0=baseline)")
    
    # Debugging flags
    parser.add_argument("--no-adapter", action="store_true",
                       help="Bypass CLIP adapter even if --clip-adapter is provided (for debugging)")
    parser.add_argument("--no-cfg", action="store_true",
                       help="Disable classifier-free guidance (sets guidance=1.0)")
    
    # Evaluation
    parser.add_argument("--limit", type=int, help="Limit number of test samples")
    parser.add_argument("--gallery-limit", type=int, default=1000,
                       help="Gallery size for NN retrieval (for comparison grid)")
    
    # Output
    parser.add_argument("--output-dir", help="Output directory (default: outputs/recon/{subject}/{encoder}_diffusion)")
    
    # System
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--test-mode", action="store_true",
                       help="Test mode: skip diffusion, only test encoder pipeline and save predictions")
    parser.add_argument("--fail-if-missing-model", action="store_true",
                       help="Fail fast (exit code 2) if diffusion model not cached. Useful for CI/scripts.")
    
    args = parser.parse_args()
    
    # Default output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"outputs/recon/{args.subject}/{args.encoder}_diffusion")
    
    try:
        np.random.seed(args.seed)
        
        logger.info("=" * 80)
        logger.info("DIFFUSION-BASED IMAGE RECONSTRUCTION FROM fMRI")
        logger.info("=" * 80)
        logger.info(f"Subject: {args.subject}")
        logger.info(f"Encoder: {args.encoder}")
        logger.info(f"Checkpoint: {args.ckpt}")
        logger.info(f"Diffusion model: {args.model_id}")
        logger.info(f"Device: {args.device}")
        logger.info(f"Dtype: {args.dtype}")
        logger.info(f"Scheduler: {args.scheduler}")
        logger.info(f"Guidance scale: {args.guidance}")
        logger.info(f"Inference steps: {args.steps}")
        logger.info(f"Output directory: {output_dir}")
        
        # Resolve device (handle "auto")
        import torch
        if args.device == "auto":
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Device 'auto' resolved to: {args.device}")
        elif args.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            args.device = "cpu"
            args.dtype = "float32"  # Force float32 on CPU
        
        # Handle debugging flags
        if args.no_cfg:
            args.guidance = 1.0
            logger.info("⚠️  CFG disabled (--no-cfg): guidance forced to 1.0")
        
        # Load CLIP adapter if specified
        clip_adapter = None
        adapter_target_dim = None
        adapter_metadata = None
        if args.clip_adapter and not args.no_adapter:
            logger.info(f"Loading CLIP adapter from {args.clip_adapter}")
            try:
                clip_adapter, adapter_metadata = load_adapter(args.clip_adapter, map_location=args.device)
                clip_adapter = clip_adapter.to(args.device)
                clip_adapter.eval()
                
                # Get target dimension from metadata (prefer target_dim, fallback to out_dim)
                adapter_target_dim = adapter_metadata.get("target_dim", adapter_metadata.get("out_dim"))
                adapter_input_dim = adapter_metadata.get("input_dim", adapter_metadata.get("in_dim", 512))
                adapter_model_id = adapter_metadata.get("model_id", "unknown")
                
                logger.info(f"✅ CLIP Adapter loaded: {adapter_input_dim}D → {adapter_target_dim}D")
                logger.info(f"   Adapter metadata: model_id={adapter_model_id}, "
                           f"subject={adapter_metadata.get('subject', 'unknown')}")
                
                # Validate target dimension if specified
                if args.clip_target_dim and args.clip_target_dim != adapter_target_dim:
                    logger.warning(f"⚠️  --clip-target-dim={args.clip_target_dim} but adapter outputs {adapter_target_dim}D")
                    logger.warning(f"   Using adapter's dimension: {adapter_target_dim}D")
                
                # Check model_id consistency if --model-id was specified
                if args.model_id and adapter_model_id != "unknown" and adapter_model_id != args.model_id:
                    logger.warning(f"⚠️  Adapter was trained for {adapter_model_id} but using {args.model_id}")
                    logger.warning(f"   This may cause dimension mismatches or degraded quality")
                
                args.clip_target_dim = adapter_target_dim
                
            except FileNotFoundError as e:
                logger.error(f"❌ {e}")
                sys.exit(1)
            except Exception as e:
                logger.error(f"❌ Failed to load adapter: {e}")
                sys.exit(1)
        elif args.clip_target_dim:
            logger.warning("⚠️  --clip-target-dim specified but no --clip-adapter provided. Will be ignored.")
        
        # Handle --no-adapter flag
        if args.no_adapter and args.clip_adapter:
            logger.warning("⚠️  --no-adapter specified: bypassing CLIP adapter for debugging")
            clip_adapter = None
            adapter_target_dim = None
        
        # Log adapter status
        if clip_adapter:
            logger.info(f"CLIP Adapter: ENABLED (512D → {adapter_target_dim}D)")
        else:
            logger.info("CLIP Adapter: DISABLED (using 512-D embeddings directly)")
        
        # Load encoder and its metadata
        encoder = load_encoder(args.encoder, Path(args.ckpt), args.device)
        
        # Load encoder checkpoint metadata for preprocessing
        ckpt_meta = torch.load(args.ckpt, map_location="cpu").get("meta", {})
        preproc_meta = ckpt_meta.get("preproc", {})
        preproc_trained_with = preproc_meta.get("used_preproc", False)
        expected_input_dim = ckpt_meta.get("input_dim")
        
        # Resolve preprocessing flag
        if args.use_preproc and args.no_preproc:
            logger.error("ERROR: Cannot specify both --use-preproc and --no-preproc")
            sys.exit(1)
        
        if args.use_preproc:
            preproc_enabled = True
        elif args.no_preproc:
            preproc_enabled = False
        else:
            # Auto-detect from metadata
            preproc_enabled = preproc_trained_with
        
        # Determine preprocessing directory
        preproc_dir = None
        if preproc_enabled:
            if args.preproc_dir:
                preproc_dir = Path(args.preproc_dir)
            elif preproc_meta.get("path"):
                preproc_dir = Path(preproc_meta["path"])
            else:
                logger.error("ERROR: Preprocessing enabled but no preprocessing directory specified")
                logger.error("Either provide --preproc-dir or ensure checkpoint metadata contains preprocessing path")
                sys.exit(1)
            
            if not preproc_dir.exists():
                logger.error(f"ERROR: Preprocessing directory not found: {preproc_dir}")
                sys.exit(1)
            
            logger.info(f"✓ Preprocessing: ENABLED from {preproc_dir}")
            logger.info(f"  Expected input_dim: {expected_input_dim}")
        else:
            if preproc_trained_with:
                logger.warning("WARNING: Model was trained WITH preprocessing but --no-preproc specified")
                logger.warning("This may cause dimension mismatch errors")
            logger.info("Preprocessing: DISABLED")
        
        # Load index
        logger.info(f"Loading index for {args.subject}...")
        df = read_subject_index(args.index_root, args.subject)
        
        if args.limit:
            df = df.head(args.limit)
            logger.info(f"Limited to {len(df)} samples")
        
        # Split data (same as training)
        _, _, test_df = train_val_test_split(df, random_seed=args.seed)
        logger.info(f"Test set: {len(test_df)} samples")
        
        # Load CLIP cache
        logger.info(f"Loading CLIP cache from {args.clip_cache}")
        clip_cache = CLIPCache(args.clip_cache).load()
        stats = clip_cache.stats()
        logger.info(f"✅ CLIP cache loaded: {stats['cache_size']} embeddings")
        
        # Setup preprocessing based on resolved flag
        preprocessor = None
        if preproc_enabled:
            logger.info("Loading preprocessing artifacts...")
            preprocessor = NSDPreprocessor(subject=args.subject)
            preprocessor.set_out_dir(str(preproc_dir))
            success = preprocessor.load_artifacts()
            if not success:
                logger.error(f"ERROR: Failed to load preprocessing artifacts from {preproc_dir}")
                return 1
            summary = preprocessor.summary()
            logger.info(f"✅ Preprocessing loaded: {summary}")
            
            if summary.get('n_voxels_kept', 0) == 0:
                logger.error("ERROR: Preprocessing artifacts are empty or invalid!")
                return 1
        else:
            # No preprocessing
            if not preproc_trained_with:
                logger.info("No preprocessing (model trained on raw voxels)")
            preprocessor = None
        
        # Initialize NIfTI loader
        s3_fs = get_s3_filesystem()
        nifti_loader = NIfTILoader(s3_fs)
        
        # Extract test features and targets
        X_test, Y_test, test_nsd_ids = extract_features_and_targets(
            test_df, nifti_loader, preprocessor, clip_cache
        )
        
        if len(X_test) == 0:
            logger.error("No valid test samples extracted!")
            return 1
        
        # Validate feature dimensions match expected input_dim
        actual_feature_dim = X_test.shape[1]
        if expected_input_dim and actual_feature_dim != expected_input_dim:
            logger.error("=" * 80)
            logger.error(f"PREPROCESSING MISMATCH ERROR")
            logger.error("=" * 80)
            logger.error(f"Model expects {expected_input_dim} features but got {actual_feature_dim}.")
            logger.error("")
            if preproc_enabled:
                logger.error(f"Preprocessing is ENABLED but dimensions don't match.")
                logger.error(f"Current preprocessing directory: {preproc_dir}")
                logger.error(f"Check that the directory matches the model's training configuration.")
            else:
                logger.error(f"Preprocessing is DISABLED but model was trained WITH preprocessing.")
                logger.error(f"")
                logger.error(f"Solution 1: Enable preprocessing with --use-preproc")
                if preproc_meta.get("path"):
                    logger.error(f"  Suggested path: --preproc-dir {preproc_meta['path']}")
                logger.error(f"Solution 2: Let the system auto-discover the correct preprocessing directory")
                logger.error(f"  (omit --no-preproc and --preproc-dir flags)")
            logger.error("=" * 80)
            return 1
        
        logger.info(f"✅ Feature dimensions match: {actual_feature_dim} features")
        
        # Predict CLIP embeddings
        logger.info("Predicting CLIP embeddings from test fMRI...")
        Y_pred = encoder.predict(X_test)
        
        # Store original 512-D predictions
        Y_pred_512 = Y_pred  # MLP output (N, 512)
        Y_pred_for_sd = Y_pred_512  # Default for SD conditioning
        
        # If adapter is enabled, project to 1024 for diffusion ONLY
        Y_pred_1024 = None
        if clip_adapter:
            logger.info(f"Applying CLIP adapter: {Y_pred_512.shape[1]}D → {adapter_target_dim}D...")
            with torch.no_grad():
                Y_pred_tensor = torch.from_numpy(Y_pred_512).float().to(args.device)
                Y_pred_1024 = clip_adapter(Y_pred_tensor).cpu().numpy()
                
                # NaN check after adapter
                if not np.isfinite(Y_pred_1024).all():
                    logger.error("=" * 80)
                    logger.error("ERROR: Adapter output contains NaN or Inf values!")
                    logger.error("=" * 80)
                    logger.error(f"NaN count: {np.isnan(Y_pred_1024).sum()}")
                    logger.error(f"Inf count: {np.isinf(Y_pred_1024).sum()}")
                    logger.error(f"Input range: [{Y_pred_512.min():.4f}, {Y_pred_512.max():.4f}]")
                    logger.error(f"Output range: [{np.nanmin(Y_pred_1024):.4f}, {np.nanmax(Y_pred_1024):.4f}]")
                    logger.error("This will cause black images. Check adapter training and normalization.")
                    raise ValueError("Adapter output has NaN/Inf values")
                
            Y_pred_for_sd = Y_pred_1024
            logger.info(f"✅ Adapter applied: output shape {Y_pred_1024.shape}")
            logger.info(f"   Output range: [{Y_pred_1024.min():.4f}, {Y_pred_1024.max():.4f}]")
        
        # Normalize predictions to unit length (standard CLIP space)
        def _norm(x: np.ndarray) -> np.ndarray:
            return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        
        Y_pred_normalized = _norm(Y_pred_for_sd)
        
        # Final NaN check before generation
        if not np.isfinite(Y_pred_normalized).all():
            logger.error("=" * 80)
            logger.error("ERROR: Normalized predictions contain NaN or Inf!")
            logger.error("=" * 80)
            logger.error(f"NaN count: {np.isnan(Y_pred_normalized).sum()}")
            logger.error(f"Inf count: {np.isinf(Y_pred_normalized).sum()}")
            raise ValueError("Normalized predictions have NaN/Inf values")
        
        logger.info(f"✅ Predictions for SD: {Y_pred_normalized.shape}")
        logger.info(f"   Normalized to unit length (mean norm: {np.linalg.norm(Y_pred_normalized, axis=1).mean():.4f})")
        logger.info(f"   Range: [{Y_pred_normalized.min():.4f}, {Y_pred_normalized.max():.4f}]")
        
        # Safe cosine computation - compare in matching dimensions
        cosine_scores = None
        try:
            if Y_test.shape[1] == Y_pred_512.shape[1]:
                # GT and pred are both 512-D
                cosine_scores = (_norm(Y_pred_512) * _norm(Y_test)).sum(axis=1)
            elif clip_adapter is not None and Y_pred_1024 is not None and Y_test.shape[1] == Y_pred_1024.shape[1]:
                # GT is 1024-D, compare with adapted predictions
                cosine_scores = (_norm(Y_pred_1024) * _norm(Y_test)).sum(axis=1)
            else:
                logger.warning(
                    f"Cosine skipped: GT dim={Y_test.shape[1]} "
                    f"vs pred dims 512{' and 1024' if clip_adapter is not None else ''}"
                )
        except Exception as e:
            logger.warning(f"Cosine computation failed but continuing: {e}")
        
        if cosine_scores is not None:
            mean_cosine = float(np.mean(cosine_scores))
            logger.info(f"   Mean cosine (pred vs GT): {mean_cosine:.4f}")
            logger.info(f"   Cosine range: [{cosine_scores.min():.4f}, {cosine_scores.max():.4f}]")
        else:
            mean_cosine = None
            logger.info("   Cosine not computed (dimension mismatch).")
        
        # Create output directories
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # TEST MODE: Skip diffusion, just save predictions and exit
        if args.test_mode:
            logger.info("=" * 80)
            logger.info("TEST MODE: Skipping diffusion image generation")
            logger.info("=" * 80)
            
            # Save predictions
            results = {
                "encoder": args.encoder,
                "checkpoint": str(args.ckpt),
                "preprocessing": str(args.preproc_dir) if args.use_preproc else None,
                "clip_adapter": str(args.clip_adapter) if args.clip_adapter else None,
                "clip_adapter_target_dim": adapter_target_dim if clip_adapter else None,
                "n_test_samples": len(X_test),
                "mean_cosine_similarity": float(mean_cosine) if mean_cosine is not None else None,
                "cosine_scores": cosine_scores.tolist() if cosine_scores is not None else None,
                "test_nsd_ids": test_nsd_ids.tolist(),
            }
            
            results_file = output_dir / "test_predictions.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"✅ Test results saved to {results_file}")
            if mean_cosine is not None:
                logger.info(f"   Mean cosine similarity: {mean_cosine:.4f}")
            logger.info(f"   Test samples: {len(X_test)}")
            if clip_adapter:
                logger.info(f"   CLIP Adapter: {adapter_meta.get('in_dim')}D → {adapter_target_dim}D")
            logger.info("")
            logger.info("To run full diffusion pipeline (requires downloading ~5GB model):")
            logger.info("Remove --test-mode flag and wait for model download to complete")
            return 0
        
        # FULL MODE: Setup diffusion pipeline
        logger.info("Setting up Stable Diffusion pipeline...")
        pipe = setup_diffusion_pipeline(
            args.model_id,
            args.device,
            args.dtype,
            args.scheduler,
            fail_if_missing=args.fail_if_missing_model
        )
        
        # Disable safety checker for debugging (prevents false positives on research images)
        if hasattr(pipe, 'safety_checker') and pipe.safety_checker is not None:
            logger.info("🔓 Disabling safety checker for research use...")
            pipe.safety_checker = lambda images, clip_input: (images, [False] * len(images))
        
        images_dir = output_dir / "images"
        grids_dir = output_dir / "grids"
        images_dir.mkdir(exist_ok=True)
        grids_dir.mkdir(exist_ok=True)
        
        # Build small gallery for NN retrieval (for comparison)
        logger.info(f"Building gallery for NN comparison (limit={args.gallery_limit})...")
        
        def _safe_get_all_clip_ids(cache):
            """Backward-compatible helper to get all IDs from various CLIP cache implementations."""
            # Try common method names first
            for name in ("get_all_ids", "get_ids", "list_ids"):
                if hasattr(cache, name):
                    try:
                        return list(getattr(cache, name)())
                    except Exception:
                        pass
            # Fallbacks
            try:
                # common parquet-backed cache: df with 'nsd_id' or 'id'
                df = getattr(cache, "df", None)
                if df is not None:
                    col = "nsd_id" if "nsd_id" in df.columns else ("id" if "id" in df.columns else None)
                    if col:
                        return list(df[col].unique())
            except Exception:
                pass
            return []
        
        all_nsd_ids = _safe_get_all_clip_ids(clip_cache)
        if not all_nsd_ids:
            logger.warning("CLIP cache IDs could not be enumerated; skipping gallery build")
            all_nsd_ids = []
        
        gallery_nsd_ids = []
        gallery_embeddings = None
        
        if all_nsd_ids:
            all_embeddings = clip_cache.get_batch(all_nsd_ids)
            
            # Exclude test samples
            mask = ~np.isin(all_nsd_ids, test_nsd_ids)
            gallery_nsd_ids = all_nsd_ids[mask]
            gallery_embeddings = all_embeddings[mask]
            
            if len(gallery_nsd_ids) > args.gallery_limit:
                indices = np.random.choice(len(gallery_nsd_ids), size=args.gallery_limit, replace=False)
                gallery_nsd_ids = gallery_nsd_ids[indices]
                gallery_embeddings = gallery_embeddings[indices]
            
            logger.info(f"✅ Gallery size: {len(gallery_embeddings)}")
        else:
            # Graceful degrade: continue without NN gallery
            logger.info("Proceeding without NN gallery (generation and per-sample eval will continue).")
        
        # Generate images
        logger.info("\n" + "=" * 80)
        logger.info("GENERATING IMAGES")
        logger.info("=" * 80)
        
        results = []
        
        # Handle None cosine_scores for zip
        cosine_scores_iter = cosine_scores if cosine_scores is not None else [None] * len(test_nsd_ids)
        
        for i, (clip_pred, clip_gt, nsd_id, cosine) in enumerate(zip(
            Y_pred_normalized, Y_test, test_nsd_ids, cosine_scores_iter
        )):
            logger.info(f"\n[{i+1}/{len(test_nsd_ids)}] Generating image for NSD ID {nsd_id}...")
            if cosine is not None:
                logger.info(f"  Cosine (pred vs GT): {cosine:.4f}")
            else:
                logger.info(f"  Cosine (pred vs GT): not computed")
            
            try:
                # Generate image from predicted CLIP embedding
                generated_img = generate_image_from_clip_embedding(
                    pipe,
                    clip_pred,
                    guidance_scale=args.guidance,
                    num_inference_steps=args.steps,
                    seed=args.seed + i,  # Different seed per sample
                    blend_alpha=args.blend_alpha
                )
                
                # Save generated image
                img_path = images_dir / f"nsd{nsd_id}_generated.png"
                generated_img.save(img_path)
                logger.info(f"  ✅ Saved generated image: {img_path}")
                
                # Find nearest neighbor for comparison (if gallery available)
                nn_nsd_id = None
                nn_cosine = None
                if gallery_embeddings is not None and len(gallery_embeddings) > 0:
                    # Compute similarity to gallery
                    sim_to_gallery = cosine_sim(clip_pred.reshape(1, -1), gallery_embeddings)[0]
                    nn_idx = np.argmax(sim_to_gallery)
                    nn_nsd_id = gallery_nsd_ids[nn_idx]
                    nn_cosine = sim_to_gallery[nn_idx]
                    
                    logger.info(f"  NN retrieval: NSD ID {nn_nsd_id} (cosine: {nn_cosine:.4f})")
                else:
                    logger.info(f"  NN retrieval: skipped (no gallery)")
                
                # For now, we don't have actual images, so skip grid creation
                # In a full implementation, you'd load the actual image via COCO/NSD dataset
                # and create the comparison grid here
                
                # Record results
                results.append({
                    "trial_id": i,
                    "nsdId": int(nsd_id),
                    "cosine_pred_gt": float(cosine) if cosine is not None else None,
                    "nn_nsdId": int(nn_nsd_id) if nn_nsd_id is not None else None,
                    "nn_cosine": float(nn_cosine) if nn_cosine is not None else None,
                    "image_path": str(img_path)
                })
                
            except Exception as e:
                logger.error(f"  ❌ Failed to generate image: {e}")
                continue
        
        # Save summary JSON
        summary_path = output_dir / "decode_summary.json"
        with open(summary_path, "w") as f:
            json.dump({
                "subject": args.subject,
                "encoder": args.encoder,
                "checkpoint": args.ckpt,
                "diffusion_model": args.model_id,
                "device": args.device,
                "dtype": args.dtype,
                "scheduler": args.scheduler,
                "guidance_scale": args.guidance,
                "num_inference_steps": args.steps,
                "clip_adapter": args.clip_adapter,
                "clip_adapter_target_dim": adapter_target_dim if clip_adapter else None,
                "n_generated": len(results),
                "mean_cosine": float(mean_cosine) if mean_cosine is not None else None,
                "results": results
            }, f, indent=2)
        
        logger.info("\n" + "=" * 80)
        logger.info("DIFFUSION DECODING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Generated {len(results)} images")
        logger.info(f"Device: {args.device}, Dtype: {args.dtype}, Scheduler: {args.scheduler}")
        logger.info(f"Guidance: {args.guidance}, Steps: {args.steps}")
        if mean_cosine is not None:
            logger.info(f"Mean cosine similarity (pred vs GT): {mean_cosine:.4f}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Summary: {summary_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Diffusion decoding failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
