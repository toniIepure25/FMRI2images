"""
Diffusion Utilities for Image Generation
========================================

Helper functions for loading and configuring Stable Diffusion pipelines.
"""

import logging
import sys
import torch
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def load_diffusion_pipeline(
    model_id: str = "stabilityai/stable-diffusion-2-1",
    device: str = "cuda",
    dtype: str = "float16",
    scheduler: str = "dpm"
):
    """
    Load Stable Diffusion pipeline with standard configuration.
    
    Args:
        model_id: HuggingFace model ID
        device: Device for computation
        dtype: "float16" or "float32"
        scheduler: "dpm", "euler", "pndm", or "default"
        
    Returns:
        pipeline: StableDiffusionPipeline ready for generation
    """
    try:
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    except ImportError:
        logger.error("diffusers not installed: pip install diffusers transformers accelerate")
        sys.exit(1)
    
    logger.info(f"Loading Stable Diffusion pipeline: {model_id}")
    
    # Determine dtype
    torch_dtype = torch.float32 if dtype == "float32" else torch.float16
    
    if torch_dtype == torch.float16 and device == "cpu":
        logger.warning("float16 not supported on CPU, using float32")
        torch_dtype = torch.float32
    
    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # Configure scheduler
    if scheduler == "dpm":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        logger.info("Using DPMSolverMultistep scheduler")
    
    # Move to device
    pipe = pipe.to(device)
    
    # Enable memory optimizations
    try:
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        logger.info("Enabled memory optimizations")
    except:
        pass
    
    logger.info(f"Pipeline loaded on {device}")
    return pipe


def generate_from_clip_embedding(
    pipe,
    clip_embedding: torch.Tensor,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    negative_prompt: str = "blurry, low quality"
):
    """
    Generate image from CLIP embedding using Stable Diffusion.
    
    Args:
        pipe: StableDiffusionPipeline
        clip_embedding: CLIP embedding, shape (1, 512) or (512,)
        num_inference_steps: Number of diffusion steps
        guidance_scale: Classifier-free guidance scale
        seed: Random seed (None for random)
        negative_prompt: Negative prompt string
        
    Returns:
        image: PIL Image
    """
    import torch
    from PIL import Image
    
    # Ensure correct shape
    if clip_embedding.dim() == 1:
        clip_embedding = clip_embedding.unsqueeze(0)
    
    # Set seed
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
    else:
        generator = None
    
    # Expand to batch size 2 for classifier-free guidance
    prompt_embeds = clip_embedding.repeat(2, 1)
    
    # Generate
    with torch.no_grad():
        output = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt=[negative_prompt],
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )
    
    return output.images[0]


def load_clip_model(device: str = "cuda"):
    """
    Load CLIP model for scoring.
    
    Returns:
        clip_model: CLIP vision encoder
        preprocess: CLIP preprocessing function
    """
    try:
        import open_clip
    except ImportError:
        logger.error("open_clip not installed: pip install open_clip_torch")
        sys.exit(1)
    
    logger.info("Loading CLIP model for scoring...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    model = model.to(device)
    model.eval()
    
    return model, preprocess
