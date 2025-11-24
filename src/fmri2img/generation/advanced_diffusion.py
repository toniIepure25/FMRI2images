"""
Advanced Diffusion Generation with Best-of-N and BOI-lite
=========================================================

SOTA image generation strategies for fMRI → image reconstruction:
1. Best-of-N sampling: Generate N candidates, select best based on CLIP similarity
2. BOI-lite refinement: Iteratively refine using encoding model feedback

Scientific Rationale:
- Best-of-N improves semantic accuracy by exploring sample space (MindEye2)
- BOI-lite uses image→fMRI encoding model to select brain-aligned candidates
- Both strategies improve reconstruction quality without retraining decoder

References:
- MindEye2 (Scotti et al. 2024): Best-of-16 sampling
- Brain-Diffuser (Ozcelik et al. 2023): BOI (Brain-Optimized Inference)
- Takagi & Nishimoto (2023): Iterative refinement strategies
"""

import logging
from typing import Optional, List, Tuple, Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


def generate_best_of_n(
    pipe,
    clip_embedding: np.ndarray,
    n: int = 8,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    seed: int = 42,
    scoring: str = "clip",
    clip_encoder: Optional[Callable] = None,
    return_all: bool = False
) -> Image.Image | Tuple[Image.Image, List[Image.Image], np.ndarray]:
    """
    Generate N images and select the best based on CLIP similarity.
    
    For each fMRI sample:
    1. Generate N images from predicted CLIP embedding (different random seeds)
    2. Encode each image with CLIP encoder
    3. Compute cosine similarity with predicted CLIP embedding
    4. Return image with highest similarity
    
    Args:
        pipe: Stable Diffusion pipeline
        clip_embedding: Predicted CLIP embedding (512,) or (1024,), L2-normalized
        n: Number of candidates to generate (default: 8)
        guidance_scale: CFG guidance scale
        num_inference_steps: Number of denoising steps
        seed: Base random seed (each candidate uses seed + i)
        scoring: "clip" (use CLIP similarity) or "random" (for ablation)
        clip_encoder: Function to encode images to CLIP embeddings
                     Should accept PIL Image and return (512,) numpy array
        return_all: If True, return (best_image, all_images, scores)
    
    Returns:
        If return_all=False: best_image (PIL Image)
        If return_all=True: (best_image, all_images, scores)
    
    Scientific Context:
    - MindEye2 uses best-of-16 sampling to improve semantic accuracy
    - Works by exploring stochastic sampling space of diffusion model
    - CLIP scoring provides semantic alignment metric without pixel-level comparison
    
    Example:
        >>> # Generate best-of-8
        >>> best_img = generate_best_of_n(
        ...     pipe=sd_pipeline,
        ...     clip_embedding=pred_clip,
        ...     n=8,
        ...     clip_encoder=lambda img: encode_image_with_clip(img, clip_model)
        ... )
    """
    if n == 1:
        # Single sample (current behavior)
        from scripts.decode_diffusion import generate_image_from_clip_embedding
        img = generate_image_from_clip_embedding(
            pipe, clip_embedding, guidance_scale, num_inference_steps, seed
        )
        if return_all:
            return img, [img], np.array([1.0])
        return img
    
    if clip_encoder is None and scoring == "clip":
        raise ValueError("clip_encoder required for CLIP scoring")
    
    logger.info(f"Generating {n} candidates (best-of-N sampling)...")
    
    # Import here to avoid circular dependency
    from scripts.decode_diffusion import generate_image_from_clip_embedding
    
    # Generate N candidates with different seeds
    candidates = []
    for i in tqdm(range(n), desc="Generating candidates", leave=False):
        candidate_seed = seed + i
        img = generate_image_from_clip_embedding(
            pipe,
            clip_embedding,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=candidate_seed
        )
        candidates.append(img)
    
    # Score candidates
    if scoring == "clip":
        # Encode each candidate with CLIP
        candidate_embeddings = []
        for img in tqdm(candidates, desc="Encoding candidates", leave=False):
            emb = clip_encoder(img)  # Should return (512,) or (1024,)
            # Normalize
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            candidate_embeddings.append(emb)
        
        candidate_embeddings = np.stack(candidate_embeddings)  # (N, D)
        
        # Normalize predicted embedding
        pred_emb = clip_embedding / (np.linalg.norm(clip_embedding) + 1e-8)
        
        # Compute cosine similarities
        scores = candidate_embeddings @ pred_emb  # (N,)
        
        # Select best
        best_idx = np.argmax(scores)
        best_img = candidates[best_idx]
        
        logger.info(f"Best-of-{n}: Selected candidate {best_idx} (score={scores[best_idx]:.4f})")
        logger.info(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    
    elif scoring == "random":
        # Random selection (ablation baseline)
        best_idx = np.random.randint(n)
        best_img = candidates[best_idx]
        scores = np.ones(n) / n  # Uniform scores
        logger.info(f"Random selection: {best_idx}")
    
    else:
        raise ValueError(f"Unknown scoring: {scoring}")
    
    if return_all:
        return best_img, candidates, scores
    
    return best_img


def refine_with_boi_lite(
    initial_image: Image.Image,
    fmri_pca: np.ndarray,
    encoding_model: Callable,
    pipe,
    clip_encoder: Callable,
    pred_clip_embedding: np.ndarray,
    steps: int = 3,
    candidates_per_step: int = 4,
    guidance_scale: float = 7.5,
    noise_strength: float = 0.3,
    seed: int = 42
) -> Image.Image:
    """
    BOI-lite refinement: Iteratively refine image using encoding model feedback.
    
    Algorithm:
    1. Start with initial image (e.g., from best-of-N)
    2. For t in 1..T:
        a. Sample K nearby images using img2img with small noise
        b. For each candidate, predict fMRI using encoding model
        c. Select candidate with highest correlation to true fMRI
        d. Use selected as new current image
    3. Return final refined image
    
    Args:
        initial_image: Starting image (PIL Image)
        fmri_pca: True fMRI PCA vector (target to match)
        encoding_model: Callable that takes PIL Image and returns predicted fMRI PCA
        pipe: Stable Diffusion pipeline (for img2img)
        clip_encoder: Function to encode images to CLIP (for sampling)
        pred_clip_embedding: Predicted CLIP embedding (for conditioning)
        steps: Number of refinement iterations (default: 3)
        candidates_per_step: Number of candidates per iteration (default: 4)
        guidance_scale: CFG guidance scale
        noise_strength: Noise strength for img2img (0.0-1.0, default: 0.3)
        seed: Base random seed
    
    Returns:
        Refined image (PIL Image)
    
    Scientific Context:
    - Inspired by Brain-Diffuser's BOI (Brain-Optimized Inference)
    - Uses encoding model to select candidates that best match brain activity
    - Iterative refinement explores local neighborhood of initial sample
    
    Example:
        >>> # Refine best-of-N result
        >>> refined_img = refine_with_boi_lite(
        ...     initial_image=best_img,
        ...     fmri_pca=true_fmri_pca,
        ...     encoding_model=lambda img: predict_fmri_from_image(img, enc_model),
        ...     pipe=sd_pipeline,
        ...     clip_encoder=lambda img: encode_image_with_clip(img, clip_model),
        ...     pred_clip_embedding=pred_clip,
        ...     steps=3,
        ...     candidates_per_step=4
        ... )
    """
    if not hasattr(pipe, "img2img"):
        logger.warning("Pipeline does not have img2img capability, skipping BOI-lite")
        return initial_image
    
    logger.info(f"Starting BOI-lite refinement: {steps} steps, {candidates_per_step} candidates/step")
    
    current_image = initial_image
    
    # Normalize true fMRI for correlation computation
    fmri_pca_norm = (fmri_pca - fmri_pca.mean()) / (fmri_pca.std() + 1e-8)
    
    for step in range(1, steps + 1):
        logger.info(f"BOI-lite step {step}/{steps}")
        
        # Sample K candidates around current image
        candidates = []
        for k in range(candidates_per_step):
            candidate_seed = seed + step * 1000 + k
            
            # Use img2img to sample nearby image
            # Note: This requires StableDiffusionImg2ImgPipeline or similar
            try:
                candidate = pipe.img2img(
                    image=current_image,
                    prompt_embeds=torch.from_numpy(pred_clip_embedding).float().unsqueeze(0).to(pipe.device),
                    strength=noise_strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=20,  # Fewer steps for img2img
                    generator=torch.Generator(device=pipe.device).manual_seed(candidate_seed)
                ).images[0]
                
                candidates.append(candidate)
            
            except Exception as e:
                logger.warning(f"img2img failed: {e}, using original image")
                candidates.append(current_image)
        
        # Predict fMRI for each candidate
        predicted_fmris = []
        for candidate in candidates:
            pred_fmri = encoding_model(candidate)  # Should return (k_pca,) array
            predicted_fmris.append(pred_fmri)
        
        # Compute correlations with true fMRI
        correlations = []
        for pred_fmri in predicted_fmris:
            # Normalize prediction
            pred_fmri_norm = (pred_fmri - pred_fmri.mean()) / (pred_fmri.std() + 1e-8)
            # Pearson correlation
            corr = np.corrcoef(fmri_pca_norm, pred_fmri_norm)[0, 1]
            correlations.append(corr)
        
        correlations = np.array(correlations)
        
        # Select best candidate
        best_idx = np.argmax(correlations)
        current_image = candidates[best_idx]
        
        logger.info(f"  Selected candidate {best_idx} (correlation={correlations[best_idx]:.4f})")
        logger.info(f"  Correlation range: [{correlations.min():.4f}, {correlations.max():.4f}]")
    
    logger.info("BOI-lite refinement complete!")
    return current_image


def generate_with_all_strategies(
    pipe,
    clip_embedding: np.ndarray,
    fmri_pca: Optional[np.ndarray] = None,
    encoding_model: Optional[Callable] = None,
    clip_encoder: Optional[Callable] = None,
    strategies: List[str] = ["single", "best_of_n"],
    best_of_n: int = 8,
    boi_lite_steps: int = 3,
    boi_lite_candidates: int = 4,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    seed: int = 42
) -> dict:
    """
    Generate images using multiple strategies for comparison.
    
    Strategies:
    - "single": Single sample (baseline)
    - "best_of_n": Best-of-N sampling
    - "boi_lite": BOI-lite refinement (requires encoding_model)
    - "best_of_n_boi": Best-of-N + BOI-lite (full pipeline)
    
    Args:
        pipe: Stable Diffusion pipeline
        clip_embedding: Predicted CLIP embedding
        fmri_pca: True fMRI PCA (for BOI-lite)
        encoding_model: Image → fMRI encoding model (for BOI-lite)
        clip_encoder: Image → CLIP encoder (for best-of-N)
        strategies: List of strategy names to run
        best_of_n: Number of candidates for best-of-N
        boi_lite_steps: Refinement steps for BOI-lite
        boi_lite_candidates: Candidates per step for BOI-lite
        guidance_scale: CFG guidance scale
        num_inference_steps: Denoising steps
        seed: Random seed
    
    Returns:
        Dictionary mapping strategy name to generated image
    
    Example:
        >>> results = generate_with_all_strategies(
        ...     pipe=sd_pipeline,
        ...     clip_embedding=pred_clip,
        ...     fmri_pca=true_fmri,
        ...     encoding_model=enc_model,
        ...     clip_encoder=clip_encoder,
        ...     strategies=["single", "best_of_n", "best_of_n_boi"]
        ... )
        >>> 
        >>> # Save results
        >>> results["single"].save("single.png")
        >>> results["best_of_n"].save("best_of_n.png")
        >>> results["best_of_n_boi"].save("best_of_n_boi.png")
    """
    results = {}
    
    # Single sample (baseline)
    if "single" in strategies:
        logger.info("Strategy: Single sample")
        from scripts.decode_diffusion import generate_image_from_clip_embedding
        results["single"] = generate_image_from_clip_embedding(
            pipe, clip_embedding, guidance_scale, num_inference_steps, seed
        )
    
    # Best-of-N
    if "best_of_n" in strategies:
        logger.info(f"Strategy: Best-of-{best_of_n}")
        if clip_encoder is None:
            logger.warning("clip_encoder required for best-of-N, skipping")
        else:
            results["best_of_n"] = generate_best_of_n(
                pipe, clip_embedding, n=best_of_n,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed,
                clip_encoder=clip_encoder
            )
    
    # BOI-lite only (on single sample)
    if "boi_lite" in strategies:
        logger.info("Strategy: BOI-lite (single + refinement)")
        if encoding_model is None or fmri_pca is None:
            logger.warning("encoding_model and fmri_pca required for BOI-lite, skipping")
        else:
            # Start with single sample
            if "single" in results:
                initial_img = results["single"]
            else:
                from scripts.decode_diffusion import generate_image_from_clip_embedding
                initial_img = generate_image_from_clip_embedding(
                    pipe, clip_embedding, guidance_scale, num_inference_steps, seed
                )
            
            results["boi_lite"] = refine_with_boi_lite(
                initial_img, fmri_pca, encoding_model, pipe, clip_encoder,
                clip_embedding, steps=boi_lite_steps,
                candidates_per_step=boi_lite_candidates,
                guidance_scale=guidance_scale, seed=seed
            )
    
    # Best-of-N + BOI-lite (full pipeline)
    if "best_of_n_boi" in strategies:
        logger.info(f"Strategy: Best-of-{best_of_n} + BOI-lite")
        if clip_encoder is None or encoding_model is None or fmri_pca is None:
            logger.warning("clip_encoder, encoding_model, and fmri_pca required, skipping")
        else:
            # Start with best-of-N
            if "best_of_n" in results:
                initial_img = results["best_of_n"]
            else:
                initial_img = generate_best_of_n(
                    pipe, clip_embedding, n=best_of_n,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    seed=seed,
                    clip_encoder=clip_encoder
                )
            
            # Refine with BOI-lite
            results["best_of_n_boi"] = refine_with_boi_lite(
                initial_img, fmri_pca, encoding_model, pipe, clip_encoder,
                clip_embedding, steps=boi_lite_steps,
                candidates_per_step=boi_lite_candidates,
                guidance_scale=guidance_scale, seed=seed
            )
    
    return results
