"""
Image Quality Metrics for fMRI Reconstruction Evaluation
========================================================

Implements perceptual metrics for evaluating generated images:
- CLIPScore: CLIP embedding similarity
- SSIM: Structural Similarity Index
- LPIPS: Learned Perceptual Image Patch Similarity

Scientific Context:
- CLIPScore measures semantic similarity in CLIP space (Hessel et al. 2021)
- SSIM measures structural similarity (Wang et al. 2004)
- LPIPS measures perceptual distance using deep features (Zhang et al. 2018)

References:
- Hessel et al. (2021). "CLIPScore: A Reference-free Evaluation Metric for Image Captioning"
- Wang et al. (2004). "Image Quality Assessment: From Error Visibility to Structural Similarity"
- Zhang et al. (2018). "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"
"""

import numpy as np
import torch
from PIL import Image
from typing import Union, List
import torchvision.transforms as transforms


def preprocess_image_for_clip(
    image: Image.Image,
    image_size: int = 224
) -> torch.Tensor:
    """
    Preprocess PIL image for CLIP encoding.
    
    Args:
        image: PIL Image (RGB)
        image_size: Target size (default: 224 for CLIP)
        
    Returns:
        tensor: Preprocessed image tensor (3, H, W), normalized
    """
    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    
    return transform(image)


def clip_score(
    generated_image: Image.Image,
    ground_truth_image: Image.Image,
    clip_model,
    device: str = "cuda"
) -> float:
    """
    Compute CLIPScore between generated and ground truth images.
    
    CLIPScore measures semantic similarity in CLIP embedding space.
    Higher is better (range: [-1, 1], typically [0.3, 0.8] for reconstructions).
    
    Args:
        generated_image: Generated PIL Image
        ground_truth_image: Ground truth PIL Image
        clip_model: CLIP vision encoder (e.g., from open_clip)
        device: Device for computation
        
    Returns:
        score: Cosine similarity in CLIP space (float)
        
    Example:
        >>> import open_clip
        >>> clip_model, _, preprocess = open_clip.create_model_and_transforms(
        ...     "ViT-L-14", pretrained="openai"
        ... )
        >>> score = clip_score(gen_img, gt_img, clip_model, "cuda")
        >>> print(f"CLIPScore: {score:.4f}")
    """
    # Preprocess images
    gen_tensor = preprocess_image_for_clip(generated_image).unsqueeze(0).to(device)
    gt_tensor = preprocess_image_for_clip(ground_truth_image).unsqueeze(0).to(device)
    
    # Encode with CLIP
    with torch.no_grad():
        gen_emb = clip_model.encode_image(gen_tensor)
        gt_emb = clip_model.encode_image(gt_tensor)
        
        # Normalize
        gen_emb = gen_emb / gen_emb.norm(dim=-1, keepdim=True)
        gt_emb = gt_emb / gt_emb.norm(dim=-1, keepdim=True)
        
        # Cosine similarity
        similarity = (gen_emb * gt_emb).sum(dim=-1).item()
    
    return similarity


def batch_clip_score(
    generated_images: List[Image.Image],
    ground_truth_images: List[Image.Image],
    clip_model,
    device: str = "cuda",
    batch_size: int = 32
) -> np.ndarray:
    """
    Compute CLIPScore for a batch of images.
    
    Args:
        generated_images: List of generated PIL Images
        ground_truth_images: List of ground truth PIL Images
        clip_model: CLIP vision encoder
        device: Device for computation
        batch_size: Batch size for processing
        
    Returns:
        scores: Array of cosine similarities, shape (n_images,)
    """
    assert len(generated_images) == len(ground_truth_images)
    
    n_images = len(generated_images)
    scores = np.zeros(n_images)
    
    for i in range(0, n_images, batch_size):
        batch_gen = generated_images[i:i+batch_size]
        batch_gt = ground_truth_images[i:i+batch_size]
        
        # Preprocess batch
        gen_tensors = torch.stack([
            preprocess_image_for_clip(img) for img in batch_gen
        ]).to(device)
        
        gt_tensors = torch.stack([
            preprocess_image_for_clip(img) for img in batch_gt
        ]).to(device)
        
        # Encode
        with torch.no_grad():
            gen_emb = clip_model.encode_image(gen_tensors)
            gt_emb = clip_model.encode_image(gt_tensors)
            
            # Normalize
            gen_emb = gen_emb / gen_emb.norm(dim=-1, keepdim=True)
            gt_emb = gt_emb / gt_emb.norm(dim=-1, keepdim=True)
            
            # Cosine similarity (element-wise)
            similarities = (gen_emb * gt_emb).sum(dim=-1).cpu().numpy()
        
        scores[i:i+len(batch_gen)] = similarities
    
    return scores


def ssim_score(
    generated_image: Image.Image,
    ground_truth_image: Image.Image,
    resize_to: int = 512,
    device: str = "cuda"
) -> float:
    """
    Compute SSIM between generated and ground truth images.
    
    Requires: pip install torchmetrics
    
    Args:
        generated_image: Generated PIL Image
        ground_truth_image: Ground truth PIL Image
        resize_to: Resize images to this size for computation
        device: Device for computation
        
    Returns:
        score: SSIM value (float in [0, 1], higher is better)
    """
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
    except ImportError:
        raise ImportError("SSIM requires torchmetrics: pip install torchmetrics")
    
    # Resize and convert to tensors
    transform = transforms.Compose([
        transforms.Resize((resize_to, resize_to)),
        transforms.ToTensor()
    ])
    
    gen_tensor = transform(generated_image).unsqueeze(0).to(device)
    gt_tensor = transform(ground_truth_image).unsqueeze(0).to(device)
    
    # Compute SSIM
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    score = ssim_fn(gen_tensor, gt_tensor).item()
    
    return score


def lpips_score(
    generated_image: Image.Image,
    ground_truth_image: Image.Image,
    resize_to: int = 512,
    net: str = "alex",
    device: str = "cuda"
) -> float:
    """
    Compute LPIPS perceptual distance.
    
    Requires: pip install lpips
    
    Args:
        generated_image: Generated PIL Image
        ground_truth_image: Ground truth PIL Image
        resize_to: Resize images to this size
        net: LPIPS network ("alex", "vgg", or "squeeze")
        device: Device for computation
        
    Returns:
        score: LPIPS distance (float, lower is better, typically [0, 1])
    """
    try:
        import lpips
    except ImportError:
        raise ImportError("LPIPS requires lpips: pip install lpips")
    
    # Resize and convert to tensors (normalized to [-1, 1])
    transform = transforms.Compose([
        transforms.Resize((resize_to, resize_to)),
        transforms.ToTensor()
    ])
    
    gen_tensor = transform(generated_image).unsqueeze(0).to(device) * 2 - 1
    gt_tensor = transform(ground_truth_image).unsqueeze(0).to(device) * 2 - 1
    
    # Compute LPIPS
    lpips_fn = lpips.LPIPS(net=net).to(device)
    
    with torch.no_grad():
        distance = lpips_fn(gen_tensor, gt_tensor).item()
    
    return distance


def compute_all_metrics(
    generated_image: Image.Image,
    ground_truth_image: Image.Image,
    clip_model,
    device: str = "cuda",
    include_lpips: bool = False,
    include_ssim: bool = False
) -> dict:
    """
    Compute all available image quality metrics.
    
    Args:
        generated_image: Generated PIL Image
        ground_truth_image: Ground truth PIL Image
        clip_model: CLIP model for CLIPScore
        device: Device for computation
        include_lpips: Compute LPIPS (slower)
        include_ssim: Compute SSIM (slower)
        
    Returns:
        metrics: Dict with all computed metrics
    """
    metrics = {}
    
    # CLIPScore (always computed)
    metrics["clip_score"] = clip_score(
        generated_image, ground_truth_image, clip_model, device
    )
    
    # SSIM (optional)
    if include_ssim:
        try:
            metrics["ssim"] = ssim_score(
                generated_image, ground_truth_image, device=device
            )
        except ImportError:
            pass
    
    # LPIPS (optional)
    if include_lpips:
        try:
            metrics["lpips"] = lpips_score(
                generated_image, ground_truth_image, device=device
            )
        except ImportError:
            pass
    
    return metrics


def pixel_mse(
    generated_image: Image.Image,
    ground_truth_image: Image.Image,
    resize_to: int = 512
) -> float:
    """
    Compute pixel-level MSE (for completeness, not recommended as primary metric).
    
    Args:
        generated_image: Generated PIL Image
        ground_truth_image: Ground truth PIL Image
        resize_to: Resize images to this size
        
    Returns:
        mse: Mean squared error (float, lower is better)
    """
    transform = transforms.Compose([
        transforms.Resize((resize_to, resize_to)),
        transforms.ToTensor()
    ])
    
    gen_tensor = transform(generated_image)
    gt_tensor = transform(ground_truth_image)
    
    mse = ((gen_tensor - gt_tensor) ** 2).mean().item()
    
    return mse
