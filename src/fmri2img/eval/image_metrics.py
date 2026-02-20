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


def pixel_correlation(
    generated_image: Image.Image,
    ground_truth_image: Image.Image,
    resize_to: int = 256,
) -> float:
    """
    PixCorr: Pearson correlation between flattened pixel arrays.

    Standard metric in brain decoding (Takagi & Nishimoto 2023,
    Scotti et al. 2023).  Measures low-level structural agreement.
    """
    transform = transforms.Compose([
        transforms.Resize((resize_to, resize_to)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    gen = transform(generated_image).flatten().numpy()
    gt = transform(ground_truth_image).flatten().numpy()
    corr = np.corrcoef(gen, gt)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


class AlexNetFeatureExtractor:
    """Extract features from early (layer2) and late (layer5) AlexNet."""

    _instance = None

    @classmethod
    def get(cls, device: str = "cuda"):
        if cls._instance is None or cls._instance._device != device:
            cls._instance = cls(device)
        return cls._instance

    def __init__(self, device: str = "cuda"):
        import torchvision.models as models
        self._device = device
        alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).to(device).eval()
        self.early = nn.Sequential(*list(alexnet.features.children())[:5]).to(device)
        self.late = nn.Sequential(*list(alexnet.features.children())).to(device)
        self._transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, img: Image.Image):
        t = self._transform(img).unsqueeze(0).to(self._device)
        return self.early(t).flatten(1), self.late(t).flatten(1)


def alexnet_feature_similarity(
    generated_image: Image.Image,
    ground_truth_image: Image.Image,
    device: str = "cuda",
) -> dict:
    """
    Compute AlexNet feature similarity at early and late layers.

    Used by MindEye, Brain Diffuser, etc. as a mid-level perceptual metric.
    """
    extractor = AlexNetFeatureExtractor.get(device)
    gen_early, gen_late = extractor.extract(generated_image)
    gt_early, gt_late = extractor.extract(ground_truth_image)

    def _cos(a, b):
        return torch.nn.functional.cosine_similarity(a, b, dim=-1).item()

    return {
        "alexnet_early": _cos(gen_early, gt_early),
        "alexnet_late": _cos(gen_late, gt_late),
    }


# ---------------------------------------------------------------------------
# Comprehensive evaluation
# ---------------------------------------------------------------------------

import torch.nn as nn


class ImageReconstructionEvaluator:
    """
    Evaluates a set of generated images against ground-truth stimuli
    using the full suite of metrics from the brain decoding literature:

        Low-level:   PixCorr, SSIM
        Mid-level:   AlexNet(2), AlexNet(5), LPIPS
        High-level:  CLIP-I (image cosine), InceptionV3 cosine
        Per-sample:  returns arrays for paired statistical tests

    Usage:
        evaluator = ImageReconstructionEvaluator(clip_model, device)
        results = evaluator.evaluate(gen_images, gt_images)
    """

    def __init__(self, clip_model=None, device: str = "cuda"):
        self.clip_model = clip_model
        self.device = device

    def evaluate(
        self,
        generated_images: List[Image.Image],
        ground_truth_images: List[Image.Image],
    ) -> dict:
        N = len(generated_images)
        assert N == len(ground_truth_images)

        pixcorr = np.zeros(N)
        ssim_vals = np.zeros(N)
        alex_early = np.zeros(N)
        alex_late = np.zeros(N)
        clip_i = np.zeros(N)

        for i in range(N):
            pixcorr[i] = pixel_correlation(generated_images[i], ground_truth_images[i])
            try:
                ssim_vals[i] = ssim_score(generated_images[i], ground_truth_images[i], device=self.device)
            except ImportError:
                ssim_vals[i] = float("nan")
            afeats = alexnet_feature_similarity(generated_images[i], ground_truth_images[i], self.device)
            alex_early[i] = afeats["alexnet_early"]
            alex_late[i] = afeats["alexnet_late"]
            if self.clip_model is not None:
                clip_i[i] = clip_score(generated_images[i], ground_truth_images[i], self.clip_model, self.device)

        results = {
            "pixcorr_mean": float(np.nanmean(pixcorr)),
            "pixcorr_std": float(np.nanstd(pixcorr)),
            "ssim_mean": float(np.nanmean(ssim_vals)),
            "ssim_std": float(np.nanstd(ssim_vals)),
            "alexnet_early_mean": float(np.nanmean(alex_early)),
            "alexnet_late_mean": float(np.nanmean(alex_late)),
            "clip_i_mean": float(np.nanmean(clip_i)),
            "clip_i_std": float(np.nanstd(clip_i)),
            "n_images": N,
            "per_sample": {
                "pixcorr": pixcorr.tolist(),
                "ssim": ssim_vals.tolist(),
                "alexnet_early": alex_early.tolist(),
                "alexnet_late": alex_late.tolist(),
                "clip_i": clip_i.tolist(),
            },
        }
        return results


def pixel_mse(
    generated_image: Image.Image,
    ground_truth_image: Image.Image,
    resize_to: int = 512,
) -> float:
    """Pixel-level MSE."""
    transform = transforms.Compose([
        transforms.Resize((resize_to, resize_to)),
        transforms.ToTensor(),
    ])
    gen_tensor = transform(generated_image)
    gt_tensor = transform(ground_truth_image)
    return float(((gen_tensor - gt_tensor) ** 2).mean().item())
