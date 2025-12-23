"""
Brain Alignment Metrics for fMRI Reconstruction Evaluation
=========================================================

Quantify how well generated images align with brain activity patterns.
Uses an encoding model (image → fMRI) to predict brain responses from
generated images and compares to actual fMRI measurements.

Scientific Context:
- Encoding models predict fMRI from images (inverse of decoding)
- Brain alignment: correlation between predicted and actual fMRI
- Complements perceptual metrics (CLIP, SSIM) with neural fidelity
- Used in BOI-lite (Brain-Optimized Inference) for candidate selection

Key Metrics:
1. **Voxel-wise correlation**: Per-voxel corr(predicted_fmri, actual_fmri)
2. **Subject-level correlation**: Corr of ROI-averaged patterns
3. **Ceiling-normalized**: Normalized by noise ceiling for fair comparison

References:
- Naselaris et al. (2011). "Encoding and decoding in fMRI"
- Ozcelik & VanRullen (2023). "Brain-optimized inference"
- Kay et al. (2008). "Identifying natural images from human brain activity"
"""

import logging
from pathlib import Path
from typing import Union, List, Optional, Tuple, Dict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_brain_alignment(
    encoding_model,
    generated_images: Union[List[Path], List[Image.Image], np.ndarray],
    fmri_targets: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    batch_size: int = 32,
    device: str = "cuda",
    return_voxelwise: bool = False
) -> Dict[str, float]:
    """
    Compute brain alignment between generated images and fMRI targets.
    
    Uses an encoding model to predict fMRI from generated images, then
    computes correlation with actual fMRI measurements.
    
    Args:
        encoding_model: Trained image→fMRI encoding model
                       Must have .predict(images) or forward() method
                       Should be on correct device already
        generated_images: Generated images, one of:
            - List of PIL Images
            - List of image paths
            - Numpy array (n_images, H, W, 3)
        fmri_targets: Ground truth fMRI, shape (n_images, n_voxels)
        roi_mask: Boolean mask for ROI, shape (n_voxels,)
                 If provided, only compute alignment for ROI voxels
        batch_size: Batch size for encoding model inference
        device: Device for computation ("cuda" or "cpu")
        return_voxelwise: If True, return per-voxel correlations
    
    Returns:
        Dictionary with:
        - voxelwise_corr_mean: Mean voxel-wise correlation
        - voxelwise_corr_median: Median voxel-wise correlation
        - voxelwise_corr_std: Std of voxel-wise correlations
        - subject_level_corr: Correlation of ROI-averaged patterns
        - n_voxels: Number of voxels
        - n_images: Number of images
        If return_voxelwise=True, also includes:
        - voxelwise_corr: Per-voxel correlation array
    
    Example:
        >>> # Load encoding model
        >>> encoding_model = load_encoding_model("checkpoints/encoding/subj01.pt")
        >>> encoding_model = encoding_model.to("cuda").eval()
        >>> 
        >>> # Load generated images
        >>> gen_images = [Image.open(f"outputs/recon/sample_{i}.png") for i in range(100)]
        >>> 
        >>> # Load fMRI targets
        >>> fmri = np.load("data/subj01/fmri_test.npy")  # (100, 5000)
        >>> 
        >>> # Compute alignment
        >>> alignment = compute_brain_alignment(
        ...     encoding_model,
        ...     gen_images,
        ...     fmri,
        ...     batch_size=32,
        ...     device="cuda"
        ... )
        >>> 
        >>> print(f"Voxel-wise corr: {alignment['voxelwise_corr_mean']:.3f}")
        >>> print(f"Subject-level corr: {alignment['subject_level_corr']:.3f}")
        
    Scientific Context:
    - Voxel-wise correlation: How well each voxel's activity is predicted
    - Subject-level correlation: Overall pattern similarity (more robust)
    - Higher correlation = images better match brain representations
    - Complements CLIP/LPIPS with direct neural evidence
    """
    # Convert images to format expected by encoding model
    if isinstance(generated_images[0], (str, Path)):
        # Load images from paths
        images = [Image.open(path).convert("RGB") for path in generated_images]
    elif isinstance(generated_images[0], Image.Image):
        images = generated_images
    elif isinstance(generated_images, np.ndarray):
        # Convert numpy to PIL
        images = [Image.fromarray(img.astype(np.uint8)) for img in generated_images]
    else:
        raise ValueError(f"Unsupported image format: {type(generated_images[0])}")
    
    n_images = len(images)
    
    if len(fmri_targets) != n_images:
        raise ValueError(
            f"Number of images ({n_images}) must match fMRI targets ({len(fmri_targets)})"
        )
    
    # Apply ROI mask if provided
    if roi_mask is not None:
        fmri_targets = fmri_targets[:, roi_mask]
        n_voxels = roi_mask.sum()
        logger.info(f"Applied ROI mask: {n_voxels} voxels")
    else:
        n_voxels = fmri_targets.shape[1]
    
    # Predict fMRI from images using encoding model
    logger.info(f"Predicting fMRI from {n_images} images...")
    
    # Check if model has predict method or needs forward
    if hasattr(encoding_model, "predict"):
        # Batch prediction
        fmri_predicted = []
        for i in tqdm(range(0, n_images, batch_size), desc="Encoding"):
            batch_images = images[i:i+batch_size]
            with torch.no_grad():
                batch_pred = encoding_model.predict(batch_images)
                if isinstance(batch_pred, torch.Tensor):
                    batch_pred = batch_pred.cpu().numpy()
                fmri_predicted.append(batch_pred)
        
        fmri_predicted = np.concatenate(fmri_predicted, axis=0)
    
    else:
        # Use forward method with image preprocessing
        from torchvision import transforms
        from fmri2img.data.preprocess import get_image_transform
        
        # Get image transform (should match encoding model training)
        try:
            transform = get_image_transform()
        except:
            # Fallback to basic transform
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        fmri_predicted = []
        encoding_model.eval()
        
        with torch.no_grad():
            for i in tqdm(range(0, n_images, batch_size), desc="Encoding"):
                batch_images = images[i:i+batch_size]
                
                # Transform and batch
                batch_tensors = torch.stack([transform(img) for img in batch_images])
                batch_tensors = batch_tensors.to(device)
                
                # Forward pass
                batch_pred = encoding_model(batch_tensors)
                
                if isinstance(batch_pred, torch.Tensor):
                    batch_pred = batch_pred.cpu().numpy()
                
                # Apply ROI mask if needed and model outputs full brain
                if roi_mask is not None and batch_pred.shape[1] != n_voxels:
                    batch_pred = batch_pred[:, roi_mask]
                
                fmri_predicted.append(batch_pred)
        
        fmri_predicted = np.concatenate(fmri_predicted, axis=0)
    
    logger.info(f"Predicted fMRI shape: {fmri_predicted.shape}")
    logger.info(f"Target fMRI shape: {fmri_targets.shape}")
    
    # Ensure shapes match
    if fmri_predicted.shape != fmri_targets.shape:
        raise ValueError(
            f"Predicted fMRI shape {fmri_predicted.shape} != "
            f"target shape {fmri_targets.shape}"
        )
    
    # Compute voxel-wise correlations
    logger.info("Computing voxel-wise correlations...")
    voxelwise_corr = np.zeros(n_voxels)
    
    for v in range(n_voxels):
        # Pearson correlation across samples for this voxel
        corr = np.corrcoef(fmri_predicted[:, v], fmri_targets[:, v])[0, 1]
        voxelwise_corr[v] = corr if not np.isnan(corr) else 0.0
    
    # Compute subject-level correlation (correlation of ROI-averaged patterns)
    roi_pred = np.mean(fmri_predicted, axis=1)  # (n_images,)
    roi_target = np.mean(fmri_targets, axis=1)  # (n_images,)
    subject_level_corr = np.corrcoef(roi_pred, roi_target)[0, 1]
    
    if np.isnan(subject_level_corr):
        subject_level_corr = 0.0
        logger.warning("Subject-level correlation is NaN, setting to 0.0")
    
    # Aggregate statistics
    result = {
        "voxelwise_corr_mean": float(np.mean(voxelwise_corr)),
        "voxelwise_corr_median": float(np.median(voxelwise_corr)),
        "voxelwise_corr_std": float(np.std(voxelwise_corr)),
        "voxelwise_corr_min": float(np.min(voxelwise_corr)),
        "voxelwise_corr_max": float(np.max(voxelwise_corr)),
        "subject_level_corr": float(subject_level_corr),
        "n_voxels": int(n_voxels),
        "n_images": int(n_images),
    }
    
    if return_voxelwise:
        result["voxelwise_corr"] = voxelwise_corr
    
    logger.info(
        f"Brain alignment: voxel-wise={result['voxelwise_corr_mean']:.4f}, "
        f"subject-level={result['subject_level_corr']:.4f}"
    )
    
    return result


def compute_brain_alignment_with_ceiling(
    encoding_model,
    generated_images: Union[List[Path], List[Image.Image]],
    fmri_targets: np.ndarray,
    noise_ceiling_map: Optional[np.ndarray] = None,
    roi_mask: Optional[np.ndarray] = None,
    batch_size: int = 32,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Compute brain alignment with noise ceiling normalization.
    
    Extends compute_brain_alignment() by also computing ceiling-normalized
    versions of alignment metrics.
    
    Args:
        encoding_model: Image→fMRI encoding model
        generated_images: Generated images
        fmri_targets: Ground truth fMRI
        noise_ceiling_map: Noise ceiling per voxel, shape (n_voxels,)
                          If None, skip ceiling normalization
        roi_mask: ROI mask
        batch_size: Batch size for inference
        device: Device
    
    Returns:
        Dictionary with raw alignment metrics plus:
        - voxelwise_corr_mean_normalized: Ceiling-normalized voxel-wise corr
        - subject_level_corr_normalized: Ceiling-normalized subject-level corr
        - roi_ceiling: ROI-averaged noise ceiling
    
    Example:
        >>> from fmri2img.reliability import load_ncsnr, compute_voxel_noise_ceiling_from_ncsnr
        >>> 
        >>> # Load noise ceiling
        >>> ncsnr = load_ncsnr("subj01", roi="nsdgeneral")
        >>> ceiling_map = compute_voxel_noise_ceiling_from_ncsnr(ncsnr)
        >>> 
        >>> # Compute alignment with ceiling
        >>> alignment = compute_brain_alignment_with_ceiling(
        ...     encoding_model,
        ...     generated_images,
        ...     fmri_targets,
        ...     noise_ceiling_map=ceiling_map
        ... )
        >>> 
        >>> print(f"Raw corr: {alignment['voxelwise_corr_mean']:.3f}")
        >>> print(f"Normalized: {alignment['voxelwise_corr_mean_normalized']:.3f}")
        >>> print(f"Ceiling: {alignment['roi_ceiling']:.3f}")
    """
    # Compute raw alignment
    alignment = compute_brain_alignment(
        encoding_model,
        generated_images,
        fmri_targets,
        roi_mask=roi_mask,
        batch_size=batch_size,
        device=device,
        return_voxelwise=True
    )
    
    # Add ceiling normalization if available
    if noise_ceiling_map is not None:
        from fmri2img.reliability import aggregate_roi_ceiling, compute_ceiling_normalized_score
        
        # Apply ROI mask to ceiling if needed
        if roi_mask is not None:
            ceiling_masked = noise_ceiling_map[roi_mask]
        else:
            ceiling_masked = noise_ceiling_map
        
        # Aggregate ROI ceiling
        roi_ceiling = aggregate_roi_ceiling(ceiling_masked, aggregation="mean")
        alignment["roi_ceiling"] = roi_ceiling
        
        # Normalize voxel-wise correlation
        voxelwise_norm = compute_ceiling_normalized_score(
            alignment["voxelwise_corr_mean"],
            roi_ceiling
        )
        alignment["voxelwise_corr_mean_normalized"] = voxelwise_norm
        
        # Normalize subject-level correlation
        subject_norm = compute_ceiling_normalized_score(
            alignment["subject_level_corr"],
            roi_ceiling
        )
        alignment["subject_level_corr_normalized"] = subject_norm
        
        logger.info(
            f"Ceiling-normalized: voxel-wise={voxelwise_norm:.4f}, "
            f"subject-level={subject_norm:.4f} (ceiling={roi_ceiling:.4f})"
        )
    else:
        logger.info("No noise ceiling provided, skipping normalization")
        alignment["roi_ceiling"] = None
        alignment["voxelwise_corr_mean_normalized"] = None
        alignment["subject_level_corr_normalized"] = None
    
    # Remove voxelwise array from return (too large)
    if "voxelwise_corr" in alignment:
        del alignment["voxelwise_corr"]
    
    return alignment
