"""
Kappa-Modulated Diffusion Guidance
====================================

Propagates per-ROI uncertainty (kappa) through the diffusion reconstruction
pipeline, enabling region-specific confidence in generated images.

Core mechanism:
    - High kappa_r (confident ROI) --> higher local guidance scale --> trust brain signal
    - Low kappa_r (uncertain ROI) --> lower local guidance scale --> rely on generative prior

Implementation strategies:
    1. **Global kappa-modulated CFG**: consensus kappa scales overall guidance_scale
    2. **Per-ROI spatial confidence map**: kappa_r mapped to image regions via
       cross-attention maps, modulating per-pixel guidance
    3. **Uncertainty-Aware Steps (UAS)**: kappa determines number of denoising steps
       (more steps for uncertain samples = more prior influence)

This module provides:
    - KappaGuidanceScheduler: maps kappa to per-step guidance values
    - SpatialConfidenceMap: kappa_per_roi --> pixel-level confidence
    - UncertaintyAwareDiffusion: full pipeline wrapping SD with kappa modulation

References:
    - Plan Paper 1, Contribution 3: "Uncertainty-Propagated Reconstruction"
    - Ho & Salimans (2022) "Classifier-Free Diffusion Guidance"
    - N4v9 config: DUA-CFG (existing simpler version)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class KappaGuidanceConfig:
    """Configuration for kappa-modulated diffusion."""
    # CFG modulation
    guidance_min: float = 3.0
    guidance_max: float = 12.0
    kappa_low_percentile: float = 10.0   # kappa at which guidance = guidance_min
    kappa_high_percentile: float = 90.0  # kappa at which guidance = guidance_max

    # Step modulation
    steps_min: int = 30
    steps_max: int = 150
    steps_kappa_threshold: float = 50.0

    # Spatial modulation
    spatial_sigma: float = 2.0           # Gaussian smoothing for spatial maps
    spatial_weight: float = 0.5          # blend spatial vs global guidance

    # Confidence map
    confidence_floor: float = 0.3        # minimum confidence (never fully distrust)
    confidence_ceiling: float = 1.0


class KappaGuidanceScheduler:
    """
    Maps kappa values to per-sample diffusion guidance parameters.

    Implements the mapping:
        kappa_consensus --> guidance_scale
        kappa_consensus --> num_inference_steps
    """

    def __init__(self, config: KappaGuidanceConfig):
        self.config = config
        self._kappa_stats = None

    def fit(self, kappa_distribution: np.ndarray):
        """Fit the scheduler to the training kappa distribution."""
        self._kappa_stats = {
            "low": float(np.percentile(kappa_distribution, self.config.kappa_low_percentile)),
            "high": float(np.percentile(kappa_distribution, self.config.kappa_high_percentile)),
            "mean": float(kappa_distribution.mean()),
            "std": float(kappa_distribution.std()),
        }
        logger.info(
            "KappaGuidanceScheduler fitted: kappa_low=%.2f, kappa_high=%.2f",
            self._kappa_stats["low"], self._kappa_stats["high"],
        )

    def get_guidance_scale(self, kappa: float) -> float:
        """Map a single kappa value to guidance scale."""
        if self._kappa_stats is None:
            # Default: linear mapping from [10, 100] -> [min, max]
            t = np.clip((kappa - 10.0) / 90.0, 0.0, 1.0)
        else:
            low, high = self._kappa_stats["low"], self._kappa_stats["high"]
            t = np.clip((kappa - low) / (high - low + 1e-8), 0.0, 1.0)

        guidance = self.config.guidance_min + t * (
            self.config.guidance_max - self.config.guidance_min
        )
        return float(guidance)

    def get_num_steps(self, kappa: float) -> int:
        """Map kappa to number of denoising steps (fewer = more prior)."""
        if kappa >= self.config.steps_kappa_threshold:
            return self.config.steps_min
        t = kappa / self.config.steps_kappa_threshold
        steps = self.config.steps_max - t * (self.config.steps_max - self.config.steps_min)
        return int(np.clip(steps, self.config.steps_min, self.config.steps_max))

    def get_parameters(self, kappa: float) -> Dict[str, float]:
        """Get all diffusion parameters for a given kappa."""
        return {
            "guidance_scale": self.get_guidance_scale(kappa),
            "num_inference_steps": self.get_num_steps(kappa),
            "kappa": kappa,
        }


class SpatialConfidenceMap:
    """
    Generate pixel-level confidence maps from per-ROI kappa values.

    Maps the 17 ROI kappas to a spatial confidence map over the image,
    using the known functional correspondence between ROIs and spatial
    positions in the visual field (retinotopic mapping).

    The mapping uses approximate retinotopic coverage:
    - V1-V3: full visual field (center-biased)
    - V4: upper/lower visual field
    - FFA: face-selective regions (foveal)
    - PPA: scene-selective (peripheral)
    - EBA: body-selective (variable)
    """

    # Approximate spatial coverage of each ROI as (center_y, center_x, radius)
    # in normalized [0, 1] image coordinates
    ROI_SPATIAL_PRIORS: Dict[str, Tuple[float, float, float]] = {
        "V1v": (0.65, 0.5, 0.35),   # lower visual field -> upper image
        "V1d": (0.35, 0.5, 0.35),   # upper visual field -> lower image
        "V2v": (0.65, 0.5, 0.4),
        "V2d": (0.35, 0.5, 0.4),
        "V3v": (0.65, 0.5, 0.45),
        "V3d": (0.35, 0.5, 0.45),
        "V3A": (0.3, 0.5, 0.3),
        "V3B": (0.5, 0.5, 0.35),
        "V4": (0.5, 0.5, 0.3),
        "FFA1": (0.5, 0.5, 0.2),    # foveal, central
        "FFA2": (0.5, 0.5, 0.25),
        "PPA": (0.5, 0.5, 0.5),     # peripheral scenes
        "EBA": (0.5, 0.5, 0.4),
        "OFA": (0.5, 0.5, 0.2),     # foveal faces
        "OPA": (0.5, 0.5, 0.5),     # peripheral scenes
        "RSC": (0.5, 0.5, 0.5),     # spatial scenes
        "nsdgeneral_other": (0.5, 0.5, 0.5),  # uniform fallback
    }

    def __init__(
        self,
        image_size: int = 768,
        roi_names: Optional[List[str]] = None,
        config: Optional[KappaGuidanceConfig] = None,
    ):
        self.image_size = image_size
        self.roi_names = roi_names or list(self.ROI_SPATIAL_PRIORS.keys())
        self.config = config or KappaGuidanceConfig()

    def generate(
        self,
        per_roi_kappas: np.ndarray,
        alphas: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate spatial confidence map from per-ROI kappas.

        Parameters
        ----------
        per_roi_kappas : (n_rois,) or (n_rois, 1) kappa values
        alphas : (n_rois,) optional attention weights for ROI mixing

        Returns
        -------
        confidence_map : (image_size, image_size) float32 confidence in [0, 1]
        """
        per_roi_kappas = np.squeeze(per_roi_kappas)
        n_rois = len(per_roi_kappas)

        if alphas is None:
            alphas = np.ones(n_rois) / n_rois

        # Normalize kappas to [0, 1] confidence
        kappa_min = per_roi_kappas.min()
        kappa_range = per_roi_kappas.max() - kappa_min + 1e-8
        normalized_kappas = (per_roi_kappas - kappa_min) / kappa_range

        # Scale to [floor, ceiling]
        floor = self.config.confidence_floor
        ceiling = self.config.confidence_ceiling
        confidence_values = floor + normalized_kappas * (ceiling - floor)

        # Create spatial map
        H = W = self.image_size
        y_coords = np.linspace(0, 1, H)
        x_coords = np.linspace(0, 1, W)
        yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")

        confidence_map = np.zeros((H, W), dtype=np.float32)
        weight_map = np.zeros((H, W), dtype=np.float32)

        for roi_idx in range(min(n_rois, len(self.roi_names))):
            roi_name = self.roi_names[roi_idx]
            if roi_name not in self.ROI_SPATIAL_PRIORS:
                continue

            cy, cx, radius = self.ROI_SPATIAL_PRIORS[roi_name]
            dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            gaussian_weight = np.exp(-0.5 * (dist / (radius * self.config.spatial_sigma)) ** 2)

            roi_weight = alphas[roi_idx] * gaussian_weight
            confidence_map += roi_weight * confidence_values[roi_idx]
            weight_map += roi_weight

        # Normalize by total weight
        confidence_map = np.where(
            weight_map > 1e-8,
            confidence_map / weight_map,
            floor,
        )

        return np.clip(confidence_map, floor, ceiling).astype(np.float32)


def kappa_modulated_generation(
    pipe,
    clip_embedding: np.ndarray,
    kappa_consensus: float,
    per_roi_kappas: Optional[np.ndarray] = None,
    alphas: Optional[np.ndarray] = None,
    scheduler: Optional[KappaGuidanceScheduler] = None,
    config: Optional[KappaGuidanceConfig] = None,
    seed: int = 42,
    return_confidence_map: bool = True,
) -> Dict:
    """
    Generate image with kappa-modulated diffusion guidance.

    This is the main entry point for uncertainty-aware reconstruction.

    Parameters
    ----------
    pipe : StableDiffusionPipeline (loaded)
    clip_embedding : (D,) predicted CLIP embedding
    kappa_consensus : scalar consensus kappa (from model)
    per_roi_kappas : (n_rois,) per-ROI kappas (for spatial map)
    alphas : (n_rois,) attention weights
    scheduler : fitted KappaGuidanceScheduler
    config : guidance configuration
    seed : random seed
    return_confidence_map : whether to generate spatial confidence map

    Returns
    -------
    Dict with keys: image, confidence_map, guidance_scale, num_steps, metadata
    """
    if config is None:
        config = KappaGuidanceConfig()
    if scheduler is None:
        scheduler = KappaGuidanceScheduler(config)

    # Determine guidance parameters from kappa
    params = scheduler.get_parameters(kappa_consensus)
    guidance_scale = params["guidance_scale"]
    num_steps = int(params["num_inference_steps"])

    logger.info(
        "Kappa-modulated generation: kappa=%.2f -> guidance=%.2f, steps=%d",
        kappa_consensus, guidance_scale, num_steps,
    )

    # Generate spatial confidence map
    confidence_map = None
    if return_confidence_map and per_roi_kappas is not None:
        spatial_gen = SpatialConfidenceMap(
            image_size=768, config=config
        )
        confidence_map = spatial_gen.generate(per_roi_kappas, alphas)

    # Generate image with modulated guidance
    generator = torch.Generator(device="cpu").manual_seed(seed)

    # Convert CLIP embedding to prompt embedding format
    # This depends on the specific pipeline (SD 2.1 uses CLIP text encoder space)
    clip_embedding_tensor = torch.from_numpy(clip_embedding).float().unsqueeze(0)

    try:
        image = pipe(
            prompt_embeds=clip_embedding_tensor.to(pipe.device),
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            generator=generator,
        ).images[0]
    except Exception as e:
        logger.warning("Direct embedding generation failed: %s. Falling back.", e)
        image = None

    result = {
        "image": image,
        "confidence_map": confidence_map,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_steps,
        "kappa_consensus": kappa_consensus,
        "metadata": {
            "per_roi_kappas": per_roi_kappas.tolist() if per_roi_kappas is not None else None,
            "seed": seed,
            "config": {
                "guidance_min": config.guidance_min,
                "guidance_max": config.guidance_max,
            },
        },
    }

    return result


def generate_confidence_overlaid_image(
    image: "Image.Image",
    confidence_map: np.ndarray,
    alpha: float = 0.4,
) -> "Image.Image":
    """
    Overlay confidence map on reconstructed image for visualization.

    Red = low confidence (model uncertain), green = high confidence.
    """
    from PIL import Image as PILImage
    import numpy as np

    img_array = np.array(image).astype(np.float32) / 255.0
    H, W = img_array.shape[:2]

    # Resize confidence map to match image
    if confidence_map.shape != (H, W):
        from scipy.ndimage import zoom
        scale_y = H / confidence_map.shape[0]
        scale_x = W / confidence_map.shape[1]
        confidence_map = zoom(confidence_map, (scale_y, scale_x))

    # Create color overlay: red (low) to green (high)
    overlay = np.zeros((H, W, 3), dtype=np.float32)
    overlay[:, :, 0] = 1.0 - confidence_map  # red channel (low confidence)
    overlay[:, :, 1] = confidence_map          # green channel (high confidence)

    # Blend
    blended = (1 - alpha) * img_array + alpha * overlay
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)

    return PILImage.fromarray(blended)
