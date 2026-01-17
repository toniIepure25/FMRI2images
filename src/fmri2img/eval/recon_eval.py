"""
Image Reconstruction Evaluation Metrics for Stage 2 (CLIP → Image)
===================================================================

Standard image quality and identification metrics for evaluating
reconstructed images from Stage 2 (diffusion generation).

Metrics:
  1. Low-level: PixCorr, SSIM, PSNR
  2. Perceptual: LPIPS
  3. Semantic: CLIP-image similarity, 2AFC identification

Scientific Context:
- PixCorr: Pixel-wise correlation (Naselaris et al. 2009)
- SSIM: Structural similarity (Wang et al. 2004)
- PSNR: Peak signal-to-noise ratio (standard)
- LPIPS: Learned perceptual similarity (Zhang et al. 2018)
- CLIP-2AFC: Semantic identification in CLIP-image space (Ozcelik et al. 2023)

References:
- Naselaris et al. (2009). "Bayesian Reconstruction of Natural Images from Human Brain Activity"
- Wang et al. (2004). "Image Quality Assessment: From Error Visibility to Structural Similarity"
- Zhang et al. (2018). "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"
- Ozcelik et al. (2023). "Brain-optimized neural networks learn non-hierarchical representations"

Author: Research-grade reconstruction evaluation
Date: January 2026
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    SKIMAGE_AVAILABLE = True
except ImportError:
    logger.warning("scikit-image not available. SSIM/PSNR will be unavailable.")
    SKIMAGE_AVAILABLE = False

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    logger.warning("lpips not available. Install with: pip install lpips")
    LPIPS_AVAILABLE = False


@dataclass
class ReconstructionEvalResults:
    """Container for reconstruction evaluation results."""
    
    # Low-level metrics (per-image then averaged)
    pixcorr_mean: float
    pixcorr_std: float
    ssim_mean: float
    ssim_std: float
    psnr_mean: float
    psnr_std: float
    
    # Perceptual metric
    lpips_mean: float
    lpips_std: float
    
    # CLIP-based semantic metrics
    clip_sim_mean: float  # CLIP-image embedding similarity
    clip_sim_std: float
    clip_2afc_accuracy: float  # Identification accuracy
    
    # Retrieval in CLIP-image space (optional)
    clip_image_top1: Optional[float] = None
    clip_image_top5: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            'low_level': {
                'pixcorr_mean': float(self.pixcorr_mean),
                'pixcorr_std': float(self.pixcorr_std),
                'ssim_mean': float(self.ssim_mean),
                'ssim_std': float(self.ssim_std),
                'psnr_mean': float(self.psnr_mean),
                'psnr_std': float(self.psnr_std),
            },
            'perceptual': {
                'lpips_mean': float(self.lpips_mean),
                'lpips_std': float(self.lpips_std),
            },
            'semantic': {
                'clip_sim_mean': float(self.clip_sim_mean),
                'clip_sim_std': float(self.clip_sim_std),
                'clip_2afc_accuracy': float(self.clip_2afc_accuracy),
            }
        }
        
        if self.clip_image_top1 is not None:
            result['semantic']['clip_image_top1'] = float(self.clip_image_top1)
        if self.clip_image_top5 is not None:
            result['semantic']['clip_image_top5'] = float(self.clip_image_top5)
        
        return result


def compute_pixcorr(img_recon: np.ndarray, img_gt: np.ndarray) -> float:
    """
    Compute pixel-wise Pearson correlation.
    
    Standard metric in fMRI reconstruction (Naselaris et al. 2009).
    
    Args:
        img_recon: Reconstructed image (H, W, C) in [0, 1]
        img_gt: Ground truth image (H, W, C) in [0, 1]
    
    Returns:
        Correlation coefficient
    """
    # Flatten to vectors
    x = img_recon.flatten()
    y = img_gt.flatten()
    
    # Pearson correlation
    corr = np.corrcoef(x, y)[0, 1]
    
    return float(corr)


def compute_ssim(img_recon: np.ndarray, img_gt: np.ndarray) -> float:
    """
    Compute Structural Similarity Index (SSIM).
    
    Args:
        img_recon: (H, W, C) in [0, 1]
        img_gt: (H, W, C) in [0, 1]
    
    Returns:
        SSIM score (higher is better, range [0, 1])
    """
    if not SKIMAGE_AVAILABLE:
        logger.error("scikit-image not available for SSIM computation")
        return 0.0
    
    # SSIM expects values in [0, 1], channel_axis=-1 for RGB
    score = ssim(
        img_gt, img_recon,
        data_range=1.0,
        channel_axis=2 if img_gt.ndim == 3 else None
    )
    
    return float(score)


def compute_psnr(img_recon: np.ndarray, img_gt: np.ndarray) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        img_recon: (H, W, C) in [0, 1]
        img_gt: (H, W, C) in [0, 1]
    
    Returns:
        PSNR in dB (higher is better, typical range 20-40)
    """
    if not SKIMAGE_AVAILABLE:
        logger.error("scikit-image not available for PSNR computation")
        return 0.0
    
    score = psnr(img_gt, img_recon, data_range=1.0)
    
    return float(score)


class LPIPSEvaluator:
    """
    LPIPS (Learned Perceptual Image Patch Similarity) evaluator.
    
    Uses pretrained network to compute perceptual distance.
    Lower is better (0 = identical, higher = more different).
    
    Note: First call will download pretrained weights (~100MB).
    """
    
    def __init__(self, net: str = 'alex', device: str = 'cuda'):
        """
        Initialize LPIPS evaluator.
        
        Args:
            net: Backbone network ('alex' or 'vgg', 'alex' is faster)
            device: Device for computation
        """
        if not LPIPS_AVAILABLE:
            raise ImportError("lpips not available. Install with: pip install lpips")
        
        self.device = device
        self.model = lpips.LPIPS(net=net).to(device)
        self.model.eval()
        logger.info(f"Initialized LPIPS with {net} backbone on {device}")
    
    def compute(self, img_recon: np.ndarray, img_gt: np.ndarray) -> float:
        """
        Compute LPIPS between reconstructed and ground truth images.
        
        Args:
            img_recon: (H, W, C) numpy array in [0, 1]
            img_gt: (H, W, C) numpy array in [0, 1]
        
        Returns:
            LPIPS distance (lower is better, typical range 0.0-1.0)
        """
        # Convert to torch tensors and normalize to [-1, 1]
        # LPIPS expects (N, C, H, W) in [-1, 1]
        recon_t = torch.from_numpy(img_recon).permute(2, 0, 1).unsqueeze(0).float()
        gt_t = torch.from_numpy(img_gt).permute(2, 0, 1).unsqueeze(0).float()
        
        recon_t = recon_t * 2 - 1  # [0, 1] -> [-1, 1]
        gt_t = gt_t * 2 - 1
        
        recon_t = recon_t.to(self.device)
        gt_t = gt_t.to(self.device)
        
        with torch.no_grad():
            distance = self.model(recon_t, gt_t)
        
        return float(distance.item())
    
    def compute_batch(
        self,
        imgs_recon: List[np.ndarray],
        imgs_gt: List[np.ndarray],
        batch_size: int = 16
    ) -> np.ndarray:
        """
        Compute LPIPS for a batch of image pairs.
        
        Args:
            imgs_recon: List of reconstructed images (H, W, C) in [0, 1]
            imgs_gt: List of ground truth images (H, W, C) in [0, 1]
            batch_size: Batch size for processing
        
        Returns:
            Array of LPIPS distances
        """
        N = len(imgs_recon)
        distances = np.zeros(N)
        
        for i in range(0, N, batch_size):
            batch_recon = imgs_recon[i:i+batch_size]
            batch_gt = imgs_gt[i:i+batch_size]
            
            # Stack into batch
            recon_batch = np.stack([img for img in batch_recon], axis=0)  # (B, H, W, C)
            gt_batch = np.stack([img for img in batch_gt], axis=0)
            
            # Convert to torch
            recon_t = torch.from_numpy(recon_batch).permute(0, 3, 1, 2).float()
            gt_t = torch.from_numpy(gt_batch).permute(0, 3, 1, 2).float()
            
            recon_t = recon_t * 2 - 1
            gt_t = gt_t * 2 - 1
            
            recon_t = recon_t.to(self.device)
            gt_t = gt_t.to(self.device)
            
            with torch.no_grad():
                dists = self.model(recon_t, gt_t)
            
            distances[i:i+len(batch_recon)] = dists.cpu().numpy().flatten()
        
        return distances


class CLIPImageEvaluator:
    """
    CLIP-based semantic evaluation using image encoder.
    
    Computes:
      - Cosine similarity between CLIP image embeddings
      - 2AFC identification
      - Optional retrieval
    """
    
    def __init__(
        self,
        clip_model,
        preprocess,
        device: str = 'cuda'
    ):
        """
        Initialize CLIP image evaluator.
        
        Args:
            clip_model: CLIP vision encoder (e.g., from open_clip)
            preprocess: CLIP image preprocessing transform
            device: Device for computation
        """
        self.clip_model = clip_model.to(device)
        self.clip_model.eval()
        self.preprocess = preprocess
        self.device = device
        logger.info(f"Initialized CLIP image evaluator on {device}")
    
    def encode_image(self, img: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Encode image to CLIP embedding.
        
        Args:
            img: PIL Image or numpy array (H, W, C)
        
        Returns:
            CLIP embedding (D,), L2-normalized
        """
        if isinstance(img, np.ndarray):
            # Convert to PIL
            img = Image.fromarray((img * 255).astype(np.uint8))
        
        img_t = self.preprocess(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.clip_model.encode_image(img_t)
            embedding = F.normalize(embedding, p=2, dim=-1)
        
        return embedding.cpu().numpy()[0]
    
    def encode_batch(self, imgs: List[Union[np.ndarray, Image.Image]]) -> np.ndarray:
        """
        Encode batch of images.
        
        Args:
            imgs: List of images
        
        Returns:
            Embeddings (N, D)
        """
        imgs_pil = []
        for img in imgs:
            if isinstance(img, np.ndarray):
                img = Image.fromarray((img * 255).astype(np.uint8))
            imgs_pil.append(img)
        
        imgs_t = torch.stack([self.preprocess(img) for img in imgs_pil]).to(self.device)
        
        with torch.no_grad():
            embeddings = self.clip_model.encode_image(imgs_t)
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings.cpu().numpy()
    
    def compute_similarity(
        self,
        imgs_recon: List[np.ndarray],
        imgs_gt: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute CLIP cosine similarity between reconstructions and ground truth.
        
        Args:
            imgs_recon: List of reconstructed images
            imgs_gt: List of ground truth images
        
        Returns:
            Array of cosine similarities
        """
        emb_recon = self.encode_batch(imgs_recon)
        emb_gt = self.encode_batch(imgs_gt)
        
        # Cosine similarity (already normalized)
        sims = np.sum(emb_recon * emb_gt, axis=1)
        
        return sims
    
    def compute_2afc(
        self,
        imgs_recon: List[np.ndarray],
        imgs_gt: List[np.ndarray],
        n_trials: int = 1000,
        seed: int = 42
    ) -> float:
        """
        Two-alternative forced choice in CLIP-image space.
        
        For each trial:
          1. Pick reconstruction and correct GT
          2. Pick random distractor GT
          3. Score: correct if sim(recon, gt_correct) > sim(recon, gt_distractor)
        
        Args:
            imgs_recon: Reconstructed images
            imgs_gt: Ground truth images
            n_trials: Number of trials
            seed: Random seed
        
        Returns:
            2AFC accuracy
        """
        rng = np.random.RandomState(seed)
        N = len(imgs_recon)
        
        # Encode all images
        emb_recon = self.encode_batch(imgs_recon)
        emb_gt = self.encode_batch(imgs_gt)
        
        correct = 0
        
        for _ in range(n_trials):
            i = rng.randint(N)
            j = rng.randint(N - 1)
            if j >= i:
                j += 1
            
            sim_correct = np.dot(emb_recon[i], emb_gt[i])
            sim_distractor = np.dot(emb_recon[i], emb_gt[j])
            
            if sim_correct > sim_distractor:
                correct += 1
        
        return correct / n_trials


def evaluate_reconstructions(
    imgs_recon: List[np.ndarray],
    imgs_gt: List[np.ndarray],
    lpips_evaluator: Optional[LPIPSEvaluator] = None,
    clip_evaluator: Optional[CLIPImageEvaluator] = None,
    device: str = 'cuda'
) -> ReconstructionEvalResults:
    """
    Comprehensive reconstruction evaluation.
    
    Args:
        imgs_recon: List of reconstructed images (H, W, C) in [0, 1]
        imgs_gt: List of ground truth images (H, W, C) in [0, 1]
        lpips_evaluator: Optional LPIPS evaluator (will create if None)
        clip_evaluator: Optional CLIP evaluator
        device: Device for computation
    
    Returns:
        ReconstructionEvalResults with all metrics
    """
    N = len(imgs_recon)
    logger.info(f"Evaluating {N} reconstructions")
    
    # Low-level metrics
    pixcorrs = []
    ssims = []
    psnrs = []
    
    for i in range(N):
        pixcorrs.append(compute_pixcorr(imgs_recon[i], imgs_gt[i]))
        if SKIMAGE_AVAILABLE:
            ssims.append(compute_ssim(imgs_recon[i], imgs_gt[i]))
            psnrs.append(compute_psnr(imgs_recon[i], imgs_gt[i]))
    
    pixcorrs = np.array(pixcorrs)
    ssims = np.array(ssims) if SKIMAGE_AVAILABLE else np.zeros(N)
    psnrs = np.array(psnrs) if SKIMAGE_AVAILABLE else np.zeros(N)
    
    # LPIPS
    if lpips_evaluator is None and LPIPS_AVAILABLE:
        lpips_evaluator = LPIPSEvaluator(device=device)
    
    if lpips_evaluator is not None:
        lpips_scores = lpips_evaluator.compute_batch(imgs_recon, imgs_gt)
    else:
        lpips_scores = np.zeros(N)
    
    # CLIP-based metrics
    clip_sims = np.zeros(N)
    clip_2afc = 0.5  # Chance
    
    if clip_evaluator is not None:
        clip_sims = clip_evaluator.compute_similarity(imgs_recon, imgs_gt)
        clip_2afc = clip_evaluator.compute_2afc(imgs_recon, imgs_gt)
    
    return ReconstructionEvalResults(
        pixcorr_mean=float(pixcorrs.mean()),
        pixcorr_std=float(pixcorrs.std()),
        ssim_mean=float(ssims.mean()),
        ssim_std=float(ssims.std()),
        psnr_mean=float(psnrs.mean()),
        psnr_std=float(psnrs.std()),
        lpips_mean=float(lpips_scores.mean()),
        lpips_std=float(lpips_scores.std()),
        clip_sim_mean=float(clip_sims.mean()),
        clip_sim_std=float(clip_sims.std()),
        clip_2afc_accuracy=float(clip_2afc),
    )
