"""
Multi-Objective Loss Functions for CLIP Alignment
=================================================

Implements SOTA loss functions for training fMRI → CLIP encoders:
1. MSE loss: L2 distance in CLIP space
2. Cosine similarity loss: Directional alignment
3. InfoNCE contrastive loss: Batch-wise discrimination

Scientific Rationale:
- MSE captures magnitude alignment (Euclidean distance)
- Cosine captures directional alignment (angular distance)
- InfoNCE provides contrastive learning signal (discrimination)
- Combining all three improves representation quality (Radford et al. 2021, Chen et al. 2020)

References:
- Radford et al. (2021): CLIP - contrastive learning of visual representations
- Chen et al. (2020): SimCLR - simple framework for contrastive learning
- Oord et al. (2018): Representation learning with contrastive predictive coding (InfoNCE)
- MindEye2 (Scotti et al. 2024): Multi-objective loss for fMRI decoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error loss in CLIP space.
    
    Measures L2 distance between predicted and target embeddings.
    Captures magnitude alignment (how close predictions are in Euclidean space).
    
    Args:
        pred: Predicted embeddings (B, D)
        target: Target embeddings (B, D)
    
    Returns:
        Scalar loss (averaged over batch)
    """
    return F.mse_loss(pred, target)


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Cosine distance loss: 1 - cosine_similarity(pred, target).
    
    Measures angular distance between predicted and target embeddings.
    Captures directional alignment (same direction in embedding space).
    
    IMPORTANT: Both pred and target should be L2-normalized for proper cosine computation.
    If not normalized, this still works but cosine similarity is not in [-1, 1].
    
    Args:
        pred: Predicted embeddings (B, D), ideally L2-normalized
        target: Target embeddings (B, D), ideally L2-normalized
    
    Returns:
        Scalar loss (averaged over batch)
    
    Scientific Context:
    - Cosine loss is standard for CLIP alignment (Radford et al. 2021)
    - Directional alignment often more important than magnitude for retrieval
    """
    # Compute cosine similarity: dot product of normalized vectors
    # If inputs are L2-normalized: cos_sim = (pred * target).sum(dim=-1)
    # Otherwise: use F.cosine_similarity which normalizes internally
    cos_sim = F.cosine_similarity(pred, target, dim=-1)  # (B,)
    
    # Cosine loss: 1 - similarity (minimizing distance)
    # Range: [0, 2] if normalized (0 = perfect match, 2 = opposite direction)
    return (1.0 - cos_sim).mean()


def info_nce_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 0.05
) -> torch.Tensor:
    """
    InfoNCE (Normalized Temperature-scaled Cross Entropy) contrastive loss.
    
    For each sample i in the batch:
    - Positive pair: (pred[i], target[i])
    - Negative pairs: (pred[i], target[j]) for all j ≠ i
    
    Encourages predicted embeddings to be close to their corresponding targets
    and far from other targets in the batch. This provides a discriminative
    learning signal that improves representation quality.
    
    Args:
        pred: Predicted embeddings (B, D), L2-normalized
        target: Target embeddings (B, D), L2-normalized
        temperature: Temperature scaling parameter (default: 0.05)
                    Lower temperature = harder discrimination
                    Typical range: [0.01, 0.1]
    
    Returns:
        Scalar loss (averaged over batch)
    
    Scientific Context:
    - InfoNCE from CPC (Oord et al. 2018), widely used in contrastive learning
    - CLIP uses symmetric InfoNCE over image-text pairs (Radford et al. 2021)
    - Temperature controls difficulty of negative discrimination
    - Requires sufficient batch size (recommend B >= 32 for meaningful negatives)
    
    Mathematical Formulation:
        L = -log(exp(sim(pred[i], target[i]) / τ) / Σ_j exp(sim(pred[i], target[j]) / τ))
        where sim() is cosine similarity, τ is temperature
    
    Example:
        >>> pred = torch.randn(64, 512)
        >>> pred = F.normalize(pred, dim=-1)  # L2 normalize
        >>> target = torch.randn(64, 512)
        >>> target = F.normalize(target, dim=-1)
        >>> loss = info_nce_loss(pred, target, temperature=0.05)
    """
    batch_size = pred.shape[0]
    
    if batch_size < 2:
        # InfoNCE requires at least 2 samples for negatives
        logger.warning(f"InfoNCE loss requires batch_size >= 2, got {batch_size}. Returning zero.")
        return torch.tensor(0.0, device=pred.device)
    
    # Compute similarity matrix: pred[i] · target[j] for all i, j
    # (B, D) @ (D, B) = (B, B)
    similarity_matrix = torch.matmul(pred, target.T)  # (B, B)
    
    # Scale by temperature
    similarity_matrix = similarity_matrix / temperature
    
    # Labels: diagonal elements are positives
    # For sample i, the positive is similarity_matrix[i, i]
    labels = torch.arange(batch_size, device=pred.device)
    
    # InfoNCE loss = cross-entropy with positive pairs on diagonal
    # For each row i: softmax over all columns, take log probability of column i
    loss = F.cross_entropy(similarity_matrix, labels)
    
    return loss


class MultiLoss(nn.Module):
    """
    Combined multi-objective loss for CLIP alignment.
    
    Combines MSE, cosine, and InfoNCE losses with configurable weights:
        L_total = w_mse * L_mse + w_cos * L_cos + w_nce * L_nce
    
    Args:
        mse_weight: Weight for MSE loss (default: 0.3)
        cosine_weight: Weight for cosine loss (default: 0.3)
        info_nce_weight: Weight for InfoNCE loss (default: 0.4)
        temperature: Temperature for InfoNCE (default: 0.05)
        log_components: Whether to return individual loss components (default: False)
    
    Scientific Rationale:
    - MSE: magnitude alignment
    - Cosine: directional alignment
    - InfoNCE: discriminative learning
    - Balanced weights (0.3/0.3/0.4) prioritize discrimination slightly
    - Can adjust weights via config for ablation studies
    
    Example:
        >>> criterion = MultiLoss(mse_weight=0.3, cosine_weight=0.3, 
        ...                       info_nce_weight=0.4, temperature=0.05)
        >>> pred = model(fmri_batch)
        >>> loss, components = criterion(pred, clip_targets, return_components=True)
        >>> print(f"Total: {loss:.3f}, MSE: {components['mse']:.3f}, "
        ...       f"Cosine: {components['cosine']:.3f}, InfoNCE: {components['info_nce']:.3f}")
    """
    
    def __init__(
        self,
        mse_weight: float = 0.3,
        cosine_weight: float = 0.3,
        info_nce_weight: float = 0.4,
        temperature: float = 0.05,
        log_components: bool = False
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
        self.info_nce_weight = info_nce_weight
        self.temperature = temperature
        self.log_components = log_components
        
        # Validate weights
        total_weight = mse_weight + cosine_weight + info_nce_weight
        if not torch.isclose(torch.tensor(total_weight), torch.tensor(1.0), atol=1e-3):
            logger.warning(f"Loss weights sum to {total_weight:.3f}, not 1.0. This is okay but may affect learning rate tuning.")
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.
        
        Args:
            pred: Predicted embeddings (B, D), L2-normalized
            target: Target embeddings (B, D), L2-normalized
            return_components: If True, return (total_loss, components_dict)
        
        Returns:
            If return_components=False: total_loss (scalar)
            If return_components=True: (total_loss, components_dict)
                components_dict = {"mse": scalar, "cosine": scalar, "info_nce": scalar}
        """
        # Compute individual losses
        loss_mse = mse_loss(pred, target)
        loss_cos = cosine_loss(pred, target)
        loss_nce = info_nce_loss(pred, target, temperature=self.temperature)
        
        # Weighted combination
        total_loss = (
            self.mse_weight * loss_mse +
            self.cosine_weight * loss_cos +
            self.info_nce_weight * loss_nce
        )
        
        if return_components or self.log_components:
            components = {
                "mse": loss_mse.item(),
                "cosine": loss_cos.item(),
                "info_nce": loss_nce.item(),
                "total": total_loss.item()
            }
            
            if return_components:
                return total_loss, components
            else:
                # Just log internally
                if self.log_components:
                    logger.debug(f"Loss components: MSE={loss_mse:.4f}, Cos={loss_cos:.4f}, NCE={loss_nce:.4f}")
        
        return total_loss


def compute_multiloss(
    pred: torch.Tensor,
    target: torch.Tensor,
    config: Optional[Dict[str, float]] = None
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    Functional interface for multi-objective loss (no nn.Module).
    
    Convenience function for computing multi-loss without creating a module.
    Useful for simple training scripts or one-off evaluations.
    
    Args:
        pred: Predicted embeddings (B, D), L2-normalized
        target: Target embeddings (B, D), L2-normalized
        config: Dictionary with keys:
                - mse_weight (default: 0.3)
                - cosine_weight (default: 0.3)
                - info_nce_weight (default: 0.4)
                - temperature (default: 0.05)
    
    Returns:
        total_loss: Scalar loss
        components: Dictionary with individual loss values
    
    Example:
        >>> config = {"mse_weight": 0.3, "cosine_weight": 0.3, 
        ...           "info_nce_weight": 0.4, "temperature": 0.05}
        >>> loss, components = compute_multiloss(pred, target, config)
    """
    if config is None:
        config = {}
    
    mse_weight = config.get("mse_weight", 0.3)
    cosine_weight = config.get("cosine_weight", 0.3)
    info_nce_weight = config.get("info_nce_weight", 0.4)
    temperature = config.get("temperature", 0.05)
    
    # Compute individual losses
    loss_mse = mse_loss(pred, target)
    loss_cos = cosine_loss(pred, target)
    loss_nce = info_nce_loss(pred, target, temperature=temperature)
    
    # Weighted combination
    total_loss = (
        mse_weight * loss_mse +
        cosine_weight * loss_cos +
        info_nce_weight * loss_nce
    )
    
    components = {
        "mse": loss_mse.item(),
        "cosine": loss_cos.item(),
        "info_nce": loss_nce.item(),
        "total": total_loss.item()
    }
    
    return total_loss, components


# Backward compatibility: keep old compose_loss function
def compose_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mse_weight: float = 0.5
) -> torch.Tensor:
    """
    Legacy combined cosine + MSE loss (for backward compatibility).
    
    This is the original loss function from train_utils.py.
    Kept for backward compatibility with existing training scripts.
    
    For new code, prefer MultiLoss or compute_multiloss which include InfoNCE.
    
    Args:
        pred: Predicted embeddings (B, D), L2-normalized
        target: Target embeddings (B, D), L2-normalized
        mse_weight: Weight for MSE term (default: 0.5)
    
    Returns:
        Scalar loss
    """
    loss_cos = cosine_loss(pred, target)
    loss_mse = mse_loss(pred, target)
    return loss_cos + mse_weight * loss_mse
