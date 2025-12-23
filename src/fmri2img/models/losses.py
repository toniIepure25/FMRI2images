"""
Loss Functions for fMRI-to-CLIP Encoding
=========================================

Implements various loss functions for training fMRI encoders:
- Cosine similarity loss (standard)
- MSE loss
- InfoNCE contrastive loss (novel contribution)
- Composable multi-objective loss

The InfoNCE loss directly optimizes retrieval performance by encouraging
batch-level discrimination in CLIP embedding space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Cosine similarity loss (1 - cosine_similarity).
    
    Args:
        pred: Predicted embeddings (B, D)
        target: Target embeddings (B, D)
        
    Returns:
        Scalar loss (mean over batch)
    """
    # Normalize to unit vectors
    pred_norm = F.normalize(pred, p=2, dim=1)
    target_norm = F.normalize(target, p=2, dim=1)
    
    # Cosine similarity: dot product of normalized vectors
    cos_sim = (pred_norm * target_norm).sum(dim=1)
    
    # Loss: 1 - similarity (range [0, 2])
    loss = 1.0 - cos_sim
    
    return loss.mean()


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error loss.
    
    Args:
        pred: Predicted embeddings (B, D)
        target: Target embeddings (B, D)
        
    Returns:
        Scalar loss (mean over batch)
    """
    return F.mse_loss(pred, target)


def infonce_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 0.07
) -> torch.Tensor:
    """
    InfoNCE contrastive loss for retrieval optimization.
    
    Treats each sample in the batch as a positive pair, and all other
    samples as negatives. Encourages the model to maximize agreement
    with the correct target while minimizing similarity to distractors.
    
    This is symmetric: we compute both pred→target and target→pred
    cross-entropy and average them (following CLIP training).
    
    Args:
        pred: Predicted embeddings (B, D)
        target: Target embeddings (B, D)
        temperature: Softmax temperature (lower = sharper distribution)
        
    Returns:
        Scalar loss (mean over batch)
        
    References:
        - van den Oord et al. (2018): Representation Learning with Contrastive Predictive Coding
        - Radford et al. (2021): Learning Transferable Visual Models (CLIP)
    """
    batch_size = pred.size(0)
    
    # Normalize embeddings
    pred_norm = F.normalize(pred, p=2, dim=1)
    target_norm = F.normalize(target, p=2, dim=1)
    
    # Compute similarity matrix: (B, B)
    # logits[i, j] = similarity between pred[i] and target[j]
    logits = torch.matmul(pred_norm, target_norm.t()) / temperature
    
    # Labels: diagonal elements are positives
    labels = torch.arange(batch_size, device=pred.device)
    
    # Symmetric loss (following CLIP)
    loss_i2t = F.cross_entropy(logits, labels)  # pred → target
    loss_t2i = F.cross_entropy(logits.t(), labels)  # target → pred
    
    loss = (loss_i2t + loss_t2i) / 2.0
    
    return loss


def compose_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    cosine_weight: float = 1.0,
    mse_weight: float = 0.0,
    infonce_weight: float = 0.0,
    temperature: float = 0.07
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    Compose multiple loss terms with configurable weights.
    
    Args:
        pred: Predicted embeddings (B, D)
        target: Target embeddings (B, D)
        cosine_weight: Weight for cosine loss
        mse_weight: Weight for MSE loss
        infonce_weight: Weight for InfoNCE loss
        temperature: Temperature for InfoNCE
        
    Returns:
        total_loss: Weighted sum of all losses
        components: Dictionary of individual loss values (for logging)
        
    Example:
        >>> pred = torch.randn(32, 512)
        >>> target = torch.randn(32, 512)
        >>> loss, components = compose_loss(
        ...     pred, target,
        ...     cosine_weight=1.0,
        ...     mse_weight=0.5,
        ...     infonce_weight=0.3
        ... )
        >>> print(f"Total: {loss.item():.4f}")
        >>> print(f"Cosine: {components['cosine']:.4f}")
        >>> print(f"MSE: {components['mse']:.4f}")
        >>> print(f"InfoNCE: {components['infonce']:.4f}")
    """
    components = {}
    total_loss = 0.0
    
    # Cosine loss
    if cosine_weight > 0:
        cos_loss = cosine_loss(pred, target)
        components['cosine'] = cos_loss.item()
        total_loss = total_loss + cosine_weight * cos_loss
    else:
        components['cosine'] = 0.0
    
    # MSE loss
    if mse_weight > 0:
        mse_loss_val = mse_loss(pred, target)
        components['mse'] = mse_loss_val.item()
        total_loss = total_loss + mse_weight * mse_loss_val
    else:
        components['mse'] = 0.0
    
    # InfoNCE loss
    if infonce_weight > 0:
        # Only compute if batch size > 1 (need negatives)
        if pred.size(0) > 1:
            info_loss = infonce_loss(pred, target, temperature)
            components['infonce'] = info_loss.item()
            total_loss = total_loss + infonce_weight * info_loss
        else:
            components['infonce'] = 0.0
    else:
        components['infonce'] = 0.0
    
    return total_loss, components


class ComposedLoss(nn.Module):
    """
    PyTorch module wrapper for compose_loss.
    
    Useful for integrating with standard PyTorch training loops.
    """
    def __init__(
        self,
        cosine_weight: float = 1.0,
        mse_weight: float = 0.0,
        infonce_weight: float = 0.0,
        temperature: float = 0.07
    ):
        super().__init__()
        self.cosine_weight = cosine_weight
        self.mse_weight = mse_weight
        self.infonce_weight = infonce_weight
        self.temperature = temperature
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass computing composed loss.
        
        Returns:
            total_loss: Weighted sum
            components: Dictionary of component values
        """
        return compose_loss(
            pred, target,
            cosine_weight=self.cosine_weight,
            mse_weight=self.mse_weight,
            infonce_weight=self.infonce_weight,
            temperature=self.temperature
        )
