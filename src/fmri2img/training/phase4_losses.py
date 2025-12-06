"""
Phase 4.2: Branch-Weighted Multi-Layer Loss
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BranchWeightedMultiLayerLoss(nn.Module):
    """
    Phase 4: Branch-weighted multi-layer loss.
    
    Structural Branch: ['layer_4', 'layer_8']
    Semantic Branch: ['layer_12', 'final', 'text']
    """
    
    def __init__(
        self,
        layer_weights: Dict[str, float],
        structural_weight: float = 1.0,
        semantic_weight: float = 1.0,
        use_mse: bool = False,
        mse_weight: float = 0.1,
        probabilistic: bool = False,
        kl_weight_max: float = 0.0001,
        kl_anneal_epochs: int = 10,
        kl_anneal_start: int = 10,
        text_clip_weight: float = 0.3
    ):
        super().__init__()
        
        logger.info("=" * 80)
        logger.info("Phase 4: Branch-Weighted Multi-Layer Loss")
        logger.info("=" * 80)
        logger.info(f"  Branch weights:")
        logger.info(f"    Structural: {structural_weight:.2f}")
        logger.info(f"    Semantic: {semantic_weight:.2f}")
        
        self.structural_layers = ['layer_4', 'layer_8']
        self.semantic_layers = ['layer_12', 'final', 'text']
        
        logger.info(f"  Structural layers: {self.structural_layers}")
        logger.info(f"  Semantic layers: {self.semantic_layers}")
        
        self.layer_weights = layer_weights
        self.structural_weight = structural_weight
        self.semantic_weight = semantic_weight
        self.use_mse = use_mse
        self.mse_weight = mse_weight
        self.probabilistic = probabilistic
        self.kl_weight_max = kl_weight_max
        self.kl_anneal_epochs = kl_anneal_epochs
        self.kl_anneal_start = kl_anneal_start
        self.text_clip_weight = text_clip_weight
        
        logger.info(f"  Probabilistic: {probabilistic}")
        if probabilistic:
            logger.info(f"  KL weight: 0.0 -> {kl_weight_max}")
        logger.info("=" * 80)
    
    def _get_kl_weight(self, current_epoch: int) -> float:
        """Compute KL weight with linear annealing."""
        if not self.probabilistic or current_epoch < self.kl_anneal_start:
            return 0.0
        
        progress = min(1.0, (current_epoch - self.kl_anneal_start) / self.kl_anneal_epochs)
        return progress * self.kl_weight_max
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        kl_loss: Optional[torch.Tensor] = None,
        current_epoch: int = 0,
        return_components: bool = False
    ):
        """Compute branch-weighted multi-layer loss."""
        device = next(iter(predictions.values())).device
        components = {}
        
        structural_loss = torch.tensor(0.0, device=device)
        semantic_loss = torch.tensor(0.0, device=device)
        
        # Compute layer-wise losses
        for layer_name, pred in predictions.items():
            if layer_name in ['structural_branch', 'semantic_branch', 'mu', 'logvar']:
                continue
            
            if layer_name not in targets:
                continue
            
            target = targets[layer_name]
            layer_weight = self.layer_weights.get(layer_name, 0.0)
            
            # Cosine similarity loss
            cos_sim = torch.nn.functional.cosine_similarity(pred, target, dim=-1)
            cos_loss = 1 - cos_sim.mean()
            
            # Optional: MSE loss
            if self.use_mse:
                mse_loss = torch.nn.functional.mse_loss(pred, target)
                layer_loss = (1 - self.mse_weight) * cos_loss + self.mse_weight * mse_loss
            else:
                layer_loss = cos_loss
            
            # Weight by layer importance
            weighted_layer_loss = layer_weight * layer_loss
            
            # Assign to branch
            if layer_name in self.structural_layers:
                structural_loss += weighted_layer_loss
            elif layer_name in self.semantic_layers:
                if layer_name == 'text':
                    weighted_layer_loss *= self.text_clip_weight
                semantic_loss += weighted_layer_loss
            
            components[layer_name] = layer_loss.item()
        
        # Apply branch weights
        components['structural_loss'] = structural_loss.item()
        components['semantic_loss'] = semantic_loss.item()
        
        total_loss = (
            self.structural_weight * structural_loss +
            self.semantic_weight * semantic_loss
        )
        
        # Add KL divergence (if probabilistic)
        if self.probabilistic and kl_loss is not None:
            kl_weight = self._get_kl_weight(current_epoch)
            weighted_kl = kl_weight * kl_loss
            total_loss += weighted_kl
            
            components['kl'] = kl_loss.item()
            components['kl_weight'] = kl_weight
            components['weighted_kl'] = weighted_kl.item()
        
        if return_components:
            return total_loss, components
        else:
            return total_loss
