"""
InfoNCE Loss with Memory Queue Support

Implements queue-augmented InfoNCE for effective contrastive learning with small batches.
Includes learnable temperature/logit_scale (CLIP-style).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class InfoNCEQueueLoss(nn.Module):
    """
    InfoNCE loss with memory queue for negative augmentation.
    
    Features:
    - Combines in-batch negatives with queue negatives
    - Learnable temperature (CLIP-style logit_scale)
    - Symmetric or asymmetric contrastive learning
    - Proper normalization and numerical stability
    
    Args:
        temperature: Initial temperature (default: 0.07, CLIP uses learnable)
        learnable_temperature: Whether temperature is learnable
        use_queue: Whether to use memory queue negatives
        symmetric: If True, compute loss in both directions (query→key and key→query)
        
    Usage:
        loss_fn = InfoNCEQueueLoss(learnable_temperature=True)
        
        # During training
        loss = loss_fn(
            query_embeddings=pred_embeddings,  # (B, D)
            key_embeddings=gt_embeddings,      # (B, D)
            queue=memory_queue,                 # MemoryQueue instance
        )
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        learnable_temperature: bool = True,
        use_queue: bool = True,
        symmetric: bool = True,
    ):
        super().__init__()
        
        self.use_queue = use_queue
        self.symmetric = symmetric
        
        # CLIP-style learnable logit scale
        # logit_scale = exp(log_scale), initialized so that 1/temperature = exp(log_scale)
        if learnable_temperature:
            log_scale = torch.log(torch.tensor(1.0 / temperature, dtype=torch.float32))
            self.logit_scale = nn.Parameter(log_scale)
            logger.info(f"Initialized learnable logit_scale: {self.logit_scale.item():.4f}")
        else:
            self.register_buffer("logit_scale", torch.log(torch.tensor(1.0 / temperature, dtype=torch.float32)))
            logger.info(f"Using fixed temperature: {temperature:.4f}")
    
    def forward(
        self,
        query_embeddings: torch.Tensor,
        key_embeddings: torch.Tensor,
        queue: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            query_embeddings: (B, D) query embeddings (predictions)
            key_embeddings: (B, D) key embeddings (targets/GT)
            queue: Optional MemoryQueue with negatives
            
        Returns:
            Scalar loss
        """
        B, D = query_embeddings.size()
        
        # Normalize in float32 for numerical stability under AMP
        query_norm = F.normalize(query_embeddings.float(), dim=1, p=2)
        key_norm = F.normalize(key_embeddings.float(), dim=1, p=2)
        
        # Clamp logit scale to prevent instability
        logit_scale = torch.clamp(self.logit_scale, max=4.6052)  # log(100)
        
        # Compute similarity matrix
        # query → key direction
        logits_qk = torch.matmul(query_norm, key_norm.T) * logit_scale.exp()
        
        # Add queue negatives if available
        if self.use_queue and queue is not None and queue.is_ready():
            queue_embeddings = queue.get_queue().float()  # (Q, D)
            queue_norm = F.normalize(queue_embeddings, dim=1, p=2)
            
            # Additional similarities with queue
            logits_q_queue = torch.matmul(query_norm, queue_norm.T) * logit_scale.exp()  # (B, Q)
            
            # Concatenate: [in-batch positives/negatives | queue negatives]
            logits_qk = torch.cat([logits_qk, logits_q_queue], dim=1)  # (B, B+Q)
        
        # Labels: diagonal elements are positives (first B columns)
        labels = torch.arange(B, device=query_embeddings.device)
        
        # Cross-entropy loss
        loss_qk = F.cross_entropy(logits_qk, labels)
        
        # Symmetric loss (key → query direction)
        if self.symmetric:
            logits_kq = torch.matmul(key_norm, query_norm.T) * logit_scale.exp()
            
            if self.use_queue and queue is not None and queue.is_ready():
                logits_k_queue = torch.matmul(key_norm, queue_norm.T) * logit_scale.exp()
                logits_kq = torch.cat([logits_kq, logits_k_queue], dim=1)
            
            loss_kq = F.cross_entropy(logits_kq, labels)
            
            loss = (loss_qk + loss_kq) / 2.0
        else:
            loss = loss_qk
        
        return loss
    
    def get_temperature(self) -> float:
        """Get current temperature value."""
        return 1.0 / self.logit_scale.exp().item()
    
    def get_logit_scale(self) -> float:
        """Get current logit scale value."""
        return self.logit_scale.exp().item()


class HardNegativeInfoNCE(nn.Module):
    """
    InfoNCE with hard negative mining from queue.
    
    Selects top-K hardest negatives (highest similarity) from queue
    to make training more challenging and efficient.
    
    Args:
        temperature: Initial temperature
        learnable_temperature: Whether temperature is learnable
        n_hard_negatives: Number of hard negatives to mine from queue
        symmetric: Symmetric contrastive loss
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        learnable_temperature: bool = True,
        n_hard_negatives: int = 512,
        symmetric: bool = True,
    ):
        super().__init__()
        
        self.n_hard_negatives = n_hard_negatives
        self.symmetric = symmetric
        
        if learnable_temperature:
            log_scale = torch.log(torch.tensor(1.0 / temperature, dtype=torch.float32))
            self.logit_scale = nn.Parameter(log_scale)
        else:
            self.register_buffer("logit_scale", torch.log(torch.tensor(1.0 / temperature, dtype=torch.float32)))
    
    def forward(
        self,
        query_embeddings: torch.Tensor,
        key_embeddings: torch.Tensor,
        queue: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Compute InfoNCE with hard negative mining."""
        B, D = query_embeddings.size()
        
        # Normalize in float32 for numerical stability under AMP
        query_norm = F.normalize(query_embeddings.float(), dim=1, p=2)
        key_norm = F.normalize(key_embeddings.float(), dim=1, p=2)
        
        logit_scale = torch.clamp(self.logit_scale, max=4.6052)
        
        # In-batch similarities
        logits_qk = torch.matmul(query_norm, key_norm.T) * logit_scale.exp()
        
        # Mine hard negatives from queue
        if queue is not None and queue.is_ready():
            queue_embeddings = queue.get_queue().float()
            queue_norm = F.normalize(queue_embeddings, dim=1, p=2)
            
            # Compute all similarities with queue
            sims_q_queue = torch.matmul(query_norm, queue_norm.T)  # (B, Q)
            
            # Select top-K hardest (highest similarity) negatives
            topk_sims, topk_indices = torch.topk(
                sims_q_queue, k=min(self.n_hard_negatives, len(queue)), dim=1
            )
            
            # Scale selected negatives
            topk_logits = topk_sims * logit_scale.exp()
            
            # Concatenate
            logits_qk = torch.cat([logits_qk, topk_logits], dim=1)
        
        labels = torch.arange(B, device=query_embeddings.device)
        loss_qk = F.cross_entropy(logits_qk, labels)
        
        if self.symmetric:
            logits_kq = torch.matmul(key_norm, query_norm.T) * logit_scale.exp()
            
            if queue is not None and queue.is_ready():
                sims_k_queue = torch.matmul(key_norm, queue_norm.T)
                topk_sims, _ = torch.topk(
                    sims_k_queue, k=min(self.n_hard_negatives, len(queue)), dim=1
                )
                topk_logits = topk_sims * logit_scale.exp()
                logits_kq = torch.cat([logits_kq, topk_logits], dim=1)
            
            loss_kq = F.cross_entropy(logits_kq, labels)
            loss = (loss_qk + loss_kq) / 2.0
        else:
            loss = loss_qk
        
        return loss
    
    def get_temperature(self) -> float:
        """Get current temperature value."""
        return 1.0 / self.logit_scale.exp().item()


def create_infonce_loss(config: dict) -> nn.Module:
    """
    Factory function to create InfoNCE loss from config.
    
    Args:
        config: Dictionary with loss configuration:
            - type: "infonce_queue" or "hard_negative_infonce"
            - temperature: float
            - learnable_temperature: bool
            - symmetric: bool
            - n_hard_negatives: int (for hard negative mining)
            
    Returns:
        Loss module
    """
    loss_type = config.get("type", "infonce_queue")
    
    if loss_type == "infonce_queue":
        return InfoNCEQueueLoss(
            temperature=config.get("temperature", 0.07),
            learnable_temperature=config.get("learnable_temperature", True),
            use_queue=config.get("use_queue", True),
            symmetric=config.get("symmetric", True),
        )
    elif loss_type == "hard_negative_infonce":
        return HardNegativeInfoNCE(
            temperature=config.get("temperature", 0.07),
            learnable_temperature=config.get("learnable_temperature", True),
            n_hard_negatives=config.get("n_hard_negatives", 512),
            symmetric=config.get("symmetric", True),
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
