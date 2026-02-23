"""
Gaussian-NCE: Distribution-Aware Contrastive Loss

Novel objective that makes uncertainty decision-relevant for retrieval.
Uses likelihood under predicted Gaussian q(z|x) as similarity score,
optimizing Bayesian retrieval performance directly.

Key idea: Instead of cosine similarity, use log-probability as score:
    s(query, key) = log N(key; mu_query, diag(exp(logvar_query)))

This makes the model learn when to be confident vs uncertain in a way
that directly improves probabilistic identification and retrieval.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging
import math

logger = logging.getLogger(__name__)


class GaussianNCELoss(nn.Module):
    """
    Gaussian-NCE: Distribution-aware contrastive learning.
    
    Computes InfoNCE-style loss using Gaussian log-likelihood as similarity:
    - Score: s(query, key) = log N(key; mu_q, Sigma_q)
    - Loss: -log softmax(s(positive)) over positives and negatives
    
    Features:
    - Uses predicted uncertainty (logvar) for scoring
    - Supports memory queue for negatives
    - Optional learnable temperature
    - Symmetric or asymmetric loss
    
    Args:
        use_temperature: Whether to use learnable temperature scaling
        temperature: Initial temperature (if learnable)
        use_queue: Whether to use memory queue
        symmetric: Compute loss in both directions
        clamp_logvar: Clamp log-variance for numerical stability
        
    Usage:
        loss_fn = GaussianNCELoss()
        
        loss = loss_fn(
            mu_query=pred_mu,           # (B, D)
            logvar_query=pred_logvar,   # (B, D)
            key_embeddings=gt_embeddings,  # (B, D)
            queue=memory_queue,
        )
    """
    
    def __init__(
        self,
        use_temperature: bool = False,
        temperature: float = 1.0,
        use_queue: bool = True,
        symmetric: bool = False,
        clamp_logvar: bool = True,
        logvar_min: float = -10.0,
        logvar_max: float = 5.0,
    ):
        super().__init__()
        
        self.use_queue = use_queue
        self.symmetric = symmetric
        self.clamp_logvar = clamp_logvar
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        
        if use_temperature:
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
            logger.info(f"Using learnable temperature for Gaussian-NCE")
        else:
            self.register_buffer("log_temperature", torch.log(torch.tensor(temperature)))
        
        logger.info(
            f"Initialized Gaussian-NCE: use_queue={use_queue}, "
            f"symmetric={symmetric}, clamp_logvar={clamp_logvar}"
        )
    
    def gaussian_log_likelihood(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Gaussian log-likelihood for diagonal covariance.
        
        log N(x; mu, diag(exp(logvar))) = -0.5 * sum(logvar + (x-mu)^2/exp(logvar) + log(2*pi))
        
        Args:
            x: (B, D) or (M, D) observations
            mu: (B, D) means
            logvar: (B, D) log-variances
            
        Returns:
            (B, M) log-likelihoods if x and mu have different batch dims,
            (B,) if same batch dim
        """
        if self.clamp_logvar:
            logvar = torch.clamp(logvar, min=self.logvar_min, max=self.logvar_max)
        
        # Always compute full pairwise (B, M) matrix for contrastive use
        x_exp = x.unsqueeze(0)           # (1, M, D)
        mu_exp = mu.unsqueeze(1)         # (B, 1, D)
        logvar_exp = logvar.unsqueeze(1) # (B, 1, D)
        
        diff = x_exp - mu_exp            # (B, M, D)
        log_prob = -0.5 * (
            logvar_exp + (diff ** 2) / torch.exp(logvar_exp) + math.log(2 * math.pi)
        )
        return log_prob.sum(dim=2)       # (B, M)
    
    def forward(
        self,
        mu_query: torch.Tensor,
        logvar_query: torch.Tensor,
        key_embeddings: torch.Tensor,
        mu_key: Optional[torch.Tensor] = None,
        logvar_key: Optional[torch.Tensor] = None,
        queue: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        Compute Gaussian-NCE loss.
        
        Args:
            mu_query: (B, D) query means (predictions)
            logvar_query: (B, D) query log-variances
            key_embeddings: (B, D) key embeddings (GT targets, deterministic)
            mu_key: Optional (B, D) key means (for symmetric loss)
            logvar_key: Optional (B, D) key log-variances (for symmetric loss)
            queue: Optional MemoryQueue with negative embeddings
            
        Returns:
            Scalar loss
        """
        B, D = mu_query.size()
        
        # Compute log-likelihood scores: query → keys
        # Score matrix: (B, B) for in-batch
        logits_qk = self.gaussian_log_likelihood(
            key_embeddings,  # (B, D) keys
            mu_query,        # (B, D) query means
            logvar_query,    # (B, D) query logvars
        )  # (B, B)
        
        # Add queue negatives
        if self.use_queue and queue is not None and queue.is_ready():
            queue_embeddings = queue.get_queue()  # (Q, D)
            
            # Scores with queue: (B, Q)
            logits_q_queue = self.gaussian_log_likelihood(
                queue_embeddings,
                mu_query,
                logvar_query,
            )  # (B, Q)
            
            # Concatenate: [in-batch | queue]
            logits_qk = torch.cat([logits_qk, logits_q_queue], dim=1)  # (B, B+Q)
        
        # Apply temperature
        temperature = torch.exp(self.log_temperature)
        logits_qk = logits_qk / temperature
        
        # Labels: diagonal (positives are at index i for query i)
        labels = torch.arange(B, device=mu_query.device)
        
        # Cross-entropy loss
        loss_qk = F.cross_entropy(logits_qk, labels)
        
        # Symmetric loss (if key distributions provided)
        if self.symmetric and mu_key is not None and logvar_key is not None:
            # key → query direction
            logits_kq = self.gaussian_log_likelihood(
                mu_query,  # Now using query means as targets
                mu_key,
                logvar_key,
            )
            
            if self.use_queue and queue is not None and queue.is_ready():
                # Keys to queue (not typically used, but for completeness)
                # In practice, queue stores GT embeddings, not distributions
                pass
            
            logits_kq = logits_kq / temperature
            loss_kq = F.cross_entropy(logits_kq, labels)
            
            loss = (loss_qk + loss_kq) / 2.0
        else:
            loss = loss_qk
        
        return loss
    
    def get_temperature(self) -> float:
        """Get current temperature."""
        return torch.exp(self.log_temperature).item()


class HybridGaussianNCE(nn.Module):
    """
    Hybrid loss combining standard InfoNCE (cosine) and Gaussian-NCE.
    
    Useful for transitioning from deterministic to probabilistic training.
    
    Args:
        gaussian_weight: Weight of Gaussian-NCE component
        cosine_weight: Weight of cosine InfoNCE component
        use_queue: Whether to use memory queue
        temperature: Temperature for both losses
    """
    
    def __init__(
        self,
        gaussian_weight: float = 1.0,
        cosine_weight: float = 0.1,
        use_queue: bool = True,
        temperature: float = 1.0,
    ):
        super().__init__()
        
        self.gaussian_weight = gaussian_weight
        self.cosine_weight = cosine_weight
        
        # Gaussian-NCE component
        self.gaussian_nce = GaussianNCELoss(
            use_temperature=True,
            temperature=temperature,
            use_queue=use_queue,
            symmetric=False,
        )
        
        # Cosine InfoNCE component
        from .infonce_queue import InfoNCEQueueLoss
        self.cosine_nce = InfoNCEQueueLoss(
            temperature=temperature,
            learnable_temperature=True,
            use_queue=use_queue,
            symmetric=True,
        )
        
        logger.info(
            f"Initialized Hybrid Gaussian-NCE: "
            f"gaussian_weight={gaussian_weight}, cosine_weight={cosine_weight}"
        )
    
    def forward(
        self,
        mu_query: torch.Tensor,
        logvar_query: torch.Tensor,
        key_embeddings: torch.Tensor,
        queue: Optional[nn.Module] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute hybrid loss.
        
        Returns:
            total_loss, loss_dict
        """
        # Gaussian-NCE
        loss_gaussian = self.gaussian_nce(
            mu_query=mu_query,
            logvar_query=logvar_query,
            key_embeddings=key_embeddings,
            queue=queue,
        )
        
        # Cosine InfoNCE (use mu as query)
        loss_cosine = self.cosine_nce(
            query_embeddings=mu_query,
            key_embeddings=key_embeddings,
            queue=queue,
        )
        
        # Combine
        total_loss = (
            self.gaussian_weight * loss_gaussian +
            self.cosine_weight * loss_cosine
        )
        
        loss_dict = {
            "gaussian_nce": loss_gaussian.item(),
            "cosine_nce": loss_cosine.item(),
            "total": total_loss.item(),
        }
        
        return total_loss, loss_dict


def create_gaussian_nce_loss(config: dict) -> nn.Module:
    """
    Factory function to create Gaussian-NCE loss from config.
    
    Args:
        config: Dictionary with loss configuration:
            - type: "gaussian_nce" or "hybrid_gaussian_nce"
            - temperature: float
            - use_queue: bool
            - symmetric: bool
            - clamp_logvar: bool
            - logvar_min: float
            - logvar_max: float
            - gaussian_weight: float (for hybrid)
            - cosine_weight: float (for hybrid)
            
    Returns:
        Loss module
    """
    loss_type = config.get("type", "gaussian_nce")
    
    if loss_type == "gaussian_nce":
        return GaussianNCELoss(
            use_temperature=config.get("use_temperature", False),
            temperature=config.get("temperature", 1.0),
            use_queue=config.get("use_queue", True),
            symmetric=config.get("symmetric", False),
            clamp_logvar=config.get("clamp_logvar", True),
            logvar_min=config.get("logvar_min", -10.0),
            logvar_max=config.get("logvar_max", 5.0),
        )
    elif loss_type == "hybrid_gaussian_nce":
        return HybridGaussianNCE(
            gaussian_weight=config.get("gaussian_weight", 1.0),
            cosine_weight=config.get("cosine_weight", 0.1),
            use_queue=config.get("use_queue", True),
            temperature=config.get("temperature", 1.0),
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
