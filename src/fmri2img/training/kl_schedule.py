"""
KL Divergence Annealing and Free-Bits for VAE-style Training

Implements proper KL regularization for probabilistic embedding prediction:
- Annealing schedule: gradually increase KL weight from 0 to target
- Free-bits: prevent posterior collapse by allowing minimum KL per dimension
"""

import torch
import torch.nn as nn
from typing import Optional, Literal
import logging
import math

logger = logging.getLogger(__name__)


class KLScheduler:
    """
    KL divergence weight scheduler with annealing and free-bits.
    
    Features:
    - Linear, cosine, or cyclical annealing
    - Free-bits mechanism to prevent collapse
    - Per-step or per-epoch updates
    
    Args:
        anneal_type: "linear", "cosine", "cyclical", or "none"
        start_weight: Initial KL weight (default: 0.0)
        end_weight: Final KL weight (default: 1.0)
        n_steps: Number of steps for annealing
        free_bits: Minimum KL per dimension (default: 0.0)
        free_bits_aggregate: How to aggregate free-bits: "dimension" or "total"
        
    Usage:
        scheduler = KLScheduler(
            anneal_type="linear",
            start_weight=0.0,
            end_weight=0.01,  # Small weight for regularization
            n_steps=10000,
            free_bits=0.5,
        )
        
        # During training
        for step in range(n_steps):
            kl_weight = scheduler.step()
            kl_loss = scheduler.apply_free_bits(kl_divergence)
            loss = reconstruction_loss + kl_weight * kl_loss
    """
    
    def __init__(
        self,
        anneal_type: Literal["linear", "cosine", "cyclical", "none"] = "linear",
        start_weight: float = 0.0,
        end_weight: float = 1.0,
        n_steps: int = 10000,
        free_bits: float = 0.0,
        free_bits_aggregate: Literal["dimension", "total"] = "dimension",
    ):
        self.anneal_type = anneal_type
        self.start_weight = start_weight
        self.end_weight = end_weight
        self.n_steps = n_steps
        self.free_bits = free_bits
        self.free_bits_aggregate = free_bits_aggregate
        
        self.current_step = 0
        self.current_weight = start_weight
        
        logger.info(
            f"Initialized KLScheduler: type={anneal_type}, "
            f"weight={start_weight:.4f}→{end_weight:.4f} over {n_steps} steps, "
            f"free_bits={free_bits:.4f} ({free_bits_aggregate})"
        )
    
    def step(self) -> float:
        """
        Update scheduler and return current KL weight.
        
        Returns:
            Current KL weight
        """
        if self.anneal_type == "none":
            self.current_weight = self.end_weight
        
        elif self.anneal_type == "linear":
            progress = min(self.current_step / self.n_steps, 1.0)
            self.current_weight = self.start_weight + progress * (self.end_weight - self.start_weight)
        
        elif self.anneal_type == "cosine":
            progress = min(self.current_step / self.n_steps, 1.0)
            # Cosine schedule: smooth increase
            cosine_progress = 0.5 * (1 - math.cos(math.pi * progress))
            self.current_weight = self.start_weight + cosine_progress * (self.end_weight - self.start_weight)
        
        elif self.anneal_type == "cyclical":
            # Cyclical annealing (for multiple cycles)
            cycle_length = self.n_steps // 4  # 4 cycles
            step_in_cycle = self.current_step % cycle_length
            progress = min(step_in_cycle / cycle_length, 1.0)
            self.current_weight = self.start_weight + progress * (self.end_weight - self.start_weight)
        
        else:
            raise ValueError(f"Unknown anneal_type: {self.anneal_type}")
        
        self.current_step += 1
        return self.current_weight
    
    def apply_free_bits(
        self,
        kl_divergence: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply free-bits mechanism to KL divergence.
        
        Free-bits: only penalize KL above a minimum threshold per dimension.
        This prevents posterior collapse while still regularizing.
        
        Args:
            kl_divergence: (B, D) or (B,) KL divergence
            
        Returns:
            KL with free-bits applied
        """
        if self.free_bits <= 0:
            return kl_divergence
        
        if self.free_bits_aggregate == "dimension":
            # Per-dimension free-bits
            # KL shape: (B, D) or reduce to (D,)
            if kl_divergence.ndim == 1:
                # Already reduced, assume per-sample
                kl_clamped = torch.clamp(kl_divergence - self.free_bits, min=0.0)
            else:
                # Per-dimension: clamp each dimension separately
                kl_per_dim = kl_divergence.mean(dim=0)  # (D,)
                kl_clamped_per_dim = torch.clamp(kl_per_dim - self.free_bits, min=0.0)
                # Scale back to match original shape
                kl_clamped = kl_clamped_per_dim.sum()
        
        elif self.free_bits_aggregate == "total":
            # Total free-bits: only penalize if total KL exceeds threshold
            total_kl = kl_divergence.sum()
            total_free_bits = self.free_bits * kl_divergence.numel()
            kl_clamped = torch.clamp(total_kl - total_free_bits, min=0.0)
        
        else:
            raise ValueError(f"Unknown free_bits_aggregate: {self.free_bits_aggregate}")
        
        return kl_clamped
    
    def get_weight(self) -> float:
        """Get current KL weight without stepping."""
        return self.current_weight
    
    def reset(self):
        """Reset scheduler to initial state."""
        self.current_step = 0
        self.current_weight = self.start_weight
        logger.info("KLScheduler reset")


class KLDivergence(nn.Module):
    """
    Compute KL divergence between q(z|x) and p(z) for diagonal Gaussian.
    
    Assumes:
    - q(z|x) = N(mu, diag(exp(logvar)))
    - p(z) = N(0, I)
    
    KL = 0.5 * sum(exp(logvar) + mu^2 - 1 - logvar)
    
    Args:
        reduction: "mean", "sum", or "none"
    """
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence.
        
        Args:
            mu: (B, D) mean
            logvar: (B, D) log-variance
            
        Returns:
            KL divergence (scalar if reduction='mean'/'sum', (B,D) if 'none')
        """
        kl = 0.5 * (torch.exp(logvar) + mu ** 2 - 1.0 - logvar)
        
        if self.reduction == "mean":
            return kl.mean()
        elif self.reduction == "sum":
            return kl.sum()
        elif self.reduction == "none":
            return kl
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class AdaptiveKLScheduler(KLScheduler):
    """
    Adaptive KL scheduler that adjusts weight based on KL value.
    
    Maintains a target KL value and adjusts weight to stay near target.
    Useful for preventing KL collapse or explosion.
    
    Args:
        target_kl: Target KL divergence value
        adjust_rate: Rate of weight adjustment
        min_weight: Minimum KL weight
        max_weight: Maximum KL weight
    """
    
    def __init__(
        self,
        target_kl: float = 0.5,
        adjust_rate: float = 0.01,
        min_weight: float = 1e-5,
        max_weight: float = 1.0,
        free_bits: float = 0.0,
    ):
        super().__init__(
            anneal_type="none",
            start_weight=min_weight,
            end_weight=max_weight,
            free_bits=free_bits,
        )
        
        self.target_kl = target_kl
        self.adjust_rate = adjust_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        logger.info(
            f"Initialized AdaptiveKLScheduler: target_kl={target_kl:.4f}, "
            f"weight_range=[{min_weight:.4f}, {max_weight:.4f}]"
        )
    
    def step(self, current_kl: Optional[float] = None) -> float:
        """
        Update weight based on current KL value.
        
        Args:
            current_kl: Current KL divergence value
            
        Returns:
            Updated KL weight
        """
        if current_kl is not None:
            # Adjust weight: increase if KL too low, decrease if too high
            kl_error = self.target_kl - current_kl
            adjustment = self.adjust_rate * kl_error
            
            self.current_weight = torch.clamp(
                torch.tensor(self.current_weight + adjustment),
                min=self.min_weight,
                max=self.max_weight,
            ).item()
        
        self.current_step += 1
        return self.current_weight


def compute_kl_divergence(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute KL divergence between N(mu, exp(logvar)) and N(0, I).
    
    Args:
        mu: (B, D) mean
        logvar: (B, D) log-variance
        reduction: "mean", "sum", or "none"
        
    Returns:
        KL divergence
    """
    kl = 0.5 * (torch.exp(logvar) + mu ** 2 - 1.0 - logvar)
    
    if reduction == "mean":
        return kl.mean()
    elif reduction == "sum":
        return kl.sum()
    elif reduction == "none":
        return kl
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def create_kl_scheduler(config: dict) -> KLScheduler:
    """
    Factory function to create KL scheduler from config.
    
    Args:
        config: Dictionary with scheduler configuration:
            - enabled: bool
            - anneal_type: str
            - start_weight: float
            - end_weight: float
            - n_steps: int
            - free_bits: float
            - free_bits_aggregate: str
            - adaptive: bool (if True, use AdaptiveKLScheduler)
            - target_kl: float (for adaptive)
            
    Returns:
        KLScheduler instance
    """
    if not config.get("enabled", False):
        logger.info("KL scheduling disabled, using fixed weight")
        return KLScheduler(
            anneal_type="none",
            end_weight=config.get("end_weight", 0.0),
        )
    
    if config.get("adaptive", False):
        return AdaptiveKLScheduler(
            target_kl=config.get("target_kl", 0.5),
            adjust_rate=config.get("adjust_rate", 0.01),
            min_weight=config.get("start_weight", 1e-5),
            max_weight=config.get("end_weight", 1.0),
            free_bits=config.get("free_bits", 0.0),
        )
    else:
        return KLScheduler(
            anneal_type=config.get("anneal_type", "linear"),
            start_weight=config.get("start_weight", 0.0),
            end_weight=config.get("end_weight", 1.0),
            n_steps=config.get("n_steps", 10000),
            free_bits=config.get("free_bits", 0.0),
            free_bits_aggregate=config.get("free_bits_aggregate", "dimension"),
        )
