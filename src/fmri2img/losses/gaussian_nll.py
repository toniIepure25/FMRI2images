"""
Gaussian Negative Log-Likelihood Loss for Heteroscedastic Regression

Implements proper probabilistic training for embedding prediction with uncertainty.
Predicts both mean (mu) and log-variance (logvar) with numerical stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class GaussianNLLLoss(nn.Module):
    """
    Gaussian Negative Log-Likelihood loss for diagonal covariance.
    
    Loss = 0.5 * sum(logvar + (err^2) / exp(logvar))
    
    Features:
    - Variance clamping for numerical stability
    - Optional learnable global variance scale for calibration
    - Per-dimension or reduced loss
    - Proper gradient flow through variance
    
    Args:
        logvar_min: Minimum log-variance (prevents variance collapse)
        logvar_max: Maximum log-variance (prevents numerical overflow)
        learnable_global_scale: If True, add learnable global variance multiplier
        reduction: 'mean', 'sum', or 'none'
        
    Usage:
        loss_fn = GaussianNLLLoss(logvar_min=-10, logvar_max=5)
        
        # Model outputs: mu (B, D), logvar (B, D)
        loss = loss_fn(mu, logvar, target)
    """
    
    def __init__(
        self,
        logvar_min: float = -10.0,
        logvar_max: float = 5.0,
        learnable_global_scale: bool = False,
        reduction: str = "mean",
    ):
        super().__init__()
        
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.reduction = reduction
        
        if learnable_global_scale:
            # Learnable log-scale (initialized to 0, so exp(0) = 1)
            self.global_log_scale = nn.Parameter(torch.zeros(1))
            logger.info("Using learnable global variance scale")
        else:
            self.register_buffer("global_log_scale", torch.zeros(1))
    
    def forward(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Gaussian NLL.
        
        Args:
            mu: (B, D) predicted mean
            logvar: (B, D) predicted log-variance
            target: (B, D) ground truth
            
        Returns:
            Scalar loss (if reduction='mean' or 'sum') or (B, D) loss (if reduction='none')
        """
        # Clamp log-variance for stability
        logvar_clamped = torch.clamp(logvar, min=self.logvar_min, max=self.logvar_max)
        
        # Apply global scale
        logvar_scaled = logvar_clamped + 2 * self.global_log_scale  # log(var * scale^2) = logvar + 2*log(scale)
        
        # Compute error
        err = target - mu
        
        # Gaussian NLL: 0.5 * (logvar + err^2 / exp(logvar))
        # Note: constant term 0.5 * log(2*pi) omitted as it doesn't affect gradients
        nll = 0.5 * (logvar_scaled + (err ** 2) / torch.exp(logvar_scaled))
        
        if self.reduction == "mean":
            return nll.mean()
        elif self.reduction == "sum":
            return nll.sum()
        elif self.reduction == "none":
            return nll
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
    
    def get_global_scale(self) -> float:
        """Get current global variance scale."""
        return torch.exp(self.global_log_scale).item()


class GaussianNLLWithPrior(nn.Module):
    """
    Gaussian NLL with optional prior regularization on variance.
    
    Adds a penalty for deviating from a target variance, useful for
    encouraging well-calibrated uncertainty estimates.
    
    Args:
        logvar_min: Minimum log-variance
        logvar_max: Maximum log-variance
        prior_logvar: Target log-variance for regularization
        prior_weight: Weight of prior regularization term
        reduction: Loss reduction method
    """
    
    def __init__(
        self,
        logvar_min: float = -10.0,
        logvar_max: float = 5.0,
        prior_logvar: float = 0.0,
        prior_weight: float = 0.01,
        reduction: str = "mean",
    ):
        super().__init__()
        
        self.nll_loss = GaussianNLLLoss(
            logvar_min=logvar_min,
            logvar_max=logvar_max,
            reduction=reduction,
        )
        
        self.register_buffer("prior_logvar", torch.tensor(prior_logvar))
        self.prior_weight = prior_weight
    
    def forward(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute NLL with prior regularization.
        
        Returns:
            total_loss, loss_dict
        """
        # Main NLL loss
        nll = self.nll_loss(mu, logvar, target)
        
        # Prior regularization (KL to prior variance)
        logvar_clamped = torch.clamp(logvar, min=self.nll_loss.logvar_min, max=self.nll_loss.logvar_max)
        prior_term = 0.5 * ((logvar_clamped - self.prior_logvar) ** 2).mean()
        
        total_loss = nll + self.prior_weight * prior_term
        
        loss_dict = {
            "nll": nll.item(),
            "prior_reg": prior_term.item(),
            "total": total_loss.item(),
        }
        
        return total_loss, loss_dict


class RobustGaussianNLLLoss(nn.Module):
    """
    Robust Gaussian NLL with outlier handling.
    
    Uses a mixture of Gaussian and uniform distribution to handle outliers,
    improving robustness to mislabeled or difficult samples.
    
    Args:
        logvar_min: Minimum log-variance
        logvar_max: Maximum log-variance
        outlier_prob: Probability of outlier (uniform) component
        reduction: Loss reduction method
    """
    
    def __init__(
        self,
        logvar_min: float = -10.0,
        logvar_max: float = 5.0,
        outlier_prob: float = 0.01,
        reduction: str = "mean",
    ):
        super().__init__()
        
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.outlier_prob = outlier_prob
        self.reduction = reduction
        
        logger.info(f"Using robust NLL with outlier_prob={outlier_prob}")
    
    def forward(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute robust Gaussian NLL."""
        logvar_clamped = torch.clamp(logvar, min=self.logvar_min, max=self.logvar_max)
        
        err = target - mu
        
        # Gaussian log-likelihood
        log_prob_gaussian = -0.5 * (logvar_clamped + (err ** 2) / torch.exp(logvar_clamped))
        
        # Uniform log-likelihood (very small constant)
        log_prob_uniform = torch.tensor(-10.0, device=mu.device)
        
        # Mixture: log(p*gaussian + (1-p)*uniform)
        log_prob_mix = torch.logsumexp(
            torch.stack([
                torch.log(torch.tensor(1 - self.outlier_prob)) + log_prob_gaussian,
                torch.log(torch.tensor(self.outlier_prob)) + log_prob_uniform,
            ], dim=0),
            dim=0,
        )
        
        # Negative log-likelihood
        nll = -log_prob_mix
        
        if self.reduction == "mean":
            return nll.mean()
        elif self.reduction == "sum":
            return nll.sum()
        elif self.reduction == "none":
            return nll
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


def compute_nll_per_dim(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Compute NLL per dimension (for evaluation).
    
    Args:
        mu: (N, D) predicted mean
        logvar: (N, D) predicted log-variance
        target: (N, D) ground truth
        
    Returns:
        (D,) per-dimension NLL
    """
    err = target - mu
    nll = 0.5 * (logvar + (err ** 2) / torch.exp(logvar))
    return nll.mean(dim=0)  # Average over samples, per dimension


def compute_delta_nll(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    baseline_variance: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute ΔNLL vs baseline (for evaluation).
    
    If baseline_variance is None, uses empirical variance of targets.
    
    Args:
        mu: (N, D) predicted mean
        logvar: (N, D) predicted log-variance
        target: (N, D) ground truth
        baseline_variance: Optional (D,) baseline variance
        
    Returns:
        Scalar ΔNLL
    """
    # Model NLL
    err = target - mu
    model_nll = 0.5 * (logvar + (err ** 2) / torch.exp(logvar))
    model_nll = model_nll.mean()
    
    # Baseline NLL (constant prediction with empirical variance)
    if baseline_variance is None:
        baseline_variance = target.var(dim=0, unbiased=True)
    
    baseline_logvar = torch.log(baseline_variance + 1e-8)
    baseline_err = target - target.mean(dim=0, keepdim=True)
    baseline_nll = 0.5 * (baseline_logvar + (baseline_err ** 2) / torch.exp(baseline_logvar))
    baseline_nll = baseline_nll.mean()
    
    return model_nll - baseline_nll


def create_gaussian_nll_loss(config: dict) -> nn.Module:
    """
    Factory function to create Gaussian NLL loss from config.
    
    Args:
        config: Dictionary with loss configuration:
            - type: "gaussian_nll", "gaussian_nll_prior", or "robust_gaussian_nll"
            - logvar_min: float
            - logvar_max: float
            - learnable_global_scale: bool
            - prior_logvar: float (for prior variant)
            - prior_weight: float (for prior variant)
            - outlier_prob: float (for robust variant)
            - reduction: str
            
    Returns:
        Loss module
    """
    loss_type = config.get("type", "gaussian_nll")
    
    if loss_type == "gaussian_nll":
        return GaussianNLLLoss(
            logvar_min=config.get("logvar_min", -10.0),
            logvar_max=config.get("logvar_max", 5.0),
            learnable_global_scale=config.get("learnable_global_scale", False),
            reduction=config.get("reduction", "mean"),
        )
    elif loss_type == "gaussian_nll_prior":
        return GaussianNLLWithPrior(
            logvar_min=config.get("logvar_min", -10.0),
            logvar_max=config.get("logvar_max", 5.0),
            prior_logvar=config.get("prior_logvar", 0.0),
            prior_weight=config.get("prior_weight", 0.01),
            reduction=config.get("reduction", "mean"),
        )
    elif loss_type == "robust_gaussian_nll":
        return RobustGaussianNLLLoss(
            logvar_min=config.get("logvar_min", -10.0),
            logvar_max=config.get("logvar_max", 5.0),
            outlier_prob=config.get("outlier_prob", 0.01),
            reduction=config.get("reduction", "mean"),
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
