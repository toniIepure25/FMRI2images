"""
Training utilities for fMRI → CLIP encoders
"""

from .losses import (
    mse_loss,
    cosine_loss,
    info_nce_loss,
    MultiLoss,
    compute_multiloss,
    compose_loss  # Backward compatibility
)

__all__ = [
    "mse_loss",
    "cosine_loss",
    "info_nce_loss",
    "MultiLoss",
    "compute_multiloss",
    "compose_loss"
]
