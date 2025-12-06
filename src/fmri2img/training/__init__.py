"""
Training utilities for fMRI → CLIP encoders
"""

# Base infrastructure
from .base import BaseTrainer, TrainerConfig

# Loss functions
from .losses import (
    mse_loss,
    cosine_loss,
    info_nce_loss,
    MultiLoss,
    compute_multiloss,
    compose_loss  # Backward compatibility
)

__all__ = [
    # Base infrastructure
    "BaseTrainer",
    "TrainerConfig",
    # Loss functions
    "mse_loss",
    "cosine_loss",
    "info_nce_loss",
    "MultiLoss",
    "compute_multiloss",
    "compose_loss"
]

