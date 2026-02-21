"""
Training utilities for fMRI-to-CLIP encoders.
"""

from .base import BaseTrainer, TrainerConfig
from .losses import (
    mse_loss,
    cosine_loss,
    info_nce_loss,
    MultiLoss,
    compute_multiloss,
    compose_loss,
    ComposedLoss,
)

__all__ = [
    "BaseTrainer",
    "TrainerConfig",
    "mse_loss",
    "cosine_loss",
    "info_nce_loss",
    "MultiLoss",
    "compute_multiloss",
    "compose_loss",
    "ComposedLoss",
]

