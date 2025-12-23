"""
fMRI-to-Image Models
===================

Neural and linear models for mapping fMRI activity to image representations.
"""

from .ridge import RidgeEncoder
from .mlp import MLPEncoder, save_mlp, load_mlp
from .clip_adapter import CLIPAdapter, save_adapter, load_adapter
from .encoders import (
    ResidualBlock,
    ResidualMLPEncoder,
    CLIPMappingHead,
    TwoStageEncoder,
    MultiLayerTwoStageEncoder,
    SelfSupervisedPretrainer,
    save_two_stage_encoder,
    load_two_stage_encoder
)
from .multi_target_decoder import (
    IPAdapterTokenHead,
    SDLatentHead,
    MultiTargetDecoder,
    MultiTaskLoss,
    save_multi_target_decoder,
    load_multi_target_decoder
)
from .encoding_model import (
    ImageEncoder,
    EncodingModel,
    save_encoding_model,
    load_encoding_model
)
from .losses import (
    cosine_loss,
    mse_loss,
    infonce_loss,
    compose_loss,
    ComposedLoss
)

__all__ = [
    # Baseline models
    "RidgeEncoder",
    "MLPEncoder",
    "save_mlp",
    "load_mlp",
    "CLIPAdapter",
    "save_adapter",
    "load_adapter",
    # Two-stage encoder
    "ResidualBlock",
    "ResidualMLPEncoder",
    "CLIPMappingHead",
    "TwoStageEncoder",
    "MultiLayerTwoStageEncoder",
    "SelfSupervisedPretrainer",
    "save_two_stage_encoder",
    "load_two_stage_encoder",
    # Multi-target decoder (novel)
    "IPAdapterTokenHead",
    "SDLatentHead",
    "MultiTargetDecoder",
    "MultiTaskLoss",
    "save_multi_target_decoder",
    "load_multi_target_decoder",
    # Encoding model (for BOI-lite)
    "ImageEncoder",
    "EncodingModel",
    "save_encoding_model",
    "load_encoding_model",
    # Loss functions (novel)
    "cosine_loss",
    "mse_loss",
    "infonce_loss",
    "compose_loss",
    "ComposedLoss"
]

