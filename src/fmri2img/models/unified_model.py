"""
Unified Model Factory for Research-Grade Experiments
===================================================

Creates models based on experiment configuration:
- Deterministic models (single output)
- Gaussian models (mu + logvar outputs)

Supports modular architecture with encoder + decoder.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Literal, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MLPEncoder(nn.Module):
    """
    Multi-layer perceptron encoder with residual connections.
    
    Args:
        input_dim: Input dimensionality (fMRI voxels)
        hidden_dims: List of hidden layer dimensions
        activation: Activation function ("relu", "gelu")
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activation: str = "relu",
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1]
        
        # Build layers
        layers = []
        in_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU() if activation == "gelu" else nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        logger.info(f"MLPEncoder: {input_dim} → {hidden_dims} (activation={activation}, dropout={dropout})")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (B, input_dim)
        Returns:
            h: Latent (B, hidden_dims[-1])
        """
        return self.encoder(x)


class DeterministicDecoder(nn.Module):
    """
    Deterministic decoder: outputs single prediction.
    
    Args:
        input_dim: Latent dimension from encoder
        output_dim: Output dimension (CLIP embedding size, typically 768)
        hidden_dims: Hidden layer dimensions
        activation: Activation function
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[list[int]] = None,
        activation: str = "relu",
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if hidden_dims is None or len(hidden_dims) == 0:
            # Simple linear projection
            self.decoder = nn.Linear(input_dim, output_dim)
        else:
            # MLP decoder
            layers = []
            in_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU() if activation == "gelu" else nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                in_dim = hidden_dim
            
            # Final projection
            layers.append(nn.Linear(in_dim, output_dim))
            self.decoder = nn.Sequential(*layers)
        
        logger.info(f"DeterministicDecoder: {input_dim} → {output_dim} (hidden={hidden_dims})")
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Latent (B, input_dim)
        Returns:
            pred: Predictions (B, output_dim), L2-normalized
        """
        pred = self.decoder(h)
        # L2 normalize (CLIP embeddings are normalized)
        pred = nn.functional.normalize(pred, p=2, dim=-1)
        return pred


class GaussianDecoder(nn.Module):
    """
    Gaussian decoder: outputs mean (mu) and log-variance (logvar).
    
    Architecture:
        Shared layers → split into mu_head and logvar_head
    
    Args:
        input_dim: Latent dimension from encoder
        output_dim: Output dimension (CLIP embedding size)
        hidden_dims: Hidden layer dimensions for shared backbone
        activation: Activation function
        dropout: Dropout probability
        logvar_min: Minimum log-variance (for clamping)
        logvar_max: Maximum log-variance (for clamping)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[list[int]] = None,
        activation: str = "relu",
        dropout: float = 0.1,
        logvar_min: float = -10.0,
        logvar_max: float = 5.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        
        # Shared backbone
        if hidden_dims is None or len(hidden_dims) == 0:
            # No hidden layers - direct projection
            self.shared_backbone = nn.Identity()
            backbone_out_dim = input_dim
        else:
            layers = []
            in_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU() if activation == "gelu" else nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                in_dim = hidden_dim
            
            self.shared_backbone = nn.Sequential(*layers)
            backbone_out_dim = hidden_dims[-1]
        
        # Prediction heads
        self.mu_head = nn.Linear(backbone_out_dim, output_dim)
        self.logvar_head = nn.Linear(backbone_out_dim, output_dim)
        
        logger.info(f"GaussianDecoder: {input_dim} → (mu, logvar) {output_dim} (hidden={hidden_dims})")
        logger.info(f"  Logvar clamping: [{logvar_min}, {logvar_max}]")
    
    def forward(
        self,
        h: torch.Tensor,
        return_normalized: bool = True,
        clamp_logvar: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: Latent (B, input_dim)
            return_normalized: Whether to L2-normalize mu
            clamp_logvar: Whether to clamp logvar
        
        Returns:
            mu: Mean predictions (B, output_dim), L2-normalized if return_normalized=True
            logvar: Log-variance (B, output_dim), clamped if clamp_logvar=True
        """
        # Shared features
        features = self.shared_backbone(h)
        
        # Predict mu and logvar
        mu = self.mu_head(features)
        logvar = self.logvar_head(features)
        
        # Normalize mu (CLIP embeddings are normalized)
        if return_normalized:
            mu = nn.functional.normalize(mu, p=2, dim=-1)
        
        # Clamp logvar for numerical stability
        if clamp_logvar:
            logvar = torch.clamp(logvar, min=self.logvar_min, max=self.logvar_max)
        
        return mu, logvar


class UnifiedModel(nn.Module):
    """
    Unified model supporting both deterministic and Gaussian outputs.
    
    Architecture:
        fMRI → Encoder → Latent → Decoder → {Prediction | (mu, logvar)}
    
    Args:
        config: Model configuration dict with keys:
            - type: "deterministic" or "gaussian"
            - encoder: encoder config (input_dim, hidden_dims, activation, dropout)
            - decoder: decoder config (input_dim, output_dim, hidden_dims, activation, dropout)
            - (for Gaussian) logvar_min, logvar_max
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.model_type = config.get("type", "deterministic")
        
        encoder_cfg = config.get("encoder", {})
        decoder_cfg = config.get("decoder", {})
        
        # Encoder
        input_dim = encoder_cfg.get("input_dim")
        if input_dim is None:
            raise ValueError("encoder.input_dim must be specified")
        
        self.encoder = MLPEncoder(
            input_dim=input_dim,
            hidden_dims=encoder_cfg.get("hidden_dims", [4096, 2048, 1024]),
            activation=encoder_cfg.get("activation", "relu"),
            dropout=encoder_cfg.get("dropout", 0.1)
        )
        
        # Decoder
        latent_dim = self.encoder.output_dim
        output_dim = decoder_cfg.get("output_dim", 768)
        
        if self.model_type == "deterministic":
            self.decoder = DeterministicDecoder(
                input_dim=latent_dim,
                output_dim=output_dim,
                hidden_dims=decoder_cfg.get("hidden_dims", [1536]),
                activation=decoder_cfg.get("activation", "relu"),
                dropout=decoder_cfg.get("dropout", 0.1)
            )
        elif self.model_type == "gaussian":
            self.decoder = GaussianDecoder(
                input_dim=latent_dim,
                output_dim=output_dim,
                hidden_dims=decoder_cfg.get("hidden_dims", [1536]),
                activation=decoder_cfg.get("activation", "relu"),
                dropout=decoder_cfg.get("dropout", 0.1),
                logvar_min=decoder_cfg.get("logvar_min", -10.0),
                logvar_max=decoder_cfg.get("logvar_max", 5.0)
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        logger.info(f"UnifiedModel created: type={self.model_type}")
    
    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input fMRI (B, input_dim)
            **kwargs: Additional arguments (e.g., return_normalized, clamp_logvar for Gaussian)
        
        Returns:
            For deterministic:
                pred: (B, output_dim), logvar=None
            For gaussian:
                mu: (B, output_dim), logvar: (B, output_dim)
        """
        # Encode
        h = self.encoder(x)
        
        # Decode
        if self.model_type == "deterministic":
            pred = self.decoder(h)
            return pred, None
        else:  # gaussian
            mu, logvar = self.decoder(h, **kwargs)
            return mu, logvar
    
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration."""
        return {
            "type": self.model_type,
            "encoder": {
                "input_dim": self.encoder.input_dim,
                "output_dim": self.encoder.output_dim
            },
            "decoder": {
                "input_dim": self.decoder.input_dim,
                "output_dim": self.decoder.output_dim
            }
        }


def create_model(config: Dict[str, Any]) -> UnifiedModel:
    """
    Factory function to create model from config.
    
    Args:
        config: Model configuration dict
    
    Returns:
        model: UnifiedModel instance
    
    Example:
        >>> config = {
        ...     "type": "gaussian",
        ...     "encoder": {"input_dim": 15724, "hidden_dims": [4096, 2048, 1024]},
        ...     "decoder": {"output_dim": 768, "hidden_dims": [1536]}
        ... }
        >>> model = create_model(config)
    """
    return UnifiedModel(config)


def load_model(checkpoint_path: Path, device: str = "cpu") -> UnifiedModel:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        model: Loaded UnifiedModel
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config
    if "config" in checkpoint:
        config = checkpoint["config"]
    elif "model_config" in checkpoint:
        config = checkpoint["model_config"]
    else:
        raise KeyError("No config found in checkpoint")
    
    # Create model
    model = create_model(config)
    
    # Load state dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        raise KeyError("No state_dict found in checkpoint")
    
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path} (type={model.model_type})")
    return model
