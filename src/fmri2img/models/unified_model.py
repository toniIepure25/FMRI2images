"""
Unified Model Factory for Research-Grade Experiments
===================================================

Creates models based on experiment configuration:
- Deterministic models (single output)
- Gaussian models (mu + logvar outputs)
- Von Mises-Fisher models (mu_normalized + log_kappa outputs)

Supports modular architecture with encoder + decoder.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Literal, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """Pre-norm residual block: LayerNorm -> Linear -> Act -> Dropout."""

    def __init__(self, dim: int, activation: str = "gelu", dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
        self.act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.act(self.linear(self.norm(x))))


class MLPEncoder(nn.Module):
    """
    Multi-layer perceptron encoder with residual connections.

    Each transition to a new hidden dimension uses a projection block
    (Linear -> LayerNorm -> Act -> Dropout).  When the dimension stays
    the same between consecutive layers, a pre-norm residual block is
    used instead.  An additional residual block is appended after the
    last projection to deepen the encoder without extra hyperparameters.

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
        activation: str = "gelu",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1]

        blocks: list[nn.Module] = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            if in_dim == hidden_dim:
                blocks.append(ResidualBlock(hidden_dim, activation, dropout))
            else:
                blocks.append(nn.Linear(in_dim, hidden_dim))
                blocks.append(nn.LayerNorm(hidden_dim))
                blocks.append(nn.GELU() if activation == "gelu" else nn.ReLU())
                blocks.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        # Extra residual block at final dimension for depth
        blocks.append(ResidualBlock(hidden_dims[-1], activation, dropout))

        self.encoder = nn.Sequential(*blocks)

        logger.info(
            f"MLPEncoder: {input_dim} -> {hidden_dims} "
            f"(activation={activation}, dropout={dropout}, residual=True)"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class VonMisesFisherDecoderLegacy(nn.Module):
    """
    Legacy vMF decoder using clamped log-kappa parameterisation.

    Kept for backward compatibility with existing checkpoints.  New code
    should use ``vmf_decoder.VonMisesFisherDecoder`` (bounded sigmoid).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[list[int]] = None,
        activation: str = "gelu",
        dropout: float = 0.1,
        log_kappa_min: float = -2.0,
        log_kappa_max: float = 8.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.log_kappa_min = log_kappa_min
        self.log_kappa_max = log_kappa_max

        if hidden_dims is None or len(hidden_dims) == 0:
            self.shared_backbone = nn.Identity()
            backbone_out_dim = input_dim
        else:
            layers: list[nn.Module] = []
            in_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU() if activation == "gelu" else nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                in_dim = hidden_dim
            self.shared_backbone = nn.Sequential(*layers)
            backbone_out_dim = hidden_dims[-1]

        self.mu_head = nn.Linear(backbone_out_dim, output_dim)
        self.kappa_head = nn.Linear(backbone_out_dim, 1)

        logger.info(
            f"VonMisesFisherDecoderLegacy: {input_dim} -> (mu, log_kappa) {output_dim} "
            f"(hidden={hidden_dims}, log_kappa=[{log_kappa_min}, {log_kappa_max}])"
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mu: (B, output_dim) L2-normalised mean direction
            log_kappa: (B, 1) clamped log-concentration
        """
        features = self.shared_backbone(h)
        mu = nn.functional.normalize(self.mu_head(features), p=2, dim=-1)
        log_kappa = torch.clamp(
            self.kappa_head(features),
            min=self.log_kappa_min,
            max=self.log_kappa_max,
        )
        return mu, log_kappa


class UnifiedModel(nn.Module):
    """
    Unified model supporting deterministic, Gaussian, vMF, and vMF-DCF outputs.

    Architecture:
        fMRI -> Encoder -> Latent -> Decoder -> {Prediction | (mu, logvar) | (mu, kappa)}

    Supported model types:
        - "deterministic": single L2-normalised prediction
        - "gaussian": mu + per-dim logvar (diagonal Gaussian)
        - "vmf": mu + scalar log_kappa (von Mises-Fisher on S^{d-1})
        - "vmf_dcf": per-ROI vMF predictions with spherical consensus fusion
                     (requires roi_transformer encoder)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.model_type = config.get("type", "deterministic")

        encoder_cfg = config.get("encoder", {})
        decoder_cfg = config.get("decoder", {})

        input_dim = encoder_cfg.get("input_dim")
        if input_dim is None:
            raise ValueError("encoder.input_dim must be specified")

        encoder_type = encoder_cfg.get("encoder_type", "mlp")

        if encoder_type == "roi_transformer":
            from fmri2img.models.roi_transformer import ROITransformerEncoder
            self.encoder = ROITransformerEncoder(
                roi_dims=encoder_cfg["roi_dims"],
                d_model=encoder_cfg.get("d_model", 512),
                nhead=encoder_cfg.get("nhead", 8),
                num_layers=encoder_cfg.get("num_layers", 4),
                dropout=encoder_cfg.get("dropout", 0.1),
                activation=encoder_cfg.get("activation", "gelu"),
            )
        else:
            self.encoder = MLPEncoder(
                input_dim=input_dim,
                hidden_dims=encoder_cfg.get("hidden_dims", [4096, 2048, 1024]),
                activation=encoder_cfg.get("activation", "gelu"),
                dropout=encoder_cfg.get("dropout", 0.1),
            )

        latent_dim = self.encoder.output_dim
        output_dim = decoder_cfg.get("output_dim", 768)

        if self.model_type == "deterministic":
            self.decoder = DeterministicDecoder(
                input_dim=latent_dim,
                output_dim=output_dim,
                hidden_dims=decoder_cfg.get("hidden_dims", [1536]),
                activation=decoder_cfg.get("activation", "gelu"),
                dropout=decoder_cfg.get("dropout", 0.1),
            )
        elif self.model_type == "gaussian":
            self.decoder = GaussianDecoder(
                input_dim=latent_dim,
                output_dim=output_dim,
                hidden_dims=decoder_cfg.get("hidden_dims", [1536]),
                activation=decoder_cfg.get("activation", "gelu"),
                dropout=decoder_cfg.get("dropout", 0.1),
                logvar_min=decoder_cfg.get("logvar_min", -10.0),
                logvar_max=decoder_cfg.get("logvar_max", 5.0),
            )
        elif self.model_type == "vmf":
            use_legacy = "log_kappa_min" in decoder_cfg or "log_kappa_max" in decoder_cfg
            use_new = "kappa_min" in decoder_cfg or "kappa_max" in decoder_cfg
            posterior = config.get("posterior", "")

            if use_new or posterior == "vmf":
                from fmri2img.models.vmf_decoder import VonMisesFisherDecoder as VmfDec
                self.decoder = VmfDec(
                    input_dim=latent_dim,
                    output_dim=output_dim,
                    hidden_dims=decoder_cfg.get("hidden_dims", [1536]),
                    activation=decoder_cfg.get("activation", "gelu"),
                    dropout=decoder_cfg.get("dropout", 0.1),
                    kappa_min=float(decoder_cfg.get("kappa_min", 1e-3)),
                    kappa_max=float(decoder_cfg.get("kappa_max", 500.0)),
                )
                self._vmf_output_is_log = False
            else:
                self.decoder = VonMisesFisherDecoderLegacy(
                    input_dim=latent_dim,
                    output_dim=output_dim,
                    hidden_dims=decoder_cfg.get("hidden_dims", [1536]),
                    activation=decoder_cfg.get("activation", "gelu"),
                    dropout=decoder_cfg.get("dropout", 0.1),
                    log_kappa_min=decoder_cfg.get("log_kappa_min", -2.0),
                    log_kappa_max=decoder_cfg.get("log_kappa_max", 8.0),
                )
                self._vmf_output_is_log = True
        elif self.model_type == "vmf_dcf":
            if encoder_type != "roi_transformer":
                raise ValueError(
                    "vmf_dcf requires encoder_type='roi_transformer'; "
                    f"got '{encoder_type}'"
                )
            from fmri2img.models.roi_dcf import ROIDCFDecoder
            n_rois = len(encoder_cfg["roi_dims"])
            dcf_cfg = decoder_cfg.get("dcf", {})
            self.decoder = ROIDCFDecoder(
                d_model=encoder_cfg.get("d_model", 512),
                output_dim=output_dim,
                n_rois=n_rois,
                shared=dcf_cfg.get("shared_heads", True),
                hidden_dim=dcf_cfg.get("hidden_dim"),
                kappa_min=float(decoder_cfg.get("kappa_min", 1e-3)),
                kappa_max=float(decoder_cfg.get("kappa_max", 500.0)),
                dropout=decoder_cfg.get("dropout", 0.1),
            )
            self._vmf_output_is_log = False
            self._dcf_return_per_roi = dcf_cfg.get("return_per_roi", True)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        logger.info(f"UnifiedModel created: type={self.model_type}")

    def forward(
        self, x: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            deterministic  -> (pred, None)
            gaussian       -> (mu, logvar)           logvar shape (B, D)
            vmf            -> (mu, kappa_or_log)     shape (B, 1)
            vmf_dcf        -> (mu_fused, kappa_consensus)  shapes (B, D), (B, 1)
                              When return_per_roi is True in config, a dict is
                              stored in self._last_dcf_extras with keys:
                              per_roi_mus, per_roi_kappas, delta, cls_to_roi_alpha.
        """
        if self.model_type == "vmf_dcf":
            return self._forward_dcf(x)

        h = self.encoder(x)

        if self.model_type == "deterministic":
            return self.decoder(h), None
        elif self.model_type == "gaussian":
            return self.decoder(h, **kwargs)
        else:  # vmf
            return self.decoder(h)

    def _forward_dcf(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for vmf_dcf: encoder with ROI output -> DCF decoder."""
        enc_out = self.encoder(x, return_roi_tokens=True)

        if self._dcf_return_per_roi:
            mu_fused, kappa_consensus, per_roi_mus, per_roi_kappas, delta = (
                self.decoder(
                    enc_out.roi_tokens,
                    enc_out.cls_to_roi_alpha,
                    return_per_roi=True,
                )
            )
            self._last_dcf_extras = {
                "per_roi_mus": per_roi_mus,
                "per_roi_kappas": per_roi_kappas,
                "delta": delta,
                "cls_to_roi_alpha": enc_out.cls_to_roi_alpha,
            }
        else:
            mu_fused, kappa_consensus = self.decoder(
                enc_out.roi_tokens, enc_out.cls_to_roi_alpha
            )
            self._last_dcf_extras = {}

        return mu_fused, kappa_consensus

    @property
    def vmf_output_is_log(self) -> bool:
        """True when the vMF decoder returns log_kappa (legacy), False for direct kappa."""
        return getattr(self, "_vmf_output_is_log", True)

    def get_config(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {
            "type": self.model_type,
            "encoder": {
                "input_dim": self.encoder.input_dim,
                "output_dim": self.encoder.output_dim,
            },
            "decoder": {
                "input_dim": self.decoder.input_dim,
                "output_dim": self.decoder.output_dim,
            },
        }
        if self.model_type == "vmf":
            cfg["vmf_output_is_log"] = self.vmf_output_is_log
        return cfg


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
