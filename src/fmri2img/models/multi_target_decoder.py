"""
Multi-Target Decoder: Novel Contribution Beyond Existing Papers
===============================================================

Extends the encoder to predict multiple conditioning signals simultaneously:
1. Global CLIP embedding (512-D): Semantic content
2. IP-Adapter token embeddings (N tokens × 1024-D): Fine-grained visual details
3. Optional SD VAE latent (4 × 64 × 64): Structural guidance

Scientific Rationale:
- **Global CLIP**: Captures semantic meaning (standard approach)
- **IP-Adapter tokens**: Provide detailed visual features for better reconstruction
  - Inspired by IP-Adapter (Ye et al. 2023) but applied to fMRI
  - Multiple tokens allow spatial/feature diversity
- **SD latent**: Coarse structural prior in diffusion latent space
  - Guides spatial layout and composition
  - Complements semantic CLIP information

This is a NOVEL COMBINATION not present in existing fMRI reconstruction papers:
- MindEye2: Only CLIP embeddings
- Brain-Diffuser: CLIP + refinement, no tokens
- NeuralDiffuser: Multi-stage but different conditioning

References:
- IP-Adapter (Ye et al. 2023): "IP-Adapter: Text Compatible Image Prompt Adapter"
- MindEye2 (Scotti et al. 2024): Single CLIP embedding
- Brain-Diffuser (Ozcelik et al. 2023): CLIP + BOI
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Literal
import logging

logger = logging.getLogger(__name__)


class IPAdapterTokenHead(nn.Module):
    """
    Predicts IP-Adapter-style token embeddings from latent representation.
    
    Generates N token vectors that can be used as image prompts in diffusion models.
    Each token is 1024-D (SD 2.1 CLIP dimension).
    
    Args:
        latent_dim: Input latent dimensionality (from Stage 1 encoder)
        n_tokens: Number of token vectors to generate (default: 16)
        token_dim: Dimension per token (default: 1024 for SD 2.1)
        hidden_dim: Hidden dimension for MLP
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        latent_dim: int,
        n_tokens: int = 16,
        token_dim: int = 1024,
        hidden_dim: int = 1024,
        dropout: float = 0.2
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_tokens = n_tokens
        self.token_dim = token_dim
        
        # Generate tokens via MLP
        self.token_generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_tokens * token_dim)
        )
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Generate token embeddings.
        
        Args:
            h: Latent representation (B, latent_dim)
        
        Returns:
            tokens: Token embeddings (B, n_tokens, token_dim)
        """
        # Generate flat tokens
        tokens_flat = self.token_generator(h)  # (B, n_tokens * token_dim)
        
        # Reshape to token format
        tokens = tokens_flat.view(-1, self.n_tokens, self.token_dim)  # (B, n_tokens, token_dim)
        
        # L2 normalize each token (standard for CLIP-like embeddings)
        tokens = torch.nn.functional.normalize(tokens, dim=-1)
        
        return tokens


class SDLatentHead(nn.Module):
    """
    Predicts coarse Stable Diffusion VAE latent from fMRI.
    
    SD VAE encodes 512×512 images to 64×64×4 latents.
    This head predicts a coarse version as structural guidance.
    
    Args:
        latent_dim: Input latent dimensionality
        spatial_size: Spatial size of SD latent (default: 64)
        latent_channels: Number of channels in SD latent (default: 4)
        hidden_dim: Hidden dimension for projection
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        latent_dim: int,
        spatial_size: int = 64,
        latent_channels: int = 4,
        hidden_dim: int = 2048,
        dropout: float = 0.2
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.spatial_size = spatial_size
        self.latent_channels = latent_channels
        self.output_size = latent_channels * spatial_size * spatial_size
        
        # Predict flat latent
        self.latent_generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.output_size)
        )
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Generate SD VAE latent.
        
        Args:
            h: Latent representation (B, latent_dim)
        
        Returns:
            sd_latent: Predicted latent (B, latent_channels, spatial_size, spatial_size)
        """
        # Generate flat latent
        latent_flat = self.latent_generator(h)  # (B, C*H*W)
        
        # Reshape to spatial format
        sd_latent = latent_flat.view(
            -1, self.latent_channels, self.spatial_size, self.spatial_size
        )  # (B, 4, 64, 64)
        
        return sd_latent


class MultiTargetDecoder(nn.Module):
    """
    Multi-target decoder: Predicts CLIP + IP-Adapter tokens + SD latent.
    
    This is a NOVEL module that extends standard fMRI → CLIP mapping
    to predict multiple conditioning signals for richer image generation.
    
    Architecture:
        Latent h (from Stage 1) → Multiple heads in parallel
            ├─ CLIP head → 512-D CLIP embedding
            ├─ IP-Adapter head → N tokens × 1024-D
            └─ SD latent head → 4 × 64 × 64 latent (optional)
    
    Args:
        latent_dim: Input latent dimensionality (from Stage 1 encoder)
        n_tokens: Number of IP-Adapter tokens (default: 16)
        predict_sd_latent: Whether to predict SD VAE latent (default: False)
        clip_head_type: "linear" or "mlp" for CLIP head
        dropout: Dropout probability
    
    Scientific Novelty:
    - First work to predict IP-Adapter tokens from fMRI
    - Multi-task learning across semantic and spatial domains
    - Richer conditioning than single CLIP embedding
    
    Example:
        >>> # Create multi-target decoder
        >>> decoder = MultiTargetDecoder(latent_dim=768, n_tokens=16, predict_sd_latent=True)
        >>> 
        >>> # Forward pass
        >>> h = torch.randn(32, 768)  # Latent from Stage 1
        >>> outputs = decoder(h)
        >>> print(outputs.keys())
        >>> # dict_keys(['clip', 'tokens', 'sd_latent'])
        >>> 
        >>> # Access outputs
        >>> clip_emb = outputs['clip']  # (32, 512)
        >>> tokens = outputs['tokens']  # (32, 16, 1024)
        >>> sd_latent = outputs['sd_latent']  # (32, 4, 64, 64)
    """
    
    def __init__(
        self,
        latent_dim: int,
        n_tokens: int = 16,
        predict_sd_latent: bool = False,
        clip_head_type: Literal["linear", "mlp"] = "linear",
        dropout: float = 0.2
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_tokens = n_tokens
        self.predict_sd_latent = predict_sd_latent
        
        # CLIP head (standard 512-D CLIP ViT-B/32)
        if clip_head_type == "linear":
            self.clip_head = nn.Linear(latent_dim, 512)
        elif clip_head_type == "mlp":
            self.clip_head = nn.Sequential(
                nn.Linear(latent_dim, 512),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(512, 512)
            )
        else:
            raise ValueError(f"Unknown clip_head_type: {clip_head_type}")
        
        # IP-Adapter token head
        self.token_head = IPAdapterTokenHead(
            latent_dim=latent_dim,
            n_tokens=n_tokens,
            token_dim=1024,  # SD 2.1 CLIP dimension
            dropout=dropout
        )
        
        # Optional SD latent head
        if predict_sd_latent:
            self.sd_latent_head = SDLatentHead(
                latent_dim=latent_dim,
                spatial_size=64,
                latent_channels=4,
                dropout=dropout
            )
        else:
            self.sd_latent_head = None
    
    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict all targets from latent representation.
        
        Args:
            h: Latent brain representation (B, latent_dim)
        
        Returns:
            Dictionary with keys:
            - 'clip': Global CLIP embedding (B, 512), L2-normalized
            - 'tokens': IP-Adapter tokens (B, n_tokens, 1024), L2-normalized
            - 'sd_latent': SD VAE latent (B, 4, 64, 64) [if predict_sd_latent=True]
        """
        outputs = {}
        
        # Global CLIP embedding
        clip_emb = self.clip_head(h)  # (B, 512)
        clip_emb = torch.nn.functional.normalize(clip_emb, dim=-1)  # L2 normalize
        outputs['clip'] = clip_emb
        
        # IP-Adapter tokens
        tokens = self.token_head(h)  # (B, n_tokens, 1024), already normalized
        outputs['tokens'] = tokens
        
        # Optional SD latent
        if self.sd_latent_head is not None:
            sd_latent = self.sd_latent_head(h)  # (B, 4, 64, 64)
            outputs['sd_latent'] = sd_latent
        
        return outputs


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for training MultiTargetDecoder.
    
    Combines losses for all prediction targets:
    1. CLIP loss: MSE + Cosine + InfoNCE (from training.losses)
    2. Token loss: Mean of per-token MSE
    3. SD latent loss: MSE in latent space (if applicable)
    
    Args:
        clip_weight: Weight for CLIP losses (default: 1.0)
        token_weight: Weight for token loss (default: 0.5)
        sd_latent_weight: Weight for SD latent loss (default: 0.3)
        clip_loss_config: Config dict for CLIP multi-loss
    """
    
    def __init__(
        self,
        clip_weight: float = 1.0,
        token_weight: float = 0.5,
        sd_latent_weight: float = 0.3,
        clip_loss_config: Optional[Dict] = None
    ):
        super().__init__()
        self.clip_weight = clip_weight
        self.token_weight = token_weight
        self.sd_latent_weight = sd_latent_weight
        
        # Import CLIP multi-loss
        from fmri2img.training.losses import MultiLoss
        
        if clip_loss_config is None:
            clip_loss_config = {
                "mse_weight": 0.3,
                "cosine_weight": 0.3,
                "info_nce_weight": 0.4,
                "temperature": 0.05
            }
        
        self.clip_loss_fn = MultiLoss(**clip_loss_config)
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        return_components: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict]:
        """
        Compute multi-task loss.
        
        Args:
            predictions: Dict with keys 'clip', 'tokens', 'sd_latent' (if applicable)
            targets: Dict with same keys as predictions
            return_components: If True, return (loss, components_dict)
        
        Returns:
            If return_components=False: total_loss
            If return_components=True: (total_loss, components_dict)
        """
        components = {}
        total_loss = 0.0
        
        # CLIP loss (multi-objective)
        if 'clip' in predictions and 'clip' in targets:
            clip_loss, clip_components = self.clip_loss_fn(
                predictions['clip'],
                targets['clip'],
                return_components=True
            )
            total_loss += self.clip_weight * clip_loss
            components['clip_total'] = clip_loss.item()
            components['clip_mse'] = clip_components['mse']
            components['clip_cosine'] = clip_components['cosine']
            components['clip_info_nce'] = clip_components['info_nce']
        
        # Token loss (mean per-token MSE)
        if 'tokens' in predictions and 'tokens' in targets:
            token_loss = nn.functional.mse_loss(predictions['tokens'], targets['tokens'])
            total_loss += self.token_weight * token_loss
            components['tokens'] = token_loss.item()
        
        # SD latent loss (MSE in latent space)
        if 'sd_latent' in predictions and 'sd_latent' in targets:
            sd_latent_loss = nn.functional.mse_loss(predictions['sd_latent'], targets['sd_latent'])
            total_loss += self.sd_latent_weight * sd_latent_loss
            components['sd_latent'] = sd_latent_loss.item()
        
        if return_components:
            components['total'] = total_loss.item()
            return total_loss, components
        
        return total_loss


def save_multi_target_decoder(
    model: MultiTargetDecoder,
    path: str,
    meta: Dict
) -> None:
    """Save MultiTargetDecoder with metadata."""
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "state_dict": model.state_dict(),
        "meta": meta,
        "model_type": "multi_target_decoder"
    }
    
    torch.save(checkpoint, path)
    logger.info(f"Saved MultiTargetDecoder to {path}")


def load_multi_target_decoder(
    path: str,
    map_location: str = "cpu"
) -> Tuple[MultiTargetDecoder, Dict]:
    """Load MultiTargetDecoder from checkpoint."""
    if map_location == "auto":
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint = torch.load(path, map_location=map_location)
    meta = checkpoint.get("meta", {})
    
    model = MultiTargetDecoder(
        latent_dim=meta["latent_dim"],
        n_tokens=meta.get("n_tokens", 16),
        predict_sd_latent=meta.get("predict_sd_latent", False),
        clip_head_type=meta.get("clip_head_type", "linear"),
        dropout=meta.get("dropout", 0.2)
    )
    
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    logger.info(f"Loaded MultiTargetDecoder from {path}")
    
    return model, meta
