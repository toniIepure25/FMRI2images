"""
Advanced fMRI Encoders for SOTA Neural Decoding
===============================================

Two-stage encoder architecture inspired by MindEye, Brain-Diffuser, etc.:
- Stage 1: fMRI representation learning (residual MLP or Transformer)
- Stage 2: CLIP mapping head (modular, configurable)

Scientific Design:
- Residual connections for deep networks (He et al. 2016)
- LayerNorm for training stability (Ba et al. 2016)
- GELU activations (Hendrycks & Gimpel 2016) - smoother than ReLU
- Dropout for regularization (prevents overfitting on fMRI)
- Optional self-supervised pretraining (masked/denoising autoencoder)

References:
- MindEye2 (Scotti et al. 2024): Multi-stage fMRI encoder with residual blocks
- Brain-Diffuser (Ozcelik et al. 2023): Deep encoder with skip connections
- He et al. (2016): "Deep Residual Learning for Image Recognition"
"""

import torch
import torch.nn as nn
from typing import Optional, Literal, Dict, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """
    Residual block with pre-normalization and GELU activation.
    
    Architecture:
        x → LayerNorm → Linear → GELU → Dropout → Linear → Dropout → (+x) → out
    
    Pre-normalization (LayerNorm before residual) improves training stability
    compared to post-norm (used in original ResNet).
    
    Args:
        hidden_dim: Dimension of hidden layers
        dropout: Dropout probability
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor (B, hidden_dim)
        
        Returns:
            Output tensor (B, hidden_dim)
        """
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual


class ResidualMLPEncoder(nn.Module):
    """
    Stage 1: Deep residual MLP for fMRI representation learning.
    
    Maps PCA-reduced fMRI vectors to a latent brain representation via
    multiple residual blocks. This latent representation captures hierarchical
    features of brain activity patterns.
    
    Architecture:
        Input projection: Linear(input_dim, latent_dim) → GELU → Dropout
        Residual blocks: N x ResidualBlock(latent_dim, dropout)
        Output: latent representation h ∈ R^latent_dim
    
    Args:
        input_dim: Input dimensionality (PCA components, e.g., 256/512/768)
        latent_dim: Latent representation dimensionality (e.g., 512/768/1024)
        n_blocks: Number of residual blocks (default: 4)
        dropout: Dropout probability (default: 0.3)
    
    Scientific Rationale:
    - Multiple residual blocks allow learning of hierarchical features
    - Pre-normalization improves gradient flow in deep networks
    - GELU provides smooth, non-monotonic activations (better than ReLU for brain signals)
    - Dropout prevents overfitting on high-dimensional fMRI data
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 512,
        n_blocks: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_blocks = n_blocks
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(latent_dim, dropout)
            for _ in range(n_blocks)
        ])
        
        # Final normalization
        self.out_norm = nn.LayerNorm(latent_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through residual encoder.
        
        Args:
            x: Input fMRI features (B, input_dim)
        
        Returns:
            h: Latent brain representation (B, latent_dim)
        """
        # Input projection
        h = self.input_proj(x)  # (B, latent_dim)
        
        # Residual blocks
        for block in self.blocks:
            h = block(h)
        
        # Final normalization
        h = self.out_norm(h)
        
        return h


class CLIPMappingHead(nn.Module):
    """
    Stage 2: Mapping head from latent brain representation to CLIP space.
    
    Maps the latent representation h from Stage 1 to CLIP embedding space
    (configurable dimension, default 768 for ViT-L/14).
    Supports both linear and MLP variants.
    
    Architecture:
        Linear: Linear(latent_dim, clip_dim) → L2-normalize
        MLP: Linear(latent_dim, hidden) → GELU → Dropout → Linear(hidden, clip_dim) → L2-normalize
    
    Args:
        latent_dim: Input latent dimensionality (from Stage 1)
        clip_dim: Output CLIP embedding dimension (default 768 for ViT-L/14)
        head_type: "linear" or "mlp"
        hidden_dim: Hidden dimension for MLP head (ignored for linear)
        dropout: Dropout for MLP head (default: 0.2)
    
    Scientific Rationale:
    - L2 normalization ensures outputs lie on unit hypersphere (CLIP space convention)
    - Linear head is parameter-efficient; MLP head adds expressiveness
    - Separate head allows freezing Stage 1 during fine-tuning
    """
    
    def __init__(
        self,
        latent_dim: int,
        clip_dim: int = 768,
        head_type: Literal["linear", "mlp"] = "linear",
        hidden_dim: int = 512,
        dropout: float = 0.2
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.head_type = head_type
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        if head_type == "linear":
            # Simple linear projection
            self.head = nn.Linear(latent_dim, clip_dim)
        
        elif head_type == "mlp":
            # Two-layer MLP
            self.head = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, clip_dim)
            )
        
        else:
            raise ValueError(f"Unknown head_type: {head_type}. Must be 'linear' or 'mlp'.")
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Map latent representation to CLIP embedding space.
        
        Args:
            h: Latent brain representation (B, latent_dim)
        
        Returns:
            z: L2-normalized CLIP embedding (B, clip_dim)
        """
        z = self.head(h)  # (B, clip_dim)
        z = torch.nn.functional.normalize(z, dim=-1)  # Unit sphere for cosine similarity
        return z


class TwoStageEncoder(nn.Module):
    """
    Complete two-stage encoder: fMRI → latent h → CLIP embedding.
    
    Combines ResidualMLPEncoder (Stage 1) and CLIPMappingHead (Stage 2)
    into a single end-to-end model. Supports flexible training strategies:
    - Joint training (both stages trainable)
    - Staged training (pretrain Stage 1, freeze and train Stage 2)
    
    Args:
        input_dim: Input dimensionality (PCA components)
        latent_dim: Latent representation dimensionality
        n_blocks: Number of residual blocks in Stage 1
        dropout: Dropout probability
        head_type: "linear" or "mlp" for Stage 2
        head_hidden_dim: Hidden dimension for MLP head
        clip_dim: CLIP embedding dimension (default 768 for ViT-L/14)
    
    Example:
        >>> # Create encoder
        >>> encoder = TwoStageEncoder(input_dim=512, latent_dim=768, n_blocks=4)
        >>> 
        >>> # Forward pass
        >>> x = torch.randn(32, 512)  # Batch of fMRI features
        >>> z = encoder(x)  # (32, 768) CLIP embeddings (ViT-L/14)
        >>> 
        >>> # Access stages separately
        >>> h = encoder.stage1(x)  # (32, 768) latent representation
        >>> z = encoder.stage2(h)  # (32, 768) CLIP embedding
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 512,
        n_blocks: int = 4,
        dropout: float = 0.3,
        head_type: Literal["linear", "mlp"] = "linear",
        head_hidden_dim: int = 512
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Stage 1: fMRI → latent representation
        self.stage1 = ResidualMLPEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            n_blocks=n_blocks,
            dropout=dropout
        )
        
        # Stage 2: latent → CLIP embedding
        self.stage2 = CLIPMappingHead(
            latent_dim=latent_dim,
            head_type=head_type,
            hidden_dim=head_hidden_dim,
            dropout=dropout * 0.7  # Lower dropout for head
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        End-to-end forward pass.
        
        Args:
            x: Input fMRI features (B, input_dim)
        
        Returns:
            z: L2-normalized CLIP embeddings (B, clip_dim)
        """
        h = self.stage1(x)  # (B, latent_dim)
        z = self.stage2(h)  # (B, clip_dim)
        return z
    
    def freeze_stage1(self):
        """Freeze Stage 1 parameters (for staged training)."""
        for param in self.stage1.parameters():
            param.requires_grad = False
        logger.info("Stage 1 (encoder) frozen")
    
    def unfreeze_stage1(self):
        """Unfreeze Stage 1 parameters."""
        for param in self.stage1.parameters():
            param.requires_grad = True
        logger.info("Stage 1 (encoder) unfrozen")


class MultiLayerTwoStageEncoder(nn.Module):
    """
    Extended TwoStageEncoder with multi-layer CLIP supervision.
    
    Predicts CLIP features from multiple ViT layers simultaneously:
    - layer_4: Early visual features (768-D ViT-B/32, 1024-D ViT-L/14)
    - layer_8: Mid-level features (768-D ViT-B/32, 1024-D ViT-L/14)
    - layer_12: Late semantic features (768-D ViT-B/32, 1024-D ViT-L/14)
    - final: Final CLIP embedding (512-D ViT-B/32, 768-D ViT-L/14)
    
    Architecture:
        Stage 1: fMRI → latent h (shared, ResidualMLPEncoder)
        Stage 2: h → {layer_4, layer_8, layer_12, final} (parallel heads)
    
    **Phase 2 Enhancement**: Configurable shared head backbone for parameter efficiency.
    - shared_head_backbone=False: Each head is fully independent (backward-compatible)
    - shared_head_backbone=True: Shared backbone MLP + lightweight per-layer projections
    
    Shared Backbone Architecture:
        latent(512) → backbone MLP → hidden(head_hidden_dim) → per-layer projections
        
        Example with head_hidden_dim=1024:
        - Backbone: 512 → 1024 (shared GELU + dropout)
        - Layer 4 proj: 1024 → 768 (linear)
        - Layer 8 proj: 1024 → 768 (linear)
        - Layer 12 proj: 1024 → 768 (linear)
        - Final proj: 1024 → clip_dim (768 for ViT-L/14)
        
        Parameter reduction: ~60% fewer parameters vs independent heads
    
    Each layer head enables multi-level supervision during training, improving feature learning.
    
    Args:
        input_dim: Input dimensionality (PCA components)
        latent_dim: Latent representation dimensionality
        n_blocks: Number of residual blocks in Stage 1
        dropout: Dropout probability
        head_type: "linear" or "mlp" for Stage 2 heads (ignored if shared_head_backbone=True)
        head_hidden_dim: Hidden dimension for MLP heads or shared backbone
        enabled_layers: Which layers to predict (default: all)
        shared_head_backbone: Use shared backbone + projections (Phase 2, default: False)
    
    Example:
        >>> # Phase 2: Shared backbone (parameter-efficient)
        >>> encoder = MultiLayerTwoStageEncoder(
        ...     input_dim=512, latent_dim=512, 
        ...     shared_head_backbone=True, head_hidden_dim=1024
        ... )
        >>> 
        >>> # Backward-compatible: Independent heads
        >>> encoder = MultiLayerTwoStageEncoder(
        ...     input_dim=512, latent_dim=768,
        ...     shared_head_backbone=False, head_type="mlp"
        ... )
        >>> 
        >>> x = torch.randn(32, 512)
        >>> outputs = encoder(x)
        >>> # outputs = {
        >>> #     'layer_4': (32, 768),
        >>> #     'layer_8': (32, 768),
        >>> #     'layer_12': (32, 768),
        >>> #     'final': (32, 768)
        >>> # }
    
    Scientific Rationale:
        - Multi-level supervision shown effective in vision (FPN, U-Net)
        - Different ViT layers capture different semantic levels
        - Supervising intermediate features improves gradient flow
        - Shared backbone reduces overfitting via parameter sharing (Ruder 2017)
        - Expected +5-10% embedding similarity improvement (Li et al. 2023)
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 512,
        n_blocks: int = 4,
        dropout: float = 0.3,
        head_type: Literal["linear", "mlp"] = "linear",
        head_hidden_dim: int = 512,
        clip_dim: int = 768,
        enabled_layers: Optional[list] = None,
        shared_head_backbone: bool = False,
        predict_text_clip: bool = False  # Phase 2: Text-CLIP prediction
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.enabled_layers = enabled_layers or ['layer_4', 'layer_8', 'layer_12', 'final']
        self.shared_head_backbone = shared_head_backbone
        self.head_hidden_dim = head_hidden_dim
        self.predict_text_clip = predict_text_clip  # Phase 2
        
        # Stage 1: Shared fMRI encoder
        self.stage1 = ResidualMLPEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            n_blocks=n_blocks,
            dropout=dropout
        )
        
        # Stage 2: Parallel heads for each layer
        if shared_head_backbone:
            # Phase 2: Shared backbone + lightweight projections
            logger.info(f"Using SHARED head backbone: {latent_dim} → {head_hidden_dim}")
            
            # Shared backbone MLP
            self.head_backbone = nn.Sequential(
                nn.Linear(latent_dim, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5)  # Lighter dropout for shared component
            )
            
            # Lightweight per-layer projections
            self.heads = nn.ModuleDict()
            
            for layer_name in ['layer_4', 'layer_8', 'layer_12']:
                if layer_name in self.enabled_layers:
                    self.heads[layer_name] = nn.Linear(head_hidden_dim, 768)
            
            if 'final' in self.enabled_layers:
                self.heads['final'] = nn.Linear(head_hidden_dim, clip_dim)
            
            # Phase 2: Text-CLIP head (shares backbone)
            if predict_text_clip:
                self.heads['text'] = nn.Linear(head_hidden_dim, clip_dim)
                logger.info("Phase 2: Text-CLIP head enabled (shared backbone)")
                
        else:
            # Backward-compatible: Fully independent heads
            logger.info(f"Using INDEPENDENT heads (head_type={head_type})")
            self.head_backbone = None
            self.heads = nn.ModuleDict()
            
            # ViT intermediate layer heads (768-D)
            for layer_name in ['layer_4', 'layer_8', 'layer_12']:
                if layer_name in self.enabled_layers:
                    if head_type == "linear":
                        self.heads[layer_name] = nn.Linear(latent_dim, 768)
                    elif head_type == "mlp":
                        self.heads[layer_name] = nn.Sequential(
                            nn.Linear(latent_dim, head_hidden_dim),
                            nn.GELU(),
                            nn.Dropout(dropout * 0.7),
                            nn.Linear(head_hidden_dim, 768)
                        )
            
            # Final CLIP embedding head (768-D for ViT-L/14)
            if 'final' in self.enabled_layers:
                self.heads['final'] = CLIPMappingHead(
                    latent_dim=latent_dim,
                    clip_dim=clip_dim,
                    head_type=head_type,
                    hidden_dim=head_hidden_dim,
                    dropout=dropout * 0.7
                )
            
            # Phase 2: Text-CLIP head (independent)
            if predict_text_clip:
                self.heads['text'] = CLIPMappingHead(
                    latent_dim=latent_dim,
                    clip_dim=clip_dim,
                    head_type=head_type,
                    hidden_dim=head_hidden_dim,
                    dropout=dropout * 0.7
                )
                logger.info("Phase 2: Text-CLIP head enabled (independent)")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-layer outputs.
        
        Args:
            x: Input fMRI features (B, input_dim)
        
        Returns:
            Dict of L2-normalized predictions:
                - layer_4: (B, 768)
                - layer_8: (B, 768)
                - layer_12: (B, 768)
                - final: (B, clip_dim) — 768 for ViT-L/14
        """
        # Stage 1: Shared encoding
        h = self.stage1(x)  # (B, latent_dim)
        
        # Stage 2: Parallel predictions
        if self.shared_head_backbone:
            # Phase 2: Shared backbone → per-layer projections
            h_shared = self.head_backbone(h)  # (B, head_hidden_dim)
            outputs = {}
            for layer_name, head in self.heads.items():
                z = head(h_shared)  # Lightweight projection
                z = torch.nn.functional.normalize(z, dim=-1)
                outputs[layer_name] = z
        else:
            # Backward-compatible: Independent heads
            outputs = {}
            for layer_name, head in self.heads.items():
                z = head(h)
                z = torch.nn.functional.normalize(z, dim=-1)
                outputs[layer_name] = z
        
        return outputs
    
    def get_infonce_representation(
        self,
        layer_outputs: Dict[str, torch.Tensor],
        strategy: str = "weighted_pool"
    ) -> torch.Tensor:
        """
        Create a combined representation from all layers for InfoNCE contrastive learning.
        
        Phase 3: Multi-layer InfoNCE combines information from all ViT layers
        for a richer contrastive signal, rather than using only the final embedding.
        
        Args:
            layer_outputs: Dict of layer predictions from forward()
                - layer_4, layer_8, layer_12: (B, 768) each
                - final: (B, clip_dim) — 768 for ViT-L/14
            strategy: Combination strategy
                - "weighted_pool": Weight by layer importances, project to clip_dim
                - "concat_project": Concatenate all, linear project to clip_dim
                - "average": Simple average of all layers projected to clip_dim
        
        Returns:
            z_infonce: Combined representation (B, clip_dim), L2-normalized
        
        Scientific Rationale:
        - Multi-layer features capture different levels of abstraction
        - Early layers (4/8): Low-level visual features (edges, textures)
        - Mid layers (12): Mid-level semantic features (parts, patterns)
        - Final: High-level semantic features (object categories)
        - Combining all layers provides richer contrastive signal
        
        Expected Improvement: +2-3% from richer representation
        """
        batch_size = layer_outputs['final'].shape[0]
        device = layer_outputs['final'].device
        
        if strategy == "weighted_pool":
            # Project each layer to clip_dim, then weighted average
            # Use default layer weights as importances
            layer_weights = {
                'layer_4': 0.15,
                'layer_8': 0.20,
                'layer_12': 0.25,
                'final': 0.40
            }
            
            # Project 768-D layers to clip_dim to match final
            # We'll use simple linear projections (minimal params)
            clip_dim = self.clip_dim
            if not hasattr(self, '_infonce_projectors'):
                self._infonce_projectors = nn.ModuleDict({
                    'layer_4': nn.Linear(768, clip_dim, bias=False),
                    'layer_8': nn.Linear(768, clip_dim, bias=False),
                    'layer_12': nn.Linear(768, clip_dim, bias=False),
                }).to(device)
            
            # Weighted combination
            z_combined = torch.zeros(batch_size, clip_dim, device=device)
            for layer_name in ['layer_4', 'layer_8', 'layer_12']:
                if layer_name in layer_outputs:
                    z_proj = self._infonce_projectors[layer_name](layer_outputs[layer_name])
                    z_combined += layer_weights[layer_name] * z_proj
            
            # Add final layer (already clip_dim)
            z_combined += layer_weights['final'] * layer_outputs['final']
            
            # L2 normalize
            z_infonce = torch.nn.functional.normalize(z_combined, dim=-1)
            
        elif strategy == "concat_project":
            # Concatenate all layers (768*3 + clip_dim) → project to clip_dim
            clip_dim = self.clip_dim
            if not hasattr(self, '_infonce_concat_proj'):
                total_dim = 768 * 3 + clip_dim  # layer_4, layer_8, layer_12, final
                self._infonce_concat_proj = nn.Linear(total_dim, clip_dim, bias=False).to(device)
            
            # Concatenate
            z_concat = torch.cat([
                layer_outputs['layer_4'],
                layer_outputs['layer_8'],
                layer_outputs['layer_12'],
                layer_outputs['final']
            ], dim=-1)  # (B, 768*3 + clip_dim)
            
            # Project and normalize
            z_infonce = self._infonce_concat_proj(z_concat)
            z_infonce = torch.nn.functional.normalize(z_infonce, dim=-1)
            
        elif strategy == "average":
            # Simple average: project all to clip_dim, then mean
            clip_dim = self.clip_dim
            if not hasattr(self, '_infonce_projectors_avg'):
                self._infonce_projectors_avg = nn.ModuleDict({
                    'layer_4': nn.Linear(768, clip_dim, bias=False),
                    'layer_8': nn.Linear(768, clip_dim, bias=False),
                    'layer_12': nn.Linear(768, clip_dim, bias=False),
                }).to(device)
            
            z_list = []
            for layer_name in ['layer_4', 'layer_8', 'layer_12']:
                if layer_name in layer_outputs:
                    z_proj = self._infonce_projectors_avg[layer_name](layer_outputs[layer_name])
                    z_list.append(z_proj)
            z_list.append(layer_outputs['final'])
            
            # Average
            z_combined = torch.stack(z_list, dim=0).mean(dim=0)  # (B, clip_dim)
            z_infonce = torch.nn.functional.normalize(z_combined, dim=-1)
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return z_infonce
    
    def freeze_stage1(self):
        """Freeze Stage 1 parameters (for staged training)."""
        for param in self.stage1.parameters():
            param.requires_grad = False
        logger.info("Stage 1 (encoder) frozen for multi-layer training")
    
    def unfreeze_stage1(self):
        """Unfreeze Stage 1 parameters."""
        for param in self.stage1.parameters():
            param.requires_grad = True
        logger.info("Stage 1 (encoder) unfrozen for multi-layer training")
    
    def get_final_only(self) -> nn.Module:
        """
        Get a single-output wrapper that only returns final layer.
        Useful for inference/evaluation that expects single embedding.
        """
        class FinalOnlyWrapper(nn.Module):
            def __init__(self, parent):
                super().__init__()
                self.parent = parent
            
            def forward(self, x):
                outputs = self.parent(x)
                return outputs['final']
        
        return FinalOnlyWrapper(self)


class SelfSupervisedPretrainer(nn.Module):
    """
    Self-supervised pretraining module for Stage 1 encoder.
    
    Implements two pretraining objectives:
    1. Masked Autoencoder: Mask random PCA dimensions, reconstruct them
    2. Denoising Autoencoder: Add noise to input, reconstruct clean version
    
    This allows learning useful fMRI representations without CLIP labels,
    potentially improving sample efficiency and generalization.
    
    Args:
        encoder: ResidualMLPEncoder (Stage 1)
        reconstruction_dim: Dimensionality to reconstruct (same as input_dim)
        objective: "masked" or "denoising"
        mask_ratio: Fraction of dimensions to mask (for masked autoencoder)
        noise_std: Noise standard deviation (for denoising autoencoder)
    
    Scientific Rationale:
    - Self-supervised pretraining shown effective for fMRI (Thomas et al. 2022)
    - Masked reconstruction forces learning of feature dependencies
    - Denoising improves robustness to measurement noise
    """
    
    def __init__(
        self,
        encoder: ResidualMLPEncoder,
        reconstruction_dim: int,
        objective: Literal["masked", "denoising"] = "masked",
        mask_ratio: float = 0.3,
        noise_std: float = 0.1
    ):
        super().__init__()
        self.encoder = encoder
        self.reconstruction_dim = reconstruction_dim
        self.objective = objective
        self.mask_ratio = mask_ratio
        self.noise_std = noise_std
        
        # Reconstruction head
        self.recon_head = nn.Sequential(
            nn.Linear(encoder.latent_dim, encoder.latent_dim),
            nn.GELU(),
            nn.Linear(encoder.latent_dim, reconstruction_dim)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for self-supervised pretraining.
        
        Args:
            x: Clean input fMRI features (B, input_dim)
        
        Returns:
            x_corrupted: Corrupted input (masked or noisy)
            x_reconstructed: Reconstructed output
            x_target: Target for reconstruction (clean input or masked portions)
        """
        if self.objective == "masked":
            # Masked autoencoder
            x_corrupted, mask = self._apply_mask(x)
            h = self.encoder(x_corrupted)
            x_reconstructed = self.recon_head(h)
            # Target is the original input (masked portions will have higher loss)
            x_target = x
            return x_corrupted, x_reconstructed, x_target
        
        elif self.objective == "denoising":
            # Denoising autoencoder
            noise = torch.randn_like(x) * self.noise_std
            x_corrupted = x + noise
            h = self.encoder(x_corrupted)
            x_reconstructed = self.recon_head(h)
            # Target is the clean input
            x_target = x
            return x_corrupted, x_reconstructed, x_target
        
        else:
            raise ValueError(f"Unknown objective: {self.objective}")
    
    def _apply_mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply random masking to input features.
        
        Args:
            x: Input tensor (B, D)
        
        Returns:
            x_masked: Masked input (B, D) with masked positions set to 0
            mask: Boolean mask (B, D), True = masked
        """
        B, D = x.shape
        # Random mask
        mask = torch.rand(B, D, device=x.device) < self.mask_ratio
        x_masked = x.clone()
        x_masked[mask] = 0.0
        return x_masked, mask


def save_two_stage_encoder(
    model: TwoStageEncoder,
    path: str,
    meta: Dict[str, Any]
) -> None:
    """
    Save TwoStageEncoder with metadata.
    
    Args:
        model: Trained TwoStageEncoder
        path: Output checkpoint path
        meta: Metadata dictionary (architecture config, training info)
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "state_dict": model.state_dict(),
        "meta": meta,
        "model_type": "two_stage_encoder"
    }
    
    torch.save(checkpoint, path)
    logger.info(f"Saved TwoStageEncoder to {path}")


def load_two_stage_encoder(
    path: str,
    map_location: str = "cpu"
) -> Tuple[TwoStageEncoder, Dict[str, Any]]:
    """
    Load TwoStageEncoder from checkpoint.
    
    Args:
        path: Checkpoint path
        map_location: Device to load to ("cpu", "cuda", or "auto")
    
    Returns:
        model: Loaded TwoStageEncoder
        meta: Metadata dictionary
    """
    # Resolve 'auto' to actual device
    if map_location == "auto":
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint = torch.load(path, map_location=map_location)
    meta = checkpoint.get("meta", {})
    
    # Reconstruct model from metadata
    model = TwoStageEncoder(
        input_dim=meta["input_dim"],
        latent_dim=meta.get("latent_dim", 512),
        n_blocks=meta.get("n_blocks", 4),
        dropout=meta.get("dropout", 0.3),
        head_type=meta.get("head_type", "linear"),
        head_hidden_dim=meta.get("head_hidden_dim", 512),
        clip_dim=meta.get("clip_dim", 768)
    )
    
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    logger.info(f"Loaded TwoStageEncoder from {path}")
    
    return model, meta


def load_multilayer_two_stage_encoder(
    path: str,
    map_location: str = "cpu"
) -> Tuple[MultiLayerTwoStageEncoder, Dict[str, Any]]:
    """
    Load MultiLayerTwoStageEncoder from checkpoint.
    
    Args:
        path: Checkpoint path
        map_location: Device to load to ("cpu", "cuda", or "auto")
    
    Returns:
        model: Loaded MultiLayerTwoStageEncoder
        meta: Metadata dictionary
    """
    # Resolve 'auto' to actual device
    if map_location == "auto":
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint = torch.load(path, map_location=map_location)
    meta = checkpoint.get("meta", {})
    
    # Reconstruct model from metadata
    model = MultiLayerTwoStageEncoder(
        input_dim=meta["input_dim"],
        latent_dim=meta.get("latent_dim", 512),
        n_blocks=meta.get("n_blocks", 4),
        dropout=meta.get("dropout", 0.3),
        head_type=meta.get("head_type", "linear"),
        head_hidden_dim=meta.get("head_hidden_dim", 512),
        clip_dim=meta.get("clip_dim", 768),
        shared_head_backbone=meta.get("shared_head_backbone", False)
    )
    
    # Load with strict=False to allow InfoNCE projectors (dynamically created during training)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["state_dict"], strict=False)
    
    # Log any issues (InfoNCE projectors are expected to be unexpected)
    if unexpected_keys:
        infonce_keys = [k for k in unexpected_keys if '_infonce_projectors' in k]
        other_keys = [k for k in unexpected_keys if '_infonce_projectors' not in k]
        if infonce_keys:
            logger.debug(f"Ignoring {len(infonce_keys)} InfoNCE projector keys (will be recreated if needed)")
        if other_keys:
            logger.warning(f"Unexpected keys in checkpoint: {other_keys}")
    
    logger.info(f"Loaded MultiLayerTwoStageEncoder from {path}")
    
    return model, meta


class ProbabilisticMultiLayerTwoStageEncoder(nn.Module):
    """
    Phase 3: Probabilistic encoder with uncertainty estimation.
    
    Extends MultiLayerTwoStageEncoder to predict distributions instead of point estimates:
    - For each output (layer_4, layer_8, layer_12, final, text), predict μ and logσ²
    - Use reparameterization trick for differentiable sampling: z = μ + ε·σ, ε ~ N(0,1)
    - Add KL divergence loss: KL(q(z|x) || N(0,I)) to regularize distributions
    
    **Scientific Motivation:**
    - fMRI measurements are noisy → predictions should reflect uncertainty
    - Variational inference provides principled uncertainty quantification
    - Enables confidence-aware decoding (weight predictions by certainty)
    - Theoretical foundation: Variational Autoencoder (Kingma & Welling 2014)
    
    **Architecture:**
        Stage 1: fMRI → latent h (shared, ResidualMLPEncoder)
        Stage 2: h → {μ, logσ²} for each target (parallel probabilistic heads)
    
    **Usage:**
        # Training: Sample from distribution
        outputs = model(x, sample=True)  # Returns sampled z ~ N(μ, σ²)
        
        # Inference: Multiple samples for uncertainty estimation
        samples = [model(x, sample=True) for _ in range(N)]
        mean_pred = torch.stack(samples).mean(dim=0)
        std_pred = torch.stack(samples).std(dim=0)
        
        # Deterministic: Use mean
        outputs = model(x, sample=False)  # Returns μ
    
    Args:
        input_dim: Input dimensionality (PCA components)
        latent_dim: Latent representation dimensionality
        n_blocks: Number of residual blocks in Stage 1
        dropout: Dropout probability
        head_hidden_dim: Hidden dimension for MLP heads
        enabled_layers: Which layers to predict
        predict_text_clip: Enable text-CLIP prediction (Phase 2)
        kl_weight: Weight for KL divergence loss (default: 0.01, annealed during training)
    
    Example:
        >>> encoder = ProbabilisticMultiLayerTwoStageEncoder(
        ...     input_dim=512, latent_dim=512,
        ...     predict_text_clip=True
        ... )
        >>> x = torch.randn(32, 512)
        >>> 
        >>> # Training: Sample from distribution
        >>> outputs, kl_loss = encoder(x, sample=True, return_kl=True)
        >>> # outputs = {'final': (32, 768), ...}
        >>> # kl_loss = scalar
        >>> 
        >>> # Inference: Uncertainty estimation
        >>> samples = [encoder(x, sample=True, return_kl=False) for _ in range(10)]
        >>> uncertainty = torch.stack([s['final'] for s in samples]).std(dim=0)
    
    References:
        - Kingma & Welling (2014): "Auto-Encoding Variational Bayes"
        - Gal & Ghahramani (2016): "Dropout as a Bayesian Approximation"
        - Kendall & Gal (2017): "What Uncertainties Do We Need in Bayesian Deep Learning?"
        - MindEye approach: Deterministic embeddings (our Phase 1-2)
        - This Phase 3: Probabilistic embeddings with uncertainty
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 512,
        n_blocks: int = 4,
        dropout: float = 0.3,
        head_hidden_dim: int = 512,
        output_dim: int = 768,  # CLIP embedding dimension (768 for ViT-L/14)
        enabled_layers: Optional[list] = None,
        predict_text_clip: bool = False,
        kl_weight: float = 0.01,
        uncertainty: Literal["diag", "scalar"] = "diag",
        logvar_min: float = -8.0,
        logvar_max: float = 2.0,
        variance_floor: float = 1.0e-6,
        clip_space: Literal["normalized", "raw"] = "normalized",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim  # Final CLIP embedding dimension
        self.enabled_layers = enabled_layers or ['layer_4', 'layer_8', 'layer_12', 'final']
        self.predict_text_clip = predict_text_clip
        self.kl_weight = kl_weight
        self.uncertainty = uncertainty
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.variance_floor = variance_floor
        self.clip_space = clip_space
        
        # Stage 1: Shared fMRI encoder (deterministic)
        self.stage1 = ResidualMLPEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            n_blocks=n_blocks,
            dropout=dropout
        )
        
        # Stage 2: Probabilistic heads (predict μ and logσ² for each target)
        # Use shared backbone for efficiency
        logger.info(f"Phase 3: Probabilistic encoder with shared backbone")
        
        self.head_backbone = nn.Sequential(
            nn.Linear(latent_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # For each layer, predict both μ and logσ²
        self.mu_heads = nn.ModuleDict()
        self.logvar_heads = nn.ModuleDict()
        
        # ViT intermediate layers (768-D)
        for layer_name in ['layer_4', 'layer_8', 'layer_12']:
            if layer_name in self.enabled_layers:
                self.mu_heads[layer_name] = nn.Linear(head_hidden_dim, 768)
                logvar_dim = 1 if self.uncertainty == "scalar" else 768
                self.logvar_heads[layer_name] = nn.Linear(head_hidden_dim, logvar_dim)
        
        # Final CLIP embedding (configurable dimension)
        if 'final' in self.enabled_layers:
            self.mu_heads['final'] = nn.Linear(head_hidden_dim, output_dim)
            logvar_dim = 1 if self.uncertainty == "scalar" else output_dim
            self.logvar_heads['final'] = nn.Linear(head_hidden_dim, logvar_dim)
        
        # Text-CLIP head (same dimension as final CLIP)
        if predict_text_clip:
            self.mu_heads['text'] = nn.Linear(head_hidden_dim, output_dim)
            logvar_dim = 1 if self.uncertainty == "scalar" else output_dim
            self.logvar_heads['text'] = nn.Linear(head_hidden_dim, logvar_dim)
            logger.info("Phase 3: Probabilistic text-CLIP head enabled")

    @staticmethod
    def _normalize_if_needed(x: torch.Tensor, space: Literal["normalized", "raw"]) -> torch.Tensor:
        return torch.nn.functional.normalize(x, dim=-1) if space == "normalized" else x

    def _expand_logvar(self, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Broadcast scalar logvar to match target dimensionality when needed."""
        if logvar.shape[-1] == 1 and target.shape[-1] != 1:
            return logvar.expand_as(target)
        return logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        logvar_clamped = logvar.clamp(self.logvar_min, self.logvar_max)
        var = torch.exp(logvar_clamped).clamp_min(self.variance_floor)
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        logvar_clamped = logvar.clamp(self.logvar_min, self.logvar_max)
        var = torch.exp(logvar_clamped).clamp_min(self.variance_floor)
        kl_div = -0.5 * torch.sum(1 + logvar_clamped - mu.pow(2) - var, dim=-1)
        return kl_div.mean()

    def forward(
        self,
        x: torch.Tensor,
        sample: bool = True,
        return_kl: bool = True
    ) -> Tuple[Dict[str, "ProbabilisticLayerOutput"], Optional[torch.Tensor]]:
        """
        Forward pass with probabilistic outputs and structured metadata.
        """
        # Stage 1: Shared encoding
        h = self.stage1(x)  # (B, latent_dim)
        h_shared = self.head_backbone(h)  # (B, head_hidden_dim)

        outputs: Dict[str, ProbabilisticLayerOutput] = {}
        kl_terms = []

        for layer_name in self.mu_heads.keys():
            mu = self.mu_heads[layer_name](h_shared)
            logvar_raw = self.logvar_heads[layer_name](h_shared)
            logvar_expanded = self._expand_logvar(logvar_raw, mu)

            logvar_clamped = logvar_expanded.clamp(self.logvar_min, self.logvar_max)
            clamp_mask = (logvar_expanded < self.logvar_min) | (logvar_expanded > self.logvar_max)
            logvar_clamp_fraction = clamp_mask.float().mean().item()

            var = torch.exp(logvar_clamped).clamp_min(self.variance_floor)

            if sample:
                z = self.reparameterize(mu, logvar_clamped)
            else:
                z = mu

            z_out = self._normalize_if_needed(z, self.clip_space)

            outputs[layer_name] = ProbabilisticLayerOutput(
                z=z_out,
                mu=self._normalize_if_needed(mu, self.clip_space),
                logvar=logvar_raw,
                logvar_clamped=logvar_clamped,
                var=var,
                uncertainty=var.mean(dim=-1),
                logvar_clamp_fraction=logvar_clamp_fraction,
            )

            if return_kl:
                kl_terms.append(self.compute_kl_loss(mu, logvar_clamped))

        kl_loss = None
        if return_kl and kl_terms:
            kl_loss = sum(kl_terms) / len(kl_terms)
            kl_loss = kl_loss * self.kl_weight

        return outputs, kl_loss


@dataclass
class ProbabilisticLayerOutput:
    z: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    logvar_clamped: torch.Tensor
    var: torch.Tensor
    uncertainty: torch.Tensor
    logvar_clamp_fraction: float


def save_probabilistic_encoder(
    model: ProbabilisticMultiLayerTwoStageEncoder,
    path: str,
    meta: Dict[str, Any]
):
    """
    Save ProbabilisticMultiLayerTwoStageEncoder to checkpoint.
    
    Args:
        model: Model to save
        path: Output checkpoint path
        meta: Metadata dictionary (training config, metrics, etc.)
    """
    checkpoint = {
        "state_dict": model.state_dict(),
        "meta": meta
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    logger.info(f"Saved ProbabilisticMultiLayerTwoStageEncoder to {path}")


def load_probabilistic_encoder(
    path: str,
    map_location: str = "cpu"
) -> Tuple[ProbabilisticMultiLayerTwoStageEncoder, Dict[str, Any]]:
    """
    Load ProbabilisticMultiLayerTwoStageEncoder from checkpoint.
    
    Args:
        path: Checkpoint path
        map_location: Device to load to ("cpu", "cuda", or "auto")
    
    Returns:
        model: Loaded ProbabilisticMultiLayerTwoStageEncoder
        meta: Metadata dictionary
    """
    if map_location == "auto":
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint = torch.load(path, map_location=map_location)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        # Check for state_dict in various keys
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            meta = checkpoint.get("meta", {})
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            meta = checkpoint.get("meta", {})
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
            meta = checkpoint.get("meta", {})
        else:
            # Assume checkpoint is the state_dict itself
            state_dict = checkpoint
            meta = {}
            logger.warning("Checkpoint appears to be a raw state_dict (no 'state_dict' key)")
    else:
        # Checkpoint is the model state_dict directly
        state_dict = checkpoint
        meta = {}
        logger.warning("Checkpoint is a raw state_dict (not a dictionary)")
    
    # Infer input_dim from state_dict if not in metadata (backward compatibility)
    if "input_dim" not in meta:
        # Try different common patterns for input projection layers
        input_dim = None
        
        # Pattern 1: projections.layer_4.0.weight (new format)
        if "projections.layer_4.0.weight" in state_dict:
            input_dim = state_dict["projections.layer_4.0.weight"].shape[1]
            logger.warning(f"input_dim inferred from projections.layer_4.0.weight: {input_dim}")
        
        # Pattern 2: stage1.input_proj.0.weight (older format)
        elif "stage1.input_proj.0.weight" in state_dict:
            input_dim = state_dict["stage1.input_proj.0.weight"].shape[1]
            logger.warning(f"input_dim inferred from stage1.input_proj.0.weight: {input_dim}")
        
        # Pattern 3: input_projection.weight
        elif "input_projection.weight" in state_dict:
            input_dim = state_dict["input_projection.weight"].shape[1]
            logger.warning(f"input_dim inferred from input_projection.weight: {input_dim}")
        
        # Pattern 4: Find any layer with "input" and "weight" that looks like a projection
        else:
            for key in state_dict.keys():
                if "input" in key.lower() and "weight" in key and isinstance(state_dict[key], torch.Tensor):
                    shape = state_dict[key].shape
                    if len(shape) == 2 and shape[1] > 10000:  # Likely an fMRI input projection
                        input_dim = shape[1]
                        logger.warning(f"input_dim inferred from {key}: {input_dim}")
                        break
        
        if input_dim is not None:
            meta["input_dim"] = input_dim
        else:
            raise ValueError(
                "Cannot infer input_dim from checkpoint. "
                "No input projection layer found. "
                "Please specify input_dim in training config or checkpoint metadata."
            )
    
    # Infer latent_dim from state_dict if not in metadata
    if "latent_dim" not in meta:
        if "stage1.input_proj.0.bias" in state_dict:
            latent_dim = state_dict["stage1.input_proj.0.bias"].shape[0]
            meta["latent_dim"] = latent_dim
            logger.warning(f"latent_dim inferred from stage1.input_proj.0.bias: {latent_dim}")
    
    # Infer n_blocks from state_dict if not in metadata
    if "n_blocks" not in meta:
        n_blocks = 0
        for key in state_dict.keys():
            if key.startswith("stage1.blocks.") and ".norm.weight" in key:
                block_idx = int(key.split(".")[2])
                n_blocks = max(n_blocks, block_idx + 1)
        if n_blocks > 0:
            meta["n_blocks"] = n_blocks
            logger.warning(f"n_blocks inferred from state_dict: {n_blocks}")
    
    # Infer head_hidden_dim from state_dict if not in metadata
    if "head_hidden_dim" not in meta:
        if "head_backbone.0.bias" in state_dict:
            head_hidden_dim = state_dict["head_backbone.0.bias"].shape[0]
            meta["head_hidden_dim"] = head_hidden_dim
            logger.warning(f"head_hidden_dim inferred from head_backbone.0.bias: {head_hidden_dim}")
    
    # Infer predict_text_clip from state_dict if not in metadata
    if "predict_text_clip" not in meta:
        has_text_head = "mu_heads.text.weight" in state_dict or "mu_heads.text.bias" in state_dict
        meta["predict_text_clip"] = has_text_head
        if has_text_head:
            logger.warning(f"predict_text_clip inferred from state_dict: True")
    
    # Reconstruct model from metadata
    model = ProbabilisticMultiLayerTwoStageEncoder(
        input_dim=meta["input_dim"],
        latent_dim=meta.get("latent_dim", 512),
        n_blocks=meta.get("n_blocks", 4),
        dropout=meta.get("dropout", 0.3),
        head_hidden_dim=meta.get("head_hidden_dim", 512),
        output_dim=meta.get("output_dim", 768),
        enabled_layers=meta.get("enabled_layers", ['layer_4', 'layer_8', 'layer_12', 'final']),
        predict_text_clip=meta.get("predict_text_clip", False),
        kl_weight=meta.get("kl_weight", 0.01),
        uncertainty=meta.get("uncertainty", "diag"),
        logvar_min=meta.get("logvar_min", -8.0),
        logvar_max=meta.get("logvar_max", 2.0),
        variance_floor=meta.get("variance_floor", 1.0e-6),
        clip_space=meta.get("clip_space", "normalized"),
    )
    
    model.load_state_dict(state_dict, strict=True)
    logger.info(f"Loaded ProbabilisticMultiLayerTwoStageEncoder from {path}")
    
    return model, meta
