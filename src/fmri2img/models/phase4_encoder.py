"""
Phase 4: Structural/Semantic Branch Encoder Addition

This file extends the Phase 3 probabilistic encoder with explicit structural/semantic factorization.
To be integrated into src/fmri2img/models/encoders.py
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class StructuralSemanticEncoder(nn.Module):
    """
    Phase 4: Encoder with explicit structural/semantic branch factorization.
    
    **Scientific Motivation:**
    Brain processes visual information through separate pathways:
    - **Ventral stream** (semantic): "what" pathway → object identity, meaning
    - **Dorsal stream** (structural): "where/how" pathway → spatial layout, edges, motion
    
    This architecture mirrors that factorization:
    - **Structural branch**: Targets early/mid CLIP layers (layer_4, layer_8)
      → Low-level features (edges, textures, spatial structure)
    - **Semantic branch**: Targets late CLIP layers (layer_12, final) + text-CLIP
      → High-level concepts (object categories, semantics)
    
    **Architecture:**
        Stage 1: fMRI → shared latent h (ResidualMLPEncoder)
        Stage 2a: h → structural_latent → early layer predictions (L4, L8)
        Stage 2b: h → semantic_latent → late layer predictions (L12, final, text)
    
    **Key Innovation:**
    - Explicit disentanglement allows differential supervision
    - Can weight structural vs semantic losses independently
    - Enables multi-condition diffusion guidance (structural + semantic)
    - Helps address layer_12 underperformance by dedicated semantic modeling
    
    **Usage:**
        # Training with probabilistic mode
        outputs = encoder(x, sample=True, return_kl=True)
        # outputs = {
        #     'structural_branch': (B, struct_dim),
        #     'semantic_branch': (B, sem_dim),
        #     'layer_4': (B, 768), 'layer_8': (B, 768),  # from structural
        #     'layer_12': (B, 768), 'final': (B, 512),   # from semantic
        #     'text': (B, 512)  # from semantic
        # }
        
        # Multi-condition diffusion
        z_struct = outputs['structural_branch']  # for structural guidance
        z_sem = outputs['semantic_branch']       # for semantic guidance
        z_final = outputs['final']               # for image-CLIP conditioning
        z_text = outputs['text']                 # for text-CLIP conditioning
    
    Args:
        input_dim: Input dimensionality (PCA components)
        latent_dim: Shared latent representation dim
        structural_dim: Structural branch latent dim (default: 256)
        semantic_dim: Semantic branch latent dim (default: 512)
        n_blocks: Number of residual blocks in Stage 1
        dropout: Dropout probability
        head_hidden_dim: Hidden dimension for prediction heads
        predict_text_clip: Enable text-CLIP prediction (Phase 2)
        probabilistic: Enable probabilistic predictions (Phase 3)
        kl_weight: Weight for KL divergence (if probabilistic)
    
    Example:
        >>> # Phase 4: Structural + Semantic branches
        >>> encoder = StructuralSemanticEncoder(
        ...     input_dim=512,
        ...     latent_dim=512,
        ...     structural_dim=256,
        ...     semantic_dim=512,
        ...     probabilistic=True,
        ...     predict_text_clip=True
        ... )
        >>> x = torch.randn(32, 512)
        >>> outputs, kl_loss = encoder(x, sample=True, return_kl=True)
        >>> 
        >>> # Access branch latents for multi-condition diffusion
        >>> struct_latent = outputs['structural_branch']  # (32, 256)
        >>> sem_latent = outputs['semantic_branch']       # (32, 512)
        >>> 
        >>> # Access layer predictions
        >>> early_layers = {k: outputs[k] for k in ['layer_4', 'layer_8']}
        >>> late_layers = {k: outputs[k] for k in ['layer_12', 'final', 'text']}
    
    References:
        - Goodale & Milner (1992): "Separate visual pathways for perception and action"
        - Ungerleider & Mishkin (1982): "Two cortical visual systems"
        - Phase 3: Probabilistic embeddings with uncertainty
        - Phase 4 (this): Structural/semantic factorization
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 512,
        structural_dim: int = 256,
        semantic_dim: int = 512,
        n_blocks: int = 4,
        dropout: float = 0.3,
        head_hidden_dim: int = 512,
        predict_text_clip: bool = False,
        probabilistic: bool = False,
        kl_weight: float = 0.01
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.structural_dim = structural_dim
        self.semantic_dim = semantic_dim
        self.predict_text_clip = predict_text_clip
        self.probabilistic = probabilistic
        self.kl_weight = kl_weight
        
        logger.info("=" * 80)
        logger.info("Phase 4: Structural/Semantic Branch Encoder")
        logger.info("=" * 80)
        logger.info(f"  Structural branch: {structural_dim}-D → early layers (L4, L8)")
        logger.info(f"  Semantic branch: {semantic_dim}-D → late layers (L12, final, text)")
        logger.info(f"  Probabilistic mode: {probabilistic}")
        
        # Stage 1: Shared fMRI encoder (deterministic backbone)
        from fmri2img.models.encoders import ResidualMLPEncoder
        self.stage1 = ResidualMLPEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            n_blocks=n_blocks,
            dropout=dropout
        )
        
        # Stage 2a: Structural branch (targets early/mid layers)
        # Brain-consistency loss: forces CLIP predictions to produce realistic fMRI patterns
        logger.info("  Creating structural branch (ventral-like stream)...")
        self.structural_backbone = nn.Sequential(
            nn.Linear(latent_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_hidden_dim, structural_dim),
            nn.GELU()
        )
        
        # Stage 2b: Semantic branch (targets late layers + text)
        # Multi-task semantics: shared latent predicts both image and text CLIP embeddings
        logger.info("  Creating semantic branch (dorsal-like stream)...")
        self.semantic_backbone = nn.Sequential(
            nn.Linear(latent_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_hidden_dim, semantic_dim),
            nn.GELU()
        )
        
        # Prediction heads from branches
        if probabilistic:
            logger.info("  Probabilistic decoding: models uncertainty in CLIP space (μ, σ)")
            self._create_probabilistic_heads(head_hidden_dim, dropout)
        else:
            logger.info("  Deterministic decoding: point estimates")
            self._create_deterministic_heads(head_hidden_dim)
        
        logger.info("=" * 80)
    
    def _create_deterministic_heads(self, head_hidden_dim: int):
        """Create deterministic prediction heads."""
        self.mu_heads = nn.ModuleDict()
        self.logvar_heads = None  # Not used in deterministic mode
        
        # Structural branch → early layers (768-D ViT features)
        for layer_name in ['layer_4', 'layer_8']:
            self.mu_heads[layer_name] = nn.Linear(self.structural_dim, 768)
        
        # Semantic branch → late layers (768-D ViT + 512-D CLIP)
        self.mu_heads['layer_12'] = nn.Linear(self.semantic_dim, 768)
        self.mu_heads['final'] = nn.Linear(self.semantic_dim, 512)
        
        if self.predict_text_clip:
            self.mu_heads['text'] = nn.Linear(self.semantic_dim, 512)
            logger.info("    Text-CLIP head enabled (semantic branch)")
    
    def _create_probabilistic_heads(self, head_hidden_dim: int, dropout: float):
        """Create probabilistic prediction heads (μ and logσ²)."""
        self.mu_heads = nn.ModuleDict()
        self.logvar_heads = nn.ModuleDict()
        
        # Structural branch → early layers (768-D ViT features)
        for layer_name in ['layer_4', 'layer_8']:
            self.mu_heads[layer_name] = nn.Linear(self.structural_dim, 768)
            self.logvar_heads[layer_name] = nn.Linear(self.structural_dim, 768)
        
        # Semantic branch → late layers (768-D ViT + 512-D CLIP)
        self.mu_heads['layer_12'] = nn.Linear(self.semantic_dim, 768)
        self.logvar_heads['layer_12'] = nn.Linear(self.semantic_dim, 768)
        
        self.mu_heads['final'] = nn.Linear(self.semantic_dim, 512)
        self.logvar_heads['final'] = nn.Linear(self.semantic_dim, 512)
        
        if self.predict_text_clip:
            self.mu_heads['text'] = nn.Linear(self.semantic_dim, 512)
            self.logvar_heads['text'] = nn.Linear(self.semantic_dim, 512)
            logger.info("    Probabilistic text-CLIP head enabled (semantic branch)")
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = μ + ε·σ where ε ~ N(0,1)
        
        Args:
            mu: Mean (B, D)
            logvar: Log variance (B, D)
        
        Returns:
            z: Sampled embedding (B, D)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence: KL(q(z|x) || N(0,I))
        
        Args:
            mu: Mean (B, D)
            logvar: Log variance (B, D)
        
        Returns:
            kl_loss: Scalar KL divergence
        """
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl_div.mean()
    
    def forward(
        self,
        x: torch.Tensor,
        sample: bool = True,
        return_kl: bool = True,
        return_branch_latents: bool = True
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass with structural/semantic branching.
        
        Args:
            x: Input fMRI features (B, input_dim)
            sample: If True (and probabilistic), sample from distribution
            return_kl: If True (and probabilistic), return KL divergence
            return_branch_latents: If True, include branch latents in output
        
        Returns:
            outputs: Dict with keys:
                - 'structural_branch': Structural latent (B, structural_dim)
                - 'semantic_branch': Semantic latent (B, semantic_dim)
                - 'layer_4', 'layer_8': Early layer predictions from structural
                - 'layer_12', 'final': Late layer predictions from semantic
                - 'text': Text-CLIP prediction from semantic (if enabled)
            kl_loss: KL divergence loss (if probabilistic and return_kl=True)
        """
        # Stage 1: Shared encoding
        h = self.stage1(x)  # (B, latent_dim)
        
        # Stage 2: Branch into structural and semantic pathways
        structural_latent = self.structural_backbone(h)  # (B, structural_dim)
        semantic_latent = self.semantic_backbone(h)      # (B, semantic_dim)
        
        outputs = {}
        total_kl = 0.0
        
        # Include branch latents for multi-condition diffusion
        if return_branch_latents:
            outputs['structural_branch'] = structural_latent
            outputs['semantic_branch'] = semantic_latent
        
        # Structural branch → early layers (L4, L8)
        for layer_name in ['layer_4', 'layer_8']:
            mu = self.mu_heads[layer_name](structural_latent)
            
            if self.probabilistic and self.logvar_heads is not None:
                logvar = self.logvar_heads[layer_name](structural_latent)
                z = self.reparameterize(mu, logvar) if sample else mu
                if return_kl:
                    total_kl += self.compute_kl_loss(mu, logvar)
            else:
                z = mu
            
            # L2 normalize
            z = torch.nn.functional.normalize(z, dim=-1)
            outputs[layer_name] = z
        
        # Semantic branch → late layers (L12, final) + text
        for layer_name in ['layer_12', 'final']:
            mu = self.mu_heads[layer_name](semantic_latent)
            
            if self.probabilistic and self.logvar_heads is not None:
                logvar = self.logvar_heads[layer_name](semantic_latent)
                z = self.reparameterize(mu, logvar) if sample else mu
                if return_kl:
                    total_kl += self.compute_kl_loss(mu, logvar)
            else:
                z = mu
            
            # L2 normalize
            z = torch.nn.functional.normalize(z, dim=-1)
            outputs[layer_name] = z
        
        # Text-CLIP from semantic branch
        if self.predict_text_clip and 'text' in self.mu_heads:
            mu = self.mu_heads['text'](semantic_latent)
            
            if self.probabilistic and self.logvar_heads is not None:
                logvar = self.logvar_heads['text'](semantic_latent)
                z = self.reparameterize(mu, logvar) if sample else mu
                if return_kl:
                    total_kl += self.compute_kl_loss(mu, logvar)
            else:
                z = mu
            
            z = torch.nn.functional.normalize(z, dim=-1)
            outputs['text'] = z
        
        # Average KL loss across all heads
        if self.probabilistic and return_kl:
            n_heads = len(self.mu_heads)
            kl_loss = (total_kl / n_heads) * self.kl_weight
            return outputs, kl_loss
        else:
            return outputs, None


def save_structural_semantic_encoder(
    model: StructuralSemanticEncoder,
    path: str,
    meta: Dict[str, Any]
):
    """Save StructuralSemanticEncoder to checkpoint."""
    checkpoint = {
        "state_dict": model.state_dict(),
        "meta": meta
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    logger.info(f"Saved StructuralSemanticEncoder to {path}")


def load_structural_semantic_encoder(
    path: str,
    map_location: str = "cpu"
) -> Tuple[StructuralSemanticEncoder, Dict[str, Any]]:
    """Load StructuralSemanticEncoder from checkpoint."""
    if map_location == "auto":
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint = torch.load(path, map_location=map_location)
    meta = checkpoint.get("meta", {})
    
    model = StructuralSemanticEncoder(
        input_dim=meta["input_dim"],
        latent_dim=meta.get("latent_dim", 512),
        structural_dim=meta.get("structural_dim", 256),
        semantic_dim=meta.get("semantic_dim", 512),
        n_blocks=meta.get("n_blocks", 4),
        dropout=meta.get("dropout", 0.3),
        head_hidden_dim=meta.get("head_hidden_dim", 512),
        predict_text_clip=meta.get("predict_text_clip", False),
        probabilistic=meta.get("probabilistic", False),
        kl_weight=meta.get("kl_weight", 0.01)
    )
    
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    logger.info(f"Loaded StructuralSemanticEncoder from {path}")
    
    return model, meta
