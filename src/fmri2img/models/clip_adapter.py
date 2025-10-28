"""
CLIP Adapter for Dimension Alignment
=====================================

Lightweight trainable adapter to map 512-D CLIP embeddings (ViT-B/32) to higher
dimensions required by diffusion models (768-D for SD-1.5, 1024-D for SD-2.1).

Scientific Design:
- Linear projection with optional LayerNorm for stable training
- L2-normalized outputs maintain cosine similarity metric in target CLIP space
- Trained with MSE + cosine loss on ground-truth CLIP pairs
- Reduces representation gap between encoder output and diffusion conditioning

Usage:
    # Training
    adapter = CLIPAdapter(in_dim=512, out_dim=1024, use_layernorm=True)
    pred_512d = encoder(fmri)
    target_1024d = diffusion_clip(images)
    loss = mse_loss(adapter(pred_512d), target_1024d)
    
    # Inference
    adapter.load(checkpoint_path)
    adapted_emb = adapter(pred_512d)
    images = diffusion_pipeline(adapted_emb)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Tuple, Optional


class CLIPAdapter(nn.Module):
    """
    Lightweight adapter for CLIP embedding dimension alignment.
    
    Maps 512-D embeddings (ViT-B/32) to target dimension (768/1024) for diffusion
    model compatibility. Preserves angular relationships in CLIP space.
    
    Architecture:
        Linear(in_dim, out_dim) → [LayerNorm(out_dim)] → L2-normalize
    
    Args:
        in_dim: Input dimension (default: 512, ViT-B/32)
        out_dim: Output dimension (768 for SD-1.5, 1024 for SD-2.1)
        use_layernorm: Apply LayerNorm before normalization (default: True)
    """
    
    def __init__(
        self,
        in_dim: int = 512,
        out_dim: int = 1024,
        use_layernorm: bool = True
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_layernorm = use_layernorm
        
        # Linear projection
        self.linear = nn.Linear(in_dim, out_dim)
        
        # Optional layer normalization for training stability
        self.layernorm = nn.LayerNorm(out_dim) if use_layernorm else None
        
        # Initialize weights (Xavier/Glorot for better convergence)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project and normalize embeddings to target dimension.
        
        Args:
            x: Input embeddings (B, in_dim)
        
        Returns:
            z: L2-normalized embeddings (B, out_dim)
        """
        # Linear projection
        z = self.linear(x)  # (B, out_dim)
        
        # Optional layer normalization
        if self.layernorm is not None:
            z = self.layernorm(z)
        
        # L2 normalization for cosine similarity metric
        z = F.normalize(z, dim=-1)
        
        return z
    
    def save(self, path: str, meta: Optional[Dict] = None) -> None:
        """
        Save adapter checkpoint with metadata.
        
        Args:
            path: Output checkpoint path
            meta: Optional metadata dictionary (training info, dataset, etc.)
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Build checkpoint with architecture info
        checkpoint = {
            "state_dict": self.state_dict(),
            "meta": {
                "in_dim": self.in_dim,
                "out_dim": self.out_dim,
                "use_layernorm": self.use_layernorm,
                **(meta or {})
            }
        }
        
        torch.save(checkpoint, path)
    
    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> Tuple["CLIPAdapter", Dict]:
        """
        Load adapter from checkpoint.
        
        Args:
            path: Checkpoint path
            map_location: Device to load to (default: "cpu", accepts "auto")
        
        Returns:
            adapter: Loaded CLIPAdapter
            meta: Metadata dictionary
        """
        # Resolve 'auto' to actual device
        if map_location == "auto":
            map_location = "cuda" if torch.cuda.is_available() else "cpu"
        
        checkpoint = torch.load(path, map_location=map_location)
        meta = checkpoint.get("meta", {})
        
        # Reconstruct model from metadata
        adapter = cls(
            in_dim=meta.get("in_dim", 512),
            out_dim=meta.get("out_dim", 1024),
            use_layernorm=meta.get("use_layernorm", True)
        )
        
        adapter.load_state_dict(checkpoint["state_dict"], strict=True)
        
        return adapter, meta


def save_adapter(adapter: CLIPAdapter, path: str, meta: Dict) -> None:
    """
    Convenience function to save adapter with metadata.
    
    Args:
        adapter: Trained CLIPAdapter
        path: Output checkpoint path
        meta: Metadata dictionary
    """
    adapter.save(path, meta)


def load_adapter(path: str, map_location: str = "cpu") -> Tuple[CLIPAdapter, Dict]:
    """
    Convenience function to load adapter from checkpoint.
    
    Args:
        path: Checkpoint path
        map_location: Device to load to (default: "cpu")
    
    Returns:
        adapter: Loaded CLIPAdapter
        meta: Metadata dictionary
    """
    return CLIPAdapter.load(path, map_location)
