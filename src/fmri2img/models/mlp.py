"""
MLP Encoder for fMRI → CLIP Embedding Mapping
==============================================

Lightweight feedforward network with L2-normalized outputs for alignment
with CLIP embedding space.

Scientific Design:
- Outputs are L2-normalized so cosine is a proper similarity metric in CLIP space
- Single hidden layer with ReLU activation (standard for regression tasks)
- Dropout for regularization (prevents overfitting on high-dimensional fMRI)
- Combined cosine+MSE loss aligns both direction and magnitude with CLIP embeddings
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Tuple


class MLPEncoder(nn.Module):
    """
    Multilayer perceptron encoder for fMRI → CLIP embedding mapping.
    
    Architecture:
        Linear(input_dim, hidden) → ReLU → Dropout → Linear(hidden, 512) → L2-normalize
    
    Args:
        input_dim: Input feature dimensionality (after preprocessing)
        hidden: Hidden layer size (default: 1024)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, input_dim: int, hidden: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden = hidden
        self.dropout = dropout
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 512)  # CLIP ViT-B/32 embedding dimension
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with L2 normalization.
        
        Outputs are L2-normalized so cosine is a proper similarity metric
        in CLIP space (standard practice for CLIP alignment).
        
        Args:
            x: Input features (B, input_dim)
        
        Returns:
            z: L2-normalized CLIP embeddings (B, 512)
        """
        z = self.net(x)  # (B, 512)
        z = torch.nn.functional.normalize(z, dim=-1)  # Unit sphere for cosine
        return z


def save_mlp(model: MLPEncoder, path: str, meta: Dict) -> None:
    """
    Save MLP model with metadata.
    
    Args:
        model: Trained MLPEncoder
        path: Output checkpoint path
        meta: Metadata dictionary (input_dim, hidden, dropout, training info)
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "state_dict": model.state_dict(),
        "meta": meta
    }
    
    torch.save(checkpoint, path)


def _resolve_map_location(location: str) -> str:
    """
    Resolve 'auto' device to 'cuda' or 'cpu'.
    
    Args:
        location: Device string ('auto', 'cuda', 'cpu', etc.)
    
    Returns:
        Resolved device string
    """
    if location == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return location


def load_mlp(path: str, map_location: str = "cpu") -> Tuple[MLPEncoder, Dict]:
    """
    Load MLP model from checkpoint.
    
    Args:
        path: Checkpoint path
        map_location: Device to load to (default: "cpu", accepts "auto")
    
    Returns:
        model: Loaded MLPEncoder
        meta: Metadata dictionary
    """
    # Resolve 'auto' to actual device
    map_location = _resolve_map_location(map_location)
    
    checkpoint = torch.load(path, map_location=map_location)
    meta = checkpoint.get("meta", {})
    
    # Reconstruct model from metadata
    model = MLPEncoder(
        input_dim=meta["input_dim"],
        hidden=meta.get("hidden", 1024),
        dropout=meta.get("dropout", 0.1)
    )
    
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    
    return model, meta
