"""
Improved MLP Encoder with Residual Connections
===============================================

State-of-the-art architecture for fMRI → CLIP embedding mapping based on:
- Ozcelik et al. (2023): Brain-Diffuser
- Takagi & Nishimoto (2023): High-resolution image reconstruction
- Gu et al. (2023): Decoding natural images with deep learning

Key improvements over baseline:
1. Residual connections for deeper networks
2. LayerNorm for training stability
3. Multi-objective loss (cosine + MSE + triplet)
4. Configurable depth and width
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple


class ResidualBlock(nn.Module):
    """
    Residual block with pre-activation and layer normalization.
    
    Architecture:
        LayerNorm → ReLU → Linear → Dropout → LayerNorm → ReLU → Linear → Dropout
        + skip connection
    
    Args:
        in_dim: Input dimensionality
        out_dim: Output dimensionality
        dropout: Dropout probability
    """
    
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection (project if dimensions don't match)
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        # First sub-block
        out = self.norm1(x)
        out = F.relu(out, inplace=True)
        out = self.fc1(out)
        out = self.dropout(out)
        
        # Second sub-block
        out = self.norm2(out)
        out = F.relu(out, inplace=True)
        out = self.fc2(out)
        out = self.dropout(out)
        
        # Add skip connection
        return out + identity


class ImprovedMLPEncoder(nn.Module):
    """
    Improved MLP encoder with residual connections for fMRI → CLIP mapping.
    
    Architecture:
        Input → Projection → [ResidualBlock × N] → Output → LayerNorm → L2-normalize
    
    Args:
        input_dim: Input feature dimensionality (after preprocessing)
        hidden: List of hidden layer sizes (default: [2048, 2048, 1024])
        dropout: Dropout probability (default: 0.3)
        output_dim: Output dimension (default: 512 for CLIP ViT-B/32)
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden: List[int] = [2048, 2048, 1024],
        dropout: float = 0.3,
        output_dim: int = 512
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden = hidden
        self.dropout = dropout
        self.output_dim = output_dim
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.LayerNorm(hidden[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList()
        for i in range(len(hidden) - 1):
            self.blocks.append(ResidualBlock(hidden[i], hidden[i+1], dropout))
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden[-1], output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with L2-normalization.
        
        Args:
            x: Input features (B, input_dim)
        
        Returns:
            Normalized CLIP embeddings (B, output_dim)
        """
        # Project input
        x = self.input_proj(x)
        
        # Apply residual blocks
        for block in self.blocks:
            x = block(x)
        
        # Project to output dimension
        x = self.output_proj(x)
        
        # L2 normalize (critical for CLIP space)
        x = F.normalize(x, dim=-1)
        
        return x


class TripletLoss(nn.Module):
    """
    Triplet loss for metric learning in CLIP space.
    
    Encourages predicted embeddings to be:
    - Close to ground truth (positive)
    - Far from other samples (negatives)
    
    Args:
        margin: Margin for triplet loss (default: 0.2)
    """
    
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            pred: Predicted embeddings (B, D)
            target: Ground truth embeddings (B, D)
        
        Returns:
            Triplet loss (scalar)
        """
        batch_size = pred.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=pred.device)
        
        # Normalize embeddings
        pred_norm = F.normalize(pred, dim=-1)
        target_norm = F.normalize(target, dim=-1)
        
        # Positive pair distance (1 - cosine similarity)
        pos_sim = (pred_norm * target_norm).sum(dim=-1)
        pos_dist = 1 - pos_sim
        
        # Compute similarity matrix for negatives
        sim_matrix = torch.mm(pred_norm, target_norm.t())
        
        # Mask diagonal (positive pairs)
        mask = torch.eye(batch_size, device=pred.device, dtype=torch.bool)
        sim_matrix.masked_fill_(mask, -1e9)
        
        # Hardest negative (highest similarity)
        neg_sim, _ = sim_matrix.max(dim=-1)
        neg_dist = 1 - neg_sim
        
        # Triplet loss with margin
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        return loss.mean()


def advanced_loss(
    pred: torch.Tensor, 
    target: torch.Tensor,
    mse_weight: float = 0.3,
    triplet_weight: float = 0.2,
    triplet_margin: float = 0.2
) -> torch.Tensor:
    """
    Multi-objective loss combining cosine, MSE, and triplet losses.
    
    Args:
        pred: Predicted embeddings (B, D)
        target: Ground truth embeddings (B, D)
        mse_weight: Weight for MSE loss (default: 0.3)
        triplet_weight: Weight for triplet loss (default: 0.2)
        triplet_margin: Margin for triplet loss (default: 0.2)
    
    Returns:
        Combined loss (scalar)
    """
    # Cosine loss (primary objective)
    cosine_sim = (pred * target).sum(dim=-1)
    cos_loss = (1 - cosine_sim).mean()
    
    # MSE loss (magnitude alignment)
    mse_loss = F.mse_loss(pred, target)
    
    # Triplet loss (metric learning)
    triplet_loss_fn = TripletLoss(margin=triplet_margin)
    triplet_loss = triplet_loss_fn(pred, target)
    
    # Combine losses
    total_loss = (
        (1 - mse_weight - triplet_weight) * cos_loss +
        mse_weight * mse_loss +
        triplet_weight * triplet_loss
    )
    
    return total_loss


def save_improved_mlp(model: ImprovedMLPEncoder, path: str, meta: Dict):
    """Save improved MLP model with metadata."""
    torch.save({
        "state_dict": model.state_dict(),
        "meta": {
            **meta,
            "architecture": "ImprovedMLPEncoder",
            "hidden": model.hidden,
            "dropout": model.dropout,
            "output_dim": model.output_dim
        }
    }, path)


def load_improved_mlp(path: str, map_location: str = "cpu") -> Tuple[ImprovedMLPEncoder, Dict]:
    """
    Load improved MLP model from checkpoint.
    
    Args:
        path: Checkpoint path
        map_location: Device to load to
    
    Returns:
        model: Loaded ImprovedMLPEncoder
        meta: Metadata dictionary
    """
    checkpoint = torch.load(path, map_location=map_location)
    meta = checkpoint.get("meta", {})
    
    # Reconstruct model from metadata
    model = ImprovedMLPEncoder(
        input_dim=meta["input_dim"],
        hidden=meta.get("hidden", [2048, 2048, 1024]),
        dropout=meta.get("dropout", 0.3),
        output_dim=meta.get("output_dim", 512)
    )
    
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    
    return model, meta


# For backwards compatibility, keep original MLPEncoder
class MLPEncoder(nn.Module):
    """
    Original single-layer MLP encoder (maintained for compatibility).
    
    Use ImprovedMLPEncoder for better performance.
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
            nn.Linear(hidden, 512)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return F.normalize(x, dim=-1)
