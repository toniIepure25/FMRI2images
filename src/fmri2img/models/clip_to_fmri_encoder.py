"""
CLIP → fMRI Encoder for Brain-Consistency Loss
==============================================

Inverse mapping: CLIP image embedding → fMRI PCA representation

This module enables cycle-consistency (brain-consistency) loss during fMRI→CLIP
decoder training:

    fMRI --[Decoder]--> CLIP_pred --[This Module]--> fMRI_reconstructed
    
    Loss = distance(fMRI_original, fMRI_reconstructed)

Scientific Rationale:
- Cycle-consistency has been successful in image translation (CycleGAN)
- Brain-consistency ensures predictions are brain-plausible
- Acts as regularization: predicted CLIP must map back to valid brain patterns
- Novel contribution: most papers only use forward mapping (fMRI→CLIP)

References:
- Zhu et al. (2017): "Unpaired Image-to-Image Translation using Cycle-Consistent GANs"
- Ozcelik et al. (2023): Brain-Diffuser uses image→fMRI for BOI refinement
- This work: Uses CLIP→fMRI for training-time cycle loss (novel)

Architecture Options:
1. Linear (Ridge-like): Simple, interpretable, parameter-efficient
2. MLP: More expressive, can capture nonlinear relationships
3. Residual MLP: Deep architecture with skip connections
"""

import torch
import torch.nn as nn
from typing import Literal, Optional, Dict, Any
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)


class CLIPToFMRIEncoder(nn.Module):
    """
    CLIP image embedding → fMRI PCA representation.
    
    Maps 512-D CLIP embeddings back to fMRI PCA space (e.g., 512-D).
    Used for cycle-consistency loss during decoder training.
    
    Args:
        clip_dim: CLIP embedding dimension (default: 512)
        fmri_dim: fMRI PCA dimension (default: 512)
        architecture: "linear", "mlp", or "residual"
        hidden_dim: Hidden dimension for MLP/residual (default: 1024)
        n_layers: Number of layers for residual (default: 2)
        dropout: Dropout probability (default: 0.2)
    
    Example:
        >>> # Train CLIP→fMRI encoder
        >>> encoder = CLIPToFMRIEncoder(clip_dim=512, fmri_dim=512)
        >>> 
        >>> # Training loop
        >>> for clip_emb, fmri_pca in dataloader:
        >>>     pred_fmri = encoder(clip_emb)
        >>>     loss = F.mse_loss(pred_fmri, fmri_pca)
        >>>     loss.backward()
        >>> 
        >>> # Use in decoder training for cycle loss
        >>> z_pred = decoder(fmri_pca)  # fMRI → CLIP
        >>> fmri_reconstructed = encoder(z_pred)  # CLIP → fMRI
        >>> cycle_loss = F.mse_loss(fmri_reconstructed, fmri_pca)
    """
    
    def __init__(
        self,
        clip_dim: int = 512,
        fmri_dim: int = 512,
        architecture: Literal["linear", "mlp", "residual"] = "mlp",
        hidden_dim: int = 1024,
        n_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.clip_dim = clip_dim
        self.fmri_dim = fmri_dim
        self.architecture = architecture
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        
        if architecture == "linear":
            # Simple linear projection (like Ridge regression)
            self.encoder = nn.Linear(clip_dim, fmri_dim)
        
        elif architecture == "mlp":
            # Two-layer MLP with GELU and dropout
            self.encoder = nn.Sequential(
                nn.Linear(clip_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, fmri_dim)
            )
        
        elif architecture == "residual":
            # Residual MLP with multiple blocks
            layers = [
                nn.Linear(clip_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ]
            
            # Residual blocks
            for _ in range(n_layers):
                layers.extend([
                    ResidualBlock(hidden_dim, dropout),
                ])
            
            # Output projection
            layers.extend([
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, fmri_dim)
            ])
            
            self.encoder = nn.Sequential(*layers)
        
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
    
    def forward(self, clip_emb: torch.Tensor) -> torch.Tensor:
        """
        Map CLIP embeddings to fMRI PCA space.
        
        Args:
            clip_emb: CLIP embeddings (B, 512), L2-normalized
        
        Returns:
            fmri_pred: Predicted fMRI PCA (B, fmri_dim)
        """
        return self.encoder(clip_emb)
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration for saving."""
        return {
            "clip_dim": self.clip_dim,
            "fmri_dim": self.fmri_dim,
            "architecture": self.architecture,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "dropout": self.dropout
        }


class ResidualBlock(nn.Module):
    """Residual block for deep CLIP→fMRI encoder."""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual


def save_clip_to_fmri_encoder(
    model: CLIPToFMRIEncoder,
    path: Path,
    metrics: Optional[Dict[str, float]] = None,
    subject: str = "subj01"
) -> None:
    """
    Save CLIP→fMRI encoder checkpoint.
    
    Args:
        model: CLIPToFMRIEncoder instance
        path: Path to save checkpoint
        metrics: Optional training metrics
        subject: Subject ID
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "state_dict": model.state_dict(),
        "config": model.get_config(),
        "subject": subject,
        "model_type": "clip_to_fmri_encoder"
    }
    
    if metrics:
        checkpoint["metrics"] = metrics
    
    torch.save(checkpoint, path)
    logger.info(f"Saved CLIP→fMRI encoder to {path}")


def load_clip_to_fmri_encoder(
    path: Path,
    device: str = "cuda"
) -> CLIPToFMRIEncoder:
    """
    Load CLIP→fMRI encoder from checkpoint.
    
    Args:
        path: Path to checkpoint
        device: Device to load model on
    
    Returns:
        CLIPToFMRIEncoder instance
    """
    checkpoint = torch.load(path, map_location=device)
    
    config = checkpoint["config"]
    model = CLIPToFMRIEncoder(**config)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded CLIP→fMRI encoder from {path}")
    logger.info(f"  Architecture: {config['architecture']}")
    logger.info(f"  CLIP dim: {config['clip_dim']}, fMRI dim: {config['fmri_dim']}")
    
    return model


def train_clip_to_fmri_encoder(
    X_clip: np.ndarray,
    Y_fmri: np.ndarray,
    architecture: str = "mlp",
    hidden_dim: int = 1024,
    learning_rate: float = 1e-3,
    batch_size: int = 256,
    epochs: int = 50,
    device: str = "cuda",
    validation_split: float = 0.1
) -> CLIPToFMRIEncoder:
    """
    Train CLIP→fMRI encoder from scratch.
    
    Convenience function for training without using a separate script.
    
    Args:
        X_clip: CLIP embeddings (N, 512)
        Y_fmri: fMRI PCA (N, fmri_dim)
        architecture: "linear", "mlp", or "residual"
        hidden_dim: Hidden dimension
        learning_rate: Learning rate
        batch_size: Batch size
        epochs: Number of epochs
        device: Device
        validation_split: Fraction for validation
    
    Returns:
        Trained CLIPToFMRIEncoder
    """
    from torch.utils.data import TensorDataset, DataLoader
    from tqdm import tqdm
    
    # Create model
    clip_dim = X_clip.shape[1]
    fmri_dim = Y_fmri.shape[1]
    model = CLIPToFMRIEncoder(
        clip_dim=clip_dim,
        fmri_dim=fmri_dim,
        architecture=architecture,
        hidden_dim=hidden_dim
    ).to(device)
    
    # Train/val split
    n_val = int(len(X_clip) * validation_split)
    n_train = len(X_clip) - n_val
    
    indices = np.random.permutation(len(X_clip))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    X_train, Y_train = X_clip[train_idx], Y_fmri[train_idx]
    X_val, Y_val = X_clip[val_idx], Y_fmri[val_idx]
    
    # DataLoaders
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(Y_train).float()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(Y_val).float()
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = nn.functional.mse_loss(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(x_batch)
        
        train_loss /= len(train_dataset)
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = model(x_batch)
                loss = nn.functional.mse_loss(y_pred, y_batch)
                val_loss += loss.item() * len(x_batch)
        
        val_loss /= len(val_dataset)
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    
    logger.info(f"Training complete! Best val loss: {best_val_loss:.4f}")
    return model
