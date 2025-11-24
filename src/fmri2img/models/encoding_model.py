"""
Encoding Model: Image → fMRI PCA
================================

Trains a model to predict fMRI PCA vectors from images.
Used for BOI-lite refinement (selecting candidates that best match brain activity).

Architecture:
- Use pretrained vision encoder (CLIP, DINO, or ResNet)
- Add regression head to predict PCA components
- Train on same NSD training set as decoder

Scientific Rationale:
- Encoding model provides feedback signal for generative refinement
- Brain-Diffuser (Ozcelik et al. 2023) uses similar approach for BOI
- Complements decoder: decoder predicts images, encoder evaluates brain alignment

References:
- Brain-Diffuser (Ozcelik et al. 2023): Brain-Optimized Inference
- Takagi & Nishimoto (2023): Iterative refinement with encoding models
"""

import logging
from typing import Optional, Literal, Dict, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)


class ImageEncoder(nn.Module):
    """
    Image encoder for extracting visual features.
    
    Supports multiple pretrained backbones:
    - CLIP ViT (clip_vit_b32, clip_vit_l14)
    - DINO (dino_vits16, dino_vitb16)
    - ResNet (resnet50, resnet101)
    
    Args:
        backbone: Pretrained backbone name
        freeze: Whether to freeze backbone weights
    """
    
    def __init__(
        self,
        backbone: str = "clip_vit_b32",
        freeze: bool = True
    ):
        super().__init__()
        self.backbone_name = backbone
        self.freeze = freeze
        
        if backbone == "clip_vit_b32":
            # Use CLIP ViT-B/32
            try:
                import open_clip
                model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
                self.backbone = model.visual
                self.feature_dim = 768  # ViT-B/32 hidden dim
                self.preprocess = preprocess
            except ImportError:
                raise ImportError("open_clip_torch required. Install: pip install open-clip-torch")
        
        elif backbone == "clip_vit_l14":
            # Use CLIP ViT-L/14 (larger)
            try:
                import open_clip
                model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
                self.backbone = model.visual
                self.feature_dim = 1024
                self.preprocess = preprocess
            except ImportError:
                raise ImportError("open_clip_torch required")
        
        elif backbone.startswith("dino"):
            # Use DINO (self-supervised ViT)
            try:
                import torch.hub
                if backbone == "dino_vits16":
                    self.backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
                    self.feature_dim = 384
                elif backbone == "dino_vitb16":
                    self.backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
                    self.feature_dim = 768
                
                # DINO preprocessing
                self.preprocess = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            except Exception as e:
                raise ImportError(f"DINO loading failed: {e}")
        
        elif backbone.startswith("resnet"):
            # Use torchvision ResNet
            import torchvision.models as models
            if backbone == "resnet50":
                resnet = models.resnet50(pretrained=True)
                self.feature_dim = 2048
            elif backbone == "resnet101":
                resnet = models.resnet101(pretrained=True)
                self.feature_dim = 2048
            else:
                raise ValueError(f"Unknown ResNet variant: {backbone}")
            
            # Remove final FC layer
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            
            # ResNet preprocessing
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Freeze backbone if requested
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info(f"Froze {backbone} backbone weights")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images.
        
        Args:
            x: Image tensor (B, 3, H, W), preprocessed
        
        Returns:
            Features (B, feature_dim)
        """
        with torch.set_grad_enabled(not self.freeze):
            features = self.backbone(x)
        
        # Handle different output formats
        if features.dim() > 2:
            features = features.flatten(1)  # (B, D)
        
        return features


class EncodingModel(nn.Module):
    """
    Complete encoding model: Image → fMRI PCA.
    
    Architecture:
        Pretrained vision encoder → Regression head → PCA predictions
    
    Args:
        output_dim: Output dimensionality (number of PCA components)
        backbone: Pretrained backbone name
        freeze_backbone: Whether to freeze backbone
        hidden_dim: Hidden dimension for regression head
        dropout: Dropout probability
    
    Example:
        >>> # Create encoding model
        >>> model = EncodingModel(output_dim=512, backbone="clip_vit_b32")
        >>> 
        >>> # Forward pass
        >>> img_tensor = preprocess_image(pil_image)
        >>> pred_fmri = model(img_tensor.unsqueeze(0))  # (1, 512)
    """
    
    def __init__(
        self,
        output_dim: int,
        backbone: str = "clip_vit_b32",
        freeze_backbone: bool = True,
        hidden_dim: int = 1024,
        dropout: float = 0.3
    ):
        super().__init__()
        self.output_dim = output_dim
        
        # Image encoder
        self.encoder = ImageEncoder(backbone, freeze=freeze_backbone)
        
        # Regression head
        self.head = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict fMRI PCA from image.
        
        Args:
            x: Image tensor (B, 3, H, W), preprocessed
        
        Returns:
            Predicted fMRI PCA (B, output_dim)
        """
        features = self.encoder(x)  # (B, feature_dim)
        pred_fmri = self.head(features)  # (B, output_dim)
        return pred_fmri
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess PIL image for encoding.
        
        Args:
            image: PIL Image
        
        Returns:
            Preprocessed tensor (1, 3, H, W)
        """
        return self.encoder.preprocess(image).unsqueeze(0)
    
    def predict(self, image: Image.Image) -> np.ndarray:
        """
        Predict fMRI PCA from PIL image (convenience method).
        
        Args:
            image: PIL Image
        
        Returns:
            Predicted fMRI PCA (output_dim,)
        """
        self.eval()
        with torch.no_grad():
            x = self.preprocess(image).to(next(self.parameters()).device)
            pred = self.forward(x)
            return pred.cpu().numpy().squeeze(0)


def save_encoding_model(
    model: EncodingModel,
    path: str,
    meta: Dict
) -> None:
    """Save encoding model with metadata."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "state_dict": model.state_dict(),
        "meta": meta,
        "model_type": "encoding_model"
    }
    
    torch.save(checkpoint, path)
    logger.info(f"Saved EncodingModel to {path}")


def load_encoding_model(
    path: str,
    map_location: str = "cpu"
) -> Tuple[EncodingModel, Dict]:
    """Load encoding model from checkpoint."""
    if map_location == "auto":
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint = torch.load(path, map_location=map_location)
    meta = checkpoint.get("meta", {})
    
    # Reconstruct model
    model = EncodingModel(
        output_dim=meta["output_dim"],
        backbone=meta.get("backbone", "clip_vit_b32"),
        freeze_backbone=meta.get("freeze_backbone", True),
        hidden_dim=meta.get("hidden_dim", 1024),
        dropout=meta.get("dropout", 0.3)
    )
    
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    logger.info(f"Loaded EncodingModel from {path}")
    
    return model, meta
