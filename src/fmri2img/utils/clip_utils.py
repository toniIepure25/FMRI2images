"""
CLIP Model Utilities
===================

Centralized CLIP model loading and configuration.
Single source of truth: configs/clip.yaml
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Tuple, Any
import numpy as np
import yaml

log = logging.getLogger(__name__)

# Import CLIP
try:
    import torch
    import open_clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


def load_clip_config(config_path: str = "configs/clip.yaml") -> dict:
    """
    Load CLIP configuration from YAML file.
    
    Args:
        config_path: Path to clip.yaml config file
        
    Returns:
        Dictionary with CLIP configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"CLIP config not found at {config_path}. "
            "Create configs/clip.yaml with model_name and other settings."
        )
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    required_fields = ['model_name', 'embedding_dim']
    missing = [f for f in required_fields if f not in config]
    if missing:
        raise ValueError(
            f"CLIP config missing required fields: {missing}. "
            f"Check {config_path}"
        )
    
    return config


def load_clip_model(
    config_path: str = "configs/clip.yaml",
    device: str = None
) -> Tuple[Any, Any, dict]:
    """
    Load CLIP model from configuration.
    
    Args:
        config_path: Path to clip.yaml config file
        device: Device override (cuda/cpu). If None, uses config default.
        
    Returns:
        Tuple of (model, preprocess_fn, config_dict)
        
    Raises:
        ImportError: If CLIP libraries not available
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    if not CLIP_AVAILABLE:
        raise ImportError(
            "CLIP libraries not available. "
            "Install with: pip install open-clip-torch torch"
        )
    
    # Load config
    config = load_clip_config(config_path)
    
    # Override device if provided
    if device is None:
        device = config.get('device', 'cuda')
    
    # Extract model settings
    model_name = config['model_name']
    pretrained = config.get('pretrained', 'openai')
    
    log.info(f"Loading CLIP model: {model_name} (pretrained={pretrained})")
    
    # Load model and preprocessing
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        model = model.to(device).eval()
        
        log.info(f"✓ CLIP model loaded on {device}")
        
    except Exception as e:
        raise RuntimeError(
            f"Failed to load CLIP model '{model_name}' with pretrained='{pretrained}': {e}"
        )
    
    return model, preprocess, config


def encode_images(
    model: Any,
    preprocess: Any,
    images: list,
    device: str = "cuda",
    normalize: bool = True
) -> np.ndarray:
    """
    Encode images to CLIP embeddings.
    
    Args:
        model: CLIP model
        preprocess: CLIP preprocessing function
        images: List of PIL Images
        device: Device for computation
        normalize: If True, L2-normalize embeddings
        
    Returns:
        (N, D) float32 array of embeddings (L2-normalized if normalize=True)
    """
    if not CLIP_AVAILABLE:
        raise ImportError("CLIP libraries not available")
    
    import torch
    from contextlib import nullcontext
    
    # Preprocess images
    imgs_tensor = torch.stack([preprocess(img) for img in images]).to(device)
    
    # Autocast context
    if device == "cuda" and torch.cuda.is_available():
        autocast_ctx = torch.amp.autocast("cuda")
    else:
        autocast_ctx = nullcontext()
    
    # Extract embeddings
    with torch.no_grad(), autocast_ctx:
        features = model.encode_image(imgs_tensor)
        
        # L2 normalize if requested
        if normalize:
            features = features / features.norm(dim=-1, keepdim=True)
    
    return features.cpu().numpy().astype(np.float32)


def encode_images_multilayer(
    model: Any,
    preprocess: Any,
    images: list,
    layers: list[int] = [4, 8, 12],
    device: str = "cuda",
    normalize: bool = True
) -> dict[str, np.ndarray]:
    """
    Encode images to multi-layer CLIP features.
    
    Extracts intermediate features from Vision Transformer (ViT) layers
    for multi-level supervision. Returns features from specified layers
    plus the final output.
    
    Args:
        model: CLIP model (must be ViT-based)
        preprocess: CLIP preprocessing function
        images: List of PIL Images
        layers: Layer indices to extract (e.g., [4, 8, 12] for ViT-B/32)
        device: Device for computation
        normalize: If True, L2-normalize all embeddings
        
    Returns:
        Dictionary mapping layer names to (N, D) float32 arrays:
        - 'layer_4': Features from 4th transformer block
        - 'layer_8': Features from 8th transformer block  
        - 'layer_12': Features from 12th transformer block
        - 'final': Final CLIP embeddings (after projection head)
        
    Note:
        ViT-B/32 has 12 transformer blocks. Common choices:
        - Early: layer 4 (low-level features)
        - Middle: layer 8 (mid-level features)
        - Late: layer 12 (high-level features)
        - Final: projection head output (semantic features)
    """
    if not CLIP_AVAILABLE:
        raise ImportError("CLIP libraries not available")
    
    import torch
    from contextlib import nullcontext
    
    # Preprocess images
    imgs_tensor = torch.stack([preprocess(img) for img in images]).to(device)
    
    # Autocast context
    if device == "cuda" and torch.cuda.is_available():
        autocast_ctx = torch.amp.autocast("cuda")
    else:
        autocast_ctx = nullcontext()
    
    # Extract multi-layer features
    features_dict = {}
    
    with torch.no_grad(), autocast_ctx:
        # Access visual encoder (ViT)
        visual = model.visual
        
        # Patch embedding + position embedding
        x = visual.conv1(imgs_tensor)  # (B, D, H, W)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (B, D, N)
        x = x.permute(0, 2, 1)  # (B, N, D)
        
        # Add class token
        class_token = visual.class_embedding.unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1)  # (B, 1, D)
        x = torch.cat([class_token, x], dim=1)  # (B, N+1, D)
        x = x + visual.positional_embedding
        
        # Pre-LayerNorm
        x = visual.ln_pre(x)
        
        # Transformer blocks with intermediate extraction
        for i, block in enumerate(visual.transformer.resblocks):
            x = block(x)
            
            # Extract features from specified layers
            if i + 1 in layers:  # +1 because i is 0-indexed
                # Take CLS token (first token)
                layer_feat = x[:, 0, :]  # (B, D)
                
                # Normalize if requested
                if normalize:
                    layer_feat = layer_feat / layer_feat.norm(dim=-1, keepdim=True)
                
                features_dict[f'layer_{i+1}'] = layer_feat.cpu().numpy().astype(np.float32)
        
        # Final projection head
        x = visual.ln_post(x[:, 0, :])
        if visual.proj is not None:
            x = x @ visual.proj
        
        # Normalize final output if requested
        if normalize:
            x = x / x.norm(dim=-1, keepdim=True)
        
        features_dict['final'] = x.cpu().numpy().astype(np.float32)
    
    return features_dict


def verify_embedding_dimension(
    embeddings: np.ndarray,
    config_path: str = "configs/clip.yaml"
) -> None:
    """
    Verify that embeddings match expected dimension from config.
    
    Args:
        embeddings: Array of embeddings (N, D)
        config_path: Path to clip.yaml config
        
    Raises:
        ValueError: If dimension mismatch
    """
    config = load_clip_config(config_path)
    expected_dim = config['embedding_dim']
    
    actual_dim = embeddings.shape[-1] if embeddings.ndim > 1 else embeddings.shape[0]
    
    if actual_dim != expected_dim:
        raise ValueError(
            f"CLIP embedding dimension mismatch!\n"
            f"  Expected: {expected_dim} (from {config_path})\n"
            f"  Got: {actual_dim}\n"
            f"  Model: {config.get('model_name', 'unknown')}\n"
            f"This usually means the CLIP model changed. "
            f"Rebuild cache with current config."
        )
