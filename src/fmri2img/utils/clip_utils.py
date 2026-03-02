"""
CLIP Model Utilities
===================

Centralized CLIP model loading and configuration.
Single source of truth: configs/system/clip.yaml
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


def load_clip_config(config_path: str = "configs/system/clip.yaml") -> dict:
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
            "Create configs/system/clip.yaml with model_name and other settings."
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
    config_path: str = "configs/system/clip.yaml",
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
    normalize: bool = True,
    project_intermediate: bool = False,
    fuse_alpha: float = 0.0,
) -> dict[str, np.ndarray]:
    """
    Encode images to multi-layer CLIP features.

    Extracts intermediate CLS-token features from Vision Transformer layers
    for multi-level supervision, plus the final projected output.

    Args:
        model: CLIP model (must be ViT-based).
        preprocess: CLIP preprocessing function.
        images: List of PIL Images.
        layers: Layer indices to extract.
            ViT-B/32 (12 blocks): [4, 8, 12]
            ViT-L/14 (24 blocks): [12, 18] recommended for semantic + detail.
        device: Device for computation.
        normalize: If True, L2-normalize all embeddings.
        project_intermediate: If True, also project intermediate 1024-D features
            through CLIP's projection head to produce 768-D ``layer_X_proj``
            columns.  Only meaningful when the model has a projection matrix
            (i.e., ViT-L/14 with 1024-D hidden, 768-D output).
        fuse_alpha: If > 0, create a ``fused`` column that blends the
            *projected* intermediate features with the final features:
            ``fused = alpha * intermed_proj + (1-alpha) * final``, L2-normed.
            Uses the *last* layer in ``layers`` for the intermediate input.
            Requires ``project_intermediate=True`` (set automatically).

    Returns:
        Dictionary mapping layer names to (N, D) float32 arrays.
        Always includes ``final`` (projected CLIP dim, e.g. 768-D).
        Intermediate keys are ``layer_X`` (hidden dim, e.g. 1024-D).
        If ``project_intermediate``, also ``layer_X_proj`` (768-D).
        If ``fuse_alpha > 0``, also ``fused`` (768-D).

    Note:
        Intermediate CLS tokens are in the ViT hidden dimension:
        - ViT-B/32: 768-D
        - ViT-L/14: 1024-D
        The ``final`` output is the projected CLIP embedding (768-D for ViT-L/14).
    """
    if not CLIP_AVAILABLE:
        raise ImportError("CLIP libraries not available")

    import torch
    from contextlib import nullcontext

    if fuse_alpha > 0:
        project_intermediate = True

    imgs_tensor = torch.stack([preprocess(img) for img in images]).to(device)

    if device == "cuda" and torch.cuda.is_available():
        autocast_ctx = torch.amp.autocast("cuda")
    else:
        autocast_ctx = nullcontext()

    features_dict = {}

    with torch.no_grad(), autocast_ctx:
        visual = model.visual

        x = visual.conv1(imgs_tensor)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        class_token = visual.class_embedding.unsqueeze(0).unsqueeze(0).expand(
            x.shape[0], -1, -1
        )
        x = torch.cat([class_token, x], dim=1)
        x = x + visual.positional_embedding

        x = visual.ln_pre(x)

        last_intermed_proj = None
        for i, block in enumerate(visual.transformer.resblocks):
            x = block(x)

            if i + 1 in layers:
                layer_feat = x[:, 0, :].float()  # (B, hidden_dim)

                if normalize:
                    layer_feat = layer_feat / layer_feat.norm(dim=-1, keepdim=True)

                features_dict[f'layer_{i+1}'] = (
                    layer_feat.cpu().numpy().astype(np.float32)
                )

                if project_intermediate and visual.proj is not None:
                    projected = layer_feat @ visual.proj.float()
                    if normalize:
                        projected = projected / projected.norm(
                            dim=-1, keepdim=True
                        )
                    features_dict[f'layer_{i+1}_proj'] = (
                        projected.cpu().numpy().astype(np.float32)
                    )
                    last_intermed_proj = projected

        x = visual.ln_post(x[:, 0, :]).float()
        if visual.proj is not None:
            x = x @ visual.proj.float()

        if normalize:
            x = x / x.norm(dim=-1, keepdim=True)

        final_np = x.cpu().numpy().astype(np.float32)
        features_dict['final'] = final_np

        if fuse_alpha > 0 and last_intermed_proj is not None:
            fused = fuse_alpha * last_intermed_proj + (1.0 - fuse_alpha) * x
            fused = fused / fused.norm(dim=-1, keepdim=True)
            features_dict['fused'] = fused.cpu().numpy().astype(np.float32)

    return features_dict


def verify_embedding_dimension(
    embeddings: np.ndarray,
    config_path: str = "configs/system/clip.yaml"
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
