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
        
        # Build checkpoint with architecture info and metadata
        checkpoint = {
            "state_dict": self.state_dict(),
            "metadata": {
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
        Load adapter from checkpoint with robust metadata handling.
        
        Handles legacy checkpoints by wrapping raw state_dicts and inferring
        missing metadata fields with sensible defaults.
        
        Args:
            path: Checkpoint path
            map_location: Device to load to (default: "cpu", accepts "auto")
        
        Returns:
            adapter: Loaded CLIPAdapter
            metadata: Metadata dictionary with required fields
        """
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Resolve 'auto' to actual device
        if map_location == "auto":
            map_location = "cuda" if torch.cuda.is_available() else "cpu"
        
        checkpoint = torch.load(path, map_location=map_location)
        
        # Handle raw state_dict (legacy format)
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            logger.info(f"Adapter metadata repaired: wrapping raw state_dict")
            checkpoint = {
                "state_dict": checkpoint if isinstance(checkpoint, dict) else checkpoint.state_dict(),
                "metadata": {}
            }
        
        # Get metadata (support both "metadata" and legacy "meta" keys)
        metadata = checkpoint.get("metadata", checkpoint.get("meta", {}))
        
        # Ensure required metadata fields with sensible defaults
        repaired_fields = []
        
        if "subject" not in metadata:
            metadata["subject"] = "unknown"
            repaired_fields.append("subject=unknown")
        
        if "model_id" not in metadata:
            metadata["model_id"] = "stabilityai/stable-diffusion-2-1"
            repaired_fields.append("model_id=stabilityai/stable-diffusion-2-1")
        
        if "input_dim" not in metadata:
            metadata["input_dim"] = metadata.get("in_dim", 512)
            repaired_fields.append(f"input_dim={metadata['input_dim']}")
        
        if "target_dim" not in metadata:
            metadata["target_dim"] = metadata.get("out_dim", 1024)
            repaired_fields.append(f"target_dim={metadata['target_dim']}")
        
        if "in_dim" not in metadata:
            metadata["in_dim"] = metadata["input_dim"]
        
        if "out_dim" not in metadata:
            metadata["out_dim"] = metadata["target_dim"]
        
        # Infer use_layernorm from state_dict if missing
        if "use_layernorm" not in metadata:
            # Check if layernorm weights exist in state_dict
            has_layernorm = "layernorm.weight" in checkpoint["state_dict"]
            metadata["use_layernorm"] = has_layernorm
            if has_layernorm:
                repaired_fields.append("use_layernorm=True (inferred from state_dict)")
            else:
                repaired_fields.append("use_layernorm=False (inferred from state_dict)")
        
        if repaired_fields:
            logger.info(f"Adapter metadata repaired: {{{', '.join(repaired_fields)}}}")
        
        # Reconstruct model from metadata
        adapter = cls(
            in_dim=metadata.get("in_dim", 512),
            out_dim=metadata.get("out_dim", 1024),
            use_layernorm=metadata.get("use_layernorm", True)
        )
        
        adapter.load_state_dict(checkpoint["state_dict"], strict=True)
        
        return adapter, metadata


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
    Load adapter from checkpoint with robust metadata handling.
    
    Handles legacy checkpoints and missing metadata gracefully. Logs information
    about metadata repairs and dimension configuration.
    
    Args:
        path: Checkpoint path
        map_location: Device to load to (default: "cpu")
    
    Returns:
        adapter: Loaded CLIPAdapter
        metadata: Metadata dictionary with required fields
    
    Raises:
        FileNotFoundError: If checkpoint path does not exist
    """
    import logging
    from pathlib import Path
    
    logger = logging.getLogger(__name__)
    
    # Check file exists
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Adapter checkpoint not found: {path}\n"
            f"Hint: Train an adapter first with scripts/train_clip_adapter.py"
        )
    
    adapter, metadata = CLIPAdapter.load(path, map_location)
    
    # Log loading summary
    target_dim = metadata.get("target_dim", metadata.get("out_dim", 1024))
    logger.info(f"Loaded adapter (target_dim={target_dim}) with metadata: "
               f"{{subject={metadata.get('subject', 'unknown')}, "
               f"model_id={metadata.get('model_id', 'unknown')}, "
               f"input_dim={metadata.get('input_dim', 512)}, "
               f"target_dim={target_dim}}}")
    
    return adapter, metadata
