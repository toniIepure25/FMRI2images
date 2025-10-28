#!/usr/bin/env python3
"""
Repair Adapter Metadata Script
===============================

Backfills missing metadata in CLIP adapter checkpoints for compatibility
with one-click reconstruction workflow.

Problem:
- Old adapter checkpoints use "meta" key (training convention)
- Orchestrator expects "metadata" key with "target_dim" field
- This script harmonizes the checkpoint format

Solution:
1. Load checkpoint
2. Check if metadata is missing or incomplete
3. Infer dimensions from projection layer weights
4. Add complete metadata with standard keys
5. Save backup and write repaired checkpoint

Usage:
    # Repair single adapter
    python scripts/repair_adapter_metadata.py
    
    # Or via Makefile
    make repair-adapter
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def repair_adapter_checkpoint(
    ckpt_path: Path,
    subject: str = "subj01",
    model_id: str = "stabilityai/stable-diffusion-2-1"
) -> None:
    """
    Repair adapter checkpoint by adding/harmonizing metadata.
    
    Args:
        ckpt_path: Path to adapter checkpoint
        subject: Subject identifier (default: subj01)
        model_id: Diffusion model identifier
    """
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)
    
    logger.info(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    # Check if repair is needed
    needs_repair = False
    
    if "metadata" not in ckpt:
        logger.info("✓ Missing 'metadata' key - will add")
        needs_repair = True
    elif "target_dim" not in ckpt["metadata"]:
        logger.info("✓ Missing 'target_dim' in metadata - will add")
        needs_repair = True
    else:
        logger.info("✅ Checkpoint already has complete metadata")
        logger.info(f"   target_dim={ckpt['metadata']['target_dim']}")
        logger.info("   No repair needed")
        return
    
    # Infer dimensions from model weights
    if "state_dict" not in ckpt:
        logger.error("ERROR: Checkpoint missing 'state_dict' - cannot infer dimensions")
        sys.exit(1)
    
    # The linear projection weight has shape [out_dim, in_dim]
    projection_weight = ckpt["state_dict"].get("linear.weight")
    if projection_weight is None:
        logger.error("ERROR: Cannot find 'linear.weight' in state_dict")
        sys.exit(1)
    
    out_dim, in_dim = projection_weight.shape
    logger.info(f"✓ Inferred from projection weight: {in_dim}D → {out_dim}D")
    
    # Check if there's existing "meta" key (training convention)
    existing_meta = ckpt.get("meta", {})
    use_layernorm = existing_meta.get("use_layernorm", True)
    
    # Build complete metadata
    metadata = {
        "format_version": 1,
        "created_at": datetime.now().isoformat(),
        "subject": subject,
        "model_id": model_id,
        "in_dim": in_dim,
        "target_dim": out_dim,  # Key field for orchestrator
        "out_dim": out_dim,      # Alias for compatibility
        "use_layernorm": use_layernorm,
        "notes": "Backfilled metadata by repair_adapter_metadata.py"
    }
    
    # Preserve any additional fields from existing meta
    for key, value in existing_meta.items():
        if key not in metadata:
            metadata[key] = value
    
    # Add metadata to checkpoint
    ckpt["metadata"] = metadata
    
    # Also keep "meta" for backward compatibility
    if "meta" not in ckpt:
        ckpt["meta"] = metadata
    
    # Create backup
    backup_path = ckpt_path.with_suffix(".pt.bak")
    logger.info(f"Creating backup: {backup_path}")
    shutil.copy2(ckpt_path, backup_path)
    
    # Save repaired checkpoint
    logger.info(f"Saving repaired checkpoint to {ckpt_path}")
    torch.save(ckpt, ckpt_path)
    
    logger.info("✅ Repair complete!")
    logger.info(f"   in_dim: {in_dim}")
    logger.info(f"   target_dim: {out_dim}")
    logger.info(f"   subject: {subject}")
    logger.info(f"   model_id: {model_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Repair CLIP adapter checkpoint metadata"
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path("checkpoints/clip_adapter/subj01/adapter.pt"),
        help="Path to adapter checkpoint (default: checkpoints/clip_adapter/subj01/adapter.pt)"
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="subj01",
        help="Subject identifier (default: subj01)"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="stabilityai/stable-diffusion-2-1",
        help="Diffusion model identifier (default: stabilityai/stable-diffusion-2-1)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("CLIP ADAPTER METADATA REPAIR")
    logger.info("=" * 80)
    
    repair_adapter_checkpoint(args.adapter, args.subject, args.model_id)
    
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
