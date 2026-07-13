"""Quick script to inspect a checkpoint and extract per-ROI kappas."""
import sys
sys.path.insert(0, "/home/jovyan/work/FMRI2images/src")

import json
import torch
import numpy as np
from pathlib import Path

# Step 1: Inspect available checkpoints
checkpoints = [
    "experimental_results/V66a_roi_pretrain/subj01/checkpoint_best.pt",
    "experimental_results/V57a_roi_transformer_dual_head/subj01/checkpoint_best.pt",
]

for ckpt_path in checkpoints:
    if not Path(ckpt_path).exists():
        continue
    print(f"\n=== {ckpt_path} ===")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"Keys: {list(ckpt.keys())[:10]}")
    if "config" in ckpt:
        cfg = ckpt["config"]
        model_cfg = cfg.get("model", {})
        print(f"Model type: {model_cfg.get('type', 'N/A')}")
        print(f"Encoder type: {model_cfg.get('encoder_type', 'N/A')}")
        print(json.dumps(model_cfg, indent=2, default=str)[:1000])
    elif "model_config" in ckpt:
        print(f"model_config: {json.dumps(ckpt['model_config'], indent=2, default=str)[:500]}")
    else:
        # Check state dict keys for clues
        if "model_state_dict" in ckpt:
            keys = list(ckpt["model_state_dict"].keys())[:15]
            print(f"State dict keys (first 15): {keys}")
        elif "state_dict" in ckpt:
            keys = list(ckpt["state_dict"].keys())[:15]
            print(f"State dict keys (first 15): {keys}")
