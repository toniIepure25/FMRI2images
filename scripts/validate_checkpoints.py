#!/usr/bin/env python3
"""
Validate all checkpoints to find the last healthy one before NaN corruption.
"""

import torch
from pathlib import Path
import sys

checkpoint_dir = Path("/bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/runs/20260113_234334_ultimate_novel_subj01/checkpoints")

print("=" * 100)
print("CHECKPOINT VALIDATION - Finding Last Healthy Checkpoint")
print("=" * 100)

for epoch in range(1, 8):
    checkpoint_path = checkpoint_dir / f"epoch_{epoch:02d}.pt"
    
    if not checkpoint_path.exists():
        print(f"❌ Epoch {epoch:02d}: File not found")
        continue
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Check model weights for NaN/Inf
        model_state = checkpoint['model_state_dict']
        has_nan = False
        has_inf = False
        
        for name, param in model_state.items():
            if torch.isnan(param).any():
                has_nan = True
                print(f"⚠️  Epoch {epoch:02d}: NaN found in {name}")
                break
            if torch.isinf(param).any():
                has_inf = True
                print(f"⚠️  Epoch {epoch:02d}: Inf found in {name}")
                break
        
        if has_nan or has_inf:
            print(f"❌ Epoch {epoch:02d}: CORRUPTED (contains NaN/Inf)")
        else:
            # Check optimizer state too
            optimizer_state = checkpoint.get('optimizer_state_dict', {})
            opt_corrupt = False
            
            if 'state' in optimizer_state:
                for state in optimizer_state['state'].values():
                    if 'exp_avg' in state and torch.isnan(state['exp_avg']).any():
                        opt_corrupt = True
                        break
                    if 'exp_avg_sq' in state and torch.isnan(state['exp_avg_sq']).any():
                        opt_corrupt = True
                        break
            
            if opt_corrupt:
                print(f"⚠️  Epoch {epoch:02d}: Model weights OK, but optimizer state CORRUPTED")
            else:
                print(f"✅ Epoch {epoch:02d}: HEALTHY (model + optimizer)")
                
        # Print metrics
        metrics = checkpoint.get('metrics', {})
        recon = metrics.get('recon_loss', 'N/A')
        kl = metrics.get('kl_loss', 'N/A')
        print(f"   Metrics: recon={recon}, kl={kl}")
        
    except Exception as e:
        print(f"❌ Epoch {epoch:02d}: ERROR loading - {e}")
    
    print()

print("=" * 100)
print("RECOMMENDATION: Resume from the last HEALTHY checkpoint")
print("=" * 100)
