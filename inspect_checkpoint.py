#!/usr/bin/env python3
"""
Inspect checkpoint structure to debug loading issues.
"""
import torch
import sys
from pathlib import Path

checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "best_model.pt"

print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location="cpu")

print("\n" + "="*60)
print("CHECKPOINT STRUCTURE")
print("="*60)

if isinstance(checkpoint, dict):
    print(f"\nCheckpoint is a dictionary with keys:")
    for key in checkpoint.keys():
        value = checkpoint[key]
        if isinstance(value, dict):
            print(f"  {key}: dict with {len(value)} keys")
            if len(value) < 20:
                for k in list(value.keys())[:5]:
                    print(f"    - {k}")
                if len(value) > 5:
                    print(f"    ... and {len(value) - 5} more")
        elif isinstance(value, torch.Tensor):
            print(f"  {key}: Tensor {value.shape}")
        else:
            print(f"  {key}: {type(value).__name__}")
    
    # Check for state_dict
    print("\n" + "="*60)
    print("LOOKING FOR STATE_DICT")
    print("="*60)
    
    state_dict = None
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        print("\n✓ Found 'state_dict' key")
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print("\n✓ Found 'model_state_dict' key")
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
        print("\n✓ Found 'model' key")
    else:
        print("\n✗ No standard state_dict key found")
        print("\nChecking if checkpoint itself is a state_dict...")
        # Check if keys look like model parameters
        keys = list(checkpoint.keys())
        if any("weight" in k or "bias" in k for k in keys[:10]):
            state_dict = checkpoint
            print("✓ Checkpoint appears to be a raw state_dict")
        else:
            print("✗ Checkpoint doesn't look like a state_dict")
    
    if state_dict is not None:
        print(f"\nState dict has {len(state_dict)} keys")
        print("\nFirst 20 keys:")
        for i, key in enumerate(list(state_dict.keys())[:20]):
            tensor = state_dict[key]
            if isinstance(tensor, torch.Tensor):
                print(f"  {i+1:2d}. {key:60s} {str(tensor.shape):20s}")
            else:
                print(f"  {i+1:2d}. {key:60s} {type(tensor).__name__}")
        
        if len(state_dict) > 20:
            print(f"\n  ... and {len(state_dict) - 20} more keys")
        
        # Look for projection layers
        print("\n" + "="*60)
        print("SEARCHING FOR PROJECTION LAYERS")
        print("="*60)
        
        projection_keys = [k for k in state_dict.keys() if "projection" in k.lower()]
        if projection_keys:
            print(f"\nFound {len(projection_keys)} projection-related keys:")
            for key in projection_keys[:10]:
                tensor = state_dict[key]
                if isinstance(tensor, torch.Tensor):
                    print(f"  {key:60s} {str(tensor.shape):20s}")
        else:
            print("\n✗ No projection layers found")
        
        # Look for any layer with weight that could indicate input_dim
        print("\n" + "="*60)
        print("SEARCHING FOR INPUT DIM HINTS")
        print("="*60)
        
        for key in list(state_dict.keys())[:30]:
            if "weight" in key and isinstance(state_dict[key], torch.Tensor):
                shape = state_dict[key].shape
                if len(shape) == 2:
                    print(f"  {key:60s} {str(shape):20s} <- input_dim could be {shape[1]}")
                    
else:
    print(f"\nCheckpoint is NOT a dictionary, it's a {type(checkpoint).__name__}")
    if isinstance(checkpoint, torch.Tensor):
        print(f"Shape: {checkpoint.shape}")

print("\n" + "="*60)
print("END OF INSPECTION")
print("="*60)
