#!/usr/bin/env python3
"""
Verification Script for Phase 1: Brain-Consistency Loss
=======================================================

Tests that all Phase 1 components are properly integrated.

Usage:
    python scripts/verify_phase1.py
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("PHASE 1: BRAIN-CONSISTENCY VERIFICATION")
print("=" * 70)
print()

# Test 1: Import ClipToFmriEncoder
print("✓ Test 1: Importing ClipToFmriEncoder...")
try:
    from fmri2img.models.clip_to_fmri_encoder import (
        CLIPToFMRIEncoder,
        save_clip_to_fmri_encoder,
        load_clip_to_fmri_encoder
    )
    print("  ✓ ClipToFmriEncoder module imported successfully")
except Exception as e:
    print(f"  ✗ Failed to import: {e}")
    sys.exit(1)

# Test 2: Create encoder
print("\n✓ Test 2: Creating ClipToFmriEncoder...")
try:
    import torch
    
    # Test linear architecture
    encoder_linear = CLIPToFMRIEncoder(
        clip_dim=512, fmri_dim=512, architecture="linear"
    )
    print(f"  ✓ Linear encoder: {sum(p.numel() for p in encoder_linear.parameters()):,} params")
    
    # Test MLP architecture
    encoder_mlp = CLIPToFMRIEncoder(
        clip_dim=512, fmri_dim=512, architecture="mlp", hidden_dim=1024
    )
    print(f"  ✓ MLP encoder: {sum(p.numel() for p in encoder_mlp.parameters()):,} params")
    
    # Test residual architecture
    encoder_residual = CLIPToFMRIEncoder(
        clip_dim=512, fmri_dim=512, architecture="residual", hidden_dim=1024, n_layers=2
    )
    print(f"  ✓ Residual encoder: {sum(p.numel() for p in encoder_residual.parameters()):,} params")
    
except Exception as e:
    print(f"  ✗ Failed to create encoder: {e}")
    sys.exit(1)

# Test 3: Forward pass
print("\n✓ Test 3: Testing forward pass...")
try:
    x = torch.randn(8, 512)  # Batch of CLIP embeddings
    y = encoder_mlp(x)
    assert y.shape == (8, 512), f"Expected (8, 512), got {y.shape}"
    print(f"  ✓ Forward pass successful: {x.shape} → {y.shape}")
except Exception as e:
    print(f"  ✗ Forward pass failed: {e}")
    sys.exit(1)

# Test 4: Brain-consistency loss function
print("\n✓ Test 4: Testing brain_consistency_loss...")
try:
    from fmri2img.training.losses import brain_consistency_loss
    
    clip_pred = torch.randn(8, 512)
    fmri_true = torch.randn(8, 512)
    encoder_mlp.eval()
    
    loss = brain_consistency_loss(clip_pred, fmri_true, encoder_mlp)
    assert loss.ndim == 0, f"Expected scalar, got shape {loss.shape}"
    print(f"  ✓ brain_consistency_loss works: loss={loss.item():.4f}")
except Exception as e:
    print(f"  ✗ brain_consistency_loss failed: {e}")
    sys.exit(1)

# Test 5: MultiLoss with brain-consistency
print("\n✓ Test 5: Testing MultiLoss with brain-consistency...")
try:
    from fmri2img.training.losses import MultiLoss
    
    encoder_mlp.eval()
    criterion = MultiLoss(
        mse_weight=0.3,
        cosine_weight=0.3,
        info_nce_weight=0.4,
        brain_consistency_weight=0.1,
        clip_to_fmri_encoder=encoder_mlp
    )
    
    pred = torch.randn(16, 512)
    pred = torch.nn.functional.normalize(pred, dim=-1)
    target = torch.randn(16, 512)
    target = torch.nn.functional.normalize(target, dim=-1)
    fmri_input = torch.randn(16, 512)
    
    loss, components = criterion(pred, target, fmri_input=fmri_input, return_components=True)
    
    assert "brain" in components, "Missing 'brain' in loss components"
    print(f"  ✓ MultiLoss with brain-consistency:")
    print(f"    - Total: {loss.item():.4f}")
    print(f"    - MSE: {components['mse']:.4f}")
    print(f"    - Cosine: {components['cosine']:.4f}")
    print(f"    - InfoNCE: {components['info_nce']:.4f}")
    print(f"    - Brain: {components['brain']:.4f}")
    
except Exception as e:
    print(f"  ✗ MultiLoss with brain-consistency failed: {e}")
    sys.exit(1)

# Test 6: Check training script exists
print("\n✓ Test 6: Checking training script...")
train_script = Path("scripts/train_clip_to_fmri.py")
if train_script.exists():
    print(f"  ✓ Training script exists: {train_script}")
else:
    print(f"  ✗ Training script not found: {train_script}")
    sys.exit(1)

# Test 7: Check config exists
print("\n✓ Test 7: Checking configuration...")
config_file = Path("configs/clip2fmri.yaml")
if config_file.exists():
    print(f"  ✓ Config file exists: {config_file}")
    
    import yaml
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    assert "encoder" in config, "Missing 'encoder' in config"
    assert "training" in config, "Missing 'training' in config"
    print(f"  ✓ Config valid with {len(config)} sections")
else:
    print(f"  ✗ Config not found: {config_file}")
    sys.exit(1)

# Test 8: Check documentation
print("\n✓ Test 8: Checking documentation...")
doc_file = Path("docs/PHASE1_BRAIN_CONSISTENCY.md")
if doc_file.exists():
    print(f"  ✓ Documentation exists: {doc_file}")
    content = doc_file.read_text()
    assert "PHASE 1" in content, "Missing 'PHASE 1' in docs"
    assert "brain-consistency" in content.lower(), "Missing 'brain-consistency' in docs"
    print(f"  ✓ Documentation complete ({len(content):,} chars)")
else:
    print(f"  ✗ Documentation not found: {doc_file}")
    sys.exit(1)

# Test 9: Check integration in main config
print("\n✓ Test 9: Checking integration in sota_two_stage.yaml...")
main_config = Path("configs/sota_two_stage.yaml")
if main_config.exists():
    with open(main_config) as f:
        config = yaml.safe_load(f)
    
    loss_config = config.get("loss", {})
    assert "brain_consistency_weight" in loss_config, "Missing 'brain_consistency_weight'"
    assert "clip_to_fmri_encoder" in loss_config, "Missing 'clip_to_fmri_encoder'"
    
    print(f"  ✓ Main config has brain-consistency support")
    print(f"    - brain_consistency_weight: {loss_config['brain_consistency_weight']}")
    print(f"    - clip_to_fmri_encoder: {loss_config['clip_to_fmri_encoder']}")
else:
    print(f"  ✗ Main config not found: {main_config}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ PHASE 1 VERIFICATION COMPLETE!")
print("=" * 70)
print("\nAll components properly integrated. Ready to use!")
print("\nNext steps:")
print("1. Train CLIP→fMRI encoder:")
print("   python scripts/train_clip_to_fmri.py --subject subj01")
print()
print("2. Update config with encoder path:")
print("   Edit configs/sota_two_stage.yaml:")
print("   loss:")
print("     brain_consistency_weight: 0.1")
print("     clip_to_fmri_encoder: 'checkpoints/clip_to_fmri/subj01/encoder.pt'")
print()
print("3. Train fMRI→CLIP with brain-consistency:")
print("   python scripts/train_two_stage.py --config configs/sota_two_stage.yaml")
print()
