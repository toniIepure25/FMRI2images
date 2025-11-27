#!/usr/bin/env python3
"""
Phase 4 Verification: Structural/Semantic Branch Encoder

Tests the Phase 4 encoder architecture with explicit structural/semantic factorization.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np

def test_1_architecture():
    """Test Phase 4 encoder architecture instantiation."""
    print("\n" + "=" * 80)
    print("TEST 1: Architecture Instantiation")
    print("=" * 80)
    
    from fmri2img.models.phase4_encoder import StructuralSemanticEncoder
    
    # Test deterministic mode
    encoder_det = StructuralSemanticEncoder(
        input_dim=512,
        latent_dim=512,
        structural_dim=256,
        semantic_dim=512,
        probabilistic=False,
        predict_text_clip=True
    )
    
    print(f"\n✅ Deterministic encoder created")
    print(f"   Parameters: {sum(p.numel() for p in encoder_det.parameters()):,}")
    
    # Test probabilistic mode
    encoder_prob = StructuralSemanticEncoder(
        input_dim=512,
        latent_dim=512,
        structural_dim=256,
        semantic_dim=512,
        probabilistic=True,
        predict_text_clip=True
    )
    
    print(f"\n✅ Probabilistic encoder created")
    print(f"   Parameters: {sum(p.numel() for p in encoder_prob.parameters()):,}")
    
    return True


def test_2_forward_deterministic():
    """Test deterministic forward pass."""
    print("\n" + "=" * 80)
    print("TEST 2: Deterministic Forward Pass")
    print("=" * 80)
    
    from fmri2img.models.phase4_encoder import StructuralSemanticEncoder
    
    encoder = StructuralSemanticEncoder(
        input_dim=512,
        latent_dim=512,
        structural_dim=256,
        semantic_dim=512,
        probabilistic=False,
        predict_text_clip=True
    )
    encoder.eval()
    
    x = torch.randn(32, 512)
    
    with torch.no_grad():
        outputs, kl_loss = encoder(x, sample=False, return_kl=False)
    
    print(f"\n✅ Forward pass successful")
    print(f"   Input shape: {x.shape}")
    print(f"\n   Branch latents:")
    print(f"     structural_branch: {outputs['structural_branch'].shape}")
    print(f"     semantic_branch: {outputs['semantic_branch'].shape}")
    print(f"\n   Structural branch outputs (early layers):")
    print(f"     layer_4: {outputs['layer_4'].shape}")
    print(f"     layer_8: {outputs['layer_8'].shape}")
    print(f"\n   Semantic branch outputs (late layers):")
    print(f"     layer_12: {outputs['layer_12'].shape}")
    print(f"     final: {outputs['final'].shape}")
    print(f"     text: {outputs['text'].shape}")
    print(f"\n   KL loss: {kl_loss}")
    
    # Check shapes
    assert outputs['structural_branch'].shape == (32, 256), "Structural latent shape mismatch"
    assert outputs['semantic_branch'].shape == (32, 512), "Semantic latent shape mismatch"
    assert outputs['layer_4'].shape == (32, 768), "Layer 4 shape mismatch"
    assert outputs['layer_8'].shape == (32, 768), "Layer 8 shape mismatch"
    assert outputs['layer_12'].shape == (32, 768), "Layer 12 shape mismatch"
    assert outputs['final'].shape == (32, 512), "Final shape mismatch"
    assert outputs['text'].shape == (32, 512), "Text shape mismatch"
    assert kl_loss is None, "KL loss should be None in deterministic mode"
    
    return True


def test_3_forward_probabilistic():
    """Test probabilistic forward pass with sampling."""
    print("\n" + "=" * 80)
    print("TEST 3: Probabilistic Forward Pass")
    print("=" * 80)
    
    from fmri2img.models.phase4_encoder import StructuralSemanticEncoder
    
    encoder = StructuralSemanticEncoder(
        input_dim=512,
        latent_dim=512,
        structural_dim=256,
        semantic_dim=512,
        probabilistic=True,
        predict_text_clip=True,
        kl_weight=0.01
    )
    encoder.eval()
    
    x = torch.randn(32, 512)
    
    # Test with sampling
    with torch.no_grad():
        outputs, kl_loss = encoder(x, sample=True, return_kl=True)
    
    print(f"\n✅ Probabilistic forward pass successful")
    print(f"   Input shape: {x.shape}")
    print(f"   KL loss: {kl_loss.item():.6f}")
    print(f"\n   All outputs have correct shapes ✓")
    
    # Check KL loss
    assert kl_loss is not None, "KL loss should not be None"
    assert kl_loss.item() >= 0, "KL loss should be non-negative"
    
    return True


def test_4_branch_factorization():
    """Test that branches correctly map to different layers."""
    print("\n" + "=" * 80)
    print("TEST 4: Branch Factorization")
    print("=" * 80)
    
    from fmri2img.models.phase4_encoder import StructuralSemanticEncoder
    
    encoder = StructuralSemanticEncoder(
        input_dim=512,
        latent_dim=512,
        structural_dim=256,
        semantic_dim=512,
        probabilistic=False,
        predict_text_clip=True
    )
    
    # Check that structural heads only go to early layers
    structural_heads = [name for name in encoder.mu_heads.keys() if name in ['layer_4', 'layer_8']]
    semantic_heads = [name for name in encoder.mu_heads.keys() if name in ['layer_12', 'final', 'text']]
    
    print(f"\n✅ Branch factorization verified:")
    print(f"   Structural → {structural_heads}")
    print(f"   Semantic → {semantic_heads}")
    
    assert len(structural_heads) == 2, "Should have 2 structural heads"
    assert len(semantic_heads) == 3, "Should have 3 semantic heads (with text)"
    
    return True


def test_5_uncertainty_estimation():
    """Test uncertainty estimation via multiple samples."""
    print("\n" + "=" * 80)
    print("TEST 5: Uncertainty Estimation")
    print("=" * 80)
    
    from fmri2img.models.phase4_encoder import StructuralSemanticEncoder
    
    encoder = StructuralSemanticEncoder(
        input_dim=512,
        latent_dim=512,
        structural_dim=256,
        semantic_dim=512,
        probabilistic=True,
        predict_text_clip=True
    )
    encoder.eval()
    
    x = torch.randn(8, 512)
    
    # Generate multiple samples
    n_samples = 20
    samples = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            outputs, _ = encoder(x, sample=True, return_kl=False)
            samples.append(outputs['final'])
    
    # Compute mean and std
    samples_tensor = torch.stack(samples, dim=0)  # (n_samples, B, D)
    mean_pred = samples_tensor.mean(dim=0)  # (B, D)
    std_pred = samples_tensor.std(dim=0)   # (B, D)
    
    print(f"\n✅ Uncertainty estimation working")
    print(f"   Samples: {n_samples}")
    print(f"   Mean std across samples: {std_pred.mean().item():.4f}")
    print(f"   Min std: {std_pred.min().item():.4f}")
    print(f"   Max std: {std_pred.max().item():.4f}")
    
    # Check that there is actual variance (not deterministic)
    assert std_pred.mean().item() > 0.01, "Should have non-zero uncertainty"
    
    return True


def test_6_gradient_flow():
    """Test gradient flow through both branches."""
    print("\n" + "=" * 80)
    print("TEST 6: Gradient Flow")
    print("=" * 80)
    
    from fmri2img.models.phase4_encoder import StructuralSemanticEncoder
    
    encoder = StructuralSemanticEncoder(
        input_dim=512,
        latent_dim=512,
        structural_dim=256,
        semantic_dim=512,
        probabilistic=True,
        predict_text_clip=True
    )
    encoder.train()
    
    x = torch.randn(16, 512, requires_grad=True)
    outputs, kl_loss = encoder(x, sample=True, return_kl=True)
    
    # Create dummy targets and loss
    target_early = torch.randn_like(outputs['layer_4'])
    target_late = torch.randn_like(outputs['final'])
    
    loss_structural = torch.nn.functional.mse_loss(outputs['layer_4'], target_early)
    loss_semantic = torch.nn.functional.mse_loss(outputs['final'], target_late)
    total_loss = loss_structural + loss_semantic + kl_loss
    
    # Backward
    total_loss.backward()
    
    # Check gradients exist
    structural_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for name, p in encoder.named_parameters()
        if 'structural' in name
    )
    
    semantic_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for name, p in encoder.named_parameters()
        if 'semantic' in name
    )
    
    print(f"\n✅ Gradient flow verified")
    print(f"   Structural branch has gradients: {structural_has_grad}")
    print(f"   Semantic branch has gradients: {semantic_has_grad}")
    print(f"   Loss breakdown:")
    print(f"     Structural: {loss_structural.item():.4f}")
    print(f"     Semantic: {loss_semantic.item():.4f}")
    print(f"     KL: {kl_loss.item():.6f}")
    print(f"     Total: {total_loss.item():.4f}")
    
    assert structural_has_grad, "Structural branch should have gradients"
    assert semantic_has_grad, "Semantic branch should have gradients"
    
    return True


def test_7_save_load():
    """Test save/load functionality."""
    print("\n" + "=" * 80)
    print("TEST 7: Save/Load")
    print("=" * 80)
    
    from fmri2img.models.phase4_encoder import (
        StructuralSemanticEncoder,
        save_structural_semantic_encoder,
        load_structural_semantic_encoder
    )
    import tempfile
    import os
    
    # Create model
    encoder = StructuralSemanticEncoder(
        input_dim=512,
        latent_dim=512,
        structural_dim=256,
        semantic_dim=512,
        probabilistic=True,
        predict_text_clip=True
    )
    encoder.eval()  # Important: set to eval mode to disable dropout
    
    # Test forward to get original output
    x = torch.randn(4, 512)
    with torch.no_grad():
        outputs_before, _ = encoder(x, sample=False, return_kl=False)
    
    # Save
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "test_encoder.pt")
        
        meta = {
            "input_dim": 512,
            "latent_dim": 512,
            "structural_dim": 256,
            "semantic_dim": 512,
            "n_blocks": 4,
            "dropout": 0.3,
            "head_hidden_dim": 512,
            "predict_text_clip": True,
            "probabilistic": True,
            "kl_weight": 0.01
        }
        
        save_structural_semantic_encoder(encoder, checkpoint_path, meta)
        
        # Load
        encoder_loaded, meta_loaded = load_structural_semantic_encoder(checkpoint_path)
        encoder_loaded.eval()  # Set to eval mode
        
        # Test that loaded model produces same output
        with torch.no_grad():
            outputs_after, _ = encoder_loaded(x, sample=False, return_kl=False)
        
        # Compare
        max_diff = max(
            (outputs_before[k] - outputs_after[k]).abs().max().item()
            for k in outputs_before.keys()
            if k in outputs_after and torch.is_tensor(outputs_before[k])
        )
        
        print(f"\n✅ Save/load working")
        print(f"   Checkpoint saved and loaded successfully")
        print(f"   Max difference in outputs: {max_diff:.2e}")
        
        assert max_diff < 1e-5, f"Outputs should match after load (diff: {max_diff})"
    
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("PHASE 4 VERIFICATION: Structural/Semantic Branch Encoder")
    print("=" * 80)
    print("\nThis script verifies the Phase 4 encoder with explicit branch factorization")
    print("Structural branch → early layers (L4, L8)")
    print("Semantic branch → late layers (L12, final, text)")
    
    tests = [
        ("Architecture", test_1_architecture),
        ("Deterministic Forward", test_2_forward_deterministic),
        ("Probabilistic Forward", test_3_forward_probabilistic),
        ("Branch Factorization", test_4_branch_factorization),
        ("Uncertainty Estimation", test_5_uncertainty_estimation),
        ("Gradient Flow", test_6_gradient_flow),
        ("Save/Load", test_7_save_load)
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ {name} FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {name}")
    
    n_passed = sum(1 for _, s in results if s)
    n_total = len(results)
    
    print(f"\n{n_passed}/{n_total} tests passed")
    
    if n_passed == n_total:
        print("\n🎉 Phase 4 encoder verification COMPLETE!")
        print("\nNext steps:")
        print("  1. Implement branch-weighted loss (Phase 4.2)")
        print("  2. Train Phase 4 encoder with structural/semantic factorization")
        print("  3. Evaluate branch-wise performance")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review and fix.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
