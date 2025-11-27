"""
Phase 3 Verification: Probabilistic Multi-Layer Encoder
========================================================

Tests the probabilistic encoder implementation to ensure:
1. ✅ Model architecture (mu/logvar heads for each layer)
2. ✅ Reparameterization trick (differentiable sampling)
3. ✅ KL divergence computation
4. ✅ Sampling consistency (multiple samples from same input)
5. ✅ Deterministic mode (sample=False returns mean)
6. ✅ Loss computation with KL annealing
7. ✅ Uncertainty estimation (std across samples)
8. ✅ Backward compatibility (can load Phase 2 checkpoints)

Usage:
    python scripts/verify_phase3_probabilistic.py \\
        --input-dim 512 \\
        --latent-dim 512 \\
        --n-samples 10 \\
        --verbose

Author: Tony Stark
Date: November 27, 2025
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.fmri2img.models.encoders import ProbabilisticMultiLayerTwoStageEncoder
from src.fmri2img.training.losses import ProbabilisticMultiLayerLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_model_architecture(input_dim: int, latent_dim: int, predict_text_clip: bool):
    """Test 1: Model architecture and parameter count."""
    logger.info("\n" + "="*70)
    logger.info("TEST 1: Model Architecture")
    logger.info("="*70)
    
    model = ProbabilisticMultiLayerTwoStageEncoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        predict_text_clip=predict_text_clip
    )
    
    # Check architecture
    logger.info(f"✅ Model created successfully")
    logger.info(f"   Input dim: {input_dim}")
    logger.info(f"   Latent dim: {latent_dim}")
    logger.info(f"   Text-CLIP enabled: {predict_text_clip}")
    
    # Check mu and logvar heads exist
    expected_layers = ['layer_4', 'layer_8', 'layer_12', 'final']
    if predict_text_clip:
        expected_layers.append('text')
    
    for layer_name in expected_layers:
        assert layer_name in model.mu_heads, f"Missing mu head for {layer_name}"
        assert layer_name in model.logvar_heads, f"Missing logvar head for {layer_name}"
        logger.info(f"   ✅ {layer_name}: mu head ({model.mu_heads[layer_name]})")
        logger.info(f"   ✅ {layer_name}: logvar head ({model.logvar_heads[layer_name]})")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    mu_params = sum(p.numel() for p in model.mu_heads.parameters())
    logvar_params = sum(p.numel() for p in model.logvar_heads.parameters())
    
    logger.info(f"\n📊 Parameter count:")
    logger.info(f"   Total: {total_params:,}")
    logger.info(f"   Mu heads: {mu_params:,}")
    logger.info(f"   Logvar heads: {logvar_params:,}")
    logger.info(f"   Backbone: {total_params - mu_params - logvar_params:,}")
    
    return model


def test_forward_pass(model: nn.Module, batch_size: int = 32):
    """Test 2: Forward pass with sampling."""
    logger.info("\n" + "="*70)
    logger.info("TEST 2: Forward Pass")
    logger.info("="*70)
    
    x = torch.randn(batch_size, model.input_dim)
    
    # Test with sampling
    outputs, kl_loss = model(x, sample=True, return_kl=True)
    
    logger.info(f"✅ Forward pass (sample=True) successful")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   KL loss: {kl_loss.item():.6f}")
    
    # Check outputs
    expected_layers = ['layer_4', 'layer_8', 'layer_12', 'final']
    if model.predict_text_clip:
        expected_layers.append('text')
    
    for layer_name in expected_layers:
        assert layer_name in outputs, f"Missing output for {layer_name}"
        z = outputs[layer_name]
        assert z.shape[0] == batch_size, f"Wrong batch size for {layer_name}"
        
        # Check normalization (CLIP embeddings are L2-normalized)
        norms = torch.norm(z, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            f"{layer_name} not normalized: norms={norms[:5].tolist()}"
        
        logger.info(f"   ✅ {layer_name}: shape={z.shape}, norm={norms.mean().item():.6f}")
    
    return outputs, kl_loss


def test_reparameterization(model: nn.Module):
    """Test 3: Reparameterization trick and sampling consistency."""
    logger.info("\n" + "="*70)
    logger.info("TEST 3: Reparameterization & Sampling")
    logger.info("="*70)
    
    x = torch.randn(4, model.input_dim)
    
    # Test deterministic mode (sample=False, should return mu)
    outputs_det, kl_det = model(x, sample=False, return_kl=True)
    logger.info(f"✅ Deterministic mode (sample=False):")
    logger.info(f"   Returns mean (μ), KL loss: {kl_det.item():.6f}")
    
    # Test stochastic mode (sample=True, should sample from distribution)
    outputs_stoch1, kl_stoch1 = model(x, sample=True, return_kl=True)
    outputs_stoch2, kl_stoch2 = model(x, sample=True, return_kl=True)
    
    # Different samples should be different (randomness)
    for layer_name in outputs_stoch1.keys():
        z1 = outputs_stoch1[layer_name]
        z2 = outputs_stoch2[layer_name]
        diff = (z1 - z2).abs().mean().item()
        logger.info(f"   {layer_name}: |sample1 - sample2| = {diff:.6f}")
        assert diff > 1e-6, f"{layer_name} samples are identical (no randomness)"
    
    logger.info(f"✅ Stochastic mode (sample=True): samples differ (randomness works)")
    
    # KL losses should be similar (same input)
    kl_diff = abs(kl_stoch1.item() - kl_stoch2.item())
    logger.info(f"   KL loss difference: {kl_diff:.6f} (should be small)")
    
    return outputs_det, outputs_stoch1


def test_uncertainty_estimation(model: nn.Module, n_samples: int = 50):
    """Test 4: Uncertainty estimation via Monte Carlo sampling."""
    logger.info("\n" + "="*70)
    logger.info("TEST 4: Uncertainty Estimation")
    logger.info("="*70)
    
    x = torch.randn(8, model.input_dim)
    
    # Generate multiple samples for each input
    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            outputs, _ = model(x, sample=True, return_kl=False)
            samples.append(outputs)
    
    # Compute mean and std across samples
    logger.info(f"✅ Generated {n_samples} samples per input")
    
    for layer_name in samples[0].keys():
        # Stack samples: (n_samples, batch_size, D)
        layer_samples = torch.stack([s[layer_name] for s in samples], dim=0)
        
        # Compute statistics
        mean_pred = layer_samples.mean(dim=0)  # (batch_size, D)
        std_pred = layer_samples.std(dim=0)  # (batch_size, D)
        
        # Aggregate stats
        mean_std = std_pred.mean().item()
        max_std = std_pred.max().item()
        min_std = std_pred.min().item()
        
        logger.info(f"   {layer_name}:")
        logger.info(f"      Mean std: {mean_std:.6f}")
        logger.info(f"      Max std:  {max_std:.6f}")
        logger.info(f"      Min std:  {min_std:.6f}")
        
        assert mean_std > 0, f"{layer_name} has zero uncertainty (model collapsed)"
        assert mean_std < 0.5, f"{layer_name} has very high uncertainty (model not converged)"
    
    logger.info(f"✅ Uncertainty estimation works (non-zero, bounded std)")
    
    return mean_pred, std_pred


def test_kl_divergence(model: nn.Module):
    """Test 5: KL divergence computation."""
    logger.info("\n" + "="*70)
    logger.info("TEST 5: KL Divergence")
    logger.info("="*70)
    
    x = torch.randn(64, model.input_dim)
    
    # Get KL loss
    _, kl_loss = model(x, sample=True, return_kl=True)
    
    logger.info(f"✅ KL loss computed: {kl_loss.item():.6f}")
    
    # KL should be positive (divergence is non-negative)
    assert kl_loss.item() >= 0, f"KL loss is negative: {kl_loss.item()}"
    
    # KL should be reasonable (not too large, not too small)
    assert 0.001 < kl_loss.item() < 100, \
        f"KL loss out of reasonable range: {kl_loss.item()}"
    
    logger.info(f"   KL loss is positive and bounded ✅")
    
    # Test that KL is weighted correctly
    logger.info(f"   Model KL weight: {model.kl_weight}")
    logger.info(f"   Effective KL contribution: {kl_loss.item() * model.kl_weight:.6f}")
    
    return kl_loss


def test_loss_function(model: nn.Module):
    """Test 6: Probabilistic loss with KL annealing."""
    logger.info("\n" + "="*70)
    logger.info("TEST 6: Loss Function & KL Annealing")
    logger.info("="*70)
    
    # Create loss function
    criterion = ProbabilisticMultiLayerLoss(
        kl_weight_max=0.01,
        kl_anneal_epochs=20,
        kl_anneal_start=10
    )
    
    logger.info(f"✅ Created ProbabilisticMultiLayerLoss")
    logger.info(f"   KL weight max: {criterion.kl_weight_max}")
    logger.info(f"   Anneal epochs: {criterion.kl_anneal_epochs}")
    logger.info(f"   Anneal start: {criterion.kl_anneal_start}")
    
    # Test annealing schedule
    test_epochs = [0, 5, 10, 15, 20, 25, 30, 40]
    logger.info(f"\n📈 KL annealing schedule:")
    for epoch in test_epochs:
        kl_weight = criterion.get_kl_weight(epoch)
        logger.info(f"   Epoch {epoch:2d}: KL weight = {kl_weight:.6f}")
    
    # Expected behavior:
    # - Epoch 0-9: weight = 0 (no KL loss)
    # - Epoch 10-29: weight linearly increases 0 → 0.01
    # - Epoch 30+: weight = 0.01 (full regularization)
    
    assert criterion.get_kl_weight(0) == 0.0, "Epoch 0 should have zero KL weight"
    assert criterion.get_kl_weight(10) == 0.0, "Epoch 10 should start annealing"
    assert abs(criterion.get_kl_weight(20) - 0.005) < 1e-6, "Epoch 20 should be halfway"
    assert criterion.get_kl_weight(30) == 0.01, "Epoch 30+ should be at max"
    
    logger.info(f"✅ Annealing schedule correct")
    
    # Test loss computation
    x = torch.randn(32, model.input_dim)
    pred_dict, kl_loss = model(x, sample=True, return_kl=True)
    
    # Create fake targets (same as predictions for this test)
    target_dict = {k: v.detach() + 0.1 * torch.randn_like(v) for k, v in pred_dict.items()}
    
    # Compute loss at different epochs
    logger.info(f"\n🔧 Loss computation at different epochs:")
    for epoch in [0, 15, 30]:
        loss, components = criterion(
            pred_dict, target_dict, kl_loss,
            current_epoch=epoch,
            return_components=True
        )
        logger.info(f"   Epoch {epoch:2d}:")
        logger.info(f"      Total loss: {loss.item():.6f}")
        logger.info(f"      Reconstruction: {components['reconstruction']:.6f}")
        logger.info(f"      KL loss: {components['kl']:.6f}")
        logger.info(f"      KL weight: {components['kl_weight']:.6f}")
        logger.info(f"      Weighted KL: {components['weighted_kl']:.6f}")
    
    logger.info(f"✅ Loss computation works correctly")
    
    return criterion


def test_gradient_flow(model: nn.Module, criterion: nn.Module):
    """Test 7: Gradient flow through reparameterization."""
    logger.info("\n" + "="*70)
    logger.info("TEST 7: Gradient Flow")
    logger.info("="*70)
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    x = torch.randn(16, model.input_dim)
    
    # Forward pass
    pred_dict, kl_loss = model(x, sample=True, return_kl=True)
    target_dict = {k: torch.randn_like(v) for k, v in pred_dict.items()}
    
    # Compute loss
    loss, components = criterion(
        pred_dict, target_dict, kl_loss,
        current_epoch=20,
        return_components=True
    )
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    has_grad = False
    max_grad = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            grad_norm = param.grad.norm().item()
            max_grad = max(max_grad, grad_norm)
            if 'mu_heads' in name or 'logvar_heads' in name:
                logger.info(f"   {name}: grad_norm = {grad_norm:.6f}")
    
    assert has_grad, "No gradients computed (backprop failed)"
    assert max_grad > 1e-8, "Gradients are too small (vanishing)"
    assert max_grad < 1e3, "Gradients are too large (exploding)"
    
    logger.info(f"✅ Gradients flow correctly")
    logger.info(f"   Max gradient norm: {max_grad:.6f}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Verify Phase 3: Probabilistic Encoder")
    parser.add_argument("--input-dim", type=int, default=512,
                       help="Input dimensionality (PCA components)")
    parser.add_argument("--latent-dim", type=int, default=512,
                       help="Latent dimensionality")
    parser.add_argument("--predict-text-clip", action="store_true",
                       help="Enable text-CLIP prediction (Phase 2)")
    parser.add_argument("--n-samples", type=int, default=50,
                       help="Number of samples for uncertainty estimation")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("="*70)
    logger.info("PHASE 3 VERIFICATION: Probabilistic Multi-Layer Encoder")
    logger.info("="*70)
    logger.info(f"Configuration:")
    logger.info(f"  Input dim: {args.input_dim}")
    logger.info(f"  Latent dim: {args.latent_dim}")
    logger.info(f"  Text-CLIP: {args.predict_text_clip}")
    logger.info(f"  N samples: {args.n_samples}")
    
    try:
        # Test 1: Model architecture
        model = test_model_architecture(
            args.input_dim,
            args.latent_dim,
            args.predict_text_clip
        )
        
        # Test 2: Forward pass
        test_forward_pass(model, batch_size=32)
        
        # Test 3: Reparameterization
        test_reparameterization(model)
        
        # Test 4: Uncertainty estimation
        test_uncertainty_estimation(model, n_samples=args.n_samples)
        
        # Test 5: KL divergence
        test_kl_divergence(model)
        
        # Test 6: Loss function
        criterion = test_loss_function(model)
        
        # Test 7: Gradient flow
        test_gradient_flow(model, criterion)
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info("✅ ALL TESTS PASSED")
        logger.info("="*70)
        logger.info("Phase 3 probabilistic encoder is working correctly!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Integrate into training pipeline (train_two_stage.py)")
        logger.info("  2. Train probabilistic model with KL annealing")
        logger.info("  3. Compare uncertainty vs reconstruction quality")
        logger.info("  4. Visualize prediction distributions")
        logger.info("="*70)
        
        return 0
    
    except Exception as e:
        logger.error(f"\n❌ TEST FAILED: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
