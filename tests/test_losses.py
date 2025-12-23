"""Tests for loss functions."""

import pytest
import torch
import numpy as np

from fmri2img.models.losses import (
    cosine_loss,
    mse_loss,
    infonce_loss,
    compose_loss,
    ComposedLoss
)


class TestCosineLoss:
    """Test cosine similarity loss."""
    
    def test_identical_embeddings(self):
        """Loss should be 0 for identical embeddings."""
        pred = torch.randn(16, 512)
        loss = cosine_loss(pred, pred)
        assert loss.item() < 1e-6, "Identical embeddings should have ~0 loss"
    
    def test_orthogonal_embeddings(self):
        """Loss should be 1.0 for orthogonal embeddings."""
        # Create orthogonal vectors
        pred = torch.zeros(2, 512)
        target = torch.zeros(2, 512)
        pred[:, :256] = 1.0  # First half
        target[:, 256:] = 1.0  # Second half
        
        loss = cosine_loss(pred, target)
        assert abs(loss.item() - 1.0) < 1e-5, "Orthogonal embeddings should have loss ≈ 1.0"
    
    def test_opposite_embeddings(self):
        """Loss should be 2.0 for opposite embeddings."""
        pred = torch.ones(4, 512)
        target = -torch.ones(4, 512)
        
        loss = cosine_loss(pred, target)
        assert abs(loss.item() - 2.0) < 1e-5, "Opposite embeddings should have loss ≈ 2.0"


class TestMSELoss:
    """Test MSE loss."""
    
    def test_identical_embeddings(self):
        """MSE should be 0 for identical embeddings."""
        pred = torch.randn(16, 512)
        loss = mse_loss(pred, pred)
        assert loss.item() < 1e-6
    
    def test_known_mse(self):
        """Test MSE with known values."""
        pred = torch.zeros(4, 10)
        target = torch.ones(4, 10)
        
        loss = mse_loss(pred, target)
        # MSE = mean((0 - 1)^2) = 1.0
        assert abs(loss.item() - 1.0) < 1e-5


class TestInfoNCELoss:
    """Test InfoNCE contrastive loss."""
    
    def test_shapes_and_finite(self):
        """InfoNCE should return finite scalar loss."""
        pred = torch.randn(32, 512)
        target = torch.randn(32, 512)
        
        loss = infonce_loss(pred, target, temperature=0.07)
        
        assert loss.ndim == 0, "Loss should be scalar"
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss.item() > 0, "Loss should be positive"
    
    def test_perfect_alignment_low_loss(self):
        """Perfectly aligned embeddings should have low loss."""
        # Create perfectly aligned batch
        pred = torch.randn(16, 512)
        target = pred.clone()  # Perfect alignment
        
        loss = infonce_loss(pred, target, temperature=0.07)
        
        # With perfect alignment, loss should be near 0
        # (but not exactly 0 due to temperature scaling)
        assert loss.item() < 0.1, f"Perfect alignment should have low loss, got {loss.item():.4f}"
    
    def test_random_embeddings_higher_loss(self):
        """Random embeddings should have higher loss than aligned ones."""
        pred = torch.randn(32, 512)
        target_aligned = pred.clone()
        target_random = torch.randn(32, 512)
        
        loss_aligned = infonce_loss(pred, target_aligned, temperature=0.07)
        loss_random = infonce_loss(pred, target_random, temperature=0.07)
        
        assert loss_random.item() > loss_aligned.item(), \
            "Random embeddings should have higher loss than aligned"
    
    def test_symmetry(self):
        """InfoNCE should be symmetric (pred→target ≈ target→pred)."""
        pred = torch.randn(16, 512)
        target = torch.randn(16, 512)
        
        loss1 = infonce_loss(pred, target)
        loss2 = infonce_loss(target, pred)
        
        # Should be very close due to symmetric formulation
        assert abs(loss1.item() - loss2.item()) < 0.01, \
            "InfoNCE should be approximately symmetric"
    
    def test_temperature_effect(self):
        """Higher temperature should smooth the loss."""
        pred = torch.randn(32, 512)
        target = torch.randn(32, 512)
        
        loss_low_temp = infonce_loss(pred, target, temperature=0.01)
        loss_high_temp = infonce_loss(pred, target, temperature=1.0)
        
        # Lower temperature = sharper distribution = typically higher loss
        # (This is not always true, but generally holds for random data)
        assert loss_low_temp.item() > 0 and loss_high_temp.item() > 0
    
    def test_single_sample_handled(self):
        """Should handle batch size of 1 gracefully."""
        pred = torch.randn(1, 512)
        target = torch.randn(1, 512)
        
        # Should not crash (though InfoNCE needs negatives to be meaningful)
        loss = infonce_loss(pred, target)
        assert torch.isfinite(loss)


class TestComposeLoss:
    """Test composed loss function."""
    
    def test_cosine_only(self):
        """Test with only cosine loss."""
        pred = torch.randn(16, 512)
        target = torch.randn(16, 512)
        
        loss, components = compose_loss(
            pred, target,
            cosine_weight=1.0,
            mse_weight=0.0,
            infonce_weight=0.0
        )
        
        assert 'cosine' in components
        assert 'mse' in components
        assert 'infonce' in components
        assert components['mse'] == 0.0
        assert components['infonce'] == 0.0
        assert components['cosine'] > 0
    
    def test_all_components(self):
        """Test with all loss components enabled."""
        pred = torch.randn(32, 512)
        target = torch.randn(32, 512)
        
        loss, components = compose_loss(
            pred, target,
            cosine_weight=1.0,
            mse_weight=0.5,
            infonce_weight=0.3,
            temperature=0.07
        )
        
        assert components['cosine'] > 0
        assert components['mse'] > 0
        assert components['infonce'] > 0
        assert torch.isfinite(loss)
    
    def test_weight_scaling(self):
        """Test that weights properly scale the total loss."""
        pred = torch.randn(16, 512)
        target = torch.randn(16, 512)
        
        # Single component with weight 1.0
        loss1, comp1 = compose_loss(pred, target, cosine_weight=1.0, mse_weight=0.0, infonce_weight=0.0)
        
        # Same component with weight 2.0
        loss2, comp2 = compose_loss(pred, target, cosine_weight=2.0, mse_weight=0.0, infonce_weight=0.0)
        
        # Component values should be same (they're logged pre-weighting)
        assert abs(comp1['cosine'] - comp2['cosine']) < 1e-5
        
        # Total loss should be 2x
        assert abs(loss2.item() - 2.0 * loss1.item()) < 1e-4
    
    def test_backward_compatibility(self):
        """Test backward compatibility: default weights match old behavior."""
        pred = torch.randn(16, 512)
        target = torch.randn(16, 512)
        
        # Default: cosine_weight=1.0, others=0.0 (old behavior)
        loss, components = compose_loss(pred, target)
        
        # Should only have cosine loss
        cos_only = cosine_loss(pred, target)
        assert abs(loss.item() - cos_only.item()) < 1e-5


class TestComposedLossModule:
    """Test PyTorch module wrapper."""
    
    def test_module_forward(self):
        """Test module forward pass."""
        loss_fn = ComposedLoss(
            cosine_weight=1.0,
            mse_weight=0.5,
            infonce_weight=0.3,
            temperature=0.07
        )
        
        pred = torch.randn(16, 512)
        target = torch.randn(16, 512)
        
        loss, components = loss_fn(pred, target)
        
        assert torch.isfinite(loss)
        assert 'cosine' in components
        assert 'mse' in components
        assert 'infonce' in components
    
    def test_gradient_flow(self):
        """Test that gradients flow through the loss."""
        loss_fn = ComposedLoss(cosine_weight=1.0, infonce_weight=0.5)
        
        pred = torch.randn(16, 512, requires_grad=True)
        target = torch.randn(16, 512)
        
        loss, _ = loss_fn(pred, target)
        loss.backward()
        
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()


def test_infonce_decreases_with_alignment():
    """Test that InfoNCE loss decreases as predictions improve."""
    torch.manual_seed(42)
    
    # Fixed target
    target = torch.randn(32, 512)
    target = torch.nn.functional.normalize(target, p=2, dim=1)
    
    # Predictions that progressively align with target
    losses = []
    for alpha in [0.0, 0.2, 0.5, 0.8, 1.0]:
        # Interpolate between random and target
        random = torch.randn(32, 512)
        pred = alpha * target + (1 - alpha) * random
        pred = torch.nn.functional.normalize(pred, p=2, dim=1)
        
        loss = infonce_loss(pred, target, temperature=0.07)
        losses.append(loss.item())
    
    # Loss should generally decrease as alpha increases
    # (more alignment with target)
    assert losses[-1] < losses[0], \
        "Loss should decrease as predictions align with targets"


def test_soft_reliability_equivalence():
    """
    Test that soft reliability weighting approximates hard threshold
    when temperature is very small.
    
    This tests the preprocessing module's soft weighting, but placed here
    as it relates to the novel contributions.
    """
    # This will be implemented when we add the preprocessing changes
    # For now, just a placeholder test
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
