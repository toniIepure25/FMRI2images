"""
Unit tests for Centered Kernel Alignment (CKA) Loss.

Run:  pytest tests/test_cka_loss.py -v
"""

import pytest
import torch
import torch.nn.functional as F

from fmri2img.losses.cka_loss import CKALoss


class TestCKALoss:
    """Tests for the CKA representational similarity loss."""

    @pytest.fixture
    def loss_fn(self):
        return CKALoss()

    def test_output_scalar(self, loss_fn):
        pred = torch.randn(16, 32)
        target = torch.randn(16, 32)
        loss = loss_fn(pred, target)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_loss_bounded_zero_one(self, loss_fn):
        """CKA is in [0, 1], so loss = 1 - CKA is also in [0, 1]."""
        torch.manual_seed(42)
        for _ in range(10):
            pred = torch.randn(16, 32)
            target = torch.randn(16, 32)
            loss = loss_fn(pred, target)
            assert 0.0 <= loss.item() <= 1.0 + 1e-5, (
                f"CKA loss outside [0, 1]: {loss.item():.4f}"
            )

    def test_identical_inputs_zero_loss(self, loss_fn):
        """CKA(X, X) = 1, so loss should be ~0."""
        x = torch.randn(16, 32)
        loss = loss_fn(x, x)
        assert loss.item() < 0.01, (
            f"Loss should be ~0 for identical inputs, got {loss.item():.4f}"
        )

    def test_scaled_copy_zero_loss(self, loss_fn):
        """CKA is invariant to isotropic scaling, so CKA(X, 2*X) = 1."""
        x = torch.randn(16, 32)
        loss = loss_fn(x, 2.0 * x)
        assert loss.item() < 0.01, (
            f"Loss should be ~0 for scaled copy, got {loss.item():.4f}"
        )

    def test_random_inputs_higher_loss(self, loss_fn):
        """Random uncorrelated spaces should have higher loss than identical."""
        torch.manual_seed(42)
        x = torch.randn(32, 64)
        y = torch.randn(32, 64)
        loss_random = loss_fn(x, y)
        loss_self = loss_fn(x, x)
        assert loss_random.item() > loss_self.item(), (
            f"Random loss ({loss_random.item():.4f}) should be > "
            f"self loss ({loss_self.item():.4f})"
        )

    def test_gradient_flow(self, loss_fn):
        pred = torch.randn(16, 32, requires_grad=True)
        target = torch.randn(16, 32)
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()

    def test_gradient_not_trivially_zero(self, loss_fn):
        """Gradient should be non-zero for non-identical inputs."""
        pred = torch.randn(16, 32, requires_grad=True)
        target = torch.randn(16, 32)
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad.abs().sum() > 1e-8

    def test_small_batch(self, loss_fn):
        """Should work even with very small batch sizes."""
        pred = torch.randn(3, 16)
        target = torch.randn(3, 16)
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)

    def test_large_dim(self, loss_fn):
        """Should work with CLIP-scale dimensions."""
        pred = torch.randn(32, 768)
        target = torch.randn(32, 768)
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)

    def test_centering_symmetry(self):
        """CKA(X, Y) should equal CKA(Y, X)."""
        loss_fn = CKALoss()
        x = torch.randn(16, 32)
        y = torch.randn(16, 32)
        loss_xy = loss_fn(x, y)
        loss_yx = loss_fn(y, x)
        assert torch.allclose(loss_xy, loss_yx, atol=1e-5), (
            f"CKA should be symmetric: {loss_xy.item():.6f} vs {loss_yx.item():.6f}"
        )
