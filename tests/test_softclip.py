"""Tests for SoftCLIP knowledge distillation loss."""

import pytest
import torch
import torch.nn.functional as F

from fmri2img.losses.softclip import SoftCLIPLoss


class TestSoftCLIPLoss:
    """Test SoftCLIP loss properties."""

    @pytest.fixture
    def loss_fn(self):
        return SoftCLIPLoss(tau=0.07, use_queue=False, symmetric=False)

    @pytest.fixture
    def symmetric_loss_fn(self):
        return SoftCLIPLoss(tau=0.07, use_queue=False, symmetric=True)

    def test_output_is_finite_scalar(self, loss_fn):
        pred = torch.randn(32, 768)
        gt = torch.randn(32, 768)
        loss = loss_fn(pred, gt)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_perfect_alignment_low_loss(self, loss_fn):
        gt = F.normalize(torch.randn(16, 768), dim=1)
        loss_perfect = loss_fn(gt.clone(), gt)
        loss_random = loss_fn(torch.randn(16, 768), gt)
        assert loss_perfect.item() < loss_random.item(), (
            "Perfectly aligned predictions should yield lower SoftCLIP loss"
        )

    def test_gradient_flow(self, loss_fn):
        pred = torch.randn(16, 768, requires_grad=True)
        gt = torch.randn(16, 768)
        loss = loss_fn(pred, gt)
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()

    def test_symmetric_mode(self, loss_fn, symmetric_loss_fn):
        pred = torch.randn(16, 768)
        gt = torch.randn(16, 768)
        loss_asym = loss_fn(pred, gt)
        loss_sym = symmetric_loss_fn(pred, gt)
        assert loss_asym.item() != pytest.approx(loss_sym.item(), abs=1e-4), (
            "Symmetric and asymmetric modes should differ"
        )

    def test_temperature_effect(self):
        pred = torch.randn(32, 768)
        gt = torch.randn(32, 768)
        loss_cold = SoftCLIPLoss(tau=0.01, use_queue=False, symmetric=False)(pred, gt)
        loss_warm = SoftCLIPLoss(tau=1.0, use_queue=False, symmetric=False)(pred, gt)
        assert torch.isfinite(loss_cold) and torch.isfinite(loss_warm)

    def test_batch_size_one(self, loss_fn):
        pred = torch.randn(1, 768)
        gt = torch.randn(1, 768)
        loss = loss_fn(pred, gt)
        assert torch.isfinite(loss)

    def test_loss_decreases_with_alignment(self):
        """SoftCLIP loss should decrease as predictions approach targets."""
        torch.manual_seed(42)
        loss_fn = SoftCLIPLoss(tau=0.07, use_queue=False, symmetric=False)
        gt = F.normalize(torch.randn(32, 768), dim=1)

        losses = []
        for alpha in [0.0, 0.3, 0.6, 1.0]:
            noise = torch.randn_like(gt)
            pred = alpha * gt + (1.0 - alpha) * noise
            losses.append(loss_fn(pred, gt).item())

        assert losses[-1] < losses[0], (
            "Loss should decrease as predictions align with targets"
        )

    def test_soft_targets_differ_from_hard(self):
        """SoftCLIP should produce different gradients than hard InfoNCE.

        Constructs GT embeddings with high inter-class similarity (low-dim
        subspace + noise) so the teacher distribution is genuinely soft.
        """
        torch.manual_seed(0)
        tau = 0.5
        B, D = 16, 768

        # Low-rank GT so pairs have non-trivial cosine similarity
        base = torch.randn(B, 4)
        proj = torch.randn(4, D)
        gt = F.normalize(base @ proj + 0.1 * torch.randn(B, D), dim=1)

        pred = torch.randn(B, D, requires_grad=True)

        soft_loss = SoftCLIPLoss(tau=tau, use_queue=False, symmetric=False)(pred, gt)
        soft_loss.backward()
        grad_soft = pred.grad.clone()

        pred.grad = None
        logits = F.normalize(pred, dim=1) @ gt.T / tau
        hard_loss = F.cross_entropy(logits, torch.arange(B))
        hard_loss.backward()
        grad_hard = pred.grad.clone()

        cosine = F.cosine_similarity(grad_soft.flatten().unsqueeze(0),
                                     grad_hard.flatten().unsqueeze(0))
        assert cosine.item() < 0.99, (
            f"SoftCLIP gradients should differ from hard InfoNCE (cos={cosine.item():.4f})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
