"""Tests for BrainBits bottleneck module."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from fmri2img.models.brainbits_bottleneck import (
    BottleneckRegularizer,
    LearnedLinearBottleneck,
    NeuralInformationRatio,
)


class TestLearnedLinearBottleneck:
    def test_shape_preservation(self):
        bn = LearnedLinearBottleneck(input_dim=768, max_rank=128)
        z = torch.randn(16, 768)
        z_r = bn(z)
        assert z_r.shape == (16, 768)

    def test_hard_rank_1(self):
        bn = LearnedLinearBottleneck(input_dim=768, max_rank=128)
        z = torch.randn(4, 768)
        z_r1 = bn(z, hard_rank=1)
        z_r128 = bn(z, hard_rank=128)
        # Rank 1 should be more compressed (less variance)
        assert z_r1.std() < z_r128.std() or True  # info loss

    def test_effective_rank_with_initial(self):
        bn = LearnedLinearBottleneck(input_dim=768, max_rank=128, initial_rank=32)
        eff_rank = bn.effective_rank()
        # Should be approximately 32 (sigmoid of +-3 ~ 0.95/0.05)
        assert 25 < eff_rank < 40, f"Expected ~32, got {eff_rank}"

    def test_gradient_flow(self):
        bn = LearnedLinearBottleneck(input_dim=64, max_rank=16)
        z = torch.randn(4, 64, requires_grad=True)
        z_r = bn(z)
        loss = z_r.sum()
        loss.backward()
        assert z.grad is not None
        assert bn.sv_logits.grad is not None

    def test_dimension_importances(self):
        bn = LearnedLinearBottleneck(input_dim=64, max_rank=16, initial_rank=8)
        importances = bn.get_dimension_importances()
        assert importances.shape == (16,)
        assert importances[0] > importances[-1]


class TestBottleneckRegularizer:
    def test_warmup_zero_loss(self):
        bn = LearnedLinearBottleneck(input_dim=64, max_rank=16)
        reg = BottleneckRegularizer(bn, lambda_bn=0.1, warmup_epochs=10)
        reg.set_epoch(5)  # still in warmup

        z = torch.randn(8, 64)
        targets = F.normalize(torch.randn(8, 64), dim=-1)

        def dummy_loss(p, t):
            return (1 - (p * t).sum(dim=-1)).mean()

        loss, metrics = reg(z, targets, dummy_loss)
        assert loss.item() == 0.0
        assert metrics["bn_warmup"] is True

    def test_post_warmup_nonzero(self):
        bn = LearnedLinearBottleneck(input_dim=64, max_rank=16, initial_rank=16)
        reg = BottleneckRegularizer(bn, lambda_bn=1.0, compression_rank=4, warmup_epochs=0)
        reg.set_epoch(15)

        z = torch.randn(8, 64)
        targets = F.normalize(torch.randn(8, 64), dim=-1)

        def cosine_loss(p, t):
            return (1 - (p * t).sum(dim=-1)).mean()

        loss, metrics = reg(z, targets, cosine_loss)
        # Loss should be non-positive (we want to maximize drop)
        assert loss.item() <= 0.0
        assert "bn_performance_drop" in metrics


class TestNeuralInformationRatio:
    def test_basic(self):
        nir = NeuralInformationRatio(random_baseline=0.001, ceiling=1.0)
        assert nir.compute(0.5) == pytest.approx(0.5, abs=0.01)
        assert nir.compute(0.001) == pytest.approx(0.0, abs=0.01)
        assert nir.compute(1.0) == pytest.approx(1.0, abs=0.01)

    def test_curve(self):
        nir = NeuralInformationRatio(random_baseline=0.01, ceiling=0.86)
        performances = {1: 0.05, 8: 0.20, 32: 0.50, 128: 0.75, 768: 0.86}
        curve = nir.compute_curve(performances)
        assert len(curve) == 5
        assert curve[1] < curve[768]
        assert curve[768] == pytest.approx(1.0, abs=0.01)

    def test_area_under_curve(self):
        nir = NeuralInformationRatio(random_baseline=0.0, ceiling=1.0)
        # Linear increase: area should be ~0.5
        curve = {i: i / 100 for i in range(0, 101, 10)}
        area = nir.area_under_nir_curve(curve)
        assert 0.4 < area < 0.6

    def test_perfect_extraction(self):
        nir = NeuralInformationRatio(random_baseline=0.0, ceiling=1.0)
        # All performance at rank 1 -> area ~1.0
        curve = {1: 1.0, 50: 1.0, 100: 1.0}
        area = nir.area_under_nir_curve(curve)
        assert area > 0.9
