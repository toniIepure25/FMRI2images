"""
Unit tests for V11 innovations:
  - CSLS-corrected contrastive loss (differentiability)
  - Inverted softmax component
  - DirectAlignmentLoss
  - UniformityLoss
  - NCSnrAttention

Run:  pytest tests/test_v11_features.py -v
"""

import pytest
import torch
import torch.nn as nn
import numpy as np


# ── CSLS-corrected contrastive loss ──────────────────────────────────────


class TestCSLSTraining:
    """Test differentiable CSLS correction inside VonMisesFisherNCELoss."""

    @pytest.fixture
    def loss_csls(self):
        from fmri2img.losses.vmf_nce import VonMisesFisherNCELoss
        return VonMisesFisherNCELoss(
            tau=1.0,
            use_queue=False,
            use_csls_training=True,
            csls_k=3,
        )

    @pytest.fixture
    def loss_baseline(self):
        from fmri2img.losses.vmf_nce import VonMisesFisherNCELoss
        return VonMisesFisherNCELoss(
            tau=1.0,
            use_queue=False,
            use_csls_training=False,
        )

    def test_csls_loss_runs(self, loss_csls):
        B, D = 16, 64
        mu = nn.functional.normalize(torch.randn(B, D), dim=-1)
        kappa = torch.ones(B, 1) * 50.0
        keys = nn.functional.normalize(torch.randn(B, D), dim=-1)
        l = loss_csls(mu, kappa, keys)
        assert l.shape == (), f"Expected scalar, got {l.shape}"
        assert l.isfinite().all(), f"Loss not finite: {l}"

    def test_csls_loss_differentiable(self, loss_csls):
        """CSLS correction must allow gradients through topk + mean."""
        B, D = 16, 64
        raw = torch.randn(B, D, requires_grad=True)
        mu = nn.functional.normalize(raw, dim=-1)
        kappa = torch.ones(B, 1) * 50.0
        keys = nn.functional.normalize(torch.randn(B, D), dim=-1)
        l = loss_csls(mu, kappa, keys)
        l.backward()
        assert raw.grad is not None, "Gradient must flow through CSLS correction"
        assert raw.grad.isfinite().all(), "Gradients must be finite"

    def test_csls_changes_loss_value(self, loss_csls, loss_baseline):
        """CSLS correction should produce different loss than baseline."""
        B, D = 32, 64
        torch.manual_seed(42)
        mu = nn.functional.normalize(torch.randn(B, D), dim=-1)
        kappa = torch.ones(B, 1) * 50.0
        keys = nn.functional.normalize(torch.randn(B, D), dim=-1)
        l_csls = loss_csls(mu, kappa, keys).item()
        l_base = loss_baseline(mu, kappa, keys).item()
        assert l_csls != pytest.approx(l_base, abs=1e-4), \
            "CSLS correction should change the loss value"

    def test_csls_correct_method_shape(self, loss_csls):
        logits = torch.randn(8, 20)
        corrected = loss_csls._csls_correct(logits)
        assert corrected.shape == logits.shape


# ── Inverted softmax component ───────────────────────────────────────────


class TestInvertedSoftmax:
    """Test the ISF component in VonMisesFisherNCELoss."""

    @pytest.fixture
    def loss_isf(self):
        from fmri2img.losses.vmf_nce import VonMisesFisherNCELoss
        return VonMisesFisherNCELoss(
            tau=1.0,
            use_queue=False,
            isf_weight=0.3,
        )

    def test_isf_loss_runs(self, loss_isf):
        B, D = 16, 64
        mu = nn.functional.normalize(torch.randn(B, D), dim=-1)
        kappa = torch.ones(B, 1) * 50.0
        keys = nn.functional.normalize(torch.randn(B, D), dim=-1)
        l = loss_isf(mu, kappa, keys)
        assert l.shape == ()
        assert l.isfinite().all()

    def test_isf_differentiable(self, loss_isf):
        B, D = 16, 64
        raw = torch.randn(B, D, requires_grad=True)
        mu = nn.functional.normalize(raw, dim=-1)
        kappa = torch.ones(B, 1) * 50.0
        keys = nn.functional.normalize(torch.randn(B, D), dim=-1)
        l = loss_isf(mu, kappa, keys)
        l.backward()
        assert raw.grad is not None

    def test_isf_weight_zero_matches_standard(self):
        from fmri2img.losses.vmf_nce import VonMisesFisherNCELoss
        loss_std = VonMisesFisherNCELoss(tau=1.0, use_queue=False, isf_weight=0.0)
        loss_isf = VonMisesFisherNCELoss(tau=1.0, use_queue=False, isf_weight=0.0)
        B, D = 16, 64
        torch.manual_seed(99)
        mu = nn.functional.normalize(torch.randn(B, D), dim=-1)
        kappa = torch.ones(B, 1) * 50.0
        keys = nn.functional.normalize(torch.randn(B, D), dim=-1)
        assert loss_std(mu, kappa, keys).item() == pytest.approx(
            loss_isf(mu, kappa, keys).item(), abs=1e-6
        )


# ── Direct alignment loss ────────────────────────────────────────────────


class TestDirectAlignmentLoss:

    @pytest.fixture
    def loss_fn(self):
        from fmri2img.losses.direct_alignment import DirectAlignmentLoss
        return DirectAlignmentLoss()

    def test_identical_vectors_zero_loss(self, loss_fn):
        v = nn.functional.normalize(torch.randn(8, 64), dim=-1)
        assert loss_fn(v, v).item() == pytest.approx(0.0, abs=1e-5)

    def test_opposite_vectors_max_loss(self, loss_fn):
        v = nn.functional.normalize(torch.randn(8, 64), dim=-1)
        l = loss_fn(v, -v).item()
        assert l == pytest.approx(2.0, abs=1e-4)

    def test_differentiable(self, loss_fn):
        raw = torch.randn(8, 64, requires_grad=True)
        pred = nn.functional.normalize(raw, dim=-1)
        tgt = nn.functional.normalize(torch.randn(8, 64), dim=-1)
        loss_fn(pred, tgt).backward()
        assert raw.grad is not None

    def test_shape(self, loss_fn):
        l = loss_fn(torch.randn(4, 32), torch.randn(4, 32))
        assert l.shape == ()


# ── Uniformity loss ──────────────────────────────────────────────────────


class TestUniformityLoss:

    @pytest.fixture
    def loss_fn(self):
        from fmri2img.losses.uniformity import UniformityLoss
        return UniformityLoss(t=2.0)

    def test_uniform_distribution_lower(self, loss_fn):
        """A more uniform distribution should have lower uniformity loss."""
        torch.manual_seed(42)
        uniform = nn.functional.normalize(torch.randn(64, 32), dim=-1)
        clustered = nn.functional.normalize(
            torch.randn(64, 32) * 0.1 + torch.randn(1, 32), dim=-1
        )
        l_uniform = loss_fn(uniform).item()
        l_clustered = loss_fn(clustered).item()
        assert l_uniform < l_clustered, \
            f"Uniform ({l_uniform:.4f}) should be less than clustered ({l_clustered:.4f})"

    def test_differentiable(self, loss_fn):
        raw = torch.randn(16, 32, requires_grad=True)
        emb = nn.functional.normalize(raw, dim=-1)
        loss_fn(emb).backward()
        assert raw.grad is not None

    def test_shape(self, loss_fn):
        l = loss_fn(nn.functional.normalize(torch.randn(8, 16), dim=-1))
        assert l.shape == ()


# ── NCSnr attention ──────────────────────────────────────────────────────


class TestNCSnrAttention:

    @pytest.fixture
    def ncsnr(self):
        rng = np.random.default_rng(42)
        return rng.exponential(scale=2.0, size=100).astype(np.float32)

    @pytest.fixture
    def attn(self, ncsnr):
        from fmri2img.models.ncsnr_attention import NCSnrAttention
        return NCSnrAttention(n_voxels=100, ncsnr=ncsnr)

    def test_output_shape(self, attn):
        x = torch.randn(4, 100)
        out = attn(x)
        assert out.shape == (4, 100)

    def test_differentiable(self, attn):
        x = torch.randn(4, 100, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None
        assert attn.raw_weights.grad is not None

    def test_weights_nonnegative(self, attn):
        import torch.nn.functional as F
        w = F.softplus(attn.raw_weights)
        assert (w >= 0).all()

    def test_high_ncsnr_gets_higher_weight(self, ncsnr):
        from fmri2img.models.ncsnr_attention import NCSnrAttention
        import torch.nn.functional as F
        attn = NCSnrAttention(n_voxels=100, ncsnr=ncsnr)
        weights = F.softplus(attn.raw_weights).detach().numpy()
        high_mask = ncsnr > np.median(ncsnr)
        low_mask = ~high_mask
        assert weights[high_mask].mean() > weights[low_mask].mean(), \
            "High-NCSNR voxels should have higher initial weights"

    def test_uniform_init_without_ncsnr(self):
        from fmri2img.models.ncsnr_attention import NCSnrAttention
        import torch.nn.functional as F
        attn = NCSnrAttention(n_voxels=50, ncsnr=None)
        weights = F.softplus(attn.raw_weights)
        assert torch.allclose(weights, torch.ones(50), atol=0.01)

    def test_parameter_count(self, attn):
        n_params = sum(p.numel() for p in attn.parameters())
        assert n_params == 100  # one weight per voxel


# ── Combined CSLS + ISF ─────────────────────────────────────────────────


class TestCombinedCSLSAndISF:
    """Test that both CSLS and ISF can be enabled simultaneously."""

    def test_combined_runs(self):
        from fmri2img.losses.vmf_nce import VonMisesFisherNCELoss
        loss = VonMisesFisherNCELoss(
            tau=1.0, use_queue=False,
            use_csls_training=True, csls_k=5,
            isf_weight=0.3,
        )
        B, D = 16, 64
        raw = torch.randn(B, D, requires_grad=True)
        mu = nn.functional.normalize(raw, dim=-1)
        kappa = torch.ones(B, 1) * 50.0
        keys = nn.functional.normalize(torch.randn(B, D), dim=-1)
        l = loss(mu, kappa, keys)
        l.backward()
        assert l.isfinite()
        assert raw.grad is not None
