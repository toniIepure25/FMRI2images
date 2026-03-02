"""Tests for softplus kappa mode in VonMisesFisherDecoder and PerROIVMFHeads."""

import pytest
import torch
import torch.nn.functional as F

from fmri2img.models.vmf_decoder import VonMisesFisherDecoder, kappa_activation
from fmri2img.models.roi_dcf import PerROIVMFHeads, ROIDCFDecoder


# ---------------------------------------------------------------------------
# kappa_activation unit tests
# ---------------------------------------------------------------------------

class TestKappaActivation:
    def test_bounded_sigmoid_within_bounds(self):
        raw = torch.randn(32, 1)
        kappa = kappa_activation(raw, mode="bounded_sigmoid", kappa_min=1.0, kappa_max=50.0)
        assert (kappa >= 1.0).all()
        assert (kappa <= 50.0).all()

    def test_softplus_positive(self):
        raw = torch.randn(32, 1)
        kappa = kappa_activation(raw, mode="softplus")
        assert (kappa >= 1.0).all()

    def test_softplus_unbounded_above_old_max(self):
        raw = torch.full((4, 1), 100.0)
        kappa = kappa_activation(raw, mode="softplus")
        assert (kappa > 50.0).all()

    def test_softplus_amp_ceil(self):
        raw = torch.full((4, 1), 1e6)
        kappa = kappa_activation(raw, mode="softplus")
        assert (kappa <= 5000.0).all()

    def test_softplus_gradient_nonzero(self):
        raw = torch.randn(16, 1, requires_grad=True)
        kappa = kappa_activation(raw, mode="softplus")
        kappa.sum().backward()
        assert raw.grad is not None
        assert (raw.grad.abs() > 0).all()

    def test_bounded_sigmoid_gradient_vanishes_near_bounds(self):
        """Bounded sigmoid has near-zero gradients at extremes — the problem softplus fixes."""
        raw_high = torch.full((4, 1), 20.0, requires_grad=True)
        kappa = kappa_activation(raw_high, mode="bounded_sigmoid", kappa_min=1.0, kappa_max=50.0)
        kappa.sum().backward()
        assert raw_high.grad.abs().max() < 1e-4

    def test_softplus_gradient_healthy_at_same_extreme(self):
        """Softplus maintains gradient flow where bounded sigmoid fails."""
        raw_high = torch.full((4, 1), 20.0, requires_grad=True)
        kappa = kappa_activation(raw_high, mode="softplus")
        kappa.sum().backward()
        assert raw_high.grad.abs().max() > 0.5


# ---------------------------------------------------------------------------
# VonMisesFisherDecoder tests
# ---------------------------------------------------------------------------

class TestVonMisesFisherDecoderSoftplus:
    @pytest.fixture
    def decoder_softplus(self):
        return VonMisesFisherDecoder(
            input_dim=512, output_dim=768, hidden_dims=[256],
            kappa_mode="softplus",
        )

    @pytest.fixture
    def decoder_bounded(self):
        return VonMisesFisherDecoder(
            input_dim=512, output_dim=768, hidden_dims=[256],
            kappa_mode="bounded_sigmoid", kappa_min=1.0, kappa_max=50.0,
        )

    def test_forward_shapes(self, decoder_softplus):
        h = torch.randn(8, 512)
        mu, kappa = decoder_softplus(h)
        assert mu.shape == (8, 768)
        assert kappa.shape == (8, 1)

    def test_mu_unit_norm(self, decoder_softplus):
        h = torch.randn(8, 512)
        mu, _ = decoder_softplus(h)
        norms = mu.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_kappa_positive(self, decoder_softplus):
        h = torch.randn(8, 512)
        _, kappa = decoder_softplus(h)
        assert (kappa > 0).all()

    def test_backward_gradient_flow(self, decoder_softplus):
        h = torch.randn(8, 512, requires_grad=True)
        mu, kappa = decoder_softplus(h)
        loss = mu.sum() + kappa.sum()
        loss.backward()
        assert h.grad is not None
        assert h.grad.abs().sum() > 0

    def test_amp_float16_no_nan(self, decoder_softplus):
        decoder_softplus.cuda() if torch.cuda.is_available() else None
        device = "cuda" if torch.cuda.is_available() else "cpu"
        decoder_softplus = decoder_softplus.to(device)
        h = torch.randn(8, 512, device=device)
        with torch.amp.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            mu, kappa = decoder_softplus(h)
        assert not torch.isnan(mu).any()
        assert not torch.isnan(kappa).any()
        assert not torch.isinf(kappa).any()

    def test_backward_compat_bounded_sigmoid(self, decoder_bounded):
        h = torch.randn(8, 512)
        _, kappa = decoder_bounded(h)
        assert (kappa >= 1.0).all()
        assert (kappa <= 50.0).all()


# ---------------------------------------------------------------------------
# PerROIVMFHeads tests
# ---------------------------------------------------------------------------

class TestPerROIVMFHeadsSoftplus:
    @pytest.fixture
    def heads_softplus(self):
        return PerROIVMFHeads(
            d_model=768, output_dim=768, n_rois=17,
            shared=True, kappa_mode="softplus",
        )

    @pytest.fixture
    def heads_bounded(self):
        return PerROIVMFHeads(
            d_model=768, output_dim=768, n_rois=17,
            shared=True, kappa_mode="bounded_sigmoid",
            kappa_min=1.0, kappa_max=50.0,
        )

    def test_forward_shapes(self, heads_softplus):
        tokens = torch.randn(4, 17, 768)
        mus, kappas = heads_softplus(tokens)
        assert mus.shape == (4, 17, 768)
        assert kappas.shape == (4, 17, 1)

    def test_kappa_positive(self, heads_softplus):
        tokens = torch.randn(4, 17, 768)
        _, kappas = heads_softplus(tokens)
        assert (kappas > 0).all()

    def test_gradient_flow_to_tokens(self, heads_softplus):
        tokens = torch.randn(4, 17, 768, requires_grad=True)
        mus, kappas = heads_softplus(tokens)
        (mus.sum() + kappas.sum()).backward()
        assert tokens.grad is not None
        assert tokens.grad.abs().sum() > 0

    def test_bounded_mode_backward_compat(self, heads_bounded):
        tokens = torch.randn(4, 17, 768)
        _, kappas = heads_bounded(tokens)
        assert (kappas >= 1.0).all()
        assert (kappas <= 50.0).all()


# ---------------------------------------------------------------------------
# ROIDCFDecoder with softplus integration test
# ---------------------------------------------------------------------------

class TestROIDCFDecoderSoftplus:
    def test_full_forward_softplus(self):
        decoder = ROIDCFDecoder(
            d_model=768, output_dim=768, n_rois=17,
            shared=True, kappa_mode="softplus",
        )
        roi_tokens = torch.randn(4, 17, 768)
        alphas = F.softmax(torch.randn(4, 17), dim=-1)
        mu_fused, kappa_consensus = decoder(roi_tokens, alphas)
        assert mu_fused.shape == (4, 768)
        assert kappa_consensus.shape == (4, 1)
        assert (kappa_consensus > 0).all()

    def test_full_forward_with_per_roi(self):
        decoder = ROIDCFDecoder(
            d_model=768, output_dim=768, n_rois=17,
            shared=True, kappa_mode="softplus",
        )
        roi_tokens = torch.randn(4, 17, 768)
        alphas = F.softmax(torch.randn(4, 17), dim=-1)
        mu_fused, kappa_consensus, per_roi_mus, per_roi_kappas, delta = decoder(
            roi_tokens, alphas, return_per_roi=True,
        )
        assert per_roi_kappas.shape == (4, 17, 1)
        assert (per_roi_kappas > 0).all()
        assert delta.shape == (4, 1)
