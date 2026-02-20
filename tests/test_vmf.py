"""
Unit tests for the vMF decoder, vMF-NCE loss, kappa regularizer,
and backward-compatible legacy decoder path.

Run:  pytest tests/test_vmf.py -v
"""

import pytest
import torch
import numpy as np


# ── Decoder tests ─────────────────────────────────────────────────────────


class TestVonMisesFisherDecoder:
    """Tests for the new bounded-sigmoid vMF decoder."""

    @pytest.fixture
    def decoder(self):
        from fmri2img.models.vmf_decoder import VonMisesFisherDecoder
        return VonMisesFisherDecoder(
            input_dim=256,
            output_dim=128,
            hidden_dims=[512],
            kappa_min=1e-3,
            kappa_max=500.0,
        )

    def test_mu_unit_norm(self, decoder):
        h = torch.randn(16, 256)
        mu, _ = decoder(h)
        norms = mu.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            f"mu should be unit-norm, got norms: {norms}"

    def test_kappa_positive_and_bounded(self, decoder):
        h = torch.randn(64, 256)
        _, kappa = decoder(h)
        assert (kappa >= decoder.kappa_min).all(), \
            f"kappa below kappa_min: min={kappa.min().item()}"
        assert (kappa <= decoder.kappa_max).all(), \
            f"kappa above kappa_max: max={kappa.max().item()}"
        assert (kappa > 0).all(), "kappa must be strictly positive"

    def test_output_shapes(self, decoder):
        h = torch.randn(8, 256)
        mu, kappa = decoder(h)
        assert mu.shape == (8, 128)
        assert kappa.shape == (8, 1)

    def test_kappa_stays_in_bounds_with_extreme_inputs(self, decoder):
        h_neg = torch.full((4, 256), -100.0)
        h_pos = torch.full((4, 256), 100.0)
        _, kappa_neg = decoder(h_neg)
        _, kappa_pos = decoder(h_pos)
        assert (kappa_neg >= decoder.kappa_min).all()
        assert (kappa_neg <= decoder.kappa_max).all()
        assert (kappa_pos >= decoder.kappa_min).all()
        assert (kappa_pos <= decoder.kappa_max).all()

    def test_gradient_flow(self, decoder):
        h = torch.randn(8, 256, requires_grad=True)
        mu, kappa = decoder(h)
        loss = mu.sum() + kappa.sum()
        loss.backward()
        assert h.grad is not None
        assert torch.isfinite(h.grad).all()


# ── Loss tests ────────────────────────────────────────────────────────────


class TestVonMisesFisherNCELoss:
    """Tests for the Bessel-free vMF-NCE contrastive loss."""

    @pytest.fixture
    def loss_fn(self):
        from fmri2img.losses.vmf_nce import VonMisesFisherNCELoss
        return VonMisesFisherNCELoss(tau=0.07, use_queue=False, kappa_is_log=False)

    def test_no_nan_on_random_inputs(self, loss_fn):
        B, D = 32, 768
        mu = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
        kappa = torch.rand(B, 1) * 100 + 1
        keys = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
        loss = loss_fn(mu, kappa, keys)
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    def test_loss_positive(self, loss_fn):
        B, D = 16, 128
        mu = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
        kappa = torch.ones(B, 1) * 50
        keys = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
        loss = loss_fn(mu, kappa, keys)
        assert loss.item() > 0, "InfoNCE-style loss should be positive"

    def test_no_bessel_in_logits(self):
        """Verify _score does NOT call _log_vmf_normaliser."""
        import inspect
        from fmri2img.losses.vmf_nce import VonMisesFisherNCELoss
        src = inspect.getsource(VonMisesFisherNCELoss._score)
        assert "_log_vmf_normaliser" not in src, \
            "_score should NOT use _log_vmf_normaliser (Bessel-free)"

    def test_gradient_flow(self, loss_fn):
        B, D = 8, 64
        mu = torch.nn.functional.normalize(torch.randn(B, D), dim=-1).requires_grad_(True)
        kappa = (torch.rand(B, 1) * 100 + 1).requires_grad_(True)
        keys = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
        loss = loss_fn(mu, kappa, keys)
        loss.backward()
        assert mu.grad is not None and torch.isfinite(mu.grad).all()
        assert kappa.grad is not None and torch.isfinite(kappa.grad).all()

    def test_perfect_alignment_low_loss(self, loss_fn):
        B, D = 16, 64
        keys = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
        mu = keys.clone()
        kappa = torch.ones(B, 1) * 200
        loss_aligned = loss_fn(mu, kappa, keys)

        mu_rand = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
        loss_random = loss_fn(mu_rand, kappa, keys)

        assert loss_aligned.item() < loss_random.item(), \
            "Aligned predictions should have lower loss than random"


class TestVonMisesFisherNCELossLegacy:
    """Test that log-kappa path still works for legacy configs."""

    def test_log_kappa_path(self):
        from fmri2img.losses.vmf_nce import VonMisesFisherNCELoss
        loss_fn = VonMisesFisherNCELoss(tau=0.07, use_queue=False, kappa_is_log=True)
        B, D = 8, 64
        mu = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
        log_kappa = torch.randn(B, 1)
        keys = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
        loss = loss_fn(mu, log_kappa, keys)
        assert torch.isfinite(loss)


class TestVonMisesFisherNLLLoss:
    """Tests for the vMF NLL (kept with Bessel normaliser)."""

    def test_finite_on_random_inputs(self):
        from fmri2img.losses.vmf_nce import VonMisesFisherNLLLoss
        loss_fn = VonMisesFisherNLLLoss(dim=64, kappa_is_log=False)
        B = 8
        mu = torch.nn.functional.normalize(torch.randn(B, 64), dim=-1)
        kappa = torch.rand(B, 1) * 50 + 1
        target = torch.nn.functional.normalize(torch.randn(B, 64), dim=-1)
        loss = loss_fn(mu, kappa, target)
        assert torch.isfinite(loss), f"NLL loss not finite: {loss.item()}"

    def test_log_kappa_path(self):
        from fmri2img.losses.vmf_nce import VonMisesFisherNLLLoss
        loss_fn = VonMisesFisherNLLLoss(dim=64, kappa_is_log=True)
        B = 8
        mu = torch.nn.functional.normalize(torch.randn(B, 64), dim=-1)
        log_kappa = torch.randn(B, 1)
        target = torch.nn.functional.normalize(torch.randn(B, 64), dim=-1)
        loss = loss_fn(mu, log_kappa, target)
        assert torch.isfinite(loss)


class TestKappaRegularizer:
    def test_basic(self):
        from fmri2img.losses.vmf_nce import kappa_regularizer
        kappa = torch.tensor([10.0, 20.0, 30.0])
        reg = kappa_regularizer(kappa, lambda_kappa=0.01)
        expected = 0.01 * 20.0
        assert abs(reg.item() - expected) < 1e-5

    def test_zero_lambda(self):
        from fmri2img.losses.vmf_nce import kappa_regularizer
        kappa = torch.tensor([100.0, 200.0])
        reg = kappa_regularizer(kappa, lambda_kappa=0.0)
        assert reg.item() == 0.0


# ── Backward compat: legacy decoder ──────────────────────────────────────


class TestLegacyDecoder:
    def test_legacy_decoder_still_works(self):
        from fmri2img.models.unified_model import VonMisesFisherDecoderLegacy
        dec = VonMisesFisherDecoderLegacy(
            input_dim=256,
            output_dim=128,
            hidden_dims=[512],
            log_kappa_min=-2.0,
            log_kappa_max=8.0,
        )
        h = torch.randn(8, 256)
        mu, log_kappa = dec(h)
        assert mu.shape == (8, 128)
        assert log_kappa.shape == (8, 1)
        norms = mu.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
        assert (log_kappa >= -2.0).all()
        assert (log_kappa <= 8.0).all()

    def test_unified_model_legacy_config(self):
        from fmri2img.models.unified_model import create_model
        config = {
            "type": "vmf",
            "encoder": {
                "input_dim": 100,
                "hidden_dims": [256],
            },
            "decoder": {
                "output_dim": 64,
                "hidden_dims": [128],
                "log_kappa_min": -2.0,
                "log_kappa_max": 8.0,
            },
        }
        model = create_model(config)
        assert model.vmf_output_is_log is True
        x = torch.randn(4, 100)
        mu, log_kappa = model(x)
        assert mu.shape == (4, 64)
        assert log_kappa.shape == (4, 1)

    def test_unified_model_new_config(self):
        from fmri2img.models.unified_model import create_model
        config = {
            "type": "vmf",
            "posterior": "vmf",
            "encoder": {
                "input_dim": 100,
                "hidden_dims": [256],
            },
            "decoder": {
                "output_dim": 64,
                "hidden_dims": [128],
                "kappa_min": 0.001,
                "kappa_max": 500.0,
            },
        }
        model = create_model(config)
        assert model.vmf_output_is_log is False
        x = torch.randn(4, 100)
        mu, kappa = model(x)
        assert mu.shape == (4, 64)
        assert kappa.shape == (4, 1)
        assert (kappa >= 0.001).all()
        assert (kappa <= 500.0).all()


# ── Eval: CSLS + hubness ─────────────────────────────────────────────────

try:
    from fmri2img.eval.embedding_metrics import compute_csls_similarity  # noqa: F401
    _SKLEARN_OK = True
except (ImportError, ValueError):
    _SKLEARN_OK = False


@pytest.mark.skipif(not _SKLEARN_OK, reason="sklearn/numpy binary incompatibility")
class TestCSLSAndHubness:
    def test_csls_similarity_shape(self):
        from fmri2img.eval.embedding_metrics import compute_csls_similarity
        N, M, D = 20, 50, 64
        q = np.random.randn(N, D).astype(np.float32)
        g = np.random.randn(M, D).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        csls = compute_csls_similarity(q, g, k_csls=5)
        assert csls.shape == (N, M)

    def test_hubness_metrics(self):
        from fmri2img.eval.embedding_metrics import compute_hubness_metrics
        sim = np.random.randn(30, 100).astype(np.float32)
        hub = compute_hubness_metrics(sim, k=5)
        assert isinstance(hub.k_occurrence_skewness, float)
        assert len(hub.top_hubs) <= 5

    def test_retrieval_with_csls(self):
        from fmri2img.eval.embedding_metrics import compute_retrieval_metrics
        N, D = 30, 64
        emb = np.random.randn(N, D).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        ids = np.arange(N)
        metrics = compute_retrieval_metrics(
            emb, emb, ids, ids,
            use_csls=True, k_csls=5, compute_hubness=True,
        )
        assert metrics.r_at_1 >= 0.9, "Self-retrieval with CSLS should be near-perfect"
        assert metrics.hubness is not None


# ── Kappa calibration ────────────────────────────────────────────────────


class TestKappaCalibration:
    def test_normalise_kappa(self):
        from fmri2img.eval.kappa_calibration import normalise_kappa
        cal = {"q10": 10.0, "q50": 50.0, "q90": 100.0}
        assert normalise_kappa(10.0, cal) == pytest.approx(0.0)
        assert normalise_kappa(100.0, cal) == pytest.approx(1.0)
        assert normalise_kappa(55.0, cal) == pytest.approx(0.5)
        assert normalise_kappa(0.0, cal) == 0.0
        assert normalise_kappa(200.0, cal) == 1.0

    def test_save_load_roundtrip(self, tmp_path):
        from fmri2img.eval.kappa_calibration import save_calibration, load_calibration
        cal = {"q10": 5.0, "q50": 25.0, "q90": 80.0, "mean": 30.0,
               "std": 20.0, "min": 1.0, "max": 120.0, "n_samples": 500}
        path = tmp_path / "cal.json"
        save_calibration(cal, path)
        loaded = load_calibration(path)
        assert loaded == cal
