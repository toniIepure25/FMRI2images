"""
Unit tests for ROI-DCF: Per-ROI Directional Experts with Spherical Consensus
Fusion — PerROIVMFHeads, SphericalConsensusFusion, ROIDCFDecoder,
MultiTaskVMFNCELoss, and end-to-end integration with UnifiedModel.

Run:  pytest tests/test_roi_dcf.py -v
"""

import pytest
import torch
import torch.nn.functional as F


# ── PerROIVMFHeads tests ──────────────────────────────────────────────────


class TestPerROIVMFHeads:
    """Tests for per-ROI vMF prediction heads."""

    @pytest.fixture
    def shared_heads(self):
        from fmri2img.models.roi_dcf import PerROIVMFHeads
        return PerROIVMFHeads(
            d_model=64, output_dim=32, n_rois=5,
            shared=True, kappa_min=1e-3, kappa_max=500.0,
        )

    @pytest.fixture
    def independent_heads(self):
        from fmri2img.models.roi_dcf import PerROIVMFHeads
        return PerROIVMFHeads(
            d_model=64, output_dim=32, n_rois=5,
            shared=False, kappa_min=1e-3, kappa_max=500.0,
        )

    def test_output_shapes_shared(self, shared_heads):
        tokens = torch.randn(4, 5, 64)
        mus, kappas = shared_heads(tokens)
        assert mus.shape == (4, 5, 32)
        assert kappas.shape == (4, 5, 1)

    def test_output_shapes_independent(self, independent_heads):
        tokens = torch.randn(4, 5, 64)
        mus, kappas = independent_heads(tokens)
        assert mus.shape == (4, 5, 32)
        assert kappas.shape == (4, 5, 1)

    def test_mu_unit_norm(self, shared_heads):
        tokens = torch.randn(8, 5, 64)
        mus, _ = shared_heads(tokens)
        norms = mus.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            f"Per-ROI mu should be unit-norm, got norms in [{norms.min():.6f}, {norms.max():.6f}]"

    def test_kappa_bounded(self, shared_heads):
        tokens = torch.randn(16, 5, 64)
        _, kappas = shared_heads(tokens)
        assert (kappas >= shared_heads.kappa_min).all()
        assert (kappas <= shared_heads.kappa_max).all()

    def test_kappa_bounded_extreme_inputs(self, shared_heads):
        tokens_neg = torch.full((4, 5, 64), -100.0)
        tokens_pos = torch.full((4, 5, 64), 100.0)
        _, k_neg = shared_heads(tokens_neg)
        _, k_pos = shared_heads(tokens_pos)
        assert (k_neg >= shared_heads.kappa_min).all()
        assert (k_pos <= shared_heads.kappa_max).all()

    def test_gradient_flow(self, shared_heads):
        tokens = torch.randn(4, 5, 64, requires_grad=True)
        mus, kappas = shared_heads(tokens)
        loss = mus.sum() + kappas.sum()
        loss.backward()
        assert tokens.grad is not None
        assert torch.isfinite(tokens.grad).all()

    def test_with_hidden_dim(self):
        from fmri2img.models.roi_dcf import PerROIVMFHeads
        heads = PerROIVMFHeads(
            d_model=64, output_dim=32, n_rois=3,
            shared=True, hidden_dim=128,
        )
        tokens = torch.randn(4, 3, 64)
        mus, kappas = heads(tokens)
        assert mus.shape == (4, 3, 32)
        assert kappas.shape == (4, 3, 1)


# ── SphericalConsensusFusion tests ────────────────────────────────────────


class TestSphericalConsensusFusion:
    """Tests for the spherical weighted mean fusion module."""

    @pytest.fixture
    def fusion(self):
        from fmri2img.models.roi_dcf import SphericalConsensusFusion
        return SphericalConsensusFusion()

    def test_output_shapes(self, fusion):
        B, R, D = 4, 5, 32
        mus = F.normalize(torch.randn(B, R, D), dim=-1)
        kappas = torch.rand(B, R, 1) * 100 + 1
        alphas = F.softmax(torch.randn(B, R), dim=-1)
        mu_f, kappa_c, delta = fusion(mus, kappas, alphas)
        assert mu_f.shape == (B, D)
        assert kappa_c.shape == (B, 1)
        assert delta.shape == (B, 1)

    def test_mu_fused_unit_norm(self, fusion):
        B, R, D = 8, 3, 16
        mus = F.normalize(torch.randn(B, R, D), dim=-1)
        kappas = torch.ones(B, R, 1) * 50
        alphas = F.softmax(torch.randn(B, R), dim=-1)
        mu_f, _, _ = fusion(mus, kappas, alphas)
        norms = mu_f.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_delta_zero_when_all_agree(self, fusion):
        """When all ROIs predict the same direction, delta should be ~0."""
        B, R, D = 4, 5, 16
        direction = F.normalize(torch.randn(B, 1, D), dim=-1)
        mus = direction.expand(B, R, D)
        kappas = torch.ones(B, R, 1) * 100
        alphas = torch.ones(B, R) / R
        _, _, delta = fusion(mus, kappas, alphas)
        assert (delta < 0.01).all(), \
            f"Delta should be ~0 when all ROIs agree, got {delta.max():.4f}"

    def test_delta_high_when_disagreeing(self, fusion):
        """When ROIs point in opposing directions, delta should be high."""
        B, D = 2, 16
        R = 2
        mu1 = F.normalize(torch.randn(B, 1, D), dim=-1)
        mu2 = -mu1
        mus = torch.cat([mu1, mu2], dim=1)
        kappas = torch.ones(B, R, 1) * 100
        alphas = torch.ones(B, R) / R
        _, _, delta = fusion(mus, kappas, alphas)
        assert (delta > 0.9).all(), \
            f"Delta should be high when ROIs oppose, got {delta.min():.4f}"

    def test_kappa_consensus_higher_when_agreeing(self, fusion):
        """Kappa consensus should be higher when ROIs agree vs disagree."""
        B, D = 4, 16
        R = 4
        direction = F.normalize(torch.randn(B, 1, D), dim=-1)
        kappas = torch.ones(B, R, 1) * 50
        alphas = torch.ones(B, R) / R

        mus_agree = direction.expand(B, R, D)
        _, kappa_agree, _ = fusion(mus_agree, kappas, alphas)

        mus_random = F.normalize(torch.randn(B, R, D), dim=-1)
        _, kappa_random, _ = fusion(mus_random, kappas, alphas)

        assert (kappa_agree > kappa_random).all(), \
            "Consensus kappa should be higher when ROIs agree"

    def test_delta_bounded_zero_one(self, fusion):
        B, R, D = 16, 5, 32
        mus = F.normalize(torch.randn(B, R, D), dim=-1)
        kappas = torch.rand(B, R, 1) * 200
        alphas = F.softmax(torch.randn(B, R), dim=-1)
        _, _, delta = fusion(mus, kappas, alphas)
        assert (delta >= 0).all() and (delta <= 1).all(), \
            f"Delta out of [0, 1]: [{delta.min():.4f}, {delta.max():.4f}]"

    def test_gradient_flow(self, fusion):
        B, R, D = 4, 3, 16
        mus = F.normalize(torch.randn(B, R, D), dim=-1).requires_grad_(True)
        kappas = (torch.rand(B, R, 1) * 50 + 1).requires_grad_(True)
        alphas = F.softmax(torch.randn(B, R), dim=-1).requires_grad_(True)
        mu_f, kappa_c, delta = fusion(mus, kappas, alphas)
        loss = mu_f.sum() + kappa_c.sum() + delta.sum()
        loss.backward()
        assert mus.grad is not None and torch.isfinite(mus.grad).all()
        assert kappas.grad is not None and torch.isfinite(kappas.grad).all()
        assert alphas.grad is not None and torch.isfinite(alphas.grad).all()


# ── ROIDCFDecoder tests ───────────────────────────────────────────────────


class TestROIDCFDecoder:
    """Tests for the complete ROI-DCF decoder module."""

    @pytest.fixture
    def decoder(self):
        from fmri2img.models.roi_dcf import ROIDCFDecoder
        return ROIDCFDecoder(
            d_model=64, output_dim=32, n_rois=5,
            shared=True, kappa_min=1e-3, kappa_max=500.0,
        )

    def test_forward_basic(self, decoder):
        B, R = 4, 5
        roi_tokens = torch.randn(B, R, 64)
        alphas = F.softmax(torch.randn(B, R), dim=-1)
        mu_f, kappa_c = decoder(roi_tokens, alphas, return_per_roi=False)
        assert mu_f.shape == (B, 32)
        assert kappa_c.shape == (B, 1)

    def test_forward_with_per_roi(self, decoder):
        B, R = 4, 5
        roi_tokens = torch.randn(B, R, 64)
        alphas = F.softmax(torch.randn(B, R), dim=-1)
        mu_f, kappa_c, pr_mus, pr_kappas, delta = decoder(
            roi_tokens, alphas, return_per_roi=True
        )
        assert mu_f.shape == (B, 32)
        assert kappa_c.shape == (B, 1)
        assert pr_mus.shape == (B, R, 32)
        assert pr_kappas.shape == (B, R, 1)
        assert delta.shape == (B, 1)

    def test_gradient_flow(self, decoder):
        B, R = 4, 5
        roi_tokens = torch.randn(B, R, 64, requires_grad=True)
        alphas = F.softmax(torch.randn(B, R), dim=-1)
        mu_f, kappa_c = decoder(roi_tokens, alphas)
        loss = mu_f.sum() + kappa_c.sum()
        loss.backward()
        assert roi_tokens.grad is not None
        assert torch.isfinite(roi_tokens.grad).all()

    def test_batch_size_one(self, decoder):
        roi_tokens = torch.randn(1, 5, 64)
        alphas = F.softmax(torch.randn(1, 5), dim=-1)
        mu_f, kappa_c = decoder(roi_tokens, alphas)
        assert mu_f.shape == (1, 32)
        assert torch.isfinite(mu_f).all()
        assert torch.isfinite(kappa_c).all()


# ── MultiTaskVMFNCELoss tests ────────────────────────────────────────────


class TestMultiTaskVMFNCELoss:
    """Tests for the multi-task fused + per-ROI vMF-NCE loss."""

    @pytest.fixture
    def loss_fn(self):
        from fmri2img.losses.vmf_nce import MultiTaskVMFNCELoss
        return MultiTaskVMFNCELoss(
            tau=0.07, use_queue=False, lambda_aux=0.5, kappa_is_log=False,
        )

    def test_fused_only(self, loss_fn):
        """Without per-ROI inputs, should fall back to fused loss only."""
        B, D = 8, 32
        mu = F.normalize(torch.randn(B, D), dim=-1)
        kappa = torch.rand(B, 1) * 50 + 1
        keys = F.normalize(torch.randn(B, D), dim=-1)
        total, fused, aux = loss_fn(mu, kappa, keys)
        assert torch.isfinite(total)
        assert total.item() == fused.item()
        assert aux.item() == 0.0

    def test_with_per_roi(self, loss_fn):
        B, R, D = 8, 5, 32
        mu_fused = F.normalize(torch.randn(B, D), dim=-1)
        kappa_cons = torch.rand(B, 1) * 50 + 1
        keys = F.normalize(torch.randn(B, D), dim=-1)
        pr_mus = F.normalize(torch.randn(B, R, D), dim=-1)
        pr_kappas = torch.rand(B, R, 1) * 50 + 1
        total, fused, aux = loss_fn(
            mu_fused, kappa_cons, keys,
            per_roi_mus=pr_mus, per_roi_kappas=pr_kappas,
        )
        assert torch.isfinite(total)
        assert torch.isfinite(fused)
        assert torch.isfinite(aux)
        assert total.item() > fused.item(), \
            "Total loss should be > fused loss when auxiliary is active"

    def test_gradient_flow(self, loss_fn):
        B, R, D = 4, 3, 16
        mu_fused = F.normalize(torch.randn(B, D), dim=-1).requires_grad_(True)
        kappa_cons = (torch.rand(B, 1) * 50 + 1).requires_grad_(True)
        keys = F.normalize(torch.randn(B, D), dim=-1)
        pr_mus = F.normalize(torch.randn(B, R, D), dim=-1).requires_grad_(True)
        pr_kappas = (torch.rand(B, R, 1) * 50 + 1).requires_grad_(True)
        total, _, _ = loss_fn(
            mu_fused, kappa_cons, keys,
            per_roi_mus=pr_mus, per_roi_kappas=pr_kappas,
        )
        total.backward()
        assert mu_fused.grad is not None and torch.isfinite(mu_fused.grad).all()
        assert kappa_cons.grad is not None and torch.isfinite(kappa_cons.grad).all()
        assert pr_mus.grad is not None and torch.isfinite(pr_mus.grad).all()
        assert pr_kappas.grad is not None and torch.isfinite(pr_kappas.grad).all()

    def test_lambda_aux_zero_disables_auxiliary(self):
        from fmri2img.losses.vmf_nce import MultiTaskVMFNCELoss
        loss_fn = MultiTaskVMFNCELoss(tau=0.07, use_queue=False, lambda_aux=0.0)
        B, R, D = 4, 3, 16
        mu_fused = F.normalize(torch.randn(B, D), dim=-1)
        kappa_cons = torch.rand(B, 1) * 50 + 1
        keys = F.normalize(torch.randn(B, D), dim=-1)
        pr_mus = F.normalize(torch.randn(B, R, D), dim=-1)
        pr_kappas = torch.rand(B, R, 1) * 50 + 1
        total, fused, aux = loss_fn(
            mu_fused, kappa_cons, keys,
            per_roi_mus=pr_mus, per_roi_kappas=pr_kappas,
        )
        assert total.item() == pytest.approx(fused.item(), abs=1e-6)
        assert aux.item() == 0.0

    def test_aligned_lower_loss(self, loss_fn):
        """Aligned predictions should produce lower fused loss than random."""
        B, D = 16, 32
        keys = F.normalize(torch.randn(B, D), dim=-1)
        kappa = torch.ones(B, 1) * 200

        total_aligned, _, _ = loss_fn(keys.clone(), kappa, keys)
        mu_rand = F.normalize(torch.randn(B, D), dim=-1)
        total_random, _, _ = loss_fn(mu_rand, kappa, keys)

        assert total_aligned.item() < total_random.item()


# ── KappaSPCLVMFNCELoss tests ────────────────────────────────────────────


class TestKappaSPCLVMFNCELoss:
    """Tests for the concentration-aware self-paced contrastive loss."""

    @pytest.fixture
    def loss_fn(self):
        from fmri2img.losses.vmf_nce import KappaSPCLVMFNCELoss
        return KappaSPCLVMFNCELoss(
            tau=0.07, use_queue=False, kappa_is_log=False,
            initial_curriculum_t=50.0,
        )

    def test_finite_on_random_inputs(self, loss_fn):
        B, D = 16, 32
        mu = F.normalize(torch.randn(B, D), dim=-1)
        kappa = torch.rand(B, 1) * 100 + 1
        keys = F.normalize(torch.randn(B, D), dim=-1)
        loss = loss_fn(mu, kappa, keys)
        assert torch.isfinite(loss)

    def test_gradient_flow(self, loss_fn):
        B, D = 8, 32
        mu = F.normalize(torch.randn(B, D), dim=-1).requires_grad_(True)
        kappa = (torch.rand(B, 1) * 50 + 1).requires_grad_(True)
        keys = F.normalize(torch.randn(B, D), dim=-1)
        loss = loss_fn(mu, kappa, keys)
        loss.backward()
        assert mu.grad is not None and torch.isfinite(mu.grad).all()
        assert kappa.grad is not None and torch.isfinite(kappa.grad).all()

    def test_high_temperature_near_uniform(self, loss_fn):
        """At high curriculum_t, weights should be nearly uniform."""
        loss_fn.set_curriculum_temperature(1e6)
        B, D = 16, 32
        mu = F.normalize(torch.randn(B, D), dim=-1)
        kappa_varied = torch.tensor([1.0, 100.0, 200.0, 500.0] * 4).unsqueeze(-1)
        keys = F.normalize(torch.randn(B, D), dim=-1)
        loss_high_t = loss_fn(mu, kappa_varied, keys)
        assert torch.isfinite(loss_high_t)

    def test_set_curriculum_temperature(self, loss_fn):
        loss_fn.set_curriculum_temperature(10.0)
        assert loss_fn.curriculum_t == 10.0
        loss_fn.set_curriculum_temperature(0.0)
        assert loss_fn.curriculum_t == 1e-6  # Clamped

    def test_aligned_lower_loss(self, loss_fn):
        B, D = 16, 32
        keys = F.normalize(torch.randn(B, D), dim=-1)
        kappa = torch.ones(B, 1) * 200
        loss_aligned = loss_fn(keys.clone(), kappa, keys)
        mu_rand = F.normalize(torch.randn(B, D), dim=-1)
        loss_random = loss_fn(mu_rand, kappa, keys)
        assert loss_aligned.item() < loss_random.item()


# ── ROI Transformer output tests ─────────────────────────────────────────


class TestROITransformerOutput:
    """Test the extended ROI Transformer with return_roi_tokens."""

    @pytest.fixture
    def encoder(self):
        from fmri2img.models.roi_transformer import ROITransformerEncoder
        return ROITransformerEncoder(
            roi_dims={"V1": 20, "V2": 15, "FFA": 10},
            d_model=32, nhead=4, num_layers=2, dropout=0.0,
        )

    def test_default_forward_unchanged(self, encoder):
        x = torch.randn(4, 45)
        out = encoder(x)
        assert out.shape == (4, 32)

    def test_return_roi_tokens(self, encoder):
        x = torch.randn(4, 45)
        out = encoder(x, return_roi_tokens=True)
        assert out.cls_out.shape == (4, 32)
        assert out.roi_tokens.shape == (4, 3, 32)
        assert out.cls_to_roi_alpha.shape == (4, 3)

    def test_alpha_sums_to_one(self, encoder):
        x = torch.randn(8, 45)
        out = encoder(x, return_roi_tokens=True)
        alpha_sums = out.cls_to_roi_alpha.sum(dim=-1)
        assert torch.allclose(alpha_sums, torch.ones_like(alpha_sums), atol=1e-5), \
            f"Alpha should sum to 1, got sums: {alpha_sums}"

    def test_alpha_non_negative(self, encoder):
        x = torch.randn(8, 45)
        out = encoder(x, return_roi_tokens=True)
        assert (out.cls_to_roi_alpha >= 0).all()


# ── End-to-end UnifiedModel integration ──────────────────────────────────


class TestUnifiedModelVMFDCF:
    """Integration test for vmf_dcf through the UnifiedModel factory."""

    @pytest.fixture
    def model(self):
        from fmri2img.models.unified_model import create_model
        config = {
            "type": "vmf_dcf",
            "posterior": "vmf",
            "encoder": {
                "encoder_type": "roi_transformer",
                "input_dim": 45,
                "roi_dims": {"V1": 20, "V2": 15, "FFA": 10},
                "d_model": 32,
                "nhead": 4,
                "num_layers": 2,
                "dropout": 0.0,
            },
            "decoder": {
                "output_dim": 16,
                "kappa_min": 0.001,
                "kappa_max": 500.0,
                "dropout": 0.0,
                "dcf": {
                    "shared_heads": True,
                    "hidden_dim": None,
                    "return_per_roi": True,
                },
            },
        }
        return create_model(config)

    def test_forward_shapes(self, model):
        x = torch.randn(4, 45)
        mu, kappa = model(x)
        assert mu.shape == (4, 16)
        assert kappa.shape == (4, 1)

    def test_mu_unit_norm(self, model):
        x = torch.randn(8, 45)
        mu, _ = model(x)
        norms = mu.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_kappa_positive(self, model):
        x = torch.randn(8, 45)
        _, kappa = model(x)
        assert (kappa > 0).all()

    def test_dcf_extras_available(self, model):
        x = torch.randn(4, 45)
        model(x)
        extras = model._last_dcf_extras
        assert "per_roi_mus" in extras
        assert "per_roi_kappas" in extras
        assert "delta" in extras
        assert "cls_to_roi_alpha" in extras
        assert extras["per_roi_mus"].shape == (4, 3, 16)
        assert extras["per_roi_kappas"].shape == (4, 3, 1)
        assert extras["delta"].shape == (4, 1)

    def test_gradient_flow_end_to_end(self, model):
        x = torch.randn(4, 45, requires_grad=True)
        mu, kappa = model(x)
        loss = mu.sum() + kappa.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_vmf_dcf_requires_roi_transformer(self):
        from fmri2img.models.unified_model import create_model
        config = {
            "type": "vmf_dcf",
            "encoder": {
                "encoder_type": "mlp",
                "input_dim": 100,
                "hidden_dims": [64],
            },
            "decoder": {"output_dim": 16, "kappa_min": 0.001},
        }
        with pytest.raises(ValueError, match="roi_transformer"):
            create_model(config)

    def test_vmf_output_is_log_false(self, model):
        assert model.vmf_output_is_log is False

    def test_mini_training_loop(self, model):
        """Verify loss decreases over a few steps on toy data."""
        from fmri2img.losses.vmf_nce import MultiTaskVMFNCELoss

        torch.manual_seed(42)
        loss_fn = MultiTaskVMFNCELoss(
            tau=0.07, use_queue=False, lambda_aux=0.5, kappa_is_log=False,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        B = 8
        x = torch.randn(B, 45)
        targets = F.normalize(torch.randn(B, 16), dim=-1)

        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            mu, kappa = model(x)
            extras = model._last_dcf_extras
            total, _, _ = loss_fn(
                mu, kappa, targets,
                per_roi_mus=extras["per_roi_mus"],
                per_roi_kappas=extras["per_roi_kappas"],
            )
            total.backward()
            optimizer.step()
            losses.append(total.item())

        assert losses[-1] < losses[0], \
            f"Loss should decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
