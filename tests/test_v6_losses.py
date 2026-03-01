"""
Unit tests for v6 novel loss improvements:
  - DeltaSPCLVMFNCELoss (disagreement-aware curriculum)
  - VMFSoftCLIPLoss (kappa as dynamic student temperature)
  - slerp() (spherical linear interpolation for MixCo)

Run:  pytest tests/test_v6_losses.py -v
"""

import pytest
import torch
import torch.nn.functional as F


# ── DeltaSPCLVMFNCELoss ──────────────────────────────────────────────────


class TestDeltaSPCLVMFNCELoss:
    """Tests for disagreement-aware self-paced contrastive learning."""

    @pytest.fixture
    def loss_fn(self):
        from fmri2img.losses.vmf_nce import DeltaSPCLVMFNCELoss
        return DeltaSPCLVMFNCELoss(
            tau=1.0, use_queue=False, delta_weight=10.0,
            initial_curriculum_t=50.0,
        )

    @pytest.fixture
    def batch(self):
        torch.manual_seed(42)
        B, D = 8, 64
        mu = F.normalize(torch.randn(B, D), dim=-1)
        kappa = torch.rand(B, 1) * 20 + 5  # [5, 25]
        keys = F.normalize(torch.randn(B, D), dim=-1)
        return mu, kappa, keys

    def test_output_is_scalar(self, loss_fn, batch):
        mu, kappa, keys = batch
        loss = loss_fn(mu, kappa, keys)
        assert loss.dim() == 0

    def test_gradient_flow_mu_and_kappa(self, loss_fn, batch):
        mu, kappa, keys = batch
        mu.requires_grad_(True)
        kappa.requires_grad_(True)
        loss = loss_fn(mu, kappa, keys)
        loss.backward()
        assert mu.grad is not None and torch.isfinite(mu.grad).all()
        assert kappa.grad is not None and torch.isfinite(kappa.grad).all()

    def test_without_delta_fallback(self, loss_fn, batch):
        """When delta=None, should behave like kappa-only SPCL."""
        mu, kappa, keys = batch
        loss_no_delta = loss_fn(mu, kappa, keys, delta=None)
        assert torch.isfinite(loss_no_delta)

    def test_with_delta(self, loss_fn, batch):
        mu, kappa, keys = batch
        delta = torch.rand(8, 1) * 0.5
        loss = loss_fn(mu, kappa, keys, delta=delta)
        assert torch.isfinite(loss) and loss.item() > 0

    def test_nonuniform_delta_changes_loss(self, loss_fn, batch):
        """Non-uniform delta should change the sample weighting relative
        to zero delta (softmax is shift-invariant, so uniform delta has
        no effect -- we need per-sample variation)."""
        mu, kappa, keys = batch
        loss_zero_delta = loss_fn(mu, kappa, keys, delta=torch.zeros(8, 1))
        varied_delta = torch.linspace(0, 0.9, 8).unsqueeze(1)
        loss_varied_delta = loss_fn(mu, kappa, keys, delta=varied_delta)
        assert not torch.allclose(loss_zero_delta, loss_varied_delta, atol=1e-4)

    def test_curriculum_temperature_update(self, loss_fn):
        loss_fn.set_curriculum_temperature(5.0)
        assert loss_fn.curriculum_t == 5.0
        loss_fn.set_curriculum_temperature(-1.0)
        assert loss_fn.curriculum_t == 1e-6

    def test_batch_size_one(self, loss_fn):
        torch.manual_seed(0)
        mu = F.normalize(torch.randn(1, 64), dim=-1)
        kappa = torch.tensor([[10.0]])
        keys = F.normalize(torch.randn(1, 64), dim=-1)
        loss = loss_fn(mu, kappa, keys)
        assert torch.isfinite(loss)

    def test_oracle_lower_than_random(self, loss_fn):
        """Perfect match (mu == key) should yield lower loss than random."""
        torch.manual_seed(7)
        B, D = 16, 64
        keys = F.normalize(torch.randn(B, D), dim=-1)
        kappa = torch.ones(B, 1) * 20.0

        loss_oracle = loss_fn(keys.clone(), kappa, keys)
        mu_random = F.normalize(torch.randn(B, D), dim=-1)
        loss_random = loss_fn(mu_random, kappa, keys)
        assert loss_oracle.item() < loss_random.item()


# ── VMFSoftCLIPLoss ──────────────────────────────────────────────────────


class TestVMFSoftCLIPLoss:
    """Tests for probabilistic SoftCLIP with kappa-scaled student."""

    @pytest.fixture
    def loss_fn(self):
        from fmri2img.losses.softclip import VMFSoftCLIPLoss
        return VMFSoftCLIPLoss(
            teacher_tau=0.05, use_queue=False, symmetric=True,
        )

    @pytest.fixture
    def loss_fn_asym(self):
        from fmri2img.losses.softclip import VMFSoftCLIPLoss
        return VMFSoftCLIPLoss(
            teacher_tau=0.05, use_queue=False, symmetric=False,
        )

    @pytest.fixture
    def batch(self):
        torch.manual_seed(99)
        B, D = 8, 64
        mu = F.normalize(torch.randn(B, D), dim=-1)
        kappa = torch.rand(B, 1) * 20 + 5
        gt = F.normalize(torch.randn(B, D), dim=-1)
        return mu, kappa, gt

    def test_output_is_scalar(self, loss_fn, batch):
        mu, kappa, gt = batch
        loss = loss_fn(mu, kappa, gt)
        assert loss.dim() == 0

    def test_gradient_flow(self, loss_fn, batch):
        mu, kappa, gt = batch
        mu.requires_grad_(True)
        kappa.requires_grad_(True)
        loss = loss_fn(mu, kappa, gt)
        loss.backward()
        assert mu.grad is not None and torch.isfinite(mu.grad).all()
        assert kappa.grad is not None and torch.isfinite(kappa.grad).all()

    def test_symmetric_mode(self, loss_fn, loss_fn_asym, batch):
        mu, kappa, gt = batch
        loss_sym = loss_fn(mu, kappa, gt)
        loss_asym = loss_fn_asym(mu, kappa, gt)
        assert torch.isfinite(loss_sym) and torch.isfinite(loss_asym)
        assert not torch.allclose(loss_sym, loss_asym, atol=1e-4)

    def test_oracle_lower_loss(self, loss_fn):
        """mu == gt should give lower loss than random mu."""
        torch.manual_seed(3)
        B, D = 16, 64
        gt = F.normalize(torch.randn(B, D), dim=-1)
        kappa = torch.ones(B, 1) * 20.0

        loss_oracle = loss_fn(gt.clone(), kappa, gt)
        mu_random = F.normalize(torch.randn(B, D), dim=-1)
        loss_random = loss_fn(mu_random, kappa, gt)
        assert loss_oracle.item() < loss_random.item()

    def test_higher_kappa_sharpens_student(self, loss_fn_asym):
        """Higher kappa should produce a sharper student distribution,
        reducing loss when mu matches gt (oracle setting)."""
        torch.manual_seed(5)
        B, D = 16, 64
        gt = F.normalize(torch.randn(B, D), dim=-1)

        loss_low_k = loss_fn_asym(gt.clone(), torch.ones(B, 1) * 2.0, gt)
        loss_high_k = loss_fn_asym(gt.clone(), torch.ones(B, 1) * 50.0, gt)
        assert loss_high_k.item() < loss_low_k.item()

    def test_batch_size_one(self, loss_fn):
        torch.manual_seed(0)
        mu = F.normalize(torch.randn(1, 64), dim=-1)
        kappa = torch.tensor([[10.0]])
        gt = F.normalize(torch.randn(1, 64), dim=-1)
        loss = loss_fn(mu, kappa, gt)
        assert torch.isfinite(loss)


# ── slerp ─────────────────────────────────────────────────────────────────


class TestSlerp:
    """Tests for spherical linear interpolation."""

    @pytest.fixture
    def unit_vectors(self):
        torch.manual_seed(11)
        B, D = 16, 64
        v0 = F.normalize(torch.randn(B, D), dim=-1)
        v1 = F.normalize(torch.randn(B, D), dim=-1)
        return v0, v1

    def test_output_approximately_unit_norm(self, unit_vectors):
        from fmri2img.losses.mixco import slerp
        v0, v1 = unit_vectors
        t = torch.full((16,), 0.5)
        result = slerp(v0, v1, t)
        norms = result.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    def test_boundary_t0(self, unit_vectors):
        """slerp at t=0 should return v0."""
        from fmri2img.losses.mixco import slerp
        v0, v1 = unit_vectors
        t = torch.zeros(16)
        result = slerp(v0, v1, t)
        assert torch.allclose(result, v0, atol=1e-4)

    def test_boundary_t1(self, unit_vectors):
        """slerp at t=1 should return v1."""
        from fmri2img.losses.mixco import slerp
        v0, v1 = unit_vectors
        t = torch.ones(16)
        result = slerp(v0, v1, t)
        assert torch.allclose(result, v1, atol=1e-4)

    def test_midpoint_symmetry(self, unit_vectors):
        """slerp(v0, v1, 0.5) == slerp(v1, v0, 0.5) (midpoint is unique)."""
        from fmri2img.losses.mixco import slerp
        v0, v1 = unit_vectors
        t = torch.full((16,), 0.5)
        mid_01 = slerp(v0, v1, t)
        mid_10 = slerp(v1, v0, t)
        assert torch.allclose(mid_01, mid_10, atol=1e-4)

    def test_gradient_flow(self, unit_vectors):
        from fmri2img.losses.mixco import slerp
        v0, v1 = unit_vectors
        v0.requires_grad_(True)
        t = torch.full((16,), 0.3)
        result = slerp(v0, v1, t)
        result.sum().backward()
        assert v0.grad is not None and torch.isfinite(v0.grad).all()

    def test_nearly_identical_vectors(self):
        """When v0 ≈ v1, slerp should not produce NaN."""
        from fmri2img.losses.mixco import slerp
        torch.manual_seed(0)
        v = F.normalize(torch.randn(4, 32), dim=-1)
        v1 = v + 1e-7 * torch.randn_like(v)
        v1 = F.normalize(v1, dim=-1)
        t = torch.full((4,), 0.5)
        result = slerp(v, v1, t)
        assert torch.isfinite(result).all()

    def test_opposite_vectors(self):
        """When v0 = -v1, slerp should still produce a finite result."""
        from fmri2img.losses.mixco import slerp
        torch.manual_seed(1)
        v0 = F.normalize(torch.randn(4, 32), dim=-1)
        v1 = -v0
        t = torch.full((4,), 0.5)
        result = slerp(v0, v1, t)
        assert torch.isfinite(result).all()

    def test_per_element_interpolation(self):
        """Each element in batch can have a different t."""
        from fmri2img.losses.mixco import slerp
        torch.manual_seed(2)
        v0 = F.normalize(torch.randn(8, 32), dim=-1)
        v1 = F.normalize(torch.randn(8, 32), dim=-1)
        t = torch.linspace(0, 1, 8)
        result = slerp(v0, v1, t)
        assert result.shape == (8, 32)
        assert torch.isfinite(result).all()


# ── mixco_augment with slerp ─────────────────────────────────────────────


class TestMixCoSlerp:
    """Tests for mixco_augment with use_slerp=True."""

    def test_slerp_mode_runs(self):
        from fmri2img.losses.mixco import mixco_augment
        torch.manual_seed(42)
        B, V, D = 8, 100, 64
        fmri = torch.randn(B, V)
        clip = F.normalize(torch.randn(B, D), dim=-1)
        fmri_mix, clip_mix, labels = mixco_augment(
            fmri, clip, alpha=0.2, use_slerp=True,
        )
        assert fmri_mix.shape == (B, V)
        assert clip_mix.shape == (B, D)
        assert labels.shape == (B, B)

    def test_linear_vs_slerp_differ(self):
        """Slerp and linear interpolation should give different results."""
        from fmri2img.losses.mixco import mixco_augment
        import numpy as np
        np.random.seed(42)
        torch.manual_seed(42)
        B, V, D = 8, 100, 64
        fmri = torch.randn(B, V)
        clip = F.normalize(torch.randn(B, D), dim=-1)
        perm = torch.randperm(B)

        np.random.seed(42)
        _, clip_lin, _ = mixco_augment(fmri, clip, perm=perm, use_slerp=False)
        np.random.seed(42)
        _, clip_slerp, _ = mixco_augment(fmri, clip, perm=perm, use_slerp=True)
        assert not torch.allclose(clip_lin, clip_slerp, atol=1e-3)
