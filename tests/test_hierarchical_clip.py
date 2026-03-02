"""
Unit tests for Hierarchical CLIP Alignment Loss.

Run:  pytest tests/test_hierarchical_clip.py -v
"""

import pytest
import torch
import torch.nn.functional as F

from fmri2img.losses.hierarchical_clip_loss import (
    HierarchicalCLIPLoss,
    DEFAULT_TIER_INDICES,
    DEFAULT_TIER_CLIP_COLUMNS,
)


class TestHierarchicalCLIPLoss:
    """Tests for the hierarchical CLIP alignment loss."""

    @pytest.fixture
    def loss_fn(self):
        return HierarchicalCLIPLoss()

    def test_output_shape(self, loss_fn):
        B, R, D = 8, 16, 768
        per_roi_mus = F.normalize(torch.randn(B, R, D), dim=-1)
        alphas = F.softmax(torch.randn(B, R), dim=-1)
        tier_targets = {
            "early": F.normalize(torch.randn(B, D), dim=-1),
            "mid": F.normalize(torch.randn(B, D), dim=-1),
            "high": F.normalize(torch.randn(B, D), dim=-1),
        }
        loss, details = loss_fn(per_roi_mus, alphas, tier_targets)
        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert "hier_early" in details
        assert "hier_mid" in details
        assert "hier_high" in details

    def test_loss_positive(self, loss_fn):
        """Cosine distance loss should be >= 0."""
        B, R, D = 4, 16, 32
        per_roi_mus = F.normalize(torch.randn(B, R, D), dim=-1)
        alphas = F.softmax(torch.randn(B, R), dim=-1)
        tier_targets = {
            "early": F.normalize(torch.randn(B, D), dim=-1),
            "mid": F.normalize(torch.randn(B, D), dim=-1),
            "high": F.normalize(torch.randn(B, D), dim=-1),
        }
        loss, _ = loss_fn(per_roi_mus, alphas, tier_targets)
        assert loss.item() >= 0

    def test_zero_loss_when_aligned(self, loss_fn):
        """Loss should be ~0 when per-ROI mus match their tier targets."""
        B, R, D = 4, 16, 32
        # Create a single direction per tier
        early_dir = F.normalize(torch.randn(1, D), dim=-1)
        mid_dir = F.normalize(torch.randn(1, D), dim=-1)
        high_dir = F.normalize(torch.randn(1, D), dim=-1)

        per_roi_mus = torch.zeros(B, R, D)
        for idx in DEFAULT_TIER_INDICES["early"]:
            per_roi_mus[:, idx] = early_dir
        for idx in DEFAULT_TIER_INDICES["mid"]:
            per_roi_mus[:, idx] = mid_dir
        for idx in DEFAULT_TIER_INDICES["high"]:
            per_roi_mus[:, idx] = high_dir
        per_roi_mus = F.normalize(per_roi_mus, dim=-1)

        alphas = torch.ones(B, R) / R
        tier_targets = {
            "early": early_dir.expand(B, D),
            "mid": mid_dir.expand(B, D),
            "high": high_dir.expand(B, D),
        }
        loss, _ = loss_fn(per_roi_mus, alphas, tier_targets)
        assert loss.item() < 0.01, (
            f"Loss should be ~0 when aligned, got {loss.item():.4f}"
        )

    def test_gradient_flow(self, loss_fn):
        """Gradients should flow back through per_roi_mus."""
        B, R, D = 4, 16, 32
        per_roi_mus = F.normalize(
            torch.randn(B, R, D), dim=-1
        ).requires_grad_(True)
        alphas = F.softmax(torch.randn(B, R), dim=-1)
        tier_targets = {
            "early": F.normalize(torch.randn(B, D), dim=-1),
            "mid": F.normalize(torch.randn(B, D), dim=-1),
            "high": F.normalize(torch.randn(B, D), dim=-1),
        }
        loss, _ = loss_fn(per_roi_mus, alphas, tier_targets)
        loss.backward()
        assert per_roi_mus.grad is not None
        assert torch.isfinite(per_roi_mus.grad).all()

    def test_partial_tier_targets(self, loss_fn):
        """Should work with incomplete tier targets."""
        B, R, D = 4, 16, 32
        per_roi_mus = F.normalize(torch.randn(B, R, D), dim=-1)
        alphas = F.softmax(torch.randn(B, R), dim=-1)
        # Only provide early tier
        tier_targets = {
            "early": F.normalize(torch.randn(B, D), dim=-1),
        }
        loss, details = loss_fn(per_roi_mus, alphas, tier_targets)
        assert torch.isfinite(loss)
        assert "hier_early" in details
        assert "hier_mid" not in details

    def test_empty_tier_targets(self, loss_fn):
        """Should return zero loss with empty targets."""
        B, R, D = 4, 16, 32
        per_roi_mus = F.normalize(torch.randn(B, R, D), dim=-1)
        alphas = F.softmax(torch.randn(B, R), dim=-1)
        loss, details = loss_fn(per_roi_mus, alphas, {})
        assert loss.item() == 0.0

    def test_default_tier_indices(self):
        """Verify default tier indices don't overlap and cover expected ROIs."""
        all_indices = set()
        for tier_name, indices in DEFAULT_TIER_INDICES.items():
            for idx in indices:
                assert idx not in all_indices, (
                    f"ROI index {idx} appears in multiple tiers"
                )
                all_indices.add(idx)
        # Should cover indices 0-14 (16 ROIs minus nsdgeneral_other at 15)
        assert all_indices == set(range(15))

    def test_custom_tier_config(self):
        """Should accept custom tier configurations."""
        custom_tiers = {"low": [0, 1], "high": [2, 3]}
        custom_columns = {"low": "layer_12_proj", "high": "final"}
        loss_fn = HierarchicalCLIPLoss(
            tier_indices=custom_tiers,
            tier_clip_columns=custom_columns,
        )
        B, D = 4, 32
        per_roi_mus = F.normalize(torch.randn(B, 4, D), dim=-1)
        alphas = F.softmax(torch.randn(B, 4), dim=-1)
        tier_targets = {
            "low": F.normalize(torch.randn(B, D), dim=-1),
            "high": F.normalize(torch.randn(B, D), dim=-1),
        }
        loss, details = loss_fn(per_roi_mus, alphas, tier_targets)
        assert torch.isfinite(loss)
        assert "hier_low" in details
        assert "hier_high" in details
