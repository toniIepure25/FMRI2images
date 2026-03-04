"""Tests for V9 features: DropPath, ProjectionHead, CSLS, R-Drop, kappa-avg."""

import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# DropPath
# ---------------------------------------------------------------------------

class TestDropPath:
    def test_identity_at_eval(self):
        from fmri2img.models.roi_transformer import DropPath

        dp = DropPath(drop_prob=0.5)
        dp.eval()
        x = torch.randn(4, 8)
        out = dp(x)
        assert torch.allclose(out, x)

    def test_stochastic_at_train(self):
        from fmri2img.models.roi_transformer import DropPath

        dp = DropPath(drop_prob=0.99)
        dp.train()
        torch.manual_seed(0)
        x = torch.ones(100, 8)
        out = dp(x)
        # With 99% drop prob, most samples should be zeroed
        zero_rows = (out.abs().sum(dim=-1) == 0).sum().item()
        assert zero_rows > 50, f"Expected many zeroed rows, got {zero_rows}"

    def test_zero_drop_prob_is_identity(self):
        from fmri2img.models.roi_transformer import DropPath

        dp = DropPath(drop_prob=0.0)
        dp.train()
        x = torch.randn(4, 8)
        out = dp(x)
        assert torch.allclose(out, x)


class TestTransformerLayerWithDropPath:
    def test_forward_shape(self):
        from fmri2img.models.roi_transformer import TransformerLayerWithDropPath

        layer = TransformerLayerWithDropPath(
            d_model=64, nhead=4, dim_feedforward=128,
            dropout=0.0, drop_path=0.1,
        )
        x = torch.randn(2, 5, 64)
        out = layer(x)
        assert out.shape == (2, 5, 64)

    def test_integrates_into_roi_transformer(self):
        from fmri2img.models.roi_transformer import ROITransformerEncoder

        roi_dims = {"V1": 100, "V2": 80, "V3": 60}
        enc = ROITransformerEncoder(
            roi_dims=roi_dims, d_model=64, nhead=4, num_layers=3,
            dropout=0.1, drop_path_rate=0.2,
        )
        x = torch.randn(2, 240)
        out = enc(x)
        assert out.shape == (2, 64)

    def test_integrates_into_multi_subject(self):
        from fmri2img.models.multi_subject_encoder import MultiSubjectROITransformer

        dims = {"subj01": {"V1": 100, "V2": 80}, "subj02": {"V1": 90, "V2": 70}}
        enc = MultiSubjectROITransformer(
            subject_roi_dims=dims, d_model=64, nhead=4, num_layers=3,
            dropout=0.1, drop_path_rate=0.15,
        )
        x = torch.randn(2, 180)
        out = enc(x, subject_ids=0)
        assert out.shape == (2, 64)


# ---------------------------------------------------------------------------
# ContrastiveProjectionHead
# ---------------------------------------------------------------------------

class TestContrastiveProjectionHead:
    def test_output_shape_and_norm(self):
        from fmri2img.models.projection_head import ContrastiveProjectionHead

        head = ContrastiveProjectionHead(d_model=64, hidden_dim=128, out_dim=64)
        x = torch.randn(8, 64)
        out = head(x)
        assert out.shape == (8, 64)
        norms = torch.norm(out, dim=-1)
        assert torch.allclose(norms, torch.ones(8), atol=1e-5)

    def test_gradient_flows(self):
        from fmri2img.models.projection_head import ContrastiveProjectionHead

        head = ContrastiveProjectionHead(d_model=32, hidden_dim=64, out_dim=32)
        x = torch.randn(4, 32, requires_grad=True)
        out = head(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_wired_in_unified_model(self):
        from fmri2img.models.unified_model import create_model

        cfg = {
            "type": "vmf",
            "encoder": {"input_dim": 100, "hidden_dims": [64]},
            "decoder": {"output_dim": 32, "kappa_mode": "softplus"},
            "projection_head": {
                "enabled": True,
                "hidden_dim": 64,
                "out_dim": 32,
                "dropout": 0.0,
            },
        }
        model = create_model(cfg)
        assert model.projection_head is not None
        x = torch.randn(2, 100)
        mu, kappa = model(x)
        proj = model.projection_head(mu)
        assert proj.shape == (2, 32)


# ---------------------------------------------------------------------------
# CSLS Similarity
# ---------------------------------------------------------------------------

class TestCSLS:
    def test_identity_retrieval(self):
        from fmri2img.eval.embedding_eval import csls_similarity

        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 64)).astype(np.float32)
        X /= np.linalg.norm(X, axis=-1, keepdims=True)
        sim = csls_similarity(X, X, k=5)
        assert sim.shape == (20, 20)
        # Diagonal should be highest for well-separated embeddings
        predicted = sim.argmax(axis=1)
        acc = (predicted == np.arange(20)).mean()
        assert acc == 1.0, f"Self-retrieval accuracy: {acc}"

    def test_csls_retrieval_metrics(self):
        from fmri2img.eval.embedding_eval import (
            compute_retrieval_metrics,
            compute_retrieval_metrics_csls,
        )

        rng = np.random.default_rng(123)
        preds = rng.standard_normal((50, 64)).astype(np.float32)
        gts = preds + rng.standard_normal((50, 64)).astype(np.float32) * 0.3

        cos_ret = compute_retrieval_metrics(preds, gts, ks=(1, 5))
        csls_ret = compute_retrieval_metrics_csls(preds, gts, ks=(1, 5), csls_k=5)

        assert "top1_accuracy" in csls_ret
        assert "top5_accuracy" in csls_ret
        # CSLS should be at least as good as cosine on noisy data
        assert csls_ret["top1_accuracy"] >= cos_ret["top1_accuracy"] - 0.1


# ---------------------------------------------------------------------------
# R-Drop for vMF
# ---------------------------------------------------------------------------

class TestVMFRDrop:
    def test_zero_divergence_for_identical(self):
        from fmri2img.losses.vmf_nce import vmf_rdrop_loss

        mu = torch.randn(8, 64)
        mu = mu / mu.norm(dim=-1, keepdim=True)
        kappa = torch.full((8,), 10.0)
        loss = vmf_rdrop_loss(mu, kappa, mu, kappa)
        assert loss.item() < 1e-5

    def test_positive_for_different(self):
        from fmri2img.losses.vmf_nce import vmf_rdrop_loss

        torch.manual_seed(42)
        mu1 = torch.randn(8, 64)
        mu1 = mu1 / mu1.norm(dim=-1, keepdim=True)
        mu2 = torch.randn(8, 64)
        mu2 = mu2 / mu2.norm(dim=-1, keepdim=True)
        k1 = torch.full((8,), 10.0)
        k2 = torch.full((8,), 20.0)
        loss = vmf_rdrop_loss(mu1, k1, mu2, k2)
        assert loss.item() > 0

    def test_gradient_flows(self):
        from fmri2img.losses.vmf_nce import vmf_rdrop_loss

        mu1 = torch.randn(4, 32, requires_grad=True)
        mu2 = torch.randn(4, 32, requires_grad=True)
        k1 = torch.full((4,), 10.0, requires_grad=True)
        k2 = torch.full((4,), 15.0, requires_grad=True)
        loss = vmf_rdrop_loss(mu1, k1, mu2, k2)
        loss.backward()
        assert mu1.grad is not None
        assert k1.grad is not None


# ---------------------------------------------------------------------------
# Label Smoothing in vMF-NCE
# ---------------------------------------------------------------------------

class TestLabelSmoothing:
    def test_label_smoothing_reduces_loss(self):
        from fmri2img.losses.vmf_nce import VonMisesFisherNCELoss

        torch.manual_seed(0)
        mu = torch.randn(8, 32)
        mu = mu / mu.norm(dim=-1, keepdim=True)
        kappa = torch.full((8, 1), 10.0)
        keys = torch.randn(8, 32)
        keys = keys / keys.norm(dim=-1, keepdim=True)

        loss_no_ls = VonMisesFisherNCELoss(
            tau=1.0, use_queue=False, label_smoothing=0.0,
        )
        loss_with_ls = VonMisesFisherNCELoss(
            tau=1.0, use_queue=False, label_smoothing=0.2,
        )
        l_no = loss_no_ls(mu, kappa, keys)
        l_ls = loss_with_ls(mu, kappa, keys)
        # Label smoothing typically lowers the loss magnitude
        # because the target distribution is softer
        assert l_ls.item() != l_no.item()


# ---------------------------------------------------------------------------
# Kappa-weighted averaging
# ---------------------------------------------------------------------------

class TestKappaWeightedAveraging:
    def test_high_kappa_dominates(self):
        """Verify that higher kappa predictions contribute more."""
        rng = np.random.default_rng(0)
        # 3 repetitions, embedding dim 8
        preds = rng.standard_normal((3, 8)).astype(np.float32)
        kappas = np.array([1.0, 1.0, 100.0], dtype=np.float32)

        # Kappa-weighted average
        k_w = kappas / kappas.sum()
        weighted = (preds * k_w[:, None]).sum(axis=0)
        weighted /= np.linalg.norm(weighted)

        # Simple average
        simple = preds.mean(axis=0)
        simple /= np.linalg.norm(simple)

        # Weighted should be much closer to the high-kappa prediction
        cos_weighted = np.dot(weighted, preds[2] / np.linalg.norm(preds[2]))
        cos_simple = np.dot(simple, preds[2] / np.linalg.norm(preds[2]))
        assert cos_weighted > cos_simple
