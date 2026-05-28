"""
Unit tests for NeuroBridge-OT architecture modules.
"""

import pytest
import torch
import torch.nn as nn

from fmri2img.models.neurobridge_ot.roi_tokenizer import (
    FixedROISummaryTokenizer,
    LearnedROITokenizer,
    SubjectAdaptiveROITokenizer,
    HyperAdapterROITokenizer,
    create_roi_tokenizer,
)
from fmri2img.models.neurobridge_ot.subject_fingerprint import SubjectFingerprint
from fmri2img.models.neurobridge_ot.optimal_transport import (
    OptimalTransportAlignment,
    sinkhorn_transport,
    CostFunction,
)
from fmri2img.models.neurobridge_ot.semantic_transformer import SemanticTransformer
from fmri2img.models.neurobridge_ot.decoding_heads import (
    CLIPGlobalHead,
    VMFUncertaintyHead,
    CalibrationHead,
    DecodingHeads,
)
from fmri2img.models.neurobridge_ot.hyper_adapter import (
    HyperNetwork,
    LoRAAdapter,
    FiLMLayer,
    AdaptationController,
)
from fmri2img.models.neurobridge_ot.losses import (
    NeuroBridgeOTLoss,
    InfoNCELoss,
    VMFNLLLoss,
    SubjectAdversarialLoss,
    GradientReversalLayer,
    SharedImageConsistencyLoss,
)
from fmri2img.models.neurobridge_ot.model import NeuroBridgeOTModel


# --- Fixtures ---

@pytest.fixture
def roi_indices():
    """Create synthetic ROI indices for 4 ROIs."""
    return {
        "V1": torch.arange(0, 100),
        "V2": torch.arange(100, 200),
        "FFA": torch.arange(200, 250),
        "PPA": torch.arange(250, 300),
    }


@pytest.fixture
def small_config():
    """Small model config for fast testing."""
    return {
        "type": "neurobridge_ot",
        "d_model": 64,
        "output_dim": 64,
        "n_rois": 4,
        "n_canonical_tokens": 4,
        "use_fingerprint": False,
        "use_hyper_adapter": False,
        "tokenizer": {
            "type": "learned",
            "n_rois": 4,
            "d_model": 64,
            "n_attn_heads": 2,
            "dropout": 0.0,
            "roi_dropout": 0.0,
            "max_voxels_per_roi": 200,
        },
        "optimal_transport": {
            "alignment_mode": "ot",
            "epsilon": 0.1,
            "n_sinkhorn_iters": 10,
            "cost_type": "cosine",
        },
        "transformer": {
            "n_heads": 4,
            "n_layers": 2,
            "dim_feedforward": 128,
            "dropout": 0.0,
            "drop_path_rate": 0.0,
            "n_subjects": 4,
            "use_subject_embedding": False,
        },
        "heads": {
            "d_model": 64,
            "output_dim": 64,
            "use_token_head": False,
            "use_vmf_head": True,
            "use_calibration_head": False,
            "kappa_min": 0.001,
            "kappa_max": 100.0,
        },
    }


# --- ROI Tokenizer Tests ---

class TestROITokenizer:
    """Tests for ROI tokenizer variants."""

    def test_fixed_summary_shape(self, roi_indices):
        tok = FixedROISummaryTokenizer(n_rois=4, d_model=64, pool_mode="mean_std_max")
        fmri = torch.randn(8, 300)
        tokens, mask = tok(fmri, roi_indices)
        assert tokens.shape == (8, 4, 64)
        assert mask.shape == (8, 4)
        assert mask.all()

    def test_learned_tokenizer_shape(self, roi_indices):
        tok = LearnedROITokenizer(n_rois=4, d_model=64, n_attn_heads=2)
        fmri = torch.randn(4, 300)
        tokens, mask = tok(fmri, roi_indices)
        assert tokens.shape == (4, 4, 64)
        assert mask.all()

    def test_subject_adaptive_shape(self, roi_indices):
        tok = SubjectAdaptiveROITokenizer(n_rois=4, d_model=64, n_subjects=4)
        fmri = torch.randn(4, 300)
        subject_id = torch.tensor([0, 1, 2, 3])
        tokens, mask = tok(fmri, roi_indices, subject_id=subject_id)
        assert tokens.shape == (4, 4, 64)

    def test_hyper_adapter_tokenizer(self, roi_indices):
        tok = HyperAdapterROITokenizer(n_rois=4, d_model=64, fingerprint_dim=32)
        fmri = torch.randn(4, 300)
        adapters = {"fingerprint": torch.randn(4, 32)}
        tokens, mask = tok(fmri, roi_indices, adapters=adapters)
        assert tokens.shape == (4, 4, 64)

    def test_empty_roi_handling(self):
        tok = LearnedROITokenizer(n_rois=3, d_model=32, n_attn_heads=2)
        fmri = torch.randn(2, 100)
        indices = {
            "V1": torch.arange(0, 50),
            "V2": torch.tensor([], dtype=torch.long),  # Empty ROI
            "V3": torch.arange(50, 100),
        }
        tokens, mask = tok(fmri, indices)
        assert tokens.shape == (2, 3, 32)
        assert not mask[:, 1].any()  # Empty ROI marked invalid

    def test_different_voxel_counts(self):
        tok = LearnedROITokenizer(n_rois=2, d_model=32, n_attn_heads=2)
        # Subject with 200 voxels
        fmri_a = torch.randn(2, 200)
        idx_a = {"V1": torch.arange(0, 100), "V2": torch.arange(100, 200)}
        tokens_a, _ = tok(fmri_a, idx_a)
        # Subject with 500 voxels
        fmri_b = torch.randn(2, 500)
        idx_b = {"V1": torch.arange(0, 300), "V2": torch.arange(300, 500)}
        tokens_b, _ = tok(fmri_b, idx_b)
        # Both produce same output shape
        assert tokens_a.shape == tokens_b.shape == (2, 2, 32)

    def test_factory(self):
        tok = create_roi_tokenizer({"type": "fixed_summary", "n_rois": 5, "d_model": 128})
        assert isinstance(tok, FixedROISummaryTokenizer)
        tok = create_roi_tokenizer({"type": "learned", "n_rois": 5, "d_model": 128})
        assert isinstance(tok, LearnedROITokenizer)


# --- Subject Fingerprint Tests ---

class TestSubjectFingerprint:
    def test_output_shape(self):
        fp = SubjectFingerprint(n_rois=4, fingerprint_dim=64, n_subjects=4)
        roi_stats = torch.randn(8, 4, 5)
        subject_id = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        out = fp(roi_stats, subject_id)
        assert out.shape == (8, 64)

    def test_without_subject_embedding(self):
        fp = SubjectFingerprint(n_rois=4, fingerprint_dim=64, use_subject_embedding=False)
        roi_stats = torch.randn(4, 4, 5)
        out = fp(roi_stats)
        assert out.shape == (4, 64)


# --- Optimal Transport Tests ---

class TestOptimalTransport:
    def test_sinkhorn_valid_transport(self):
        cost = torch.rand(4, 5, 6)
        T = sinkhorn_transport(cost, epsilon=0.1, n_iters=100)
        assert T.shape == (4, 5, 6)
        # Rows should approximately sum to 1/5
        row_sums = T.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums) / 5, atol=0.05)
        # Columns should approximately sum to 1/6
        col_sums = T.sum(dim=-2)
        assert torch.allclose(col_sums, torch.ones_like(col_sums) / 6, atol=0.05)

    def test_ot_alignment_output(self):
        ot = OptimalTransportAlignment(
            n_canonical_tokens=4, d_model=32, n_source_tokens=4,
            epsilon=0.1, n_sinkhorn_iters=20, alignment_mode="ot"
        )
        roi_tokens = torch.randn(2, 4, 32)
        result = ot(roi_tokens)
        assert "aligned_tokens" in result
        assert result["aligned_tokens"].shape == (2, 4, 32)
        assert "transport_matrix" in result
        assert result["transport_matrix"].shape == (2, 4, 4)
        assert "ot_cost" in result

    def test_identity_alignment(self):
        ot = OptimalTransportAlignment(
            n_canonical_tokens=4, d_model=32, n_source_tokens=4,
            alignment_mode="identity"
        )
        roi_tokens = torch.randn(2, 4, 32)
        result = ot(roi_tokens)
        assert result["ot_cost"].item() == 0.0

    def test_cost_functions(self):
        cost_fn = CostFunction("cosine")
        source = torch.randn(2, 3, 32)
        target = torch.randn(2, 4, 32)
        cost = cost_fn(source, target)
        assert cost.shape == (2, 3, 4)
        assert (cost >= 0).all()  # Cosine distance is non-negative


# --- Semantic Transformer Tests ---

class TestSemanticTransformer:
    def test_output_shape(self):
        trans = SemanticTransformer(d_model=64, n_heads=4, n_layers=2, n_canonical_tokens=4)
        x = torch.randn(2, 4, 64)
        out = trans(x)
        assert out["cls_embedding"].shape == (2, 64)
        assert out["roi_tokens"].shape == (2, 4, 64)

    def test_with_subject_embedding(self):
        trans = SemanticTransformer(
            d_model=64, n_heads=4, n_layers=2,
            n_canonical_tokens=4, use_subject_embedding=True, n_subjects=4
        )
        x = torch.randn(2, 4, 64)
        subject_id = torch.tensor([0, 1])
        out = trans(x, subject_id=subject_id)
        assert out["cls_embedding"].shape == (2, 64)

    def test_all_layers_return(self):
        trans = SemanticTransformer(d_model=64, n_heads=4, n_layers=3, n_canonical_tokens=4)
        x = torch.randn(2, 4, 64)
        out = trans(x, return_all_layers=True)
        assert "all_layers" in out
        assert len(out["all_layers"]) == 3


# --- Decoding Heads Tests ---

class TestDecodingHeads:
    def test_clip_head_normalized(self):
        head = CLIPGlobalHead(d_model=64, output_dim=128)
        features = torch.randn(4, 64)
        out = head(features)
        assert out.shape == (4, 128)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_vmf_head_constraints(self):
        head = VMFUncertaintyHead(d_model=64, output_dim=128, kappa_min=0.01, kappa_max=200.0)
        features = torch.randn(8, 64)
        mu, kappa = head(features)
        # mu is unit-normalized
        assert torch.allclose(mu.norm(dim=-1), torch.ones(8), atol=1e-5)
        # kappa is positive and bounded
        assert (kappa > 0).all()
        assert (kappa <= 200.0).all()

    def test_decoding_heads_container(self):
        config = {"d_model": 64, "output_dim": 128, "use_vmf_head": True, "use_calibration_head": True}
        heads = DecodingHeads(config)
        features = torch.randn(4, 64)
        roi_tokens = torch.randn(4, 6, 64)
        out = heads(features, roi_tokens)
        assert "clip_embedding" in out
        assert "vmf_mu" in out
        assert "vmf_kappa" in out
        assert "confidence" in out


# --- Hyper-Adapter Tests ---

class TestHyperAdapter:
    def test_hypernetwork_film(self):
        hyper = HyperNetwork(fingerprint_dim=32, d_model=64, n_target_layers=3, adapter_type="film")
        fp = torch.randn(4, 32)
        adapters = hyper(fp)
        assert "layer_0_gamma" in adapters
        assert "layer_0_beta" in adapters
        assert adapters["layer_0_gamma"].shape == (4, 64)

    def test_hypernetwork_lora(self):
        hyper = HyperNetwork(
            fingerprint_dim=32, d_model=64, n_target_layers=2,
            adapter_type="lora", lora_rank=4
        )
        fp = torch.randn(2, 32)
        adapters = hyper(fp)
        assert "layer_0_A" in adapters
        assert adapters["layer_0_A"].shape == (2, 64, 4)
        assert adapters["layer_0_B"].shape == (2, 4, 64)

    def test_lora_adapter(self):
        lora = LoRAAdapter(d_in=64, d_out=64, rank=8)
        x = torch.randn(4, 64)
        out = lora(x)
        assert out.shape == (4, 64)

    def test_film_layer(self):
        film = FiLMLayer(d_model=64)
        x = torch.randn(4, 10, 64)
        gamma = torch.ones(4, 64)
        beta = torch.zeros(4, 64)
        out = film(x, gamma, beta)
        assert out.shape == (4, 10, 64)
        assert torch.allclose(out, x)  # Identity when gamma=1, beta=0


# --- Loss Tests ---

class TestLosses:
    def test_infonce_loss(self):
        loss_fn = InfoNCELoss(temperature=0.07, symmetric=True)
        query = torch.randn(8, 64)
        key = torch.randn(8, 64)
        loss = loss_fn(query, key)
        assert loss.ndim == 0
        assert loss > 0

    def test_vmf_nll_loss(self):
        loss_fn = VMFNLLLoss(kappa_reg_weight=0.1, target_kappa=50.0)
        mu = torch.nn.functional.normalize(torch.randn(4, 128), dim=-1)
        kappa = torch.ones(4) * 50.0
        target = torch.nn.functional.normalize(torch.randn(4, 128), dim=-1)
        loss = loss_fn(mu, kappa, target)
        assert loss.ndim == 0

    def test_adversarial_gradient_reversal(self):
        adv = SubjectAdversarialLoss(d_model=64, n_subjects=4)
        features = torch.randn(8, 64, requires_grad=True)
        labels = torch.randint(0, 4, (8,))
        loss = adv(features, labels)
        loss.backward()
        assert features.grad is not None

    def test_composite_loss(self):
        config = {
            "w_contrastive": 1.0,
            "w_regression": 0.5,
            "w_vmf": 0.3,
            "w_adversarial": 0.1,
            "contrastive": {"temperature": 0.07},
            "vmf": {"kappa_reg_weight": 0.1, "target_kappa": 50.0},
            "adversarial": {"d_model": 64, "n_subjects": 4},
        }
        loss_fn = NeuroBridgeOTLoss(config)
        model_outputs = {
            "clip_embedding": torch.nn.functional.normalize(torch.randn(4, 64), dim=-1),
            "vmf_mu": torch.nn.functional.normalize(torch.randn(4, 64), dim=-1),
            "vmf_kappa": torch.ones(4) * 50.0,
            "cls_embedding": torch.randn(4, 64),
        }
        targets = {
            "clip_target": torch.nn.functional.normalize(torch.randn(4, 64), dim=-1),
            "subject_id": torch.randint(0, 4, (4,)),
            "nsd_id": torch.arange(4),
        }
        losses = loss_fn(model_outputs, targets)
        assert "total_loss" in losses
        assert losses["total_loss"] > 0

    def test_shared_image_consistency(self):
        loss_fn = SharedImageConsistencyLoss(margin=0.8)
        # Two subjects viewing same image
        embeddings = torch.randn(4, 64)
        nsd_ids = torch.tensor([1, 1, 2, 2])  # Pairs
        subject_ids = torch.tensor([0, 1, 0, 1])  # Different subjects
        loss = loss_fn(embeddings, nsd_ids, subject_ids)
        assert loss.ndim == 0


# --- Full Model Tests ---

class TestNeuroBridgeOTModel:
    def test_forward_shape(self, small_config, roi_indices):
        model = NeuroBridgeOTModel(small_config)
        fmri = torch.randn(4, 300)
        subject_id = torch.tensor([0, 1, 2, 3])
        out = model(fmri, roi_indices, subject_id=subject_id)
        assert "clip_embedding" in out
        assert out["clip_embedding"].shape == (4, 64)
        assert "vmf_mu" in out
        assert out["vmf_mu"].shape == (4, 64)
        assert "vmf_kappa" in out
        assert (out["vmf_kappa"] > 0).all()

    def test_backward_pass(self, small_config, roi_indices):
        model = NeuroBridgeOTModel(small_config)
        fmri = torch.randn(4, 300, requires_grad=True)
        out = model(fmri, roi_indices)
        loss = out["clip_embedding"].sum() + out["ot_cost"]
        loss.backward()
        assert fmri.grad is not None

    def test_with_intermediates(self, small_config, roi_indices):
        model = NeuroBridgeOTModel(small_config)
        fmri = torch.randn(2, 300)
        out = model(fmri, roi_indices, return_intermediates=True)
        assert "roi_tokens_raw" in out
        assert "aligned_tokens" in out
        assert "transport_matrix" in out
        assert "canonical_bank" in out

    def test_full_config_with_fingerprint(self, roi_indices):
        config = {
            "type": "neurobridge_ot",
            "d_model": 32,
            "output_dim": 32,
            "n_rois": 4,
            "n_canonical_tokens": 4,
            "use_fingerprint": True,
            "use_hyper_adapter": True,
            "tokenizer": {"type": "hyper_adapter", "n_rois": 4, "d_model": 32,
                          "fingerprint_dim": 16, "n_attn_heads": 2},
            "fingerprint": {"fingerprint_dim": 16, "n_subjects": 4, "use_subject_embedding": True},
            "optimal_transport": {"alignment_mode": "ot", "epsilon": 0.1, "n_sinkhorn_iters": 5},
            "transformer": {"n_heads": 2, "n_layers": 1, "dim_feedforward": 64},
            "heads": {"d_model": 32, "output_dim": 32, "use_vmf_head": True},
            "hyper_adapter": {"adapter_type": "film", "hidden_dim": 64},
        }
        model = NeuroBridgeOTModel(config)
        fmri = torch.randn(2, 300)
        roi_stats = torch.randn(2, 4, 5)
        subject_id = torch.tensor([0, 1])
        out = model(fmri, roi_indices, subject_id=subject_id, roi_stats=roi_stats)
        assert out["clip_embedding"].shape == (2, 32)
