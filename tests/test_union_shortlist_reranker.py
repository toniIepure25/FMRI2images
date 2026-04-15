"""Tests for union-shortlist reranker models and utilities."""

from __future__ import annotations

import torch
import pytest

from fmri2img.models.union_shortlist_reranker import (
    CandidateReranker,
    PlattCalibrator,
    ShortlistSetTransformerReranker,
    VMFEvidenceReranker,
    shortlist_cross_entropy,
    shortlist_pairwise_margin_loss,
)


@pytest.fixture
def batch_data() -> dict[str, torch.Tensor]:
    """Synthetic shortlist batch: 4 queries, max 20 candidates, 53 features."""
    B, K, F = 4, 20, 53
    features = torch.randn(B, K, F)
    mask = torch.ones(B, K, dtype=torch.bool)
    mask[0, 15:] = False
    mask[1, 18:] = False
    labels = torch.zeros(B, K)
    labels[0, 3] = 1.0
    labels[1, 7] = 1.0
    labels[2, 0] = 1.0
    labels[3, 12] = 1.0
    return {"features": features, "mask": mask, "labels": labels}


class TestCandidateReranker:
    def test_forward_shape(self, batch_data: dict[str, torch.Tensor]) -> None:
        model = CandidateReranker(input_dim=53, hidden_dim=32, num_layers=2, dropout=0.0)
        logits = model(batch_data["features"], batch_data["mask"])
        assert logits.shape == (4, 20)

    def test_masked_positions_are_neg_inf(self, batch_data: dict[str, torch.Tensor]) -> None:
        model = CandidateReranker(input_dim=53, hidden_dim=32, num_layers=2, dropout=0.0)
        logits = model(batch_data["features"], batch_data["mask"])
        assert (logits[0, 15:] == float("-inf")).all()
        assert (logits[1, 18:] == float("-inf")).all()
        assert torch.isfinite(logits[2]).all()

    def test_gradient_flows(self, batch_data: dict[str, torch.Tensor]) -> None:
        model = CandidateReranker(input_dim=53, hidden_dim=32, num_layers=2, dropout=0.0)
        logits = model(batch_data["features"], batch_data["mask"])
        loss = shortlist_cross_entropy(logits, batch_data["labels"], batch_data["mask"])
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()


class TestVMFEvidenceReranker:
    def test_forward_shape(self, batch_data: dict[str, torch.Tensor]) -> None:
        model = VMFEvidenceReranker(input_dim=53, hidden_dim=48, num_layers=2, dropout=0.0)
        logits = model(batch_data["features"], batch_data["mask"])
        assert logits.shape == (4, 20)

    def test_masked_positions(self, batch_data: dict[str, torch.Tensor]) -> None:
        model = VMFEvidenceReranker(input_dim=53, hidden_dim=48, num_layers=2, dropout=0.0)
        logits = model(batch_data["features"], batch_data["mask"])
        assert (logits[0, 15:] == float("-inf")).all()


class TestShortlistSetTransformerReranker:
    def test_forward_shape(self, batch_data: dict[str, torch.Tensor]) -> None:
        model = ShortlistSetTransformerReranker(
            input_dim=53, hidden_dim=64, num_layers=1, dropout=0.0, num_heads=4,
        )
        logits = model(batch_data["features"], batch_data["mask"])
        assert logits.shape == (4, 20)

    def test_gradient_flows(self, batch_data: dict[str, torch.Tensor]) -> None:
        model = ShortlistSetTransformerReranker(
            input_dim=53, hidden_dim=64, num_layers=1, dropout=0.0, num_heads=4,
        )
        logits = model(batch_data["features"], batch_data["mask"])
        loss = shortlist_cross_entropy(logits, batch_data["labels"], batch_data["mask"])
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None


class TestShortlistCrossEntropy:
    def test_basic(self, batch_data: dict[str, torch.Tensor]) -> None:
        logits = torch.randn(4, 20)
        logits = logits.masked_fill(~batch_data["mask"], float("-inf"))
        loss = shortlist_cross_entropy(logits, batch_data["labels"], batch_data["mask"])
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_no_gt_returns_zero(self) -> None:
        logits = torch.randn(2, 10)
        labels = torch.zeros(2, 10)
        mask = torch.ones(2, 10, dtype=torch.bool)
        loss = shortlist_cross_entropy(logits, labels, mask)
        assert loss.item() == 0.0


class TestPairwiseMarginLoss:
    def test_basic(self, batch_data: dict[str, torch.Tensor]) -> None:
        logits = torch.randn(4, 20)
        logits = logits.masked_fill(~batch_data["mask"], float("-inf"))
        loss = shortlist_pairwise_margin_loss(
            logits, batch_data["labels"], batch_data["mask"],
            margin=0.2, hard_neg_k=3,
        )
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_no_gt(self) -> None:
        logits = torch.randn(2, 10)
        labels = torch.zeros(2, 10)
        mask = torch.ones(2, 10, dtype=torch.bool)
        loss = shortlist_pairwise_margin_loss(logits, labels, mask)
        assert loss.item() == 0.0


class TestPlattCalibrator:
    def test_fit_and_calibrate(self, batch_data: dict[str, torch.Tensor]) -> None:
        model = CandidateReranker(input_dim=53, hidden_dim=32, num_layers=2, dropout=0.0)
        with torch.no_grad():
            logits = model(batch_data["features"], batch_data["mask"])

        cal = PlattCalibrator(lr=0.01, max_iter=50)
        cal.fit(logits, batch_data["labels"], batch_data["mask"])
        assert isinstance(cal.temperature, float)
        assert isinstance(cal.bias, float)

        probs = cal.calibrate(logits, batch_data["mask"])
        assert probs.shape == logits.shape
        valid_probs = probs[batch_data["mask"]]
        assert (valid_probs >= 0).all()
        assert (valid_probs <= 1).all()

    def test_confidence(self, batch_data: dict[str, torch.Tensor]) -> None:
        model = CandidateReranker(input_dim=53, hidden_dim=32, num_layers=2, dropout=0.0)
        with torch.no_grad():
            logits = model(batch_data["features"], batch_data["mask"])

        cal = PlattCalibrator()
        cal.fit(logits, batch_data["labels"], batch_data["mask"])
        conf = cal.confidence(logits, batch_data["mask"])
        assert conf.shape == (4,)
        assert (conf >= 0).all()
        assert (conf <= 1).all()

    def test_state_dict_roundtrip(self) -> None:
        cal = PlattCalibrator()
        cal.temperature = 1.5
        cal.bias = -0.3
        state = cal.state_dict()

        cal2 = PlattCalibrator()
        cal2.load_state_dict(state)
        assert cal2.temperature == 1.5
        assert cal2.bias == -0.3
