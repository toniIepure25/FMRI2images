"""
Integration / smoke tests for NeuroBridge-OT.

Tests the full pipeline from data loading through evaluation
using synthetic data (no real NSD files needed).
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from fmri2img.models.neurobridge_ot.model import NeuroBridgeOTModel
from fmri2img.models.neurobridge_ot.losses import NeuroBridgeOTLoss


@pytest.fixture
def synthetic_dataset_dir(tmp_path):
    """Create synthetic pre-extracted data for one subject."""
    subj = "subj01"
    n_trials = 100
    n_voxels = 300
    n_images = 50

    # fMRI features
    cache_dir = tmp_path / "cache" / "preextracted" / f"subject={subj}"
    cache_dir.mkdir(parents=True)
    fmri = np.random.randn(n_trials, n_voxels).astype(np.float32)
    np.save(cache_dir / "fmri_features.npy", fmri)

    # Index parquet
    index_dir = tmp_path / "data" / "indices" / "nsd_index" / f"subject={subj}"
    index_dir.mkdir(parents=True)
    nsd_ids = np.random.choice(n_images, size=n_trials, replace=True)
    df = pd.DataFrame({
        "nsdId": nsd_ids,
        "session": np.repeat(np.arange(1, 5), n_trials // 4 + 1)[:n_trials],
        "shared1000": [i < 10 for i in range(n_trials)],
    })
    df.to_parquet(index_dir / "index.parquet")

    # CLIP embeddings
    clip_dir = tmp_path / "outputs" / "clip_cache"
    clip_dir.mkdir(parents=True)
    unique_ids = list(range(n_images))
    embs = np.random.randn(n_images, 64).astype(np.float32)
    clip_df = pd.DataFrame({
        "nsdId": unique_ids,
        "fused": [embs[i] for i in range(n_images)],
    })
    clip_df.to_parquet(clip_dir / "clip.parquet")

    return {
        "tmp_path": tmp_path,
        "cache_root": tmp_path / "cache" / "preextracted",
        "index_root": tmp_path / "data" / "indices" / "nsd_index",
        "clip_path": clip_dir / "clip.parquet",
        "subject": subj,
        "n_trials": n_trials,
        "n_voxels": n_voxels,
        "n_images": n_images,
    }


class TestDatasetIntegration:
    """Test NeuroBridgeDataset with synthetic data."""

    def test_dataset_loads(self, synthetic_dataset_dir):
        from fmri2img.data.neurobridge_dataset import NeuroBridgeDataset

        info = synthetic_dataset_dir
        clip_df = pd.read_parquet(info["clip_path"])

        ds = NeuroBridgeDataset(
            subjects=[info["subject"]],
            cache_root=info["cache_root"],
            index_root=info["index_root"],
            embeddings_df=clip_df,
            embedding_column="fused",
            exclude_shared1000=True,
            split_by_image=True,
            seed=42,
            protocol="multi_subject_seen_subject",
        )

        assert len(ds) > 0
        assert len(ds.train_indices) > 0
        assert len(ds.val_indices) > 0

        # No SHARED1000 leakage
        for idx in ds.train_indices:
            row = ds.index_df.iloc[idx]
            assert not row.get("shared1000", False)

    def test_dataset_getitem(self, synthetic_dataset_dir):
        from fmri2img.data.neurobridge_dataset import NeuroBridgeDataset

        info = synthetic_dataset_dir
        clip_df = pd.read_parquet(info["clip_path"])

        ds = NeuroBridgeDataset(
            subjects=[info["subject"]],
            cache_root=info["cache_root"],
            index_root=info["index_root"],
            embeddings_df=clip_df,
            embedding_column="fused",
            exclude_shared1000=True,
            protocol="multi_subject_seen_subject",
        )

        sample = ds[0]
        assert "fmri" in sample
        assert "clip_target" in sample
        assert "subject_id" in sample
        assert "nsd_id" in sample
        assert sample["fmri"].shape == (info["n_voxels"],)

    def test_collate_function(self, synthetic_dataset_dir):
        from fmri2img.data.neurobridge_dataset import NeuroBridgeDataset, neurobridge_collate

        info = synthetic_dataset_dir
        clip_df = pd.read_parquet(info["clip_path"])

        ds = NeuroBridgeDataset(
            subjects=[info["subject"]],
            cache_root=info["cache_root"],
            index_root=info["index_root"],
            embeddings_df=clip_df,
            embedding_column="fused",
            protocol="multi_subject_seen_subject",
        )

        batch = [ds[i] for i in range(4)]
        collated = neurobridge_collate(batch)
        assert collated["fmri"].shape[0] == 4
        assert collated["clip_target"].shape[0] == 4
        assert collated["subject_id"].shape[0] == 4


class TestFullPipelineSmokeTest:
    """End-to-end smoke test: model forward + loss backward."""

    def test_smoke_forward_backward(self):
        config = {
            "type": "neurobridge_ot",
            "d_model": 32,
            "output_dim": 32,
            "n_rois": 3,
            "n_canonical_tokens": 3,
            "use_fingerprint": False,
            "use_hyper_adapter": False,
            "tokenizer": {"type": "learned", "n_rois": 3, "d_model": 32,
                          "n_attn_heads": 2, "dropout": 0.0},
            "optimal_transport": {"alignment_mode": "ot", "epsilon": 0.1,
                                  "n_sinkhorn_iters": 5},
            "transformer": {"n_heads": 2, "n_layers": 1, "dim_feedforward": 64,
                            "dropout": 0.0, "drop_path_rate": 0.0},
            "heads": {"d_model": 32, "output_dim": 32, "use_vmf_head": True,
                      "kappa_min": 0.01, "kappa_max": 100.0},
        }
        loss_config = {
            "w_contrastive": 1.0,
            "w_regression": 0.5,
            "w_vmf": 0.3,
            "contrastive": {"temperature": 0.07},
            "vmf": {"kappa_reg_weight": 0.1, "target_kappa": 30.0},
        }

        model = NeuroBridgeOTModel(config)
        loss_fn = NeuroBridgeOTLoss(loss_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Synthetic batch
        B, V = 8, 200
        fmri = torch.randn(B, V)
        roi_indices = {
            "V1": torch.arange(0, 70),
            "V2": torch.arange(70, 140),
            "FFA": torch.arange(140, 200),
        }
        clip_target = torch.nn.functional.normalize(torch.randn(B, 32), dim=-1)
        subject_ids = torch.randint(0, 4, (B,))

        # Forward
        outputs = model(fmri, roi_indices, subject_id=subject_ids)
        targets = {
            "clip_target": clip_target,
            "subject_id": subject_ids,
            "nsd_id": torch.arange(B),
        }
        losses = loss_fn(outputs, targets, ot_cost=outputs["ot_cost"])

        # Backward
        optimizer.zero_grad()
        losses["total_loss"].backward()
        optimizer.step()

        # Verify parameter update — most parameters should have gradients
        params_with_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
        total_params = sum(1 for p in model.parameters() if p.requires_grad)
        assert params_with_grad > total_params * 0.8, (
            f"Only {params_with_grad}/{total_params} parameters have gradients"
        )

    def test_evaluation_metrics_on_synthetic(self):
        """Test that retrieval metrics computation works."""
        from scripts.neurobridge_ot.eval_neurobridge_ot import compute_retrieval_metrics

        preds = torch.randn(50, 64)
        targets = preds + torch.randn_like(preds) * 0.1  # Near-perfect

        metrics = compute_retrieval_metrics(preds, targets, csls_k=0)
        assert "r@1" in metrics
        assert "r@5" in metrics
        assert "mrr" in metrics
        assert 0.0 <= metrics["r@1"] <= 1.0

    def test_model_from_create_model_factory(self):
        """Test integration with unified create_model factory."""
        from fmri2img.models.unified_model import create_model

        config = {
            "type": "neurobridge_ot",
            "d_model": 32,
            "output_dim": 32,
            "n_rois": 2,
            "n_canonical_tokens": 2,
            "use_fingerprint": False,
            "use_hyper_adapter": False,
            "tokenizer": {"type": "fixed_summary", "n_rois": 2, "d_model": 32},
            "optimal_transport": {"alignment_mode": "identity"},
            "transformer": {"n_heads": 2, "n_layers": 1, "dim_feedforward": 64},
            "heads": {"d_model": 32, "output_dim": 32, "use_vmf_head": False},
        }
        model = create_model(config)
        assert isinstance(model, NeuroBridgeOTModel)
