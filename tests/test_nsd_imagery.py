"""Tests for NSD-Imagery integration (non-IO parts)."""

import numpy as np
import pandas as pd
import pytest

from fmri2img.data.nsd_imagery import (
    NSDImageryDataset,
    compute_perception_imagery_kappa_comparison,
    create_paired_perception_imagery_split,
    IMAGERY_SUBJECTS,
)


class TestImageryDataset:
    def test_basic_creation(self):
        features = np.random.randn(50, 15000).astype(np.float32)
        meta_df = pd.DataFrame({
            "trial_idx": range(50),
            "subject": "subj01",
            "condition": "imagery",
            "nsd_id": np.random.randint(0, 10000, size=50),
        })
        ds = NSDImageryDataset(features, meta_df)
        assert len(ds) == 50
        item = ds[0]
        assert "fmri" in item
        assert item["fmri"].shape == (15000,)
        assert item["condition"] == "imagery"

    def test_with_embeddings(self):
        features = np.random.randn(10, 100).astype(np.float32)
        nsd_ids = list(range(100, 110))
        meta_df = pd.DataFrame({
            "trial_idx": range(10),
            "subject": "subj01",
            "condition": "imagery",
            "nsd_id": nsd_ids,
        })
        embeddings_df = pd.DataFrame({
            "nsdId": nsd_ids,
            "fused": [np.random.randn(768).astype(np.float32) for _ in range(10)],
        })
        ds = NSDImageryDataset(features, meta_df, embeddings_df)
        item = ds[0]
        assert "clip_target" in item
        assert item["clip_target"].shape == (768,)


class TestPairedSplit:
    def test_basic_pairing(self):
        perc_meta = pd.DataFrame({"nsdId": [1, 2, 3, 4, 5]})
        img_meta = pd.DataFrame({"nsd_id": [3, 4, 5, 6, 7]})
        perc_idx, img_idx = create_paired_perception_imagery_split(perc_meta, img_meta)
        assert len(perc_idx) == 3  # shared: 3, 4, 5
        assert len(img_idx) == 3


class TestKappaComparison:
    def test_perception_higher_kappa(self):
        np.random.seed(42)
        n_pairs = 100
        n_rois = 17
        # Perception has higher kappa (our hypothesis)
        perc_kappas = np.random.exponential(50, size=(n_pairs, n_rois)).astype(np.float32)
        img_kappas = np.random.exponential(30, size=(n_pairs, n_rois)).astype(np.float32)

        report = compute_perception_imagery_kappa_comparison(perc_kappas, img_kappas)
        assert report["hypothesis_confirmed"] is True
        assert report["overall"]["kappa_drop_fraction"] > 0
        assert report["n_pairs"] == 100

    def test_per_roi_statistics(self):
        np.random.seed(0)
        perc_kappas = np.random.exponential(50, size=(200, 17)).astype(np.float32)
        img_kappas = np.random.exponential(30, size=(200, 17)).astype(np.float32)

        report = compute_perception_imagery_kappa_comparison(perc_kappas, img_kappas)
        assert len(report["per_roi"]) == 17
        for roi_data in report["per_roi"].values():
            assert "cohens_d" in roi_data
            assert "p_value" in roi_data


class TestSubjects:
    def test_expected_subjects(self):
        assert "subj01" in IMAGERY_SUBJECTS
        assert "subj02" in IMAGERY_SUBJECTS
        assert "subj05" in IMAGERY_SUBJECTS
        assert "subj07" in IMAGERY_SUBJECTS
        assert len(IMAGERY_SUBJECTS) == 4
