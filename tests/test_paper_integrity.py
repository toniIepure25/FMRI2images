import numpy as np
import pandas as pd
import pytest
import torch

from fmri2img.data.nsd_index_builder import NSDIndexBuilder
from fmri2img.data.nsd_index_reader import read_subject_index
from fmri2img.training.train_two_stage import LazyDataset
from fmri2img.data.clip_cache import CLIPCache


class _FakeImg:
    def __init__(self, shape):
        self.shape = shape

    def get_fdata(self):
        return np.zeros(self.shape, dtype=np.float32)


class _FakeNiftiLoader:
    def __init__(self, shape=(2, 2, 2, 2)):
        self.shape = shape
        self.loaded = []

    def load(self, path, *_, **__):
        self.loaded.append(path)
        return _FakeImg(self.shape)


def test_validate_index_checks_beta_bounds(tmp_path, monkeypatch):
    # Build a tiny, consistent index
    beta_path = "s3://bucket/betas_session01.nii.gz"
    df = pd.DataFrame(
        {
            "subject": ["subj01", "subj01"],
            "session": [1, 1],
            "trial_in_session": [0, 1],
            "global_trial_index": [0, 1],
            "nsdId": [10, 11],
            "cocoId": [0, 1],
            "cocoSplit": ["train", "train"],
            "shared1000": [False, False],
            "filename": ["a", "b"],
            "beta_path": [beta_path, beta_path],
            "beta_index": [0, 1],
        }
    )

    builder = NSDIndexBuilder()
    monkeypatch.setattr(builder, "nifti_loader", _FakeNiftiLoader(shape=(2, 2, 2, 2)))
    builder.validate_index(df)  # should not raise


def test_read_subject_index_blocks_fallback(tmp_path):
    # Single beta_path should fail when allow_fallback_index is False
    beta_path = "s3://bucket/only.nii.gz"
    df = pd.DataFrame(
        {
            "subject": ["subj01"],
            "beta_path": [beta_path],
        }
    )
    path = tmp_path / "index.parquet"
    df.to_parquet(path)

    with pytest.raises(ValueError):
        read_subject_index(str(path), "subj01", allow_fallback_index=False)

    # Allowing fallback should succeed (no replacement performed)
    df_loaded = read_subject_index(str(path), "subj01", allow_fallback_index=True)
    assert len(df_loaded) == 1


def test_lazy_dataset_streams_and_normalizes(monkeypatch):
    df = pd.DataFrame(
        {
            "beta_path": ["s3://bucket/a.nii.gz", "s3://bucket/a.nii.gz"],
            "beta_index": [0, 1],
            "nsdId": [1, 2],
        }
    )

    loader = _FakeNiftiLoader(shape=(1, 1, 1, 2))
    cache = CLIPCache(cache_path="dummy.parquet")
    cache._df = pd.DataFrame({"nsdId": [1, 2], "clip512": [np.ones(512), np.ones(512)]})
    cache._is_loaded = True

    dataset = LazyDataset(df, loader, preprocessor=None, clip_cache=cache)
    x0, y0 = dataset[0]
    assert x0.dtype == torch.float32
    assert y0.shape[0] == 512
    assert len(loader.loaded) == 1  # cached beta file
