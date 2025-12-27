import numpy as np
import pandas as pd
import pytest

from fmri2img.models.train_utils import extract_features_and_targets


class _DummyImg:
    def __init__(self, data):
        self._data = data

    def get_fdata(self):
        return self._data


class _DummyLoader:
    def __init__(self, data):
        self._data = data

    def load(self, path):
        return _DummyImg(self._data)


class _DummyClipCache:
    def __init__(self, embeddings):
        self._embeddings = embeddings

    def get(self, ids):
        return {i: self._embeddings[i] for i in ids if i in self._embeddings}


def test_extract_skips_out_of_bounds_and_reports_stats():
    data = np.zeros((2, 2, 2, 3), dtype=np.float32)  # 3 volumes along last axis
    loader = _DummyLoader(data)
    clip_cache = _DummyClipCache({1: np.ones(4, dtype=np.float32)})

    df = pd.DataFrame(
        [
            {"beta_path": "dummy", "beta_index": 1, "nsdId": 1},  # valid
            {"beta_path": "dummy", "beta_index": 5, "nsdId": 1},  # out of bounds
        ]
    )

    X, Y, nsd_ids, stats = extract_features_and_targets(
        df,
        loader,
        preprocessor=None,
        clip_cache=clip_cache,
        desc="test",
        return_stats=True,
    )

    assert X.shape == (1, 8)  # 2*2*2 flattened
    assert Y.shape == (1, 4)
    assert nsd_ids.tolist() == [1]
    assert stats["used"] == 1
    assert stats["skipped_invalid_beta"] == 1
    assert stats["total_rows"] == 2


def test_extract_raises_when_all_invalid():
    data = np.zeros((2, 2, 2, 1), dtype=np.float32)  # only 1 volume
    loader = _DummyLoader(data)
    clip_cache = _DummyClipCache({1: np.ones(4, dtype=np.float32)})

    df = pd.DataFrame(
        [
            {"beta_path": "dummy", "beta_index": 5, "nsdId": 1},
        ]
    )

    with pytest.raises(ValueError):
        extract_features_and_targets(
            df,
            loader,
            preprocessor=None,
            clip_cache=clip_cache,
            desc="test",
        )
