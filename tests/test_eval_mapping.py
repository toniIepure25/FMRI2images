import numpy as np
import pandas as pd

from fmri2img.eval.stimulus import (
    parse_stimulus_id_from_filename,
    resolve_from_cache,
    resolve_from_index,
    StimulusKey,
)
from fmri2img.eval.retrieval import compute_ranking_metrics


def test_parse_stimulus_id_from_filename_basic(tmp_path):
    p = tmp_path / "67574.png"
    p.write_text("stub")
    assert parse_stimulus_id_from_filename(p) == 67574


def test_resolve_from_cache_prefers_nsdId():
    df = pd.DataFrame({
        "nsdId": [67574],
        "nsd_id": [123],
        "clip512": [[0.0] * 512],
    })
    stim, field = resolve_from_cache(67574, df)
    assert field == "nsdId"
    assert stim.nsdId == 67574
    assert stim.nsd_id == 123


def test_resolve_from_index_handles_both_columns():
    df = pd.DataFrame({"nsdId": [10], "nsd_id": [3]})
    stim = resolve_from_index(3, df)
    assert isinstance(stim, StimulusKey)
    assert stim.nsd_id == 3
    assert stim.nsdId == 10


def test_ranks_are_one_based():
    # recon matches first gallery vector exactly -> rank 1
    recon = np.array([[1.0, 0.0]], dtype=np.float32)
    gallery = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    metrics = compute_ranking_metrics(recon, gallery, np.array([0]))
    assert metrics["mean_rank"] >= 1
    assert metrics["median_rank"] >= 1
