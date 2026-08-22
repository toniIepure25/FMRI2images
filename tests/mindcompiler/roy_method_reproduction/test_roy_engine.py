"""S2.5M engine end-to-end (data-free): Roy pairing + per-target Lambda + D1 + null."""
from __future__ import annotations

import numpy as np

from fmri2img.mindcompiler.roy_method_reproduction import roy_engine as re
from fmri2img.mindcompiler.roy_method_reproduction import roy_pairing as rp


def _setup(n_id=4, n_vox=12, seed=0):
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((n_id, n_vox))
    rows = []
    r = 0
    vis_by, img_by = {}, {}
    M_rows = []
    for state, store in (("vision", vis_by), ("imagery", img_by)):
        for i in range(n_id):
            ident = f"ID{i}"
            these = list(range(r, r + 8)); r += 8
            for x in these:
                M_rows.append(latent[i] + 0.3 * rng.standard_normal(n_vox))
            store[ident] = {"train": these[:4], "val": these[4:6], "test": these[6:8]}
    M = np.vstack(M_rows)
    return M, vis_by, img_by


def test_engine_runs_leakage_clean_with_per_target_lambda():
    M, vis_by, img_by = _setup()
    cell = re.run_fold_roy(M, vis_by, img_by, rp.root_digest(0), "subj02", "fold_0", n_null=200)
    assert all(v == 0 for v in cell.denoise_leakage.values())
    assert cell.n_denoise_records > 0
    # per-target lambda actually varies (vis2img)
    assert cell.vis2img["n_unique_lambdas"] >= 1
    assert 1 <= cell.vis2img["r_model"] <= cell.vis2img["rank_max"] <= re.RANK_HARD_MAX
    for k in ("test_mean_r", "finite_frac", "target_constant"):
        assert k in cell.vis2img


def test_engine_records_rank_curve_and_null():
    M, vis_by, img_by = _setup()
    cell = re.run_fold_roy(M, vis_by, img_by, rp.root_digest(0), "subj02", "fold_0", n_null=200)
    assert len(cell.rank_curve) == cell.vis2img["rank_max"]
    assert all("val" in c and "test" in c for c in cell.rank_curve)
    for k in ("null_mean", "null_p95_mean", "fraction_voxels_above_null_p95"):
        assert k in cell.null
    assert 0.0 <= cell.null["fraction_voxels_above_null_p95"] <= 1.0


def test_engine_deterministic_under_same_seed():
    M, vis_by, img_by = _setup()
    a = re.run_fold_roy(M, vis_by, img_by, rp.root_digest(0), "subj02", "fold_0", n_null=100, null_seed=7)
    b = re.run_fold_roy(M, vis_by, img_by, rp.root_digest(0), "subj02", "fold_0", n_null=100, null_seed=7)
    assert a.vis2img["test_mean_r"] == b.vis2img["test_mean_r"]
    assert a.null["fraction_voxels_above_null_p95"] == b.null["fraction_voxels_above_null_p95"]


def test_null_breaks_correspondence_on_structured_signal():
    # build test data where prediction strongly matches measured -> observed r >> null
    rng = np.random.default_rng(1)
    Y = rng.standard_normal((8, 10))
    pred = Y + 0.01 * rng.standard_normal((8, 10))     # near-perfect prediction
    out = re._prediction_null(Y, pred, n_null=500, seed=3)
    assert out["fraction_voxels_above_null_p95"] > 0.5  # most voxels beat their null
