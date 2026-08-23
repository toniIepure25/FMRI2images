"""S2.6R geometry engine (data-free): replay-consistency with S2.5M + geometry outputs."""
from __future__ import annotations

import numpy as np

from fmri2img.mindcompiler.roy_method_reproduction import roy_engine as re
from fmri2img.mindcompiler.roy_method_reproduction import roy_geometry_engine as ge
from fmri2img.mindcompiler.roy_method_reproduction import roy_pairing as rp


def _setup(n_id=13, n_vox=30, seed=0):
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((n_id, n_vox))
    vis_by, img_by, rows = {}, {}, []
    r = 0
    for state, store in (("vision", vis_by), ("imagery", img_by)):
        for i in range(n_id):
            these = list(range(r, r + 8)); r += 8
            for _ in these:
                rows.append(latent[i] + 0.3 * rng.standard_normal(n_vox))
            store[f"ID{i}"] = {"train": these[:4], "val": these[4:6], "test": these[6:8]}
    return np.vstack(rows), vis_by, img_by


def test_geometry_replays_s2_5m_vis2img_curve_within_tol():
    M, vis_by, img_by = _setup()
    root = rp.root_digest(0)
    ref = re.run_fold_roy(M, vis_by, img_by, root, "subj02", "fold_0", n_null=50)
    geo = ge.run_fold_geometry(M, vis_by, img_by, root, "subj02", "V1", "fold_0", n_null=50)
    assert len(geo.vis2img_curve) == len(ref.rank_curve)
    for a, b in zip(geo.vis2img_curve, ref.rank_curve):
        assert abs(a["val"] - b["val"]) <= 1e-10
        assert abs(a["test"] - b["test"]) <= 1e-10


def test_geometry_outputs_present_and_valid():
    M, vis_by, img_by = _setup()
    geo = ge.run_fold_geometry(M, vis_by, img_by, rp.root_digest(0), "subj02", "V1", "fold_0", n_null=100)
    # dimensionality
    for d in (geo.d_vis, geo.d_img):
        assert "d_report" in d and "reason" in d and "argmax_rank" in d
    # bases
    assert geo.basis["V_vis_cols"] >= 1 and geo.basis["V_img_cols"] >= 1
    # alignment
    assert set(("TV_vis", "TV_img", "a_g", "evaluable", "d")).issubset(geo.alignment)
    # null: exactly 100 draws when evaluable
    if geo.alignment["evaluable"]:
        assert geo.provenance["n_null"] == 100
        assert len(geo.alignment_null["raw"]) == 100
        assert geo.alignment["d"] == geo.d_img["d_report"] or geo.alignment["d"] <= geo.d_img["d_report"]


def test_geometry_deterministic():
    M, vis_by, img_by = _setup()
    a = ge.run_fold_geometry(M, vis_by, img_by, rp.root_digest(0), "subj02", "V1", "fold_0", n_null=40)
    b = ge.run_fold_geometry(M, vis_by, img_by, rp.root_digest(0), "subj02", "V1", "fold_0", n_null=40)
    assert a.alignment["a_g"] == b.alignment["a_g"] or (
        np.isnan(a.alignment["a_g"]) and np.isnan(b.alignment["a_g"]))
    assert a.basis["vis_projector_hash"] == b.basis["vis_projector_hash"]
    assert a.alignment_null["raw"] == b.alignment_null["raw"]


def test_alignment_uses_same_d_in_num_and_denom():
    M, vis_by, img_by = _setup()
    geo = ge.run_fold_geometry(M, vis_by, img_by, rp.root_digest(0), "subj02", "V1", "fold_0", n_null=10)
    if geo.alignment["evaluable"]:
        # d recorded on the alignment equals the imagery dimensionality (bounded by basis cols)
        assert geo.alignment["d"] == min(geo.d_img["d_report"], geo.basis["V_vis_cols"], geo.basis["V_img_cols"])
