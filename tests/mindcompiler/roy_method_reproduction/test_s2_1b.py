"""S2.1B data-free tests: D1b symmetric cross-fit, RAW mode, diagnostic formulas."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fmri2img.mindcompiler.roy_method_reproduction import s2_1b_diagnostics as dg
from fmri2img.mindcompiler.roy_method_reproduction import s2_denoising as dn
from fmri2img.mindcompiler.roy_method_reproduction import s2_folds as sf
from fmri2img.mindcompiler.roy_method_reproduction import s2_reconstruction as rc
from fmri2img.mindcompiler.roy_method_reproduction.s2_preprocessing import PREPROC_ZSCORE_TRAIN_ONLY


def _setup(n_id=3, n_vox=10, seed=0):
    rng = np.random.default_rng(seed)
    train_by, val_by, test_by, row_identity, row_beta = {}, {}, {}, {}, {}
    r = 0
    for i in range(n_id):
        ident = f"ID{i}"; rows = list(range(r, r + 8)); r += 8
        for x in rows:
            row_identity[x] = ident; row_beta[x] = 1000 + x
        train_by[ident] = rows[:4]; val_by[ident] = rows[4:6]; test_by[ident] = rows[6:8]
    M = rng.standard_normal((r, n_vox))
    return M, train_by, val_by, test_by, row_identity, row_beta


# --- D1b symmetric cross-fit --------------------------------------------------

def test_d1b_no_self_leak_and_val_test_safe():
    M, tr, va, te, rid, rb = _setup()
    den, recs = dn.d1b_symmetric_denoise(M, "fold_0", tr, va, te, rid, rb,
                                         lam=10.0, rank=3, pairing_policy="all_ordered_distinct",
                                         pairing_seed=1234, preproc_policy=PREPROC_ZSCORE_TRAIN_ONLY)
    fold_train = {int(x) for rows in tr.values() for x in rows}
    v = dn.validate_denoising_leakage(recs, fold_train, rid)
    assert all(x == 0 for x in v.values())
    for r in recs:
        assert r.denoised_row not in r.train_target_rows   # trial never denoises itself
    # symmetric: val/test use the ensemble model id
    assert any(r.model_id.endswith("D1b_ensemble") for r in recs if r.split in ("val", "test"))


def test_d1b_differs_from_d1():
    M, tr, va, te, rid, rb = _setup()
    d1, _ = dn.d1_crossfit_denoise(M, "f", tr, va, te, rid, rb, lam=10.0, rank=3,
                                   pairing_policy="all_ordered_distinct", pairing_seed=None,
                                   preproc_policy=PREPROC_ZSCORE_TRAIN_ONLY)
    d1b, _ = dn.d1b_symmetric_denoise(M, "f", tr, va, te, rid, rb, lam=10.0, rank=3,
                                      pairing_policy="all_ordered_distinct", pairing_seed=1234,
                                      preproc_policy=PREPROC_ZSCORE_TRAIN_ONLY)
    row = next(iter(tr.values()))[0]
    assert not np.allclose(d1[row], d1b[row])   # genuinely different denoising


# --- RAW mode -----------------------------------------------------------------

def test_raw_mode_uses_raw_vision_and_runs():
    rng = np.random.default_rng(1)
    n_id, n_vox = 4, 12
    rows = []
    bi = 0
    for state in ("vision", "imagery"):
        for i in range(n_id):
            for rep in range(8):
                rows.append(dict(identity=f"ID{i}", state=state, repeat=rep, beta_index0=bi)); bi += 1
    tt = pd.DataFrame(rows)
    latent = rng.standard_normal((n_id, n_vox))
    M = np.vstack([latent[int(r.identity[2:])] + 0.3 * rng.standard_normal(n_vox)
                   for _, r in tt.iterrows()])
    folds = sf.build_four_folds(tt, 1234)
    rid = {r: tt.loc[r, "identity"] for r in tt.index}
    rb = {r: int(tt.loc[r, "beta_index0"]) for r in tt.index}
    res, recs = rc.run_fold(M, folds["fold_0"], rid, rb, beta_version="syn",
                            preproc_policy=PREPROC_ZSCORE_TRAIN_ONLY, vis2vis_pairing="all_ordered_distinct",
                            pairing_seed=None, fold_name="fold_0", return_records=True, denoising="RAW")
    assert res.denoising == "RAW_S2_MATCHED" and recs == []
    assert "mean" in res.vis2img and 0.0 <= res.vis2img["finite_frac"] <= 1.0


def test_unknown_denoising_rejected():
    M, tr, va, te, rid, rb = _setup()
    tt = pd.DataFrame([dict(identity="ID0", state="vision", repeat=0, beta_index0=0)])
    with pytest.raises(ValueError, match="unknown denoising"):
        rc.run_fold(M, {("ID0", "vision"): {"train": np.array([0]), "val": np.array([]), "test": np.array([])}},
                    rid, rb, beta_version="x", preproc_policy=PREPROC_ZSCORE_TRAIN_ONLY,
                    vis2vis_pairing="all_ordered_distinct", pairing_seed=None, denoising="BOGUS")


# --- diagnostic formulas ------------------------------------------------------

def test_paired_similarity_identity_case():
    M = np.random.default_rng(2).standard_normal((10, 8))
    s = dg.paired_trial_similarity(M, M.copy(), list(range(10)))
    assert abs(s["pearson"]["mean"] - 1.0) < 1e-9 and abs(s["norm_ratio"]["mean"] - 1.0) < 1e-9


def test_shrinkage_pure_scale_has_no_rotation_gain():
    M0 = np.random.default_rng(3).standard_normal((20, 6))
    M1 = 0.5 * M0 + 2.0                       # pure global scale+shift
    out = dg.shrinkage_vs_rotation(M0, M1, list(range(20)))
    assert out["scale_only"]["r2"] > 0.999
    assert abs(out["rotation_gain_over_scale"]) < 1e-6


def test_principal_angles_same_subspace_zero():
    X = np.random.default_rng(4).standard_normal((30, 8))
    out = dg.principal_angles(X, X.copy(), dim=3)
    assert out["mean_angle_deg"] < 1e-6


def test_cross_state_matching_perfect_when_aligned():
    C = np.random.default_rng(5).standard_normal((6, 10))
    out = dg.cross_state_matching(C, C.copy())
    assert out["top1_accuracy"] == 1.0 and out["separation"] > 0


def test_covariance_spectrum_and_reliability_run():
    M = np.random.default_rng(6).standard_normal((24, 9))
    cs = dg.covariance_spectrum(M)
    assert cs["participation_ratio"] > 0 and len(cs["eigen_spectrum"]) == 9
    rel = dg.repeat_reliability(M, {"ID0": [0, 1, 2, 3], "ID1": [4, 5, 6, 7]})
    assert "mean" in rel and "per_identity" in rel
