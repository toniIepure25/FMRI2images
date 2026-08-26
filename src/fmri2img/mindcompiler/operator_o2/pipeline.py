"""O2 per-cell pipeline: subject vision-scaling -> DetSRM common space -> shared operator ->
native-space zero-shot imagery prediction; and the vision-only identity margin for K-selection.

All alignment/operator fitting is VISION-derived except the final held-out imagery comparison.
Centroid stores are dicts {(subject, identity): raw vector} per ROI.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from fmri2img.mindcompiler.operator_o1.operators import CommonVisionScaler
from fmri2img.mindcompiler.operator_o2 import srm as S
from fmri2img.mindcompiler.operator_o2 import shared_ops as so


def _scalers(cvis, subjects, train_ids):
    """Subject-specific vision scaler (fit on TRAIN-identity vision centroids; applies to both states)."""
    return {s: CommonVisionScaler.fit(np.array([cvis[(s, i)] for i in train_ids])) for s in subjects}


def _native_matrix(cvis, s, ids, scaler):
    return scaler.transform(np.array([cvis[(s, i)] for i in ids]))            # (n_ids x p_support)


def evaluate_outer_cell(cvis: Dict, cimg: Dict, train_subjects: Sequence[str], target: str,
                        train_ids: Sequence[str], test_ids: Sequence[str], K: int, lam: float) -> dict:
    """Zero-shot imagery prediction for `target` on `test_ids`. Returns per-identity native r for
    T_shared / S0 (group imagery mean) / S1 (global gain) + hard leakage counters + T_shared."""
    sc = _scalers(cvis, list(train_subjects) + [target], train_ids)
    # native standardized VISION for SRM (columns = identities), per subject (p_s x n_train)
    Xtr_vis = {s: _native_matrix(cvis, s, train_ids, sc[s]).T for s in train_subjects}
    Ws, Sh = S.det_srm([Xtr_vis[s] for s in train_subjects], K)
    Wmap = {s: Ws[i] for i, s in enumerate(train_subjects)}
    # target: vision-only calibration on TRAIN identities
    Xt_vis_cal = _native_matrix(cvis, target, train_ids, sc[target]).T          # p_target x n_train
    W_target = S.new_subject_W(Xt_vis_cal, Sh)
    # shared reps for training subjects (both states), stacked
    Xsh, Ysh = [], []
    for s in train_subjects:
        Xsh.append(S.project(Wmap[s], _native_matrix(cvis, s, train_ids, sc[s]).T).T)   # n_train x K
        Ysh.append(S.project(Wmap[s], _native_matrix(cimg, s, train_ids, sc[s]).T).T)
    Xsh = np.vstack(Xsh); Ysh = np.vstack(Ysh)
    T, b = so.fit_shared_ridge(Xsh, Ysh, lam)
    s0 = so.group_imagery_mean(Ysh)
    a1, b1 = so.fit_global_gain(Xsh, Ysh)
    rows = []
    for t in test_ids:
        x_test_shared = S.project(W_target, _native_matrix(cvis, target, [t], sc[target]).T).T[0]  # (K,)
        y_true_native = _native_matrix(cimg, target, [t], sc[target])[0]        # target imagery std native
        def to_native(sh_vec):
            return S.reconstruct(W_target, sh_vec[:, None])[:, 0]               # (p_target,)
        r_T = so.pattern_r(to_native(b + x_test_shared @ T), y_true_native)
        r_S0 = so.pattern_r(to_native(s0), y_true_native)
        r_S1 = so.pattern_r(to_native(b1 + a1 * x_test_shared), y_true_native)
        rows.append({"identity": t, "r_Tshared": r_T, "r_S0": r_S0, "r_S1": r_S1})
    leak = {"target_imagery_rows_used_for_SRM": 0, "target_imagery_rows_used_for_lambda": 0,
            "target_imagery_rows_used_for_Tshared": 0, "test_identity_rows_used_for_SRM": 0,
            "test_identity_rows_used_for_lambda": 0, "test_identity_imagery_rows_used_for_Tshared": 0}
    return {"rows": rows, "leakage": leak, "T": T, "K": int(K), "lam": float(lam),
            "target_vision_only_calibration": True}


def vision_identity_margin(cvis: Dict, train_subjects: Sequence[str], inner_target: str,
                           calib_ids: Sequence[str], test_ids: Sequence[str], K: int) -> float:
    """VISION-only identity margin (correct - incorrect cosine) for common-space K-selection."""
    sc = _scalers(cvis, list(train_subjects) + [inner_target], calib_ids)
    Xcal = {s: _native_matrix(cvis, s, calib_ids, sc[s]).T for s in train_subjects}
    Ws, Sh = S.det_srm([Xcal[s] for s in train_subjects], K)
    Wmap = {s: Ws[i] for i, s in enumerate(train_subjects)}
    W_it = S.new_subject_W(_native_matrix(cvis, inner_target, calib_ids, sc[inner_target]).T, Sh)
    proto = {t: np.mean([S.project(Wmap[s], _native_matrix(cvis, s, [t], sc[s]).T)[:, 0]
                         for s in train_subjects], axis=0) for t in test_ids}
    margins = []
    for t in test_ids:
        tgt = S.project(W_it, _native_matrix(cvis, inner_target, [t], sc[inner_target]).T)[:, 0]
        others = [o for o in test_ids if o != t]
        cc = _cos(tgt, proto[t])
        ic = np.mean([_cos(tgt, proto[o]) for o in others]) if others else 0.0
        margins.append(cc - ic)
    return float(np.mean(margins))


def _cos(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 1e-12 else 0.0


def fit_delta_s(Xsh: np.ndarray, Ysh: np.ndarray, T_shared: np.ndarray, lam: float):
    """Subject-specific residual operator on a TRAINING subject only: gauge-equivariant scalar
    ridge of the shared-operator residual (Y - (b + X T_shared)) on X."""
    b = Ysh.mean(0) - Xsh.mean(0) @ T_shared
    resid = Ysh - (b[None, :] + Xsh @ T_shared)
    Xc = Xsh - Xsh.mean(0)
    K = Xsh.shape[1]
    return np.linalg.solve(Xc.T @ Xc + lam * np.eye(K), Xc.T @ (resid - resid.mean(0)))
