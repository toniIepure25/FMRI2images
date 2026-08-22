"""S2.5M public-method-aligned per-fold engine: Roy pairings + per-target ridge + D1 + null.

One fold, one participant, one ROI, one beta. vis2vis uses within-split derangement
pairing and per-target Lambda; D1 denoises (LOTO train / full-train val-test) with the
FIXED fold-level (Lambda, rank); vis2img uses random within-identity pairing on denoised
vision -> imagery with its own per-target Lambda; a Figure-3-style prediction null breaks
the test input<->output correspondence. All train-only preprocessing; test evaluated once.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from fmri2img.mindcompiler.roy_method_reproduction import roy_pairing as rp
from fmri2img.mindcompiler.roy_method_reproduction import roy_public_rrr as rr
from fmri2img.mindcompiler.roy_method_reproduction.metrics import per_voxel_report
from fmri2img.mindcompiler.roy_method_reproduction.s2_preprocessing import FittedScaler, PREPROC_ZSCORE_TRAIN_ONLY

RANK_HARD_MAX = 12
N_NULL_DEFAULT = 1000


def _rank_cap(n_conditions, src, tgt, Xtr):
    eff = int(min(np.linalg.matrix_rank(Xtr), Xtr.shape[0]))
    return max(1, min(RANK_HARD_MAX, n_conditions, src, tgt, eff))


def _fit_raw(M, src_rows, tgt_rows, lambdas, rank, preproc, predict_rows):
    """Fit per-target-Lambda reduced-rank ridge on (src->tgt) and predict predict_rows in RAW units."""
    sx = FittedScaler.fit(M[src_rows], preproc); sy = FittedScaler.fit(M[tgt_rows], preproc)
    W = rr.fit_W_lambda(sx.transform(M[src_rows]), sy.transform(M[tgt_rows]), lambdas)
    W_rrr = rr._rrr(W, sx.transform(M[src_rows]), rank)
    return sy.inverse_transform(sx.transform(M[list(predict_rows)]) @ W_rrr)


def _prediction_null(Y_te, pred, n_null, seed):
    """Break test input<->output correspondence by permuting measured rows; per-voxel percentile."""
    obs = _pvp(Y_te, pred)
    rng = np.random.default_rng(seed)
    n = Y_te.shape[0]
    null = np.empty((n_null, Y_te.shape[1]))
    for k in range(n_null):
        null[k] = _pvp(Y_te[rng.permutation(n)], pred)
    p95 = np.nanpercentile(null, 95, axis=0)
    with np.errstate(invalid="ignore"):
        emp_p = np.nanmean(null >= obs[None, :], axis=0)
    frac_above = float(np.nanmean(obs > p95))
    return dict(null_mean=float(np.nanmean(null)), null_p95_mean=float(np.nanmean(p95)),
                fraction_voxels_above_null_p95=frac_above,
                median_empirical_p=float(np.nanmedian(emp_p)))


def _pvp(Yt, Yp):
    yt = Yt - Yt.mean(0); yp = Yp - Yp.mean(0)
    st = np.sqrt((yt ** 2).sum(0)); sp = np.sqrt((yp ** 2).sum(0))
    d = st * sp
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(d > 1e-12, (yt * yp).sum(0) / d, np.nan)


@dataclass
class RoyCell:
    vis2vis: dict
    vis2img: dict
    denoise_leakage: dict
    n_denoise_records: int
    rank_curve: list
    null: dict


def run_fold_roy(M, vis_by: Dict[str, Dict[str, Sequence[int]]],
                 img_by: Dict[str, Dict[str, Sequence[int]]], root_hex: str, participant: str,
                 fold_name: str, *, preproc: str = PREPROC_ZSCORE_TRAIN_ONLY,
                 n_null: int = N_NULL_DEFAULT, null_seed: int = 0) -> RoyCell:
    idents = sorted(vis_by)
    nv = M.shape[1]
    tr = {i: list(map(int, vis_by[i]["train"])) for i in idents}
    va = {i: list(map(int, vis_by[i]["val"])) for i in idents}
    te = {i: list(map(int, vis_by[i]["test"])) for i in idents}

    # --- vis2vis: Roy within-split derangement + per-target Lambda ---
    S_tr, T_tr = rp.pooled_vis2vis(tr, root_hex, participant, fold_name, "train")
    S_va, T_va = rp.pooled_vis2vis(va, root_hex, participant, fold_name, "val")
    sx = FittedScaler.fit(M[S_tr], preproc); sy = FittedScaler.fit(M[T_tr], preproc)
    rmax_v = _rank_cap(len(idents), nv, nv, sx.transform(M[S_tr]))
    v2v = rr.fit_select(sx.transform(M[S_tr]), sy.transform(M[T_tr]),
                        sx.transform(M[S_va]), sy.transform(M[T_va]), rmax_v)
    lam_v, rank_v = v2v.lambdas, v2v.r_model

    # --- D1: LOTO train / full-train val-test with FIXED (Lambda, rank) ---
    denoised: Dict[int, np.ndarray] = {}
    leak = {"self_target": 0, "self_source": 0, "val_test_in_training": 0,
            "cross_identity_target": 0, "training_row_not_train_split": 0}
    fold_train_rows = {r for rows in tr.values() for r in rows}
    n_records = 0
    for ident in idents:
        for t in tr[ident]:
            keep = (S_tr != t) & (T_tr != t)
            denoised[t] = _fit_raw(M, S_tr[keep], T_tr[keep], lam_v, rank_v, preproc, [t])[0]
            n_records += 1
            if t in set(S_tr[keep]) or t in set(T_tr[keep]):
                leak["self_source"] += 1  # (never, by construction)
    full_pred_rows = [r for i in idents for r in (va[i] + te[i])]
    full_raw = _fit_raw(M, S_tr, T_tr, lam_v, rank_v, preproc, full_pred_rows)
    for k, r in enumerate(full_pred_rows):
        denoised[r] = full_raw[k]
        n_records += 1
        if r in fold_train_rows:
            leak["val_test_in_training"] += 1

    # --- vis2img: Roy random pairing on denoised vision -> imagery, per-target Lambda ---
    def pairs(split):
        Xr, Yr = [], []
        for ident in idents:
            cs = rp.child_seed(root_hex, participant, fold_name, split, ident, "vis2img")
            v, im = rp.vis2img_pairs(vis_by[ident][split], img_by[ident][split], cs)
            Xr += list(v); Yr += list(im)
        return Xr, Yr

    def build(split):
        Xr, Yr = pairs(split)
        return np.vstack([denoised[r] for r in Xr]), M[Yr]
    Xtr, Ytr = build("train"); Xva, Yva = build("val"); Xte, Yte = build("test")
    ix = FittedScaler.fit(Xtr, preproc); iy = FittedScaler.fit(Ytr, preproc)
    rmax_i = _rank_cap(len(idents), Xtr.shape[1], Ytr.shape[1], ix.transform(Xtr))
    v2i = rr.fit_select(ix.transform(Xtr), iy.transform(Ytr), ix.transform(Xva), iy.transform(Yva), rmax_i)
    W_rrr = rr._rrr(v2i.W_lambda, ix.transform(Xtr), v2i.r_model)
    Yte_s = iy.transform(Yte); pred_te = ix.transform(Xte) @ W_rrr
    rep = per_voxel_report(Yte_s, pred_te)
    curve = rr.rank_curve(ix.transform(Xtr), iy.transform(Ytr), ix.transform(Xva), iy.transform(Yva),
                          ix.transform(Xte), Yte_s, v2i.W_lambda, rmax_i)
    null = _prediction_null(Yte_s, pred_te, n_null, null_seed)

    d = rep.as_dict(); d["finite_frac"] = d.pop("finite_fraction")
    return RoyCell(
        vis2vis=dict(lambda_min=float(lam_v.min()), lambda_median=float(np.median(lam_v)),
                     lambda_max=float(lam_v.max()), n_unique_lambdas=int(np.unique(lam_v).size),
                     r_model=int(rank_v), rank_max=int(rmax_v), val_score=float(v2v.val_score)),
        vis2img=dict(lambda_min=float(v2i.lambdas.min()), lambda_median=float(np.median(v2i.lambdas)),
                     lambda_max=float(v2i.lambdas.max()), n_unique_lambdas=int(np.unique(v2i.lambdas).size),
                     r_model=int(v2i.r_model), rank_max=int(rmax_i), val_score=float(v2i.val_score),
                     test_mean_r=d["mean"], test_median_r=d["median"], finite_frac=d["finite_frac"],
                     target_constant=d["target_constant"], prediction_constant=d["prediction_constant"]),
        denoise_leakage=leak, n_denoise_records=n_records, rank_curve=curve, null=null)
