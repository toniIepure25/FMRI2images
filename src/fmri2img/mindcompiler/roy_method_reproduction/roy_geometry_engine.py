"""S2.6R per-fold geometry engine: deterministic replay of the S2.5M fold + Roy geometry.

Re-runs one S2.5M fold EXACTLY (same pairings, per-target Lambda, D1 denoising, operational
rank selection) and then derives the deferred Roy quantities without touching the model:

* vis2vis TEST rank curve (S2.5M persisted only the vis2img curve);
* vis2img rank curve (recomputed -> replay-audited against the committed S2.5M curve);
* d_vis_fold / d_img_fold via the frozen first-crossing 99%-of-peak rule;
* V_vis / V_img output-space bases (SVD of the fitted TRAIN predictions);
* alignment ratio a_g and its 100-draw random-visual-subspace null.

The model math is imported unchanged from roy_public_rrr / roy_pairing / s2_preprocessing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from fmri2img.mindcompiler.roy_method_reproduction import roy_geometry as rg
from fmri2img.mindcompiler.roy_method_reproduction import roy_pairing as rp
from fmri2img.mindcompiler.roy_method_reproduction import roy_public_rrr as rr
from fmri2img.mindcompiler.roy_method_reproduction.s2_preprocessing import (
    PREPROC_ZSCORE_TRAIN_ONLY, FittedScaler,
)


@dataclass
class GeoCell:
    participant: str
    roi: str
    fold: str
    vis2vis_curve: list          # [{rank,val,test}]
    vis2img_curve: list          # recomputed (for replay audit)
    d_vis: dict                  # {d_report, reason, argmax_rank, test_peak}
    d_img: dict
    basis: dict                  # dims + projector hashes + lambda hashes
    alignment: dict              # TV_vis, TV_img, a_g, evaluable, reason, d
    alignment_null: dict         # summary + raw list
    provenance: dict


def _pool(by, split):
    return {i: list(map(int, by[i][split])) for i in sorted(by)}


def _lambda_hash(lam: np.ndarray) -> str:
    import hashlib
    return hashlib.sha256(np.round(lam, 9).astype(np.float64).tobytes()).hexdigest()[:16]


def run_fold_geometry(M, vis_by: Dict[str, Dict[str, Sequence[int]]],
                      img_by: Dict[str, Dict[str, Sequence[int]]], root_hex: str,
                      participant: str, roi: str, fold_name: str, *,
                      preproc: str = PREPROC_ZSCORE_TRAIN_ONLY,
                      n_null: int = rg.N_ALIGNMENT_NULL) -> GeoCell:
    idents = sorted(vis_by)
    nv = M.shape[1]
    tr, va, te = _pool(vis_by, "train"), _pool(vis_by, "val"), _pool(vis_by, "test")

    # === vis2vis (IDENTICAL to S2.5M run_fold_roy) ===
    S_tr, T_tr = rp.pooled_vis2vis(tr, root_hex, participant, fold_name, "train")
    S_va, T_va = rp.pooled_vis2vis(va, root_hex, participant, fold_name, "val")
    S_te, T_te = rp.pooled_vis2vis(te, root_hex, participant, fold_name, "test")
    sx = FittedScaler.fit(M[S_tr], preproc); sy = FittedScaler.fit(M[T_tr], preproc)
    Xtr_v = sx.transform(M[S_tr]); Ytr_v = sy.transform(M[T_tr])
    rmax_v = rr_rank_cap(len(idents), nv, nv, Xtr_v)
    v2v = rr.fit_select(Xtr_v, Ytr_v, sx.transform(M[S_va]), sy.transform(M[T_va]), rmax_v)
    lam_v, rank_v = v2v.lambdas, v2v.r_model

    # vis2vis TEST rank curve (Roy Figure-4, previously not persisted)
    v2v_curve = rr.rank_curve(Xtr_v, Ytr_v, sx.transform(M[S_va]), sy.transform(M[T_va]),
                              sx.transform(M[S_te]), sy.transform(M[T_te]), v2v.W_lambda, rmax_v)
    V_vis = rg.extract_output_basis(Xtr_v, v2v.W_lambda)

    # === D1 denoising (IDENTICAL to S2.5M) ===
    denoised: Dict[int, np.ndarray] = {}
    for ident in idents:
        for t in tr[ident]:
            keep = (S_tr != t) & (T_tr != t)
            denoised[t] = _fit_raw(M, S_tr[keep], T_tr[keep], lam_v, rank_v, preproc, [t])[0]
    full_rows = [r for i in idents for r in (va[i] + te[i])]
    full_raw = _fit_raw(M, S_tr, T_tr, lam_v, rank_v, preproc, full_rows)
    for k, r in enumerate(full_rows):
        denoised[r] = full_raw[k]

    # === vis2img (IDENTICAL to S2.5M) ===
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
    Xtr_i = ix.transform(Xtr)
    rmax_i = rr_rank_cap(len(idents), Xtr.shape[1], Ytr.shape[1], Xtr_i)
    v2i = rr.fit_select(Xtr_i, iy.transform(Ytr), ix.transform(Xva), iy.transform(Yva), rmax_i)
    v2i_curve = rr.rank_curve(Xtr_i, iy.transform(Ytr), ix.transform(Xva), iy.transform(Yva),
                              ix.transform(Xte), iy.transform(Yte), v2i.W_lambda, rmax_i)
    V_img = rg.extract_output_basis(Xtr_i, v2i.W_lambda)

    # === dimensionality ===
    d_vis_r, d_vis_reason = rg.d99_from_curve([c["test"] for c in v2v_curve])
    d_img_r, d_img_reason = rg.d99_from_curve([c["test"] for c in v2i_curve])
    d_vis = dict(d_report=d_vis_r, reason=d_vis_reason,
                 argmax_rank=rg.argmax_rank([c["test"] for c in v2v_curve]),
                 test_peak=float(np.nanmax([c["test"] for c in v2v_curve])))
    d_img = dict(d_report=d_img_r, reason=d_img_reason,
                 argmax_rank=rg.argmax_rank([c["test"] for c in v2i_curve]),
                 test_peak=float(np.nanmax([c["test"] for c in v2i_curve])))

    # === alignment (d = d_img_fold for BOTH numerator and denominator) ===
    test_vis_rows = [r for i in idents for r in te[i]]
    X_test_vis = sx.transform(M[test_vis_rows])          # train-fitted VISUAL z-score
    if d_img_r is None:
        align = dict(TV_vis=float("nan"), TV_img=float("nan"), a_g=float("nan"),
                     evaluable=False, reason="NON_EVALUABLE_DIMG_" + d_img_reason, d=None)
        null_sum = rg.null_summary(float("nan"), [])
        null_raw = []
    else:
        d = int(min(d_img_r, V_vis.shape[1], V_img.shape[1]))
        align = rg.alignment_ratio(X_test_vis, V_vis, V_img, d)
        null_raw = rg.alignment_null(X_test_vis, V_vis, align["TV_vis"], d,
                                     participant, roi, fold_name, n=n_null)
        null_sum = rg.null_summary(align["a_g"], null_raw)

    sig_vis = rg.output_spectrum(Xtr_v, v2v.W_lambda)
    sig_img = rg.output_spectrum(Xtr_i, v2i.W_lambda)
    basis = dict(nv=int(nv), V_vis_cols=int(V_vis.shape[1]), V_img_cols=int(V_img.shape[1]),
                 vis_projector_hash=(rg.projector_hash(V_vis[:, :d_vis_r]) if d_vis_r else None),
                 img_projector_hash=(rg.projector_hash(V_img[:, :d_img_r]) if d_img_r else None),
                 v2v_lambda_hash=_lambda_hash(lam_v), v2i_lambda_hash=_lambda_hash(v2i.lambdas),
                 v2v_r_model=int(rank_v), v2i_r_model=int(v2i.r_model),
                 rank_max_vis=int(rmax_v), rank_max_img=int(rmax_i),
                 vis_spectral_gap_at_dvis=rg.relative_spectral_gap(sig_vis, d_vis_r),
                 img_spectral_gap_at_dimg=rg.relative_spectral_gap(sig_img, d_img_r),
                 vis_degenerate=bool(rg.relative_spectral_gap(sig_vis, d_vis_r) < rg.DEGENERATE_GAP_TOL),
                 img_degenerate=bool(rg.relative_spectral_gap(sig_img, d_img_r) < rg.DEGENERATE_GAP_TOL))
    provenance = dict(
        vis2vis_source_rows=list(map(int, S_tr)), vis2vis_target_rows=list(map(int, T_tr)),
        vis2vis_test_source=list(map(int, S_te)), vis2vis_test_target=list(map(int, T_te)),
        test_vis_rows=test_vis_rows, n_train_vis=int(S_tr.size),
        alignment_space=rg.ALIGNMENT_SPACE, null_basis=rg.ALIGNMENT_NULL_BASIS,
        n_null=int(len(null_raw)))
    return GeoCell(participant, roi, fold_name, v2v_curve, v2i_curve, d_vis, d_img,
                   basis, align, dict(summary=null_sum, raw=null_raw), provenance)


# ---- helpers reused verbatim from the S2.5M engine (kept local to avoid import cycles) ----
RANK_HARD_MAX = 12


def rr_rank_cap(n_conditions, src, tgt, Xtr):
    eff = int(min(np.linalg.matrix_rank(Xtr), Xtr.shape[0]))
    return max(1, min(RANK_HARD_MAX, n_conditions, src, tgt, eff))


def _fit_raw(M, src_rows, tgt_rows, lambdas, rank, preproc, predict_rows):
    sx = FittedScaler.fit(M[src_rows], preproc); sy = FittedScaler.fit(M[tgt_rows], preproc)
    W = rr.fit_W_lambda(sx.transform(M[src_rows]), sy.transform(M[tgt_rows]), lambdas)
    W_rrr = rr._rrr(W, sx.transform(M[src_rows]), rank)
    return sy.inverse_transform(sx.transform(M[list(predict_rows)]) @ W_rrr)
