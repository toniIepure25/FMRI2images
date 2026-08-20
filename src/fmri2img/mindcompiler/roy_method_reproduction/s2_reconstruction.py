"""S2.0 PART D+F -- per-fold vis2vis selection, D1 denoising, and vis2img mapping.

One fold's reconstruction, given the response matrix ``M`` (trial × voxel, PSC) and
that fold's per-(identity, state) train/val/test rows:

1. **vis2vis** (PART D): pooled within-identity pairs; select ``(lam, rank)`` on
   train→val with the rank cap ``min(12, src_dim, tgt_dim, effective_train_rank)``
   and the 99%-of-peak parsimony rule.
2. **D1 denoising** (PART E): leave-one-out for train vision, full-train for
   val/test, using the selected ``(lam, rank)``; leakage-validated.
3. **vis2img** (PART F): X = denoised vision, Y = imagery, paired within identity
   and split (index-aligned); train-only preprocessing; select ``(lam, rank)`` on
   val; evaluate test ONCE with the structured metrics module.

Everything is train-only fit; validation never sees test; test is evaluated once.
Operates on arrays so it is data-free testable; a separate driver supplies real M.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from fmri2img.mindcompiler.roy_method_reproduction.metrics import MetricReport, per_voxel_report
from fmri2img.mindcompiler.roy_method_reproduction.reduced_rank_ridge import (
    fit_reduced_rank, select_hyperparameters,
)
from fmri2img.mindcompiler.roy_method_reproduction import s2_denoising as dn
from fmri2img.mindcompiler.roy_method_reproduction.s2_preprocessing import FittedScaler

RANK_HARD_MAX = 12  # future_roy_rank_max (paper condition bound)


def rank_cap(n_train_conditions: int, src_dim: int, tgt_dim: int,
             effective_train_rank: int, hard: int = RANK_HARD_MAX) -> int:
    """rank_max = min(12, #train conditions, src dim, tgt dim, effective train rank)."""
    return max(1, min(hard, n_train_conditions, src_dim, tgt_dim, effective_train_rank))


def _by_ident(assignment: Dict[Tuple[str, str], Dict[str, np.ndarray]], state: str
              ) -> Dict[str, Dict[str, np.ndarray]]:
    """Reshape fold assignment -> {identity: {split: rows}} for one state."""
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for (ident, st), splits in assignment.items():
        if st == state:
            out[ident] = splits
    return out


def _split_rows(by_ident: Dict[str, Dict[str, np.ndarray]], split: str) -> Dict[str, np.ndarray]:
    return {ident: d[split] for ident, d in by_ident.items()}


@dataclass
class FoldResult:
    fold: str
    beta_version: str
    preproc_policy: str
    vis2vis_pairing: str
    pairing_seed: Optional[int]
    denoising: str
    vis2vis: dict
    vis2img: dict
    denoise_leakage: dict
    n_denoise_records: int


def _select_and_report(Xtr, Ytr, Xva, Yva, Xte, Yte, n_train_conditions):
    """Ridge+reduced-rank select on val, evaluate test once. Returns (sel, MetricReport)."""
    eff = int(min(np.linalg.matrix_rank(Xtr), Xtr.shape[0]))
    rmax = rank_cap(n_train_conditions, Xtr.shape[1], Ytr.shape[1], eff)
    sel = select_hyperparameters(Xtr, Ytr, Xva, Yva,
                                 "smallest_rank_at_threshold", "argmax_validation", r_max=rmax)
    W = fit_reduced_rank(Xtr, Ytr, sel.lam, sel.rank)
    rep = per_voxel_report(Yte, Xte @ W)
    return sel, rep, rmax


def run_fold(M: np.ndarray, assignment, row_identity: Dict[int, str],
             row_beta_index: Dict[int, int], *, beta_version: str, preproc_policy: str,
             vis2vis_pairing: str, pairing_seed: Optional[int]) -> FoldResult:
    """Reconstruct one fold end to end (vis2vis → D1 → vis2img)."""
    fold_name = "fold"  # label supplied by caller via assignment context
    vis = _by_ident(assignment, "vision")
    img = _by_ident(assignment, "imagery")
    n_conditions = len(vis)  # identities in the fold
    nv = M.shape[1]

    # --- PART D: vis2vis selection on train->val (pooled within-identity pairs) ---
    tr_by = _split_rows(vis, "train")
    va_by = _split_rows(vis, "val")
    S_tr, T_tr = dn.pooled_pairs(tr_by, vis2vis_pairing, pairing_seed)
    S_va, T_va = dn.pooled_pairs(va_by, vis2vis_pairing, pairing_seed)
    sx = FittedScaler.fit(M[S_tr], preproc_policy)
    sy = FittedScaler.fit(M[T_tr], preproc_policy)
    v2v_sel, _rep_ignored, v2v_rmax = _select_and_report(
        sx.transform(M[S_tr]), sy.transform(M[T_tr]),
        sx.transform(M[S_va]), sy.transform(M[T_va]),
        sx.transform(M[S_va]), sy.transform(M[T_va]),  # test slot unused for v2v (selection only)
        n_conditions)

    # --- PART E: D1 denoising with selected (lam, rank) ---
    te_by = _split_rows(vis, "test")
    denoised, recs = dn.d1_crossfit_denoise(
        M, fold_name, tr_by, va_by, te_by, row_identity, row_beta_index,
        lam=v2v_sel.lam, rank=v2v_sel.rank, pairing_policy=vis2vis_pairing,
        pairing_seed=pairing_seed, preproc_policy=preproc_policy)
    fold_train_vis = {int(x) for rows in tr_by.values() for x in rows}
    leak = dn.validate_denoising_leakage(recs, fold_train_vis, row_identity)

    # --- PART F: vis2img (denoised vision -> imagery), within-identity index-aligned ---
    def pairs(split):
        Xrows, Yrows = [], []
        for ident in sorted(vis):
            vr = sorted(int(x) for x in vis[ident][split])
            ir = sorted(int(x) for x in img[ident][split])
            for a, b in zip(vr, ir):
                Xrows.append(a); Yrows.append(b)
        return Xrows, Yrows

    def build(split):
        Xr, Yr = pairs(split)
        X = np.vstack([denoised[r] for r in Xr])
        Y = M[Yr]
        return X, Y

    Xtr, Ytr = build("train")
    Xva, Yva = build("val")
    Xte, Yte = build("test")
    ix = FittedScaler.fit(Xtr, preproc_policy)
    iy = FittedScaler.fit(Ytr, preproc_policy)
    v2i_sel, v2i_rep, v2i_rmax = _select_and_report(
        ix.transform(Xtr), iy.transform(Ytr), ix.transform(Xva), iy.transform(Yva),
        ix.transform(Xte), iy.transform(Yte), n_conditions)

    return FoldResult(
        fold=fold_name, beta_version=beta_version, preproc_policy=preproc_policy,
        vis2vis_pairing=vis2vis_pairing, pairing_seed=pairing_seed, denoising=dn.DENOISE_LABEL,
        vis2vis=dict(lam=v2v_sel.lam, rank=v2v_sel.rank, val_score=v2v_sel.val_score,
                     rank_max=v2v_rmax, n_train_pairs=int(len(S_tr)),
                     n_zero_var_src=sx.n_zero_var, n_zero_var_tgt=sy.n_zero_var),
        vis2img=dict(lam=v2i_sel.lam, rank=v2i_sel.rank, val_score=v2i_sel.val_score,
                     rank_max=v2i_rmax, n_train=int(Xtr.shape[0]),
                     **_metric_dict(v2i_rep)),
        denoise_leakage=leak, n_denoise_records=len(recs))


def _metric_dict(rep: MetricReport) -> dict:
    d = rep.as_dict()
    d["finite_frac"] = d.pop("finite_fraction")
    return d
