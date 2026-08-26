"""O1 operator geometry on the empirically supported visual-centroid span (Parts S-V).

Characterizes the FULL_RRR (O4) operator restricted to the span the experiment actually
sampled (12 visual identity centroids). Never interprets unsupported voxel directions.
Operator matrix convention: prediction is y = x @ W (row vectors), so the linear map on
column direction vectors is A = W^T (A @ v applies the operator to direction v).
"""
from __future__ import annotations

from typing import List

import numpy as np

STABILITY_HIGH = 0.80          # frozen: mean pairwise supported-action cosine >= => HIGH
STABILITY_MODERATE = 0.50      # >= => MODERATE, else LOW
_EPS = 1e-12


def supported_basis(vis_centroids_scaled: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Orthonormal basis B_v (p x k) of the centered visual identity-centroid span."""
    Xc = vis_centroids_scaled - vis_centroids_scaled.mean(0)
    U, s, Vt = np.linalg.svd(Xc.T, full_matrices=False)      # columns of U span the row space
    k = int((s > tol * (s[0] if s.size and s[0] > 0 else 1.0)).sum())
    return U[:, :max(1, k)]


def _eff_ranks(sv: np.ndarray) -> dict:
    sv = np.asarray(sv, float); sv = sv[sv > _EPS]
    if sv.size == 0:
        return {"numerical_rank": 0, "participation_ratio": 0.0, "energy90_rank": 0}
    pr = float((sv.sum() ** 2) / (np.square(sv).sum()))
    energy = np.cumsum(sv ** 2) / (sv ** 2).sum()
    e90 = int(np.searchsorted(energy, 0.90) + 1)
    numrank = int((sv > 1e-6 * sv.max()).sum())
    return {"numerical_rank": numrank, "participation_ratio": pr, "energy90_rank": e90}


def operator_geometry(W: np.ndarray, B_v: np.ndarray) -> dict:
    """Structural diagnostics of operator W restricted to supported span B_v (Part S/T)."""
    A = W.T                                        # column-direction operator
    AB = A @ B_v                                   # p x k action on supported directions
    k = B_v.shape[1]
    P_out = B_v @ B_v.T
    ident_dev = float(np.linalg.norm(AB - B_v) / (np.linalg.norm(B_v) + _EPS))
    oos = float((np.linalg.norm((np.eye(A.shape[0]) - P_out) @ AB) ** 2) / (np.linalg.norm(AB) ** 2 + _EPS))
    sv = np.linalg.svd(AB, compute_uv=False)
    resid = AB - B_v
    sv_res = np.linalg.svd(resid, compute_uv=False)
    Ab = B_v.T @ A @ B_v                           # k x k operator in supported coords
    offdiag = float((np.linalg.norm(Ab - np.diag(np.diag(Ab))) ** 2) / (np.linalg.norm(Ab) ** 2 + _EPS))
    return {
        "supported_source_rank": int(k),
        "identity_deviation": ident_dev,
        "out_of_visual_span_fraction": oos,
        "restricted_singular_values": {"median": float(np.median(sv)), "min": float(sv.min()),
                                       "max": float(sv.max()), "frac_lt_1": float((sv < 1).mean()),
                                       "frac_gt_1": float((sv > 1).mean())},
        "effective_output_rank": _eff_ranks(sv),
        "residual_operator": {"frobenius": float(np.linalg.norm(resid)), **_eff_ranks(sv_res)},
        "off_diagonal_mixing_energy": offdiag,
        "_Ab": Ab, "_sv": sv.tolist(),
    }


def polar_decomposition(Ab: np.ndarray) -> dict:
    """Polar decomposition of the supported-coord operator: Ab = R P (rotation x stretch)."""
    U, s, Vt = np.linalg.svd(Ab)
    R = U @ Vt                                     # nearest orthogonal (rotation/reflection)
    rot_frob = float(np.linalg.norm(R - np.eye(R.shape[0])))
    cos_ang = np.clip((np.trace(R) - (R.shape[0] - 2)) / 2.0, -1.0, 1.0) if R.shape[0] >= 2 else 1.0
    return {
        "rotation_frobenius": rot_frob,
        "rotation_mean_angle_proxy": float(np.arccos(np.clip((np.trace(R) / R.shape[0]), -1, 1))),
        "stretch_singular_values": {"mean": float(s.mean()), "min": float(s.min()), "max": float(s.max())},
        "stretch_anisotropy": float(s.max() / (s.min() + _EPS)),
        "mean_contraction": float(s[s < 1].mean()) if (s < 1).any() else None,
        "mean_expansion": float(s[s > 1].mean()) if (s > 1).any() else None,
        "fraction_contracted": float((s < 1).mean()), "fraction_expanded": float((s > 1).mean()),
        "determinant_sign": float(np.sign(np.linalg.det(R))),
    }


def operator_stability(W_list: List[np.ndarray], B_v: np.ndarray) -> dict:
    """Compare fold operators' action on the COMMON supported span B_v (Part V)."""
    actions = [(W.T @ B_v) for W in W_list]
    vecs = [a.ravel() / (np.linalg.norm(a) + _EPS) for a in actions]
    svs = [np.linalg.svd(a, compute_uv=False) for a in actions]
    n = len(actions)
    cos_pairs, subspace_pairs, sv_pairs = [], [], []
    for i in range(n):
        Qi, _ = np.linalg.qr(actions[i])
        for j in range(i + 1, n):
            cos_pairs.append(float(vecs[i] @ vecs[j]))
            Qj, _ = np.linalg.qr(actions[j])
            sv_ij = np.linalg.svd(Qi.T @ Qj, compute_uv=False)
            subspace_pairs.append(float(sv_ij.mean()))        # mean cos principal angle
            sv_pairs.append(float(np.corrcoef(svs[i], svs[j])[0, 1]) if svs[i].size > 1 else float("nan"))
    mean_cos = float(np.mean(cos_pairs)) if cos_pairs else float("nan")
    label = ("OPERATOR_STABILITY_HIGH" if mean_cos >= STABILITY_HIGH else
             "OPERATOR_STABILITY_MODERATE" if mean_cos >= STABILITY_MODERATE else "OPERATOR_STABILITY_LOW")
    return {"mean_pairwise_action_cosine": mean_cos,
            "mean_subspace_principal_cosine": float(np.mean(subspace_pairs)) if subspace_pairs else float("nan"),
            "mean_singular_profile_corr": float(np.nanmean(sv_pairs)) if sv_pairs else float("nan"),
            "n_pairs": len(cos_pairs), "label": label}
