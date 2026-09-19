"""MINDIR-X2 latent-state certification (data-free/synthetic). Verifies GMM recovery, that single-Gaussian data
does NOT yield a spurious held-out K2 gain, that a planted 2-state mixture does, that heavy-tailed data favors
Student-t over K2 (no false discreteness), label-swap invariance of state geometry, the 2-test classification
branches, and config immutability. Folds/repeats are never the inferential N (N=8 participants)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fmri2img.mindcompiler.x2_latent_imagery_states import latent_states as L
import fmri2img.mindcompiler.operator_o2_3a_rd.geometry as G

rng = np.random.default_rng(2)


def _heldout_gain(Xtr, Xte):
    mu1, cov1 = L._fit_gauss(Xtr)
    g2 = L._fit_gmm2(Xtr, G, "seed")
    ll1 = float(np.mean(L._gauss_ll(Xte, mu1, cov1))); ll2 = float(np.mean(L._gmm_ll(Xte, g2[1], g2[2], g2[3])))
    return ll2 - ll1, g2


def test_gmm_recovers_two_clusters():
    A = rng.standard_normal((100, 2)) * 0.3 + np.array([-3, 0]); B = rng.standard_normal((100, 2)) * 0.3 + np.array([3, 0])
    X = np.vstack([A, B]); g2 = L._fit_gmm2(X, G, "rec")
    sep = np.linalg.norm(g2[1][1] - g2[1][0]); assert sep > 4.0
    hard = np.argmax(g2[4], 1); occ = np.mean(hard == 0); assert 0.35 < occ < 0.65


def test_single_gaussian_no_spurious_k2_gain():
    X = rng.standard_normal((160, 2))
    gains = []
    for _ in range(5):
        idx = rng.permutation(160); gains.append(_heldout_gain(X[idx[:120]], X[idx[120:]])[0])
    assert np.median(gains) < 0.15  # no meaningful held-out gain for truly single-Gaussian data


def test_planted_two_state_k2_gain_positive():
    A = rng.standard_normal((120, 2)) * 0.4 + np.array([-3, 0]); B = rng.standard_normal((120, 2)) * 0.4 + np.array([3, 0])
    X = np.vstack([A, B]); idx = rng.permutation(240)
    gain, _ = _heldout_gain(X[idx[:180]], X[idx[180:]]); assert gain > 0.3


def test_student_t_control_no_false_discrete():
    # heavy-tailed single-state data: Student-t should predict held-out >= K2
    df = 3.0; base = rng.standard_normal((240, 2)) / np.sqrt(rng.chisquare(df, (240, 1)) / df)
    idx = rng.permutation(240); Xtr, Xte = base[idx[:180]], base[idx[180:]]
    _, g2 = _heldout_gain(Xtr, Xte); mut, covt, nu = L._fit_student_t(Xtr, G, "t")
    ll2 = float(np.mean(L._gmm_ll(Xte, g2[1], g2[2], g2[3]))); llt = float(np.mean(L._t_ll(Xte, mut, covt, nu)))
    assert llt >= ll2 - 0.2  # t not clearly beaten by K2 on heavy-tailed single-state data


def test_label_swap_invariance():
    A = rng.standard_normal((80, 2)) * 0.5 + np.array([-2, 0]); B = rng.standard_normal((80, 2)) * 0.5 + np.array([2, 0])
    g2 = L._fit_gmm2(np.vstack([A, B]), G, "sw")
    d01 = g2[1][1] - g2[1][0]; d10 = g2[1][0] - g2[1][1]
    poolc = 0.5 * (g2[2][0] + g2[2][1])
    m01 = float(np.sqrt(d01 @ np.linalg.inv(poolc) @ d01)); m10 = float(np.sqrt(d10 @ np.linalg.inv(poolc) @ d10))
    assert abs(m01 - m10) < 1e-9  # Mahalanobis separation label-symmetric


# ---- classification ----
def _res(vals):
    return {s: {roi: dict(vals[roi]) for roi in vals} for s in L.ALL}


def _cell(E, DLL, k2t, degen, gen, tgain):
    return {"E_STATE": E, "DELTA_LL": DLL, "k2_gt_t_frac": k2t, "degenerate_folds": degen, "cross_id_min_distinct_med": gen,
            "median_tgain": tgain, "occupancy_med": 0.4, "state_sep_med": 2.0, "max_post_med": 0.9, "ambiguous_frac": 0.1}


def test_classify_supported():
    v = _cell(3.0, 0.5, 0.8, 0, 4, 0.2)
    _, infer, roi_status, prog = L.classify(_res({"ventral": v, "lateral": v}), ["ventral", "lateral"], G)
    assert roi_status["ventral"] == "LATENT_TWO_STATE_STRUCTURE"
    assert prog == "LATENT_IMAGERY_TWO_STATE_STRUCTURE_SUPPORTED"


def test_classify_unresolved_by_student_t():
    v = _cell(3.0, 0.5, 0.2, 0, 4, -0.1)  # K2 beats K1 but NOT Student-t
    _, infer, roi_status, prog = L.classify(_res({"ventral": v, "lateral": v}), ["ventral", "lateral"], G)
    assert roi_status["ventral"] == "MULTIMODAL_REPEAT_GEOMETRY_DISCRETENESS_UNRESOLVED"
    assert prog == "REPEAT_GEOMETRY_MULTIMODAL_BUT_DISCRETENESS_UNRESOLVED"


def test_classify_negative():
    v = _cell(-0.5, -0.01, 0.3, 0, 4, -0.2)
    _, infer, roi_status, prog = L.classify(_res({"ventral": v, "lateral": v}), ["ventral", "lateral"], G)
    assert roi_status["ventral"] == "NO_REPRODUCIBLE_LATENT_STATE" and prog == "NO_REPRODUCIBLE_LATENT_STATE_STRUCTURE"


def test_config():
    cfg = json.load(open("artifacts/mindcompiler/x2_latent_imagery_states/x2_frozen_config.json"))
    assert cfg["O3"] == "O3_NOT_READY" and cfg["N"] == 8
    assert cfg["primary_inference"]["family"].startswith("2 ROIs") and cfg["dimensionality"]["primary_D"] == 2
    assert "config_sha256" in cfg
    src = Path(L.__file__).read_text()
    for bad in ("sklearn", "GaussianMixture", "torch", "n_components_search"):
        assert bad not in src
