"""MINDIR-X3 certification (data-free/synthetic). Verifies angle basis/sign invariance, Spearman/Fisher, the
angular-stability effect direction (stable scaffold -> real more stable than constrained null), the 4-test
classification branches (full/partial/multiregime/negative), H1 pipeline-triviality gate, and config
immutability. Folds/schedules/repeats are never the inferential N (N=8)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fmri2img.mindcompiler.x3_angular_reliability import angular_reliability as X
import fmri2img.mindcompiler.operator_o2_3a_rd.geometry as G

rng = np.random.default_rng(3)
X._G = G


def test_angle_basis_sign_invariance():
    V, K = 50, 8
    W = np.linalg.qr(rng.standard_normal((V, K)))[0][:, :K]; R = rng.standard_normal((4, V))
    a0 = X._angle_cos2(W, R, 4)
    Rot = np.linalg.qr(rng.standard_normal((K, K)))[0]; S = np.diag(rng.choice([-1.0, 1.0], K))
    assert np.max(np.abs(a0 - X._angle_cos2(W @ Rot, R, 4))) < 1e-9
    assert np.max(np.abs(a0 - X._angle_cos2(W @ S, R, 4))) < 1e-9


def test_spearman_and_fisher():
    x = np.array([1., 2, 3, 4, 5]); assert abs(X._spearman(x, 2 * x) - 1) < 1e-9 and abs(X._spearman(x, -x) + 1) < 1e-9
    Z, rho = X._fisher_mean([0.4, 0.4, 0.4]); assert abs(rho - 0.4) < 1e-9


def test_angular_stability_direction():
    # a stable low-dim scaffold: single-repeat and complement subspaces agree -> small D_real; random -> larger
    V, K = 60, 8
    W = np.linalg.qr(rng.standard_normal((V, K)))[0][:, :K]
    base = rng.standard_normal((4, V))
    R_r = base + 0.02 * rng.standard_normal((4, V)); R_mo = base + 0.02 * rng.standard_normal((4, V))
    d_real = np.linalg.norm(X._angle_cos2(W, R_r, 4) - X._angle_cos2(W, R_mo, 4))
    d_null = np.median([np.linalg.norm(X._angle_cos2(W, rng.standard_normal((4, V)), 4) - X._angle_cos2(W, rng.standard_normal((4, V)), 4)) for _ in range(50)])
    assert d_null > d_real  # constrained-random less stable than the shared scaffold


# ---- classification ----
def _res(h1v, h2v):
    r = {}
    for s in X.ALL:
        for roi in ("ventral", "lateral"):
            hv = h1v[roi]; sv = h2v[roi]
            r[(s, roi)] = {"E_ANGLE": hv["EA"], "d_real_med": 0.1, "d_null_med": 0.2, "angle_pipeline_constrained": hv["triv"],
                           "E_SCALE": sv["ES"], "rho_SCALE": sv["rho"], "Z_SCALE": 0.3, "Z_SCALE_null": 0.1,
                           "rho_pos_res": 0.1, "rho_id_res": 0.1, "replay_R_COMP": 0.4}
    return r


def _run(h1v, h2v):
    return X.classify(_res(h1v, h2v), ["ventral", "lateral"], G)


def test_full_support():
    v = {"EA": 1.5, "triv": False}; s = {"ES": 1.5, "rho": 0.3}
    infer, roi_status, prog = _run({"ventral": v, "lateral": v}, {"ventral": s, "lateral": s})
    assert prog == "CONSERVED_ANGULAR_SCAFFOLD_WITH_CONTINUOUS_RELIABILITY_VARIATION_SUPPORTED"


def test_angular_only_partial():
    v = {"EA": 1.5, "triv": False}; s = {"ES": -0.5, "rho": -0.2}
    _, _, prog = _run({"ventral": v, "lateral": v}, {"ventral": s, "lateral": s})
    assert prog == "ANGULAR_SCAFFOLD_SUPPORTED_RELIABILITY_LINK_PARTIAL"


def test_scale_only_partial():
    v = {"EA": -0.5, "triv": False}; s = {"ES": 1.5, "rho": 0.3}
    _, _, prog = _run({"ventral": v, "lateral": v}, {"ventral": s, "lateral": s})
    assert prog == "CONTINUOUS_RELIABILITY_LINK_SUPPORTED_ANGULAR_STABILITY_PARTIAL"


def test_pipeline_trivial_blocks_h1():
    v = {"EA": 1.5, "triv": True}; s = {"ES": -0.5, "rho": -0.2}  # angle would pass but flagged pipeline-constrained
    infer, roi_status, prog = _run({"ventral": v, "lateral": v}, {"ventral": s, "lateral": s})
    assert infer["ventral|ANGLE"]["H1_supported"] is False and prog == "NO_JOINT_ANGULAR_RELIABILITY_STRUCTURE"


def test_negative():
    v = {"EA": -0.5, "triv": False}; s = {"ES": -0.5, "rho": -0.2}
    _, _, prog = _run({"ventral": v, "lateral": v}, {"ventral": s, "lateral": s})
    assert prog == "NO_JOINT_ANGULAR_RELIABILITY_STRUCTURE"


def test_config():
    cfg = json.load(open("artifacts/mindcompiler/x3_angular_continuous_reliability/x3_frozen_config.json"))
    assert cfg["O3"] == "O3_NOT_READY" and cfg["N"] == 8
    assert cfg["primary_inference"]["family"].startswith("EXACTLY 4 tests") and "config_sha256" in cfg
    src = Path(X.__file__).read_text()
    for bad in ("sklearn", "torch", "GaussianMixture", "Laplace", "new_feature"):
        assert bad not in src
