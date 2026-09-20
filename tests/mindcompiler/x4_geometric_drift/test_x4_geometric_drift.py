"""MINDIR-X4 certification (data-free/synthetic). Verifies the geometric primitives (orthonormal span,
top-direction / top-fraction, basis-rotation invariance of the Gram eigenstructure), the anisotropy contrast
(planted rank-1 drift beats isotropic noise), the mode<->distance monotonicity, the exact 4-test Holm
classification branches (full / H1-only / H2-only / negative / multiregime), the reproducibility-gate block,
the disclosed CENTRED-inference integrity resolution (a ratio sitting at ~1 under H0 cannot pass), and config
immutability. Folds/schedules/repeats/subsets/dimensions are NEVER the inferential N (participant N=8)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fmri2img.mindcompiler.x4_geometric_drift import geometric_drift as X
import fmri2img.mindcompiler.operator_o2_3a_rd.geometry as G

rng = np.random.default_rng(4)
X._G = G
ROIS = ["ventral", "lateral"]


# ---------------- geometric primitives ----------------
def test_orth_is_orthonormal():
    M = rng.standard_normal((40, 6))
    U = X._orth(M, 3)
    assert U.shape == (40, 3)
    assert np.max(np.abs(U.T @ U - np.eye(3))) < 1e-9


def test_spearman_monotone():
    x = np.array([1.0, 2, 3, 4, 5, 6])
    assert abs(X._spearman(x, 3 * x + 1) - 1) < 1e-9
    assert abs(X._spearman(x, -x) + 1) < 1e-9


def test_top_dir_planted_beats_isotropic():
    # planted rank-1 anisotropic cloud -> top-fraction near 1; isotropic -> small
    n, d = 160, 40
    direction = rng.standard_normal(d); direction /= np.linalg.norm(direction)
    scores = rng.standard_normal((n, 1))
    planted = scores @ direction[None, :] + 0.02 * rng.standard_normal((n, d))
    _, tf_planted, er_planted = X._top_dir(planted)
    _, tf_iso, er_iso = X._top_dir(rng.standard_normal((n, d)))
    assert tf_planted > 0.9 and tf_iso < 0.2
    assert er_planted < 1.5 and er_iso > 5.0  # participation ratio ~1 vs ~d


def test_basis_rotation_invariance():
    # the Gram eigenstructure (top-fraction + participation ratio) is invariant to an orthogonal basis change
    n, d = 120, 30
    Xc = rng.standard_normal((n, 5)) @ rng.standard_normal((5, d)) + 0.1 * rng.standard_normal((n, d))
    Q = np.linalg.qr(rng.standard_normal((d, d)))[0]
    _, tf0, er0 = X._top_dir(Xc)
    _, tf1, er1 = X._top_dir(Xc @ Q)
    assert abs(tf0 - tf1) < 1e-9 and abs(er0 - er1) < 1e-9


def test_mode_distance_monotone():
    # along a single dominant direction, |a_r1-a_r2| tracks the full displacement distance
    d = 20
    v = rng.standard_normal(d); v /= np.linalg.norm(v)
    coords = np.linspace(-2, 2, 8)
    Xr = coords[:, None] * v[None, :] + 0.01 * rng.standard_normal((8, d))
    gaps, dfull = [], []
    for i in range(8):
        for j in range(i + 1, 8):
            gaps.append(abs(coords[i] - coords[j])); dfull.append(np.linalg.norm(Xr[i] - Xr[j]))
    assert X._spearman(np.array(gaps), np.array(dfull)) > 0.95


# ---------------- classification ----------------
def _res(h1, h2):
    """h1[roi]=(aniso_centred, repro_pass_bool); h2[roi]=(Z_MODE, rho_FAILURE)."""
    r = {}
    for s in X.ALL:
        for roi in ROIS:
            ac, repro = h1[roi]; zm, rf = h2[roi]
            r[(s, roi)] = {"ANISO_CENTRED": ac, "E_ANISO": 1.0 + ac, "eff_rank_med": 1.2, "align_pass_folds": 6 if repro else 2,
                           "repro_pass": repro, "Z_MODE": zm, "Z_MODE_null": 0.0, "E_MODE": float("nan"),
                           "MODE_CENTRED": zm, "rho_FAILURE": rf, "dmode_dfull_med": 0.9, "repeat_pos_profile": [0.0] * 8,
                           "replay_R_COMP": 0.4}
    return r


def _run(h1, h2):
    return X.classify(_res(h1, h2), ROIS)


def test_full_support():
    _, _, prog = _run({"ventral": (0.1, True), "lateral": (0.1, True)}, {"ventral": (0.3, 0.3), "lateral": (0.3, 0.3)})
    assert prog == "LOW_DIMENSIONAL_CONTINUOUS_GEOMETRIC_DRIFT_SUPPORTED"


def test_h1_only_partial():
    _, _, prog = _run({"ventral": (0.1, True), "lateral": (0.1, True)}, {"ventral": (-0.3, -0.3), "lateral": (-0.3, -0.3)})
    assert prog == "GEOMETRIC_DRIFT_EXISTS_BUT_FAILURE_LINK_PARTIAL"


def test_h2_only_partial():
    _, _, prog = _run({"ventral": (-0.1, True), "lateral": (-0.1, True)}, {"ventral": (0.3, 0.3), "lateral": (0.3, 0.3)})
    assert prog == "FAILURE_PREDICTIVE_MODE_WITHOUT_REPRODUCIBLE_GLOBAL_ANISOTROPY"


def test_negative():
    _, _, prog = _run({"ventral": (-0.1, True), "lateral": (-0.1, True)}, {"ventral": (-0.3, -0.3), "lateral": (-0.3, -0.3)})
    assert prog == "NO_REPRODUCIBLE_LOW_DIMENSIONAL_GEOMETRIC_DRIFT"


def test_multiregime():
    _, _, prog = _run({"ventral": (0.1, True), "lateral": (-0.1, True)}, {"ventral": (0.3, 0.3), "lateral": (-0.3, -0.3)})
    assert prog == "GEOMETRIC_DRIFT_MULTIREGIME"


def test_reproducibility_gate_blocks_h1():
    # anisotropy positive + Holm-significant but cross-half reproducibility fails -> H1 cannot pass
    infer, roi_status, prog = _run({"ventral": (0.1, False), "lateral": (0.1, False)}, {"ventral": (-0.3, -0.3), "lateral": (-0.3, -0.3)})
    assert infer["ventral|ANISO"]["H1_supported"] is False and prog == "NO_REPRODUCIBLE_LOW_DIMENSIONAL_GEOMETRIC_DRIFT"


def test_ratio_at_null_cannot_pass():
    # INTEGRITY: E_ANISO ratio = 1.0 (>0) but the CENTRED statistic is exactly 0 -> must NOT be counted as support
    infer, roi_status, prog = _run({"ventral": (0.0, True), "lateral": (0.0, True)}, {"ventral": (0.0, 0.0), "lateral": (0.0, 0.0)})
    assert infer["ventral|ANISO"]["median_E_ANISO_ratio"] == 1.0
    assert infer["ventral|ANISO"]["H1_supported"] is False
    assert prog == "NO_REPRODUCIBLE_LOW_DIMENSIONAL_GEOMETRIC_DRIFT"


def test_holm_exactly_four_tests():
    infer, _, _ = _run({"ventral": (0.1, True), "lateral": (0.1, True)}, {"ventral": (0.3, 0.3), "lateral": (0.3, 0.3)})
    keys = set(infer.keys())
    assert keys == {"ventral|ANISO", "lateral|ANISO", "ventral|MODE", "lateral|MODE"}


# ---------------- config immutability + no forbidden models ----------------
def test_config_frozen():
    cfg = json.load(open("artifacts/mindcompiler/x4_geometric_drift/x4_frozen_config.json"))
    assert cfg["O3"] == "O3_NOT_READY" and cfg["N"] == 8 and cfg["unit"] == "participant"
    assert cfg["primary_family"]["exactly_four"] == ["ventral E_ANISO", "lateral E_ANISO", "ventral E_MODE", "lateral E_MODE"]
    assert cfg["final_mechanistic_gate_on_N8"] is True and "config_sha256" in cfg
    assert cfg["immutable"]["X3"] == "NO_JOINT_ANGULAR_RELIABILITY_STRUCTURE"


def test_no_forbidden_models_in_source():
    src = Path(X.__file__).read_text()
    for bad in ("sklearn", "import torch", "GaussianMixture", "umap", "TSNE", "KMeans"):
        assert bad not in src
    assert "signflip_p_onesided" in src and "CENTRED" in src  # centred-inference resolution present
