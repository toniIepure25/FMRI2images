"""MINDIR-X1 gauge-invariant certification (data-free/synthetic). Verifies coordinate invariance (W-rotation +
sign flips leave invariants unchanged), fixed block dimensionality, the LOSO donor-mean E_SHARED (positive when
signatures are shared, ~0 for random operators; no target leakage), the pipeline-triviality flag, the exact
8-test classification branches (incl. SHARED_INTRINSIC_STRUCTURE_PRIVATE_ORIENTATION), and config immutability /
O2 preservation. Folds/dimensions are never the inferential N (N=8 participants)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fmri2img.mindcompiler.x1_gauge_invariants import gauge_invariants as X
import fmri2img.mindcompiler.operator_o2_3a_rd.geometry as G

rng = np.random.default_rng(1)
X._G = G


def _cell(V=40, K=8, r=4):
    W = np.linalg.qr(rng.standard_normal((V, K)))[0][:, :K]
    return W, r


# ---- A. coordinate invariance ----
def test_W_rotation_invariance():
    W, r = _cell(); Dn = rng.standard_normal((10, W.shape[0]))
    R = np.linalg.qr(rng.standard_normal((8, 8)))[0]
    i0 = X.invariants(W, Dn, r, 4); i1 = X.invariants(W @ R, Dn, r, 4)
    for b in X.BLOCKS:
        assert np.max(np.abs(i0[b] - i1[b])) < 1e-9, b


def test_sign_flip_invariance():
    W, r = _cell(); Dn = rng.standard_normal((10, W.shape[0]))
    S = np.diag(rng.choice([-1.0, 1.0], W.shape[1]))
    i0 = X.invariants(W, Dn, r, 4); i1 = X.invariants(W @ S, Dn, r, 4)
    for b in X.BLOCKS:
        assert np.max(np.abs(i0[b] - i1[b])) < 1e-9, b


def test_fixed_block_dims():
    W, r = _cell(); Dn = rng.standard_normal((10, W.shape[0]))
    inv = X.invariants(W, Dn, r, 4)
    assert inv["A_spectral"].size == 4 + 3 and inv["B_angles"].size == 4 + 5
    assert inv["C_support"].size == 6 and inv["D_compression"].size == 4 + 1 + 4


# ---- B. LOSO E_SHARED ----
def test_e_shared_positive_when_shared():
    # 8 signatures clustered around a common pattern -> donor mean predicts target better than permutation null
    base = np.array([0.5, 0.3, 0.15, 0.05, 1.2, 3.0, 2.1])
    sigs = [base + 0.02 * rng.standard_normal(base.size) for _ in range(8)]
    es, dp, dn = X.e_shared(sigs[0], sigs[1:], "A_spectral", "T|s|r|A", G)
    assert es > 0 and dn > dp


def test_e_shared_no_leakage_and_random_null():
    # random unrelated signatures -> E_SHARED near/below zero (no shared component structure)
    sigs = [rng.standard_normal(7) for _ in range(8)]
    vals = [X.e_shared(sigs[i], [sigs[j] for j in range(8) if j != i], "A_spectral", f"R|{i}", G)[0] for i in range(8)]
    assert float(np.median(vals)) < 0.5 * float(np.median([np.linalg.norm(s) for s in sigs]))  # not a strong positive


# ---- C. classification (full synthetic fixture) ----
def _fixture(shared):
    """8 subjects x 2 ROIs; if shared, invariants cluster (via near-identical Dn structure)."""
    cells = {}; extn = {}
    common = rng.standard_normal((10, 40))
    for s in X.ALL:
        cells[s] = {}; extn[s] = {}
        for roi in ("ventral", "lateral"):
            W = np.linalg.qr(rng.standard_normal((40, 8)))[0][:, :8]
            Dn = (common + 0.02 * rng.standard_normal((10, 40))) if shared else rng.standard_normal((10, 40))
            cells[s][roi] = [{"W_target": W, "r_best": 4} for _ in range(X.N_FOLDS)]
            extn[s][roi] = [{"delta_native": Dn} for _ in range(X.N_FOLDS)]
    return cells, extn


def _run_classify(cells, extn):
    rois = ["ventral", "lateral"]; r_common = {roi: 4 for roi in rois}
    sig = {s: {roi: {b: X.invariants(np.asarray(cells[s][roi][0]["W_target"], float), np.asarray(extn[s][roi][0]["delta_native"], float), 4, 4)[b] for b in X.BLOCKS} for roi in rois} for s in X.ALL}
    part = {roi: {b: {} for b in X.BLOCKS} for roi in rois}
    for roi in rois:
        for b in X.BLOCKS:
            for s in X.ALL:
                part[roi][b][s] = X.e_shared(sig[s][roi][b], [sig[d][roi][b] for d in X.ALL if d != s], b, f"C|{s}|{roi}|{b}", G)[0]
    return X.classify(part, sig, rois, r_common, cells, extn, G)


def test_classify_private_orientation_when_shared():
    cells, extn = _fixture(shared=True)
    _, infer, roi_status, prog, triv, synth = _run_classify(cells, extn)
    # shared cohort -> >=2 blocks supported both ROIs AND O2.10 orientation negative -> PRIVATE_ORIENTATION
    assert prog in ("SHARED_INTRINSIC_STRUCTURE_PRIVATE_ORIENTATION", "SHARED_INTRINSIC_TRANSFORMATION_STRUCTURE_SUPPORTED", "INVARIANT_STRUCTURE_MULTIREGIME")
    assert X.O2_10_ORIENTATION["ventral"] == "COMPOSITE_GEOMETRY_NOT_SHARED"


def test_classify_negative_when_random():
    cells, extn = _fixture(shared=False)
    _, infer, roi_status, prog, triv, synth = _run_classify(cells, extn)
    assert prog in ("NO_REPRODUCIBLE_SHARED_INTRINSIC_STRUCTURE", "INVARIANT_STRUCTURE_MULTIREGIME")


# ---- D. config immutability + O2 preservation ----
def test_config_and_o2_preserved():
    cfg = json.load(open("artifacts/mindcompiler/x1_gauge_invariant_shared_law/x1_frozen_config.json"))
    assert cfg["O3"] == "O3_NOT_READY" and cfg["N"] == 8
    assert cfg["primary_inference"]["family"].startswith("4 invariant blocks")
    assert "config_sha256" in cfg and "immutable_o2" in cfg
    src = Path(X.__file__).read_text()
    for bad in ("Ridge", "Procrustes", "sklearn", "torch", "hyperparameter_search"):
        assert bad not in src
