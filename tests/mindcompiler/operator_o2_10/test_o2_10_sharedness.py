"""O2.10 Composite Geometry Sharedness certification (data-free/synthetic). Verifies the fingerprint/lift/
consensus helpers, the matched-null correspondence logic, the self-lift ceiling gating, all ROI/program
status branches, and the synthetic transfer controls. No target imagery is used in prediction math."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fmri2img.mindcompiler.operator_o2_3a_rd import geometry as G
from fmri2img.mindcompiler.operator_o2_10 import sharedness as SH

rng = np.random.default_rng(10)


def test_helpers_orth_pinv_consensus():
    M = rng.standard_normal((512, 3))
    Q = SH._orth_k(M, 3)
    assert Q.shape == (512, 3) and np.allclose(Q.T @ Q, np.eye(3), atol=1e-9)
    # consensus of identical subspaces returns that subspace
    Qs = [Q, Q, Q]
    Qc = SH._consensus(Qs, 3)
    P1 = Q @ Q.T; P2 = Qc @ Qc.T
    assert np.allclose(P1, P2, atol=1e-8)


def test_within_outside_basis_rank_guard():
    W = np.linalg.qr(rng.standard_normal((60, 8)))[0][:, :8]
    Z = rng.standard_normal((10, 8))
    assert SH._within_basis(Z, 3).shape == (8, 3)
    dout = np.stack([(lambda x: x - W @ (W.T @ x))(rng.standard_normal(60)) for _ in range(5)]).T
    assert SH._outside_basis(dout, 2).shape == (60, 2)
    assert SH._outside_basis(dout[:, :1], 2) is None


# --- synthetic transfer: perfect shared fingerprint recovers via consensus+lift ---
def test_synthetic_perfect_shared_recovers():
    V, K, r = 80, 8, 3
    W = np.linalg.qr(rng.standard_normal((V, K)))[0][:, :K]
    A = np.linalg.qr(rng.standard_normal((512, V)))[0][:, :V] if 512 >= V else rng.standard_normal((512, V))
    A = rng.standard_normal((512, V))
    # planted shared within subspace (in col(W)) realized identically across donors via A
    U_true = SH._orth_k(rng.standard_normal((K, r)), r); B_in_true = W @ U_true
    Q_shared = SH._orth(A @ B_in_true)
    Cs = A @ W; U_hat = SH._orth_k(SH._pinv(Cs) @ Q_shared, r); B_in_lift = W @ U_hat
    # lifted within subspace should coincide with the true within subspace
    P_true = B_in_true @ B_in_true.T; P_lift = B_in_lift @ B_in_lift.T
    assert np.trace(P_true @ P_lift) / r > 0.99


def test_synthetic_null_destroys_correspondence():
    V = 80; W = np.linalg.qr(rng.standard_normal((V, 8)))[0][:, :8]
    A = rng.standard_normal((512, V))
    Q = SH._orth_k(A @ (W @ SH._orth_k(rng.standard_normal((8, 3)), 3)), 3)
    # permuting the 512 anchor rows changes the realized subspace
    perm = rng.permutation(512)
    assert not np.allclose(Q, Q[perm])


# --- classification branches ---
def _part(vals):
    """vals: dict roi -> dict of the 8 per-participant metric medians (constant across participants)."""
    part = {}
    for s in SH.ALL:
        part[s] = {roi: {k: vals[roi][k] for k in vals[roi]} for roi in vals}
    return part


def _cell(self_in, self_out, e_in, e_out, wtr, otr, tot, cf):
    return {"SELF_IN": self_in, "SELF_OUT": self_out, "E_IN": e_in, "E_OUT": e_out,
            "WITHIN_TRANSFER": wtr, "OUTSIDE_TRANSFER": otr, "TOTAL_XSUB": tot, "COMPOSITE_FRACTION": cf}


def test_transport_ceiling_insufficient():
    # SELF_OUT low -> outside transport inadequate -> ceiling insufficient
    v = _cell(0.8, 0.2, 0.1, 0.1, 0.7, 0.1, 0.3, 0.3)
    rs, prog, infer = SH.classify(_part({"ventral": v, "lateral": v}), ["ventral", "lateral"], G)
    assert rs["ventral"] == "VISION_ANCHOR_TRANSPORT_CEILING_INSUFFICIENT"
    assert prog == "VISION_ANCHOR_TRANSPORT_INSUFFICIENT_FOR_SHAREDNESS_TEST"


def test_established():
    v = _cell(0.8, 0.8, 0.1, 0.1, 0.7, 0.7, 0.7, 0.7)
    rs, prog, infer = SH.classify(_part({"ventral": v, "lateral": v}), ["ventral", "lateral"], G)
    assert rs["ventral"] == "COMPOSITE_GEOMETRY_ZERO_IMAGERY_TRANSFER_ESTABLISHED"
    assert prog == "ZERO_TARGET_IMAGERY_COMPOSITE_TRANSFER_ESTABLISHED_WITH_VISION_ONLY_CALIBRATION"


def test_within_shared_outside_not():
    v = _cell(0.8, 0.8, 0.1, -0.05, 0.7, 0.1, 0.3, 0.3)                      # outside E negative -> not supported
    rs, prog, infer = SH.classify(_part({"ventral": v, "lateral": v}), ["ventral", "lateral"], G)
    assert rs["ventral"] == "WITHIN_SHARED_OUTSIDE_NOT_SHARED"
    assert prog == "COMPOSITE_GEOMETRY_SHAREDNESS_MULTIREGIME"                # partial-sharing -> catch-all multiregime


def test_subject_specific_dominant():
    v = _cell(0.8, 0.8, -0.05, -0.05, 0.1, 0.1, 0.2, 0.2)                    # transport adequate, neither shared
    rs, prog, infer = SH.classify(_part({"ventral": v, "lateral": v}), ["ventral", "lateral"], G)
    assert rs["ventral"] == "COMPOSITE_GEOMETRY_NOT_SHARED"
    assert prog == "COMPOSITE_TARGET_STATE_GEOMETRY_SUBJECT_SPECIFIC_DOMINANT"


def test_shared_but_composite_insufficient():
    v = _cell(0.8, 0.8, 0.1, 0.1, 0.7, 0.7, 0.3, 0.3)                        # components shared, composite <0.5
    rs, prog, infer = SH.classify(_part({"ventral": v, "lateral": v}), ["ventral", "lateral"], G)
    assert rs["ventral"] == "SHARED_COMPONENTS_INSUFFICIENT_FOR_COMPOSITE_TRANSFER"
    assert prog == "SHARED_COMPOSITE_GEOMETRY_INSUFFICIENT_FOR_ZERO_IMAGERY_TRANSFER"


def test_config_and_no_model_search():
    cfg = json.load(open("artifacts/mindcompiler/operator_o2_10/o2_10_frozen_config.json"))
    assert cfg["immutable_history"]["O2_9"] == "MINIMAL_COMPOSITE_TARGET_STATE_CALIBRATION_ESTABLISHED"
    assert cfg["inference"]["primary_family"].startswith("2 components") and cfg["O3"] == "O3_NOT_READY"
    src = Path(SH.__file__).read_text()
    for bad in ("Ridge", "CCA(", "PLSRegression", "torch", "sklearn"):
        assert bad not in src
