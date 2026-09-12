"""O2.7A post-seal audit helper checks (data-free). Verification-only; no scientific recompute."""
from __future__ import annotations

import itertools
import json

import numpy as np

from fmri2img.mindcompiler.operator_o2_7a_postaudit import audit as AU


def test_pairs_25():
    assert len(AU._pairs([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])) == 25


def test_outside_axis_props():
    W = np.linalg.qr(np.random.default_rng(1).standard_normal((40, 8)))[0][:, :8]
    b, s = AU._outside_axis(np.random.default_rng(2).standard_normal(40), np.random.default_rng(3).standard_normal(40), W, want_sv=True)
    assert abs(np.linalg.norm(b) - 1) < 1e-9 and float(abs(W.T @ b).max()) < 1e-10 and s[0] > 0


def test_35_complementary_balanced_splits():
    splits = list(itertools.combinations(range(8), 4)); seen = set(); uniq = []
    for A in splits:
        B = tuple(sorted(set(range(8)) - set(A)))
        key = frozenset([A, B])
        if key not in seen:
            seen.add(key); uniq.append((A, B))
    assert len(uniq) == 35


def test_config_immutable_and_nonscientific():
    cfg = json.load(open("artifacts/mindcompiler/operator_o2_7a_postaudit/postseal_audit_config.json"))
    assert cfg["immutable_o2_7a"]["status"] == "REDUCED_TRIAL_OUTSIDE_AXIS_CALIBRATION_NOT_ESTABLISHED"
    assert cfg["class"] == "POST_SEAL_NON_SCIENTIFIC_INTEGRITY_AUDIT" and cfg["O3"] == "O3_NOT_READY"


def test_no_inference_recompute():
    import pathlib
    src = pathlib.Path(AU.__file__).read_text()
    for bad in ("signflip_p_onesided", "holm(", "T_AXIS_STAR ="):        # audit must not recompute inference
        assert bad not in src
