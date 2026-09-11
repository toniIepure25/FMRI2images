"""O2.3A-RD Phase-2 pre-outcome implementation-correction certification (data-free).

Certifies the seven static-review fixes that align the driver to the already-frozen contracts
(configs unchanged): target-specific K, CV-clean K-selection scaling, correct RD + O2.4R oracle
recovery, derangement null, cache provenance binding, and Stage-A/Stage-B access ordering.
No methodology, thresholds, or status rules are changed here."""
from __future__ import annotations

import json

import numpy as np
import pytest

from fmri2img.mindcompiler.operator_o2_3a_rd import cohort as CH
from fmri2img.mindcompiler.operator_o2_3a_rd import frontier as F
from fmri2img.mindcompiler.operator_o2_3a_rd import geometry as G

rng = np.random.default_rng(101)


# --- FIX 1: target-specific K ------------------------------------------------
def test_target_specific_K_can_differ():
    """Two targets with different expressed latent dimensionality must be able to select different K.
    Common latent rank 8; target A expresses only 2 shared dims, target B expresses all 8."""
    n, K_true, p = 80, 8, 120
    S = rng.standard_normal((K_true, n))
    block_id = np.arange(n) % 5
    C = {}
    for s in CH.ALL:
        W = np.linalg.qr(rng.standard_normal((p, K_true)))[0][:, :K_true]
        if s == "subj01":                                 # target A: only first 2 shared dims expressed
            W[:, 2:] = 0.0
        C[s] = (W @ S).T + 0.01 * rng.standard_normal((n, p))
    K_A, sA = CH.select_K_for_target("subj01", C, block_id, G)   # low-rank target
    K_B, sB = CH.select_K_for_target("subj02", C, block_id, G)   # full-rank target
    assert K_A < K_B                                       # target-specific: distinct optimal K


# --- FIX 2: CV-clean K-selection scaler --------------------------------------
def test_scaler_leakage_train_only():
    """Perturbing ONLY held-out (validation) block rows must not change the fitted TRAIN-block scaler."""
    n, p = 50, 30
    C = rng.standard_normal((n, p))
    block_id = np.arange(n) % 5
    tr = block_id != 2
    mu0, sd0 = CH._scaler_train(C, tr)
    C2 = C.copy()
    C2[block_id == 2] += 999.0                             # corrupt only the validation block
    mu1, sd1 = CH._scaler_train(C2, tr)
    assert np.array_equal(mu0, mu1) and np.array_equal(sd0, sd1)


# --- FIX 3: RD oracle recovery = R_ZERO / max(R_ORACLE, eps) ------------------
def test_rd_oracle_recovery_exact_historical_value():
    predicted = 0.1444999174837383
    oracle = 0.2308295784225333
    assert abs(CH.rd_oracle_recovery_fold(predicted, oracle) - 0.6260026053473587) < 1e-12
    # never raw oracle
    assert CH.rd_oracle_recovery_fold(0.5, 0.5) == pytest.approx(1.0)


# --- FIX 4: O2.4R calibration oracle recovery fraction ------------------------
def test_o2_4r_oracle_recovery_fraction():
    # R_0=0.10, R_M=0.30, R_oracle=0.50 -> (0.30-0.10)/(0.50-0.10)=0.5
    unclip, clip = CH.o2_4r_oracle_recovery(0.30, 0.10, 0.50)
    assert abs(unclip - 0.5) < 1e-12 and abs(clip - 0.5) < 1e-12
    # over-recovery clips to 1; negative clips to 0
    assert CH.o2_4r_oracle_recovery(0.60, 0.10, 0.50)[1] == 1.0
    assert CH.o2_4r_oracle_recovery(0.05, 0.10, 0.50)[1] == 0.0


# --- FIX 5: derangement null (zero fixed points, deterministic) ---------------
@pytest.mark.parametrize("M", [2, 4, 6, 8, 10])
def test_null_derangement_no_fixed_points(M):
    for it in range(20):
        p = F._null_perm("O2.4R|subj01|ventral|0|%d|3|%d" % (M, it), M)
        assert len(p) == M
        assert not np.any(p == np.arange(M))              # no calibration identity paired to itself


def test_null_derangement_deterministic_and_seed_specific():
    a = F._null_perm("O2.4R|subj01|ventral|0|4|3|7", 4)
    b = F._null_perm("O2.4R|subj01|ventral|0|4|3|7", 4)
    c = F._null_perm("O2.4R|subj01|ventral|0|4|3|8", 4)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_null_M2_is_the_unique_swap():
    p = F._null_perm("O2.4R|subj03|lateral|1|2|0|5", 2)
    assert np.array_equal(p, np.array([1, 0]))


# --- FIX 6: cache provenance binding -----------------------------------------
def test_cache_provenance_hash_binds_all_fields():
    base = {"anchor_manifest_sha256": "a", "expdesign_sha256": "b", "config_rd": "c1b2ddb0",
            "config_r4": "ad959446", "code_version": "v1", "n_anchor": 512, "subject": "subj01",
            "union_voxel_hash": "u", "anchor_id_ordered_hash": "o",
            "session_list": list(range(1, 31)), "scaling": "PSC_div300", "beta_lineage": "B0_b2"}
    h0 = CH._prov_hash(base)
    for field, newval in [("anchor_manifest_sha256", "A"), ("expdesign_sha256", "B"),
                          ("union_voxel_hash", "U"), ("session_list", list(range(1, 30))),
                          ("code_version", "v2"), ("anchor_id_ordered_hash", "O")]:
        d = dict(base); d[field] = newval
        assert CH._prov_hash(d) != h0                     # any provenance change invalidates reuse


# --- FIX 7 / access order: M>0 refused without RD seal + valid M0 cert --------
def test_access_order_o2_4r_requires_seal_and_cert(tmp_path):
    out = tmp_path
    (out / "state").mkdir()
    (out / "rd_results.json").write_text(json.dumps({"per_target": {}}))
    # (a) no artifacts -> refuse
    with pytest.raises(SystemExit) as e1:
        CH.run_o2_4r(CH.Path("."), out, ["ventral", "lateral"])
    assert e1.value.code == 3
    # (b) seal present but M0 cert invalid -> refuse
    (out / "RD_SEAL.json").write_text(json.dumps({"artifact": "RD_SEAL", "status": "X"}))
    (out / "M0_CERT.json").write_text(json.dumps({"artifact": "M0_CERT", "ok": False}))
    with pytest.raises(SystemExit) as e2:
        CH.run_o2_4r(CH.Path("."), out, ["ventral", "lateral"])
    assert e2.value.code == 3


def test_certify_m0_requires_rd_seal(tmp_path):
    out = tmp_path
    (out / "state").mkdir()
    assert CH.certify_m0(CH.Path("."), out, ["ventral"]) == 3   # RD_SEAL.json missing
