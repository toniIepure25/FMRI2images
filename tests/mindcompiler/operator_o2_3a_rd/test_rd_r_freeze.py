"""O2.3A-RD + O2.4R prospective-freeze tests (data-free, offline).

Assert both configs are frozen and self-consistent and that every numerically-determining detail the
PROVREC forensics found MISSING is now fully specified (DetSRM init/iter/convergence + canonical gauge;
K-block partition mod 5; rank inner-CV identity-pair folds; null RNG PCG64-from-SHA256; tie rules), that
budgets/subsets/inference/M_STAR match the frozen O2.4 family, that historical statuses are immutable and
NOT relabelled as reproduced, and that O3 stays locked. Compute results are honest PENDING placeholders.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RD = ROOT / "artifacts/mindcompiler/operator_o2_3a_rd"
R4 = ROOT / "artifacts/mindcompiler/operator_o2_4r"
PROV = ROOT / "artifacts/mindcompiler/operator_o2_4_provrec"


def _j(base, n):
    return json.loads((base / n).read_text())


def test_rd_config_sha_matches():
    c = _j(RD, "o2_3a_rd_frozen_config.json")
    stored = c.pop("config_sha256")
    assert hashlib.sha256(json.dumps(c, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == stored
    assert stored.startswith("c1b2ddb0")


def test_r4_config_sha_matches_and_refs_rd():
    c = _j(R4, "o2_4r_frozen_config.json")
    rd_ref = c["rd_config_sha256"]
    stored = c.pop("config_sha256")
    assert hashlib.sha256(json.dumps(c, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == stored
    assert stored.startswith("ad959446")
    assert rd_ref == _j(RD, "o2_3a_rd_frozen_config.json")["config_sha256"]  # O2.4R references the RD state


def test_srm_fully_specified_now():
    s = _j(RD, "o2_3a_rd_frozen_config.json")["deterministic_srm"]
    assert s["max_iter"] == 200 and s["convergence_tol"] == 1e-7 and s["dtype"] == "float64"
    assert s["initialization"].startswith("deterministic SVD")
    assert any("largest-absolute-loading" in g for g in s["gauge_convention"])


def test_kblocks_mod5_and_kcandidates_and_tie():
    c = _j(RD, "o2_3a_rd_frozen_config.json")
    assert c["K_candidates"] == [2, 4, 8, 16, 32, 64]
    assert "mod 5" in c["K_blocks"]["partition"]
    assert "1e-12" in c["K_selection"]["tie_rule"] and "smaller K" in c["K_selection"]["tie_rule"]


def test_rank_inner_cv_pairs_and_candidates():
    c = _j(RD, "o2_3a_rd_frozen_config.json")
    assert c["rank_candidates"] == [1, 2, 3, 4, 5, 6]
    assert "sorted-simple[j] + sorted-naturalistic[j]" in c["rank_inner_cv"]["inner_folds"]
    assert "smaller r" in c["rank_inner_cv"]["tie_rule"]


def test_null_rng_fully_specified():
    n = _j(RD, "o2_3a_rd_frozen_config.json")["null"]
    assert n["seed_text"].startswith("'O2.3A-RD|'")
    assert "first 16 hex -> uint64" in n["digest"]
    assert "PCG64" in n["rng"] and n["n_samples"] == 100
    r4n = _j(R4, "o2_4r_frozen_config.json")["null"]
    assert r4n["seed_text"].startswith("'O2.4R|'") and r4n["n_permutations"] == 100 and "PCG64" in r4n["digest"]


def test_r4_budgets_subsets_and_M0_reuse():
    c = _j(R4, "o2_4r_frozen_config.json")
    assert c["budgets"]["M"] == [0, 2, 4, 6, 8, 10]
    b = c["balanced_subsets"]
    assert (b["M2"], b["M4"], b["M6"], b["M8"], b["M10"]) == (25, 100, 100, 25, 1)
    assert c["M0"]["certify"] == "O2_4R_M0_EQUALS_O2_3A_RD at hash/numerical level"
    assert c["estimator"]["family"] == "ORTHOGONAL PROCRUSTES only"
    for bad in ["affine", "ridge", "CCA", "RRR", "nonlinear"]:
        assert bad in c["estimator"]["forbidden"]


def test_r4_inference_mstar_and_coherence_defined():
    c = _j(R4, "o2_4r_frozen_config.json")
    assert c["inference"]["primary_tests"] == 10 and "2^8" in c["inference"]["signflip"] and c["inference"]["alpha"] == 0.05
    assert c["M_STAR"]["if_none"] == "NOT_REACHED" and c["M_STAR"]["no_interpolation"] is True
    coh = c["status_family"]["coherently_positive_defined_now"]
    assert any(">=4/5 nonzero budgets" in x for x in coh)


def test_immutable_history_not_relabelled():
    for base, cfg in [(RD, "o2_3a_rd_frozen_config.json"), (R4, "o2_4r_frozen_config.json")]:
        h = _j(base, cfg)["immutable_history"]
        assert h["historical_O2_3A"] == "CORE_ANCHOR_TARGET_ORIENTATION_NOT_IDENTIFIABLE"
        assert h["historical_O2_4"] == "TARGET_STATE_ORIENTATION_CALIBRATION_INCONCLUSIVE"
        assert h["never_relabel_historical_o2_3a_as_reproduced"] is True
        assert h["O3"] == "O3_NOT_READY"
    # PROVREC determination untouched
    assert _j(PROV, "replay_status.json")["status"] == "O2_4_PROVREC_HISTORICAL_DETAIL_MISSING"


def test_compute_results_pending_not_fabricated():
    assert _j(RD, "scientific_status.json")["status"] == "PENDING_RD_COMPUTE"
    assert _j(R4, "scientific_status.json")["status"] == "PENDING_R4_COMPUTE"
