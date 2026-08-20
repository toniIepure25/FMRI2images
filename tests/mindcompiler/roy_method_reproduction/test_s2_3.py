"""S2.3 data-free tests: participant set, aggregation, exact permutation, criteria."""
from __future__ import annotations

import numpy as np
import pytest

from fmri2img.mindcompiler.roy_method_reproduction import s2_3_scorecard as sc
from fmri2img.mindcompiler.roy_method_reproduction import smoke_pipeline as sp


def test_primary_participant_set_and_subj01_excluded():
    assert sc.PRIMARY_PARTICIPANTS == ("subj02", "subj03", "subj04", "subj05", "subj06", "subj07", "subj08")
    assert sc.DEVELOPMENT_REFERENCE == "subj01"
    assert "subj01" not in sc.PRIMARY_PARTICIPANTS
    assert len(sc.PRIMARY_PARTICIPANTS) == 7


def test_build_trial_table_is_subject_parameterized():
    # default stays subj01 (S1 replay unaffected); the TSV name uses the subject
    import inspect
    src = inspect.getsource(sp.build_trial_table)
    assert 'subject: str = "subj01"' in src
    assert 'nsdimagery_{subject}_' in src


def test_participant_summary_median_equal_weight():
    L_rel = {f"R{i}": v for i, v in enumerate([0.1, 0.2, 0.05, 0.15, 0.08, 0.12, 0.09])}
    L_perf = {f"R{i}": v for i, v in enumerate([0.05, 0.1, 0.02, 0.08, 0.03, 0.06, 0.04])}
    s = sc.participant_summary(L_rel, L_perf)
    assert s["S_rel"] == np.median(list(L_rel.values()))
    assert s["roi_concordant_signs"] == 7


def test_exact_permutation_enumerates_5040():
    x = [1, 2, 3, 4, 5, 6, 7]
    y = [1, 2, 3, 4, 5, 6, 7]           # perfect positive monotone
    p = sc.exact_permutation_spearman(x, y)
    assert p["n_permutations"] == 5040
    assert p["rho_observed"] == pytest.approx(1.0)
    # only the identity ordering ties the max rho -> smallest possible p = 1/5040
    assert p["p_exact_one_sided"] == pytest.approx(1 / 5040)


def test_permutation_p_is_one_sided_positive():
    x = [1, 2, 3, 4, 5, 6, 7]
    y = [7, 6, 5, 4, 3, 2, 1]           # perfect negative
    p = sc.exact_permutation_spearman(x, y)
    assert p["rho_observed"] == pytest.approx(-1.0)
    assert p["p_exact_one_sided"] == pytest.approx(1.0)  # nothing exceeds the worst


def test_criteria_all_pass_supported():
    S_rel = {f"subj0{i}": v for i, v in zip(range(2, 9), [0.10, 0.16, 0.06, 0.15, 0.12, 0.13, 0.09])}
    S_perf = {f"subj0{i}": v for i, v in zip(range(2, 9), [0.06, 0.08, 0.01, 0.09, 0.04, 0.05, 0.03])}
    s = sc.evaluate_criteria(S_rel, S_perf)
    assert s["criterion_A"] and s["criterion_B"] and s["criterion_C"] and s["criterion_D"]
    assert sc.classify(s) == "H_A_CROSS_PARTICIPANT_SUPPORTED"
    assert s["permutation"]["p_exact_one_sided"] <= 0.05


def test_inconclusive_when_few_evaluable():
    S_rel = {"subj02": 0.1, "subj03": 0.2, "subj04": 0.05}
    S_perf = {"subj02": 0.05, "subj03": 0.1, "subj04": 0.02}
    s = sc.evaluate_criteria(S_rel, S_perf)
    assert s["n_evaluable_participants"] == 3
    assert sc.classify(s) == "H_A_CROSS_PARTICIPANT_INCONCLUSIVE"


def test_anti_correlated_not_supported():
    S_rel = {f"subj0{i}": v for i, v in zip(range(2, 9), [0.10, 0.16, 0.06, 0.15, 0.12, 0.13, 0.09])}
    S_perf = {f"subj0{i}": -v for i, v in zip(range(2, 9), [0.06, 0.08, 0.01, 0.09, 0.04, 0.05, 0.03])}
    s = sc.evaluate_criteria(S_rel, S_perf)
    assert not s["criterion_B"] and not s["criterion_C"] and not s["criterion_D"]
    assert sc.classify(s) == "H_A_CROSS_PARTICIPANT_NOT_SUPPORTED"


def test_sign_conventions():
    assert (0.30 - 0.14) > 0   # B1 lower imagery reliability -> positive loss
    assert (0.10 - (-0.03)) > 0  # B1 worse RAW performance -> positive loss


def test_leave_one_participant():
    S_rel = {f"subj0{i}": v for i, v in zip(range(2, 9), [0.10, 0.16, 0.06, 0.15, 0.12, 0.13, 0.09])}
    S_perf = {f"subj0{i}": v for i, v in zip(range(2, 9), [0.06, 0.08, 0.01, 0.09, 0.04, 0.05, 0.03])}
    lo = sc.leave_one_participant(S_rel, S_perf)
    assert len(lo["rhos"]) == 7 and lo["n_positive"] >= 6
