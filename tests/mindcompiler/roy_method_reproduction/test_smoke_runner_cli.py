"""CLI-contract and trial-mapping tests for the Roy S1 smoke runner.

Data-free where possible. The real-data replay itself runs only in Lane E with
authorized B0 access; these tests guard against MISLABELED artifacts and verify
the trial-mapping invariants on synthetic TSV fixtures.
"""
import json, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[3]
RUNNER = REPO / "scripts/analysis/mindcompiler/run_roy_s1_smoke.py"


@pytest.mark.parametrize("args", [
    ["--subject", "subj02"],
    ["--roi", "V2"],
    ["--beta-version", "GLMdenoise_RR"],
    ["--denoising", "D1"],
])
def test_unsupported_scope_rejected_and_leaves_no_artifacts(tmp_path, args):
    out = tmp_path / "should_not_exist"
    r = subprocess.run([sys.executable, str(RUNNER), "--output-dir", str(out), *args],
                       capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode != 0, f"expected nonzero exit for {args}"
    assert "unsupported configuration" in r.stderr.lower()
    # no misleading result artifact
    assert not (out / "smoke_result.json").exists()


def test_run_boundaries_sum_to_720_and_selected_ranges_exact():
    sys.path.insert(0, str(REPO / "src"))
    from fmri2img.mindcompiler.roy_method_reproduction import smoke_pipeline as sp
    b = sp.run_beta_boundaries()
    assert sum(e - s for s, e in b.values()) == 720
    assert b["visA"] == (0, 48) and b["visB"] == (192, 240)
    assert b["imgA_2"] == (576, 624) and b["imgB_2"] == (624, 672)


def _synth_tsv(dirpath, run, conditions):
    rows = []
    for rep in range(8):
        for k, cond in enumerate(conditions):
            rows.append(dict(SUBJECT=1, SESSION=1, RUN=1, TRIAL=rep * 6 + k + 1,
                             CONDITION=cond, CUE=cond,
                             FRAMEFILE=f"{cond}_{rep}.png", IMAGEFILE=np.nan,
                             GROUNDTRUTH=1, ISCORRECT=1, CHANGEMIND=0, TOTAL1=1, TOTAL2=0,
                             BUTTON=1, TRIALONSET=float(rep * 6 + k), TRIALEND=0.0,
                             IMAGEONSET=np.nan, IMAGESERIESENDS=np.nan,
                             FINALDECISIONTIME=0.0, RT=0.0))
    pd.DataFrame(rows).to_csv(dirpath / f"nsdimagery_subj01_{run}.tsv", sep="\t", index=False)


def test_trial_table_invariants_on_synthetic_tsvs(tmp_path):
    sys.path.insert(0, str(REPO / "src"))
    from fmri2img.mindcompiler.roy_method_reproduction import smoke_pipeline as sp
    simple = list("LVHRUD")       # 6 set-A conditions
    natural = ["n1", "n2", "n3", "n4", "n5", "n6"]
    _synth_tsv(tmp_path, "visA", simple);   _synth_tsv(tmp_path, "imgA_2", simple)
    _synth_tsv(tmp_path, "visB", natural);  _synth_tsv(tmp_path, "imgB_2", natural)
    tt = sp.build_trial_table(str(tmp_path))  # asserts internally
    assert len(tt) == 192
    assert (tt.state == "vision").sum() == 96 and (tt.state == "imagery").sum() == 96
    assert tt["identity"].nunique() == 12
    assert tt["beta_index0"].is_unique
    # cue variation must not create identities: same CONDITION -> one identity across states
    for cond in simple:
        ids = tt[tt.condition == cond]["identity"].unique()
        assert len(ids) == 1


def test_trial_table_is_deterministic(tmp_path):
    sys.path.insert(0, str(REPO / "src"))
    from fmri2img.mindcompiler.roy_method_reproduction import smoke_pipeline as sp
    for run, conds in [("visA", list("LVHRUD")), ("imgA_2", list("LVHRUD")),
                       ("visB", [f"n{i}" for i in range(6)]), ("imgB_2", [f"n{i}" for i in range(6)])]:
        _synth_tsv(tmp_path, run, conds)
    a = sp.build_trial_table(str(tmp_path)); b = sp.build_trial_table(str(tmp_path))
    pd.testing.assert_frame_equal(a, b)


def test_derangement_pairing_has_no_self_pairs():
    sys.path.insert(0, str(REPO / "src"))
    from fmri2img.mindcompiler.roy_method_reproduction import smoke_pipeline as sp
    vsplit = {"A:L": dict(train=np.array([0, 1, 2, 3]), val=np.array([4, 5]), test=np.array([6, 7])), "_tt": None}
    X, Y = sp.vis2vis_pairs(vsplit, "train", "derangement", seed=1234)
    assert len(X) == 4 and not np.any(X == Y)  # 4 train, no self
    Xp, Yp = sp.vis2vis_pairs(vsplit, "train", "all-ordered-distinct", seed=1234)
    assert len(Xp) == 12  # 4x3 ordered distinct


def test_runner_fit_eval_routes_through_fitted_pipeline():
    """The single execution path must go through the leakage-safe pipeline.

    Proof: fit_eval returns the metrics-module constant-voxel diagnostics and the
    sealed policy name, which only the FittedPipeline path produces.
    """
    import numpy as np
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "roy_runner", REPO / "scripts/analysis/mindcompiler/run_roy_s1_smoke.py")
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    rng = np.random.default_rng(0)
    M = rng.standard_normal((40, 6))
    tr = (np.arange(0, 16), np.arange(0, 16))
    va = (np.arange(16, 24), np.arange(16, 24))
    te = (np.arange(24, 32), np.arange(24, 32))
    d = runner.fit_eval(M, tr[0], tr[1], va[0], va[1], te[0], te[1])
    assert "target_constant" in d and "prediction_constant" in d and "both_constant" in d
    assert d["policy_name"] == "HISTORICAL_SMOKE_TRAIN_ONLY_CENTERING_V1"
