"""S2.0 PART D+F -- reconstruction engine end-to-end (data-free)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fmri2img.mindcompiler.roy_method_reproduction import s2_folds as sf
from fmri2img.mindcompiler.roy_method_reproduction import s2_reconstruction as rc
from fmri2img.mindcompiler.roy_method_reproduction.s2_preprocessing import PREPROC_ZSCORE_TRAIN_ONLY


def _tt(n_id=4, n_rep=8, n_vox=12, seed=0):
    rows = []
    bi = 0
    for state in ("vision", "imagery"):
        for i in range(n_id):
            for rep in range(n_rep):
                rows.append(dict(identity=f"ID{i}", state=state, repeat=rep, beta_index0=bi))
                bi += 1
    tt = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    # give identities a shared latent so vis2vis/vis2img are non-degenerate
    latent = rng.standard_normal((n_id, n_vox))
    M = np.zeros((len(tt), n_vox))
    for r, row in tt.iterrows():
        i = int(row["identity"][2:])
        M[r] = latent[i] + 0.3 * rng.standard_normal(n_vox)
    return tt, M


def _run(policy="all_ordered_distinct", seed=None):
    tt, M = _tt()
    folds = sf.build_four_folds(tt, 1234)
    row_identity = {r: tt.loc[r, "identity"] for r in tt.index}
    row_beta = {r: int(tt.loc[r, "beta_index0"]) for r in tt.index}
    return rc.run_fold(M, folds["fold_0"], row_identity, row_beta,
                       beta_version="synthetic", preproc_policy=PREPROC_ZSCORE_TRAIN_ONLY,
                       vis2vis_pairing=policy, pairing_seed=seed)


def test_fold_runs_end_to_end_leakage_clean():
    res = _run()
    assert res.denoising == "D1_STRICT_CROSSFIT"
    assert all(v == 0 for v in res.denoise_leakage.values())      # no leakage
    assert res.n_denoise_records > 0


def test_ranks_within_caps():
    res = _run()
    assert 1 <= res.vis2vis["rank"] <= res.vis2vis["rank_max"] <= rc.RANK_HARD_MAX
    assert 1 <= res.vis2img["rank"] <= res.vis2img["rank_max"] <= rc.RANK_HARD_MAX
    # 4 identities -> condition bound caps rank at 4
    assert res.vis2vis["rank_max"] <= 4 and res.vis2img["rank_max"] <= 4


def test_metrics_present_and_finite_fraction_reported():
    res = _run()
    for k in ("mean", "median", "finite_frac", "target_constant",
              "prediction_constant", "p5", "p95"):
        assert k in res.vis2img
    assert 0.0 <= res.vis2img["finite_frac"] <= 1.0


def test_deterministic_under_fixed_seed():
    a = _run(policy="single_deterministic_derangement", seed=1234)
    b = _run(policy="single_deterministic_derangement", seed=1234)
    assert a.vis2img["lam"] == b.vis2img["lam"] and a.vis2img["rank"] == b.vis2img["rank"]
    assert a.vis2img["mean"] == b.vis2img["mean"]


def test_ridge_grid_is_paper_range():
    from fmri2img.mindcompiler.roy_method_reproduction.reduced_rank_ridge import ridge_grid
    g = ridge_grid()
    assert g.size == 100 and abs(g[0] - 1e-3) < 1e-9 and abs(g[-1] - 1e5) < 1e-3
