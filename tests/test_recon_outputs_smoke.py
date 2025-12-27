import json
from pathlib import Path

import pandas as pd

from fmri2img.eval.run_reconstruct_and_eval import write_trial_results, run_calibration_assets
from fmri2img.inference.pipeline import TrialResult


def _dummy_trial(trial_id: int) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        stimulus_id=1000 + trial_id,
        split="test",
        subject="subjXX",
        uncertainty=0.1 * (trial_id + 1),
        embedding_error=None,
        nll=-1.0,
        mu_cosine=0.5,
        mu_mse=0.1,
        K_assigned=4,
        K_base=4,
        K_set=(1, 2, 4, 8),
        sampling_policy="fixed_k",
        budget_target=4,
        budget_actual=4,
        selection_rule="likelihood",
        allow_oracle=False,
        chosen_k=0,
        best_score=0.7,
        score_min=0.6,
        score_mean=0.65,
        score_max=0.7,
        diffusion_calls=4,
        diffusion_steps=50,
        guidance_scale=7.5,
        seed_base=123,
        wall_time_ms=10.0,
        is_oracle_run=False,
    )


def test_recon_outputs_smoke(tmp_path: Path):
    results = [_dummy_trial(i) for i in range(2)]
    csv_path, pq_path = write_trial_results(results, tmp_path)

    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    required_cols = {"uncertainty", "best_score", "diffusion_calls", "selection_rule"}
    assert required_cols.issubset(df.columns)

    # calibration assets
    run_calibration_assets(csv_path, tmp_path)
    paper_dir = tmp_path / "paper_assets"
    assert (paper_dir / "calibration_summary.json").exists()
    assert (paper_dir / "calibration_binned.csv").exists()
    assert (paper_dir / "calibration_reliability.png").exists()

    summary = json.loads((paper_dir / "calibration_summary.json").read_text())
    assert summary.get("n_trials") == len(results)
