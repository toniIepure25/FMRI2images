import csv
from pathlib import Path

import pytest
import torch

from fmri2img.inference.pipeline import SamplingConfig, run_probabilistic_trials
from fmri2img.eval.run_reconstruct_and_eval import write_trial_results


def _run_simple_trials(selection_rule="cosine", allow_oracle=False, seed_base=123):
    torch.manual_seed(0)
    mus = torch.tensor([[1.0, 0.0], [0.5, 0.5]], dtype=torch.float32)
    logvars = torch.zeros_like(mus)
    targets = torch.nn.functional.normalize(mus, dim=-1)
    uncertainties = torch.tensor([0.1, 0.9], dtype=torch.float32)

    cfg = SamplingConfig(
        sampling_policy="fixed_k",
        k_base=2,
        k_set=(2, 2),
        adaptive_quantile=((), (), "deterministic"),
        adaptive_mapping=((), ()),
        logvar_min=-8.0,
        logvar_max=2.0,
        variance_floor=1e-6,
        clip_space="normalized",
    )

    def generate_fn(z, seed):
        # deterministic: return normalized z so embed returns the same
        return torch.nn.functional.normalize(z, dim=-1)

    def embed_image_fn(img):
        return img  # identity embedding

    results, imgs = run_probabilistic_trials(
        mus=mus,
        logvars=logvars,
        targets=targets,
        uncertainties=uncertainties,
        sampling_cfg=cfg,
        selection_rule=selection_rule,
        allow_oracle=allow_oracle,
        generate_fn=generate_fn,
        embed_image_fn=embed_image_fn,
        seed_base=seed_base,
        guidance_scale=7.5,
        diffusion_steps=50,
        stimulus_ids=[1, 2],
        subject="subjX",
        split="test",
    )
    return results, imgs


def test_budget_and_determinism():
    res1, _ = _run_simple_trials(seed_base=111)
    res2, _ = _run_simple_trials(seed_base=111)

    # Budget exactness: each trial gets k_base=2
    assert all(r.K_assigned == 2 for r in res1)
    # Determinism with fixed seed
    assert [r.chosen_k for r in res1] == [r.chosen_k for r in res2]
    assert [round(r.best_score, 6) for r in res1] == [round(r.best_score, 6) for r in res2]


def test_oracle_gate_blocks_without_flag():
    with pytest.raises(PermissionError):
        _ = _run_simple_trials(selection_rule="oracle", allow_oracle=False)


def test_trial_csv_schema(tmp_path: Path):
    results, _ = _run_simple_trials(seed_base=222)
    csv_path, parquet_path = write_trial_results(results, tmp_path, csv_name="trials.csv")

    assert csv_path.exists()
    # Validate header contains key columns
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
    for col in ["trial_id", "stimulus_id", "uncertainty", "best_score", "sampling_policy"]:
        assert col in header

    # Parquet optional; if written ensure it matches row count
    if parquet_path is not None and parquet_path.exists():
        import pandas as pd

        df = pd.read_parquet(parquet_path)
        assert len(df) == len(results)
