"""Engineering acquisition planning: run-duration estimation (RANGE/TBD-aware), run-partition enumeration, and
session-partition options. Preserves the frozen trial counts (1536 perception, 96 imagery) and never selects a
final partition or uses scientific outcomes. For PI / MRI-physicist review only."""
from __future__ import annotations

from . import config as C


def estimate_acquisition(cfg: C.ExperimentConfig, n_perception_runs=6, n_imagery_runs=6):
    """Returns per-run and total estimates. When run/session structure or TR is not finalized, returns TBD /
    ranges rather than a false exact total."""
    t = cfg.timing
    perc_trial = (t.perception_image_s or 3.0) + (t.perception_gap_s or 1.0)
    img_trial = t.imagery_trial_s or 4.0
    perc_per_run = C.PERCEPTION_TRIALS / n_perception_runs
    img_per_run = C.IMAGERY_TRIALS / n_imagery_runs
    tr = t.TR_s
    out = {
        "TR_s": tr if tr is not None else "TBD_SITE_OPERATOR",
        "dummy_volumes": t.dummy_volumes if t.dummy_volumes is not None else "TBD_SITE_OPERATOR",
        "inter_run_rest_s": t.inter_run_rest_s if t.inter_run_rest_s is not None else "TBD_SITE_OPERATOR",
        "perception": {"n_runs": n_perception_runs, "trials_per_run": perc_per_run,
                       "task_time_per_run_s": round(perc_per_run * perc_trial, 1),
                       "total_task_time_s": round(C.PERCEPTION_TRIALS * perc_trial, 1)},
        "imagery": {"n_runs": n_imagery_runs, "trials_per_run": img_per_run,
                    "task_time_per_run_s": round(img_per_run * img_trial, 1),
                    "total_task_time_s": round(C.IMAGERY_TRIALS * img_trial, 1)},
        "functional_volumes_per_run": "TBD_SITE_OPERATOR (needs TR + dummy count)" if tr is None else None,
        "structural_time": "TBD_SITE_OPERATOR (T1w<=1mm + T2w)",
        "setup_time": "TBD_SITE_OPERATOR",
        "total_scheduled_block": "TBD_SITE_OPERATOR (range only until run/session structure + TR approved)",
        "note": "engineering estimate; not a scientific parameter; final structure requires PI/MRI physicist",
    }
    total_task = out["perception"]["total_task_time_s"] + out["imagery"]["total_task_time_s"]
    out["approx_total_task_time_min_lowerbound"] = round(total_task / 60.0, 1)
    return out


def _divisors(n, lo, hi):
    return [d for d in range(lo, hi + 1) if n % d == 0]


def run_partition_options(perc_trials=C.PERCEPTION_TRIALS, img_trials=C.IMAGERY_TRIALS):
    """Enumerate candidate run partitions that keep balanced integer trials/run. No final choice; no outcomes."""
    rows = []
    for n_runs in _divisors(perc_trials, 4, 12):
        rows.append({"task": "perception", "runs": n_runs, "trials_per_run": perc_trials // n_runs,
                     "balanced_integer": True, "total_preserved": n_runs * (perc_trials // n_runs) == perc_trials,
                     "operational_concern": "longer runs = fewer breaks; shorter runs = more restarts"})
    for n_runs in _divisors(img_trials, 4, 12):
        rows.append({"task": "imagery", "runs": n_runs, "trials_per_run": img_trials // n_runs,
                     "balanced_integer": True, "total_preserved": n_runs * (img_trials // n_runs) == img_trials,
                     "operational_concern": "keep 6 folds valid; family balance across runs"})
    return rows


def session_partition_options():
    """Feasible session partitions preserving all counts + fold validity + randomization. No final selection."""
    return [
        {"option": "1-session", "perception_trials": C.PERCEPTION_TRIALS, "imagery_trials": C.IMAGERY_TRIALS,
         "preserves_counts": True, "fold_validity": True, "burden": "high single-visit burden; long scan block"},
        {"option": "2-session", "perception_trials": C.PERCEPTION_TRIALS, "imagery_trials": C.IMAGERY_TRIALS,
         "preserves_counts": True, "fold_validity": True, "burden": "moderate; e.g. perception + imagery split or halves"},
        {"option": "multi-session", "perception_trials": C.PERCEPTION_TRIALS, "imagery_trials": C.IMAGERY_TRIALS,
         "preserves_counts": True, "fold_validity": True, "burden": "lowest per-visit; more scheduling/registration overhead"},
    ]
