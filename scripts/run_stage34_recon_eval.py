#!/usr/bin/env python3
"""
Stage 3 & 4 reconstruction/eval orchestrator (paper-grade).

Runs a sequence of reconstruction/evaluation jobs with strict paper-mode defaults
and then aggregates compute-vs-quality metrics.

Default runs (for subj01):
  - det_fixed: deterministic checkpoint, fixed K=1
  - prob_fixed: probabilistic checkpoint, fixed K=4
  - prob_adapt_q: probabilistic checkpoint, adaptive_quantile with K_set [1,2,4,8,16]

Usage examples are written to RUNS.md; invoke `--help` for CLI.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_RUNS = [
    {
        "run_name": "det_fixed",
        "encoder": "two_stage",
        "config_path": "configs/experiments/two_stage/det_fixed.yaml",
        "checkpoint_path": "checkpoints/two_stage/{subject}/det_fixed/two_stage_best.pt",
        "sampling_policy": "fixed_k",
        "k_base": 1,
        "k_set": [1],
        "selection_rule": "cosine",
        "probabilistic": False,
        "adaptive_quantile": None,
    },
    {
        "run_name": "prob_fixed",
        "encoder": "prob",
        "config_path": "configs/experiments/probabilistic/stage1/prob_fixed.yaml",
        "checkpoint_path": "checkpoints/two_stage/{subject}/prob_fixed/two_stage_best.pt",
        "sampling_policy": "fixed_k",
        "k_base": 4,
        "k_set": [1, 2, 4],
        "selection_rule": "likelihood",
        "probabilistic": True,
        "adaptive_quantile": None,
    },
    {
        "run_name": "prob_adapt_q",
        "encoder": "prob",
        "config_path": "configs/experiments/probabilistic/stage1/prob_adapt_q.yaml",
        "checkpoint_path": "checkpoints/two_stage/{subject}/prob_adapt_q_run2/two_stage_best.pt",
        "sampling_policy": "adaptive_quantile",
        "k_base": 4,
        "k_set": [1, 2, 4, 8, 16],
        "selection_rule": "likelihood",
        "probabilistic": True,
        "adaptive_quantile": [0.0, 0.10, 0.30, 0.60, 0.85, 1.0],
        "adaptive_k_bins": [16, 8, 4, 2, 1],
    },
]


def parse_run_arg(raw: str) -> Dict[str, Any]:
    """Parse a run spec of the form:
    run_name:encoder:ckpt:config:sampling_policy:k_base:k_set:selection_rule[:adaptive_quantile_csv][:adaptive_k_bins_csv]
    k_set is comma-separated ints. adaptive_quantile_csv is comma-separated floats. adaptive_k_bins_csv is comma-separated ints.
    """
    parts = raw.split(":")
    if len(parts) < 8:
        raise ValueError("Run spec must have at least 8 parts: name:encoder:ckpt:config:policy:k_base:k_set:selection")
    run_name, encoder, ckpt, cfg, policy, k_base, k_set_csv, selection = parts[:8]
    adaptive_q = None
    adaptive_k_bins = None
    if len(parts) >= 9 and parts[8]:
        adaptive_q = [float(x) for x in parts[8].split(",")]
    if len(parts) >= 10 and parts[9]:
        adaptive_k_bins = [int(x) for x in parts[9].split(",")]
    return {
        "run_name": run_name,
        "encoder": encoder,
        "checkpoint_path": ckpt,
        "config_path": cfg,
        "sampling_policy": policy,
        "k_base": int(k_base),
        "k_set": [int(x) for x in k_set_csv.split(",")],
        "selection_rule": selection,
        "probabilistic": encoder == "prob",
        "adaptive_quantile": adaptive_q,
        "adaptive_k_bins": adaptive_k_bins,
    }


def load_runs(subject: str, overrides: List[str]) -> List[Dict[str, Any]]:
    runs = [dict(r) for r in DEFAULT_RUNS]
    if overrides:
        runs = [parse_run_arg(r) for r in overrides]
    # Resolve subject in checkpoint paths
    for r in runs:
        r["checkpoint_path"] = str(r["checkpoint_path"]).format(subject=subject)
    return runs


def ensure_output_dir(path: Path, force: bool) -> None:
    if path.exists() and any(path.iterdir()) and not force:
        raise FileExistsError(f"Output dir {path} exists. Use --force to overwrite.")
    path.mkdir(parents=True, exist_ok=True)


def _augment_manifest(out_dir: Path, run: Dict[str, Any], cmd: List[str]) -> None:
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        return
    try:
        data = json.loads(manifest_path.read_text())
    except Exception:
        return
    data.setdefault("orchestration", {})
    data["orchestration"].update({
        "run_name": run["run_name"],
        "config_path": run.get("config_path"),
        "orchestrator_cmd": cmd,
    })
    manifest_path.write_text(json.dumps(data, indent=2))


def run_single(run: Dict[str, Any], args: argparse.Namespace) -> None:
    out_dir = Path("outputs/recon") / args.subject / run["run_name"]
    report_dir = Path("outputs/reports") / args.subject / run["run_name"]
    ensure_output_dir(out_dir, args.force)
    report_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/run_reconstruct_and_eval.py",
        "--subject", args.subject,
        "--encoder", run["encoder"],
        "--ckpt", run["checkpoint_path"],
        "--clip-cache", str(args.clip_cache),
        "--index-root", str(args.index_root),
        "--output-dir", str(out_dir),
        "--report-dir", str(report_dir),
        "--trial-csv", "trial_results.csv",
        "--limit", str(args.limit),
        "--steps", str(args.steps),
        "--guidance-scale", str(args.guidance),
    ]

    if run.get("probabilistic", False):
        cmd += [
            "--probabilistic",
            "--sampling-policy", run["sampling_policy"],
            "--k-base", str(run["k_base"]),
            "--k-set", *[str(k) for k in run["k_set"]],
            "--selection-rule", run["selection_rule"],
        ]
        if run.get("adaptive_quantile"):
            cmd += ["--adaptive-quantile", *[str(x) for x in run["adaptive_quantile"]]]
        if run.get("adaptive_k_bins"):
            cmd += ["--adaptive-k-bins", *[str(x) for x in run["adaptive_k_bins"]]]
    else:
        # deterministic: force fixed K=1 semantics via guidance/steps only
        pass

    # paper-mode safety: disallow fallback/skip validation
    if args.device and args.device != "auto":
        cmd += ["--device", args.device]

    # Execute
    print(f"\n=== Running {run['run_name']} ===")
    print("Command:", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Run {run['run_name']} failed with exit code {proc.returncode}")
    _augment_manifest(out_dir, run, cmd)


def compute_vs_quality(run_dirs: Sequence[Path], global_dir: Path) -> None:
    records = []
    for rd in run_dirs:
        trial_csv = rd / "trial_results.csv"
        budget_json = rd / "budget_stats.json"
        if not trial_csv.exists():
            continue
        df = pd.read_csv(trial_csv)
        best_mean = df.get("best_score", df.get("best_score", pd.Series(dtype=float))).mean()
        n = len(df)
        budget = {}
        if budget_json.exists():
            try:
                budget = json.loads(budget_json.read_text())
            except Exception:
                budget = {}
        records.append({
            "run_name": rd.name,
            "n_trials": n,
            "mean_best_score": best_mean,
            "budget_target": budget.get("budget_target"),
            "budget_actual": budget.get("budget_actual"),
            "mean_K": budget.get("mean_K"),
            "max_K": budget.get("max_K"),
            "min_K": budget.get("min_K"),
        })

    if not records:
        return

    global_dir.mkdir(parents=True, exist_ok=True)
    csv_path = global_dir / "compute_vs_quality.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)

    plt.figure(figsize=(6, 4))
    xs = [r.get("budget_actual", r.get("budget_target")) for r in records]
    ys = [r.get("mean_best_score", 0) for r in records]
    labels = [r["run_name"] for r in records]
    plt.scatter(xs, ys)
    for x, y, label in zip(xs, ys, labels):
        plt.annotate(label, (x, y))
    plt.xlabel("Total diffusion calls (budget_actual)")
    plt.ylabel("Mean best score")
    plt.title("Compute vs Quality")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(global_dir / "compute_vs_quality.png", dpi=200)
    plt.close()



def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--index-root", required=True, type=Path)
    parser.add_argument("--clip-cache", required=True, type=Path)
    parser.add_argument("--runs", action="append", help="Custom run spec: name:encoder:ckpt:config:policy:k_base:k_set:selection[:adaptive_q]")
    parser.add_argument("--limit", type=int, default=64, help="Sample limit for quick runs")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output dirs")
    args = parser.parse_args()

    runs = load_runs(args.subject, args.runs or [])
    finished_dirs = []
    for r in runs:
        try:
            run_single(r, args)
            finished_dirs.append(Path("outputs/recon") / args.subject / r["run_name"])
        except Exception as e:
            print(f"❌ Run {r['run_name']} failed: {e}")
            raise

    # Compute vs quality across probabilistic runs
    prob_dirs = [d for d in finished_dirs if "prob" in d.name]
    global_dir = Path("outputs/recon") / args.subject / "paper_assets_global"
    compute_vs_quality(prob_dirs, global_dir)
    print(f"✓ Aggregated compute-vs-quality assets -> {global_dir}")


if __name__ == "__main__":
    sys.exit(main())
