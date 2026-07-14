"""
PCD Ablation Runner — Generates and runs ablation experiments.
================================================================

Ablations:
  1. PCD full         — the standard hierarchical error-driven architecture
  2. PCD no_errors    — same hierarchy but skip prediction errors (raw tokens)
  3. PCD reversed     — reverse the hierarchy order (test if order matters)
  4. PCD random       — random hierarchy order (control)
  5. Flat ROI Trans   — standard ROI Transformer (no hierarchy, same params)
  6. MLP baseline     — MLP encoder with matched parameter count

Usage:
    python scripts/analysis/run_pcd_ablations.py \\
        --base-config configs/experiments/PCD_v1_8subject.yaml \\
        --subject subj01 \\
        --output-dir experimental_results/PCD_ablations \\
        --gpu 0

    # Or run specific ablation:
    python scripts/analysis/run_pcd_ablations.py \\
        --base-config configs/experiments/PCD_v1_8subject.yaml \\
        --subject subj01 --only no_errors
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

ABLATION_CONFIGS = {
    "full": {
        "description": "PCD full (error-driven hierarchy)",
        "model_type": "pcd",
        "pcd_overrides": {"ablation_mode": "full"},
    },
    "no_errors": {
        "description": "PCD no prediction errors (hierarchical grouping only)",
        "model_type": "pcd",
        "pcd_overrides": {"ablation_mode": "no_errors"},
    },
    "reversed": {
        "description": "PCD reversed hierarchy (high→low instead of low→high)",
        "model_type": "pcd",
        "pcd_overrides": {"ablation_mode": "reversed"},
    },
    "random": {
        "description": "PCD random hierarchy (control)",
        "model_type": "pcd",
        "pcd_overrides": {"ablation_mode": "random"},
    },
    "flat_transformer": {
        "description": "Flat ROI Transformer (no hierarchy, same ROIs)",
        "model_type": "vmf",
        "replace_model": {
            "type": "vmf",
            "encoder": {
                "encoder_type": "roi_transformer",
                "d_model": 768,
                "nhead": 12,
                "num_layers": 8,
                "dropout": 0.1,
                "activation": "gelu",
                "drop_path_rate": 0.0,
            },
            "decoder": {
                "output_dim": 768,
                "hidden_dims": [768],
                "activation": "gelu",
                "dropout": 0.1,
                "kappa_mode": "softplus",
                "kappa_min": 0.001,
                "kappa_max": 500.0,
            },
        },
    },
    "mlp_baseline": {
        "description": "MLP encoder baseline (matched params)",
        "model_type": "vmf",
        "replace_model": {
            "type": "vmf",
            "encoder": {
                "encoder_type": "mlp",
                "input_dim": None,
                "hidden_dims": [8192, 4096, 2048],
                "activation": "gelu",
                "dropout": 0.25,
                "use_residual": True,
            },
            "decoder": {
                "output_dim": 768,
                "hidden_dims": [2048],
                "activation": "gelu",
                "dropout": 0.1,
                "kappa_mode": "softplus",
                "kappa_min": 0.001,
                "kappa_max": 500.0,
            },
        },
    },
}


def generate_ablation_config(
    base_config: Dict,
    ablation_name: str,
    subject: str,
    output_dir: str,
) -> Path:
    """Generate a YAML config for one ablation experiment."""
    ablation_spec = ABLATION_CONFIGS[ablation_name]
    config = copy.deepcopy(base_config)

    exp_name = f"PCD_ablation_{ablation_name}"
    config["experiment"]["name"] = exp_name
    config["experiment"]["description"] = ablation_spec["description"]

    if "replace_model" in ablation_spec:
        model_replace = copy.deepcopy(ablation_spec["replace_model"])
        if ablation_name == "flat_transformer":
            model_replace["encoder"]["roi_dims"] = config["model"]["encoder"].get("roi_dims", {})
        config["model"] = model_replace
        config["model"].setdefault("projection_head", {"enabled": True, "hidden_dim": 2048, "out_dim": 768, "dropout": 0.1})
    elif "pcd_overrides" in ablation_spec:
        for key, val in ablation_spec["pcd_overrides"].items():
            config["model"].setdefault("pcd", {})[key] = val

    config["data"]["subject"] = subject
    if "subjects" in config["data"]:
        del config["data"]["subjects"]
    if "cross_subject" in config.get("model", {}):
        del config["model"]["cross_subject"]

    config["paths"]["output_dir"] = str(Path(output_dir) / exp_name)

    config_dir = Path("configs/experiments/ablations")
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{exp_name}.yaml"

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info("Generated ablation config: %s -> %s", ablation_name, config_path)
    return config_path


def run_ablation(
    config_path: Path,
    gpu: int = 0,
    subject: str = "subj01",
) -> int:
    """Run one ablation experiment via train_unified.py."""
    cmd = [
        sys.executable,
        "scripts/training/train_unified.py",
        "--config", str(config_path),
        "--subject", subject,
        "--gpu", str(gpu),
    ]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[2]))
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="PCD Ablation Runner")
    parser.add_argument("--base-config", type=str,
                        default="configs/experiments/PCD_v1_8subject.yaml")
    parser.add_argument("--subject", type=str, default="subj01")
    parser.add_argument("--output-dir", type=str,
                        default="experimental_results/PCD_ablations")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--only", type=str, default=None,
                        choices=list(ABLATION_CONFIGS.keys()),
                        help="Run only this ablation")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate configs only, don't train")
    args = parser.parse_args()

    with open(args.base_config) as f:
        base_config = yaml.safe_load(f)

    ablations = [args.only] if args.only else list(ABLATION_CONFIGS.keys())

    results = {}
    for name in ablations:
        logger.info("=" * 60)
        logger.info("Ablation: %s — %s", name, ABLATION_CONFIGS[name]["description"])
        logger.info("=" * 60)

        config_path = generate_ablation_config(
            base_config, name, args.subject, args.output_dir,
        )

        if args.dry_run:
            logger.info("DRY RUN — skipping training for %s", name)
            results[name] = {"status": "dry_run", "config": str(config_path)}
            continue

        rc = run_ablation(config_path, args.gpu, args.subject)
        results[name] = {
            "status": "success" if rc == 0 else "failed",
            "return_code": rc,
            "config": str(config_path),
        }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Ablation summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
