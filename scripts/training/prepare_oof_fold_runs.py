#!/usr/bin/env python3
"""Generate fold-specific configs and run script for OOF training/prediction.

This utility automates the repetitive setup for V40 OOF folds:
- clones a base experiment YAML per fold,
- injects `data.split_file` for that fold,
- sets a unique `experiment.name`,
- optionally disables shared1000 eval for speed,
- writes a bash script that runs train_unified + generate_split_predictions
  and exports fold-heldout predictions to a merge-compatible directory layout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON: {path}")
    with open(path) as f:
        return json.load(f)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing YAML: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _path_remap(path_str: str, old_prefix: str | None, new_prefix: str | None) -> str:
    if not old_prefix or not new_prefix:
        return path_str
    if path_str.startswith(old_prefix):
        return new_prefix + path_str[len(old_prefix):]
    return path_str


def _validate_json_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    text = path.read_text().strip()
    if not text:
        raise ValueError(f"JSON file is empty: {path}")
    try:
        json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e


def _bool_env_default(val: bool) -> str:
    return "1" if val else "0"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare OOF fold configs and run script")
    parser.add_argument("--base-config", type=str, required=True, help="Base experiment YAML")
    parser.add_argument("--fold-manifest", type=str, required=True, help="oof_fold_manifest.json path")
    parser.add_argument(
        "--config-output-dir",
        type=str,
        required=True,
        help="Where fold YAMLs are written",
    )
    parser.add_argument(
        "--run-script-path",
        type=str,
        required=True,
        help="Path to generated bash runner",
    )
    parser.add_argument(
        "--exp-prefix",
        type=str,
        required=True,
        help="Experiment name prefix for fold runs (e.g. V35_oof or N1v28a_oof)",
    )
    parser.add_argument("--subject", type=str, default="subj01", help="Subject override")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id used in generated script")
    parser.add_argument(
        "--fold-preds-root",
        type=str,
        required=True,
        help="Root where fold prediction outputs go (e.g. .../oof/tri_folds)",
    )
    parser.add_argument(
        "--save-checkpoints",
        type=str,
        default="best",
        choices=["all", "best", "none"],
        help="train_unified checkpoint policy in generated script",
    )
    parser.add_argument(
        "--disable-shared1000-eval",
        action="store_true",
        help="If set, force evaluation.eval_shared1000=false in fold configs",
    )
    parser.add_argument(
        "--prediction-split",
        type=str,
        default="val",
        choices=["val", "train"],
        help="Which split to export from each fold (OOF should use val)",
    )
    parser.add_argument(
        "--checkpoint-path-old-prefix",
        type=str,
        default=None,
        help="Optional old prefix for remapping checkpoint paths in config",
    )
    parser.add_argument(
        "--checkpoint-path-new-prefix",
        type=str,
        default=None,
        help="Optional new prefix for remapping checkpoint paths in config",
    )
    parser.add_argument(
        "--strip-pretrained-model",
        action="store_true",
        help=(
            "Remove model.pretrained_model_path and disable require_pretrained_model "
            "for generated fold configs. Use this for leakage-safe OOF runs."
        ),
    )
    parser.add_argument(
        "--fold-teacher-checkpoint-template",
        type=str,
        default=None,
        help=(
            "Optional per-fold teacher checkpoint path template applied to "
            "loss.legacy_teacher_distill.teacher_checkpoint_path. Supports "
            "{fold_index}, {fold_tag}, and {subject}."
        ),
    )
    args = parser.parse_args()

    base_config_path = Path(args.base_config)
    manifest_path = Path(args.fold_manifest)
    config_output_dir = Path(args.config_output_dir)
    run_script_path = Path(args.run_script_path)
    fold_preds_root = Path(args.fold_preds_root)

    base_cfg = _load_yaml(base_config_path)
    manifest = _load_json(manifest_path)

    folds = manifest.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError("Manifest must contain non-empty list: folds")

    # Preflight: fold split JSONs must all be present and valid.
    for fold in folds:
        _validate_json_file(Path(fold["split_json"]))

    generated_configs: list[dict[str, Any]] = []

    for fold in folds:
        fold_idx = int(fold["fold_index"])
        fold_tag = f"{fold_idx:02d}"

        cfg = json.loads(json.dumps(base_cfg))  # simple deep-copy via json-safe payload

        exp = cfg.setdefault("experiment", {})
        data = cfg.setdefault("data", {})
        evaluation = cfg.setdefault("evaluation", {})

        exp_name = f"{args.exp_prefix}_fold_{fold_tag}"
        exp["name"] = exp_name
        data["subject"] = args.subject
        data["split_file"] = str(fold["split_json"])

        # Optional checkpoint path remap for remote environments.
        if "model" in cfg and isinstance(cfg["model"], dict):
            if args.strip_pretrained_model:
                cfg["model"].pop("pretrained_model_path", None)
                cfg["model"]["require_pretrained_model"] = False
                cfg["model"]["require_parent_split_match"] = False
            pm = cfg["model"].get("pretrained_model_path")
            if isinstance(pm, str):
                cfg["model"]["pretrained_model_path"] = _path_remap(
                    pm,
                    args.checkpoint_path_old_prefix,
                    args.checkpoint_path_new_prefix,
                )
        if "loss" in cfg and isinstance(cfg["loss"], dict):
            for loss_name in [
                "legacy_teacher_distill",
                "legacy_compact_distill",
                "component_legacy_compact_distill",
                "tri_teacher_distill",
            ]:
                loss_cfg = cfg["loss"].get(loss_name)
                if not isinstance(loss_cfg, dict):
                    continue
                if args.fold_teacher_checkpoint_template and "teacher_checkpoint_path" in loss_cfg:
                    loss_cfg["teacher_checkpoint_path"] = args.fold_teacher_checkpoint_template.format(
                        fold_index=fold_idx,
                        fold_tag=fold_tag,
                        subject=args.subject,
                    )
                tp = loss_cfg.get("teacher_checkpoint_path")
                if isinstance(tp, str):
                    loss_cfg["teacher_checkpoint_path"] = _path_remap(
                        tp,
                        args.checkpoint_path_old_prefix,
                        args.checkpoint_path_new_prefix,
                    )

        if args.disable_shared1000_eval:
            evaluation["eval_shared1000"] = False

        out_cfg = config_output_dir / f"{exp_name}.yaml"
        _save_yaml(out_cfg, cfg)

        generated_configs.append(
            {
                "fold_index": fold_idx,
                "config_path": str(out_cfg),
                "experiment_name": exp_name,
            }
        )

    run_script_path.parent.mkdir(parents=True, exist_ok=True)
    fold_preds_root.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("#!/bin/bash")
    lines.append("set -euo pipefail")
    lines.append("")
    lines.append(f"GPU=\"${{GPU:-{args.gpu}}}\"")
    lines.append(f"SUBJECT=\"${{SUBJECT:-{args.subject}}}\"")
    lines.append(f"SAVE_CKPT=\"${{SAVE_CKPT:-{args.save_checkpoints}}}\"")
    lines.append("PYTHON_BIN=\"${PYTHON_BIN:-python3}\"")
    lines.append("export CUDA_VISIBLE_DEVICES=\"$GPU\"")
    lines.append("")
    lines.append(f"FOLD_PREDS_ROOT=\"{fold_preds_root}\"")
    lines.append("mkdir -p \"$FOLD_PREDS_ROOT\"")
    lines.append("")
    lines.append("echo \"Running OOF fold jobs...\"")

    for row in generated_configs:
        fold_idx = int(row["fold_index"])
        fold_tag = f"{fold_idx:02d}"
        cfg_path = row["config_path"]
        exp_name = row["experiment_name"]

        lines.append("")
        lines.append(f"echo \"[Fold {fold_tag}] train {exp_name}\"")
        lines.append(
            "\"$PYTHON_BIN\" scripts/training/train_unified.py "
            f"--config \"{cfg_path}\" --subject \"$SUBJECT\" --gpu 0 --save-checkpoints \"$SAVE_CKPT\""
        )
        lines.append("")
        lines.append(f"echo \"[Fold {fold_tag}] export {args.prediction_split} predictions\"")
        lines.append(
            "\"$PYTHON_BIN\" scripts/evaluation/generate_split_predictions.py "
            f"--checkpoint \"experimental_results/{exp_name}/$SUBJECT/checkpoint_best.pt\" "
            f"--output-dir \"$FOLD_PREDS_ROOT/fold_{fold_tag}\" "
            f"--split {args.prediction_split} --subject \"$SUBJECT\""
        )

    lines.append("")
    lines.append("echo \"OOF fold jobs completed.\"")

    with open(run_script_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("Generated configs:")
    for row in generated_configs:
        print(f"  fold {row['fold_index']:02d}: {row['config_path']}")
    print(f"Generated run script: {run_script_path}")
    print(f"Fold predictions root: {fold_preds_root}")
    if args.checkpoint_path_old_prefix and args.checkpoint_path_new_prefix:
        print(
            "Checkpoint path remap enabled: "
            f"{args.checkpoint_path_old_prefix} -> {args.checkpoint_path_new_prefix}"
        )


if __name__ == "__main__":
    main()
