#!/usr/bin/env python3
"""Run an experiment described by a YAML file.

This is the canonical entrypoint behind `make exp`.

Contract
--------
Inputs:
- --exp: path to an experiment YAML (see `experiments/*.yaml`)
- --override key=value: optional overrides applied after YAML is loaded

Outputs:
- Creates a unique run directory under `$OUTPUT_ROOT/runs/` (or `outputs/runs/`)
- Writes:
  - resolved_experiment.yaml   (YAML after overrides + defaults)
  - status.json                (running/success/failed + timestamps)
  - manifest.json              (environment + git + CLI + experiment config)
  - registry.csv append        (one line per run)

The actual heavy lifting is delegated to `scripts/run_full_pipeline.py`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fmri2img.utils.manifest import gather_env_info, write_manifest


def _require(p: Path, what: str) -> None:
    if not p.exists():
        raise SystemExit(f"Missing {what}: {p}")


def _truthy(v: Optional[str]) -> bool:
    if v is None:
        return False
    return v.strip().lower() not in {"0", "false", "no", "off", ""}


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: pyyaml. Install it (it should be in requirements) and retry. "
            f"Original error: {e}"
        )

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise SystemExit(f"Experiment YAML must be a mapping/dict at top-level: {path}")

    return data


def _dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: pyyaml. Install it (it should be in requirements) and retry. "
            f"Original error: {e}"
        )

    text = yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
    path.write_text(text, encoding="utf-8")


_OVERRIDE_RE = re.compile(r"^(?P<key>[A-Za-z0-9_.-]+)=(?P<value>.*)$")


def _parse_overrides(items: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for it in items:
        m = _OVERRIDE_RE.match(it)
        if not m:
            raise SystemExit(f"Invalid --override (expected key=value): {it}")
        out[m.group("key")] = m.group("value")
    return out


def _set_dotted(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur: Dict[str, Any] = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _apply_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in overrides.items():
        _set_dotted(cfg, k, v)
    return cfg


def _git_commit() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True)
            .strip()
        )
    except Exception:
        return None


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _slug(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^A-Za-z0-9_.-]+", "-", s)
    return s.strip("-") or "run"


@dataclass
class Experiment:
    name: str
    subject: str
    mode: str
    resume_from: Optional[str] = None
    force_rebuild: bool = False
    dry_run: bool = False
    skip_eval: bool = False
    root_dir: str = "."

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Experiment":
        exp = d.get("experiment", d)
        if not isinstance(exp, dict):
            raise SystemExit("Experiment YAML must contain either top-level keys or an 'experiment:' mapping")

        name = str(exp.get("name", "experiment"))
        subject = exp.get("subject")
        mode = exp.get("mode")
        if not subject or not mode:
            raise SystemExit("Experiment YAML must set 'subject' and 'mode' (baseline|novel|ablation)")

        resume_from = exp.get("resume_from")
        force_rebuild = bool(exp.get("force_rebuild", False))
        dry_run = bool(exp.get("dry_run", False))
        skip_eval = bool(exp.get("skip_eval", False))
        root_dir = str(exp.get("root_dir", "."))

        return cls(
            name=name,
            subject=str(subject),
            mode=str(mode),
            resume_from=str(resume_from) if resume_from is not None else None,
            force_rebuild=force_rebuild,
            dry_run=dry_run,
            skip_eval=skip_eval,
            root_dir=root_dir,
        )


def _resolve_output_root() -> Path:
    # OUTPUT_ROOT is the new contract; fall back to legacy `outputs/`.
    out = os.environ.get("OUTPUT_ROOT")
    if out:
        return Path(out)
    return Path("outputs")


def _run_dir(base: Path, exp: Experiment) -> Path:
    commit = _git_commit()
    suffix = commit[:8] if commit else "nogit"
    name = _slug(exp.name)
    ts = _timestamp()
    return base / "runs" / f"{ts}_{exp.subject}_{name}_{suffix}"


def _write_status(path: Path, state: str, extra: Optional[Dict[str, Any]] = None) -> None:
    payload: Dict[str, Any] = {
        "state": state,
        "timestamp": datetime.now().isoformat(),
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_registry(registry_path: Path, row: Dict[str, Any]) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    exists = registry_path.exists()

    # Deterministic column order (keep future additions stable)
    fieldnames = [
        "timestamp",
        "run_dir",
        "exp_file",
        "name",
        "subject",
        "mode",
        "git_commit",
        "dry_run",
        "skip_eval",
        "force_rebuild",
        "resume_from",
        "returncode",
        "state",
    ]

    with registry_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def _build_cmd(exp: Experiment) -> List[str]:
    cmd = [sys.executable, "scripts/run_full_pipeline.py", "--subject", exp.subject, "--mode", exp.mode]
    if exp.root_dir and exp.root_dir != ".":
        cmd += ["--root-dir", exp.root_dir]
    if exp.resume_from:
        cmd += ["--resume-from", exp.resume_from]
    if exp.force_rebuild:
        cmd += ["--force-rebuild"]
    if exp.dry_run:
        cmd += ["--dry-run"]
    if exp.skip_eval:
        cmd += ["--skip-eval"]
    return cmd


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run an experiment YAML and record provenance")
    p.add_argument("--exp", required=True, type=Path, help="Path to experiment YAML")
    p.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config key (repeatable). Supports dotted keys: experiment.force_rebuild=true",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override OUTPUT_ROOT for this run (default: $OUTPUT_ROOT or ./outputs)",
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Write the run into this directory instead of auto-naming under OUTPUT_ROOT/runs",
    )

    args = p.parse_args(argv)

    exp_path = args.exp
    _require(exp_path, "experiment YAML")

    cfg = _load_yaml(exp_path)
    overrides = _parse_overrides(args.override)
    cfg = _apply_overrides(cfg, overrides)

    exp = Experiment.from_dict(cfg)

    output_root = args.output_root or _resolve_output_root()
    run_dir = args.run_dir or _run_dir(output_root, exp)
    run_dir.mkdir(parents=True, exist_ok=False)

    resolved_yaml_path = run_dir / "resolved_experiment.yaml"
    _dump_yaml(resolved_yaml_path, cfg)

    status_path = run_dir / "status.json"
    _write_status(status_path, "running", {"exp": str(exp_path)})

    env = gather_env_info()
    git_commit = env.get("git_commit") if isinstance(env.get("git_commit"), str) else None

    cmd = _build_cmd(exp)

    # manifest.json right away (even for failures)
    write_manifest(
        run_dir / "manifest.json",
        config_dict={
            "script": "scripts/run_experiment.py",
            "exp_file": str(exp_path),
            "experiment": {
                "name": exp.name,
                "subject": exp.subject,
                "mode": exp.mode,
                "resume_from": exp.resume_from,
                "force_rebuild": exp.force_rebuild,
                "dry_run": exp.dry_run,
                "skip_eval": exp.skip_eval,
                "root_dir": exp.root_dir,
            },
            "resolved_experiment_yaml": str(resolved_yaml_path),
        },
        cli_args=sys.argv,
        env_info=env,
        input_hashes=None,
        additional_info={
            "subprocess": {
                "cmd": cmd,
                "cwd": str(Path.cwd()),
            }
        },
    )

    # Run and capture return code
    returncode: int
    try:
        proc = subprocess.run(cmd, check=False)
        returncode = int(proc.returncode)
    except KeyboardInterrupt:
        _write_status(status_path, "interrupted")
        return 130

    state = "success" if returncode == 0 else "failed"
    _write_status(status_path, state, {"returncode": returncode})

    registry_path = output_root / "runs" / "registry.csv"
    _append_registry(
        registry_path,
        {
            "timestamp": datetime.now().isoformat(),
            "run_dir": str(run_dir),
            "exp_file": str(exp_path),
            "name": exp.name,
            "subject": exp.subject,
            "mode": exp.mode,
            "git_commit": git_commit or _git_commit() or "",
            "dry_run": exp.dry_run,
            "skip_eval": exp.skip_eval,
            "force_rebuild": exp.force_rebuild,
            "resume_from": exp.resume_from or "",
            "returncode": returncode,
            "state": state,
        },
    )

    print(str(run_dir))
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
