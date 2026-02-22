#!/usr/bin/env python3
"""Doctor command: comprehensive readiness check for remote GPU pods.

This wraps:
  - preflight
  - dataset verification
  - model cache check/download

Exit codes:
  0: PASS
  2: FAIL
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str]) -> int:
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    return int(proc.returncode)


def main() -> int:
    rc = 0

    # Preflight (deps + cuda + disk + paths contract)
    rc = max(rc, _run([sys.executable, str(REPO_ROOT / "scripts" / "utils" / "preflight.py")]))

    # Dataset
    rc = max(rc, _run([sys.executable, str(REPO_ROOT / "scripts" / "utils" / "verify_dataset.py")]))

    # Models (only relevant when DIFFUSION_ENABLED=1; fetch_models itself enforces deps)
    rc = max(rc, _run([sys.executable, str(REPO_ROOT / "scripts" / "utils" / "fetch_models.py")]))

    if rc != 0:
        print("doctor: FAIL")
        return 2

    print("doctor: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
