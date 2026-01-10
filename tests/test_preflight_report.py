import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_preflight_prints_report_header(monkeypatch):
    # Ensure it runs and prints the report header (we don't require PASS in CI).
    env = os.environ.copy()
    env.pop("NSD_DATA_ROOT", None)
    env.pop("OUTPUT_ROOT", None)
    env.pop("CACHE_ROOT", None)
    env.pop("CHECKPOINT_ROOT", None)

    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "preflight.py")],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
    )

    assert "FMRI2images preflight" in proc.stdout
    # If env vars are unset, preflight should fail with rc=2.
    assert proc.returncode in (0, 2)
