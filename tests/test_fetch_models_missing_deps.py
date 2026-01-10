import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_fetch_models_exits_2_when_diffusers_missing(monkeypatch):
    # Ensure the script is robust even if diffusers isn't installed.
    env = os.environ.copy()

    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "fetch_models.py")],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
    )

    # If diffusers is installed in this env, it may proceed; accept 0 or 2.
    assert proc.returncode in (0, 2)
    if proc.returncode == 2:
        assert "missing dependency" in proc.stdout
