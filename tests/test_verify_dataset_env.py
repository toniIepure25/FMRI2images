import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_verify_dataset_returns_2_when_env_unset(monkeypatch):
    monkeypatch.delenv("NSD_DATA_ROOT", raising=False)

    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "verify_dataset.py")],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        cwd=REPO_ROOT,
    )

    assert proc.returncode == 2
    assert "NSD_DATA_ROOT" in proc.stdout
