import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("force", [False, True])
def test_write_run_manifest_creates_manifest(tmp_path: Path, force: bool):
    out_dir = tmp_path / "run"
    out_dir.mkdir()

    # Create an input file to hash
    f = tmp_path / "input.txt"
    f.write_text("hello")

    cmd = [
        sys.executable,
        "scripts/write_run_manifest.py",
        "--output-dir",
        str(out_dir),
        "--config",
        "subject=subj01",
        "--hash",
        str(f),
    ]

    # First run should always succeed
    subprocess.check_call(cmd)

    manifest_path = out_dir / "manifest.json"
    assert manifest_path.exists()

    m = json.loads(manifest_path.read_text())
    assert m["manifest_version"] == "1.0"
    assert "environment" in m
    assert m["config"]["script"] == "scripts/write_run_manifest.py"
    assert m["config"]["config"]["subject"] == "subj01"

    # Second run: requires --force
    if force:
        subprocess.check_call(cmd + ["--force"])
        assert manifest_path.exists()
    else:
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.check_call(cmd)
