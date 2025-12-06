#!/usr/bin/env python3
"""
Backward-compatible wrapper for compare_evals.py
Calls: python -m fmri2img.eval.compare_evals
"""
import sys
from pathlib import Path
import runpy

# Add src directory to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

if __name__ == "__main__":
    sys.argv[0] = "python -m fmri2img.eval.compare_evals"
    runpy.run_module("fmri2img.eval.compare_evals", run_name="__main__")
