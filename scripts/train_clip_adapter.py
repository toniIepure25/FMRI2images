#!/usr/bin/env python3
"""
Backward-compatible wrapper for train_clip_adapter.py
Calls: python -m fmri2img.training.train_clip_adapter
"""
import sys
from pathlib import Path
import runpy

# Add src directory to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

if __name__ == "__main__":
    sys.argv[0] = "python -m fmri2img.training.train_clip_adapter"
    runpy.run_module("fmri2img.training.train_clip_adapter", run_name="__main__")
