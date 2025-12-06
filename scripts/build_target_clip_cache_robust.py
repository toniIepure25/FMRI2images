#!/usr/bin/env python3
"""
Backward-compatible wrapper for build_target_clip_cache_robust.py
Calls: python -m fmri2img.data.build_target_clip_cache_robust
"""
import sys
from pathlib import Path
import runpy

# Add src directory to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

if __name__ == "__main__":
    sys.argv[0] = "python -m fmri2img.data.build_target_clip_cache_robust"
    runpy.run_module("fmri2img.data.build_target_clip_cache_robust", run_name="__main__")
