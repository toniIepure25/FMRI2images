#!/usr/bin/env python3
"""
Main evaluation script wrapper - forwards to evaluate_experiment.py

This is a convenience wrapper that calls the main evaluation script.
For more control, use evaluate_experiment.py directly.

Usage:
    python scripts/evaluate.py --checkpoint path/to/checkpoint.pt
"""

import sys
import subprocess
from pathlib import Path

if __name__ == "__main__":
    # Get the scripts directory
    scripts_dir = Path(__file__).parent
    
    # Forward all arguments to evaluate_experiment.py
    eval_script = scripts_dir / "evaluate_experiment.py"
    
    cmd = [sys.executable, str(eval_script)] + sys.argv[1:]
    
    try:
        sys.exit(subprocess.call(cmd))
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(130)
