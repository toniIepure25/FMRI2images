#!/usr/bin/env python3
"""
Main training script wrapper - forwards to train_unified.py

This is a convenience wrapper that calls the main training script.
For more control, use train_unified.py directly.

Usage:
    python scripts/train.py --config configs/experiments/exp0_baseline.yaml
"""

import sys
import subprocess
from pathlib import Path

if __name__ == "__main__":
    # Get the scripts directory
    scripts_dir = Path(__file__).parent
    
    # Forward all arguments to train_unified.py
    train_script = scripts_dir / "train_unified.py"
    
    cmd = [sys.executable, str(train_script)] + sys.argv[1:]
    
    try:
        sys.exit(subprocess.call(cmd))
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(130)
