#!/usr/bin/env python3
"""
Image reconstruction script wrapper - forwards to generate_images.py

This is a convenience wrapper that calls the image generation script.
For more control, use generate_images.py directly.

Usage:
    python scripts/reconstruct.py --checkpoint path/to/checkpoint.pt --n_images 100
"""

import sys
import subprocess
from pathlib import Path

if __name__ == "__main__":
    # Get the scripts directory
    scripts_dir = Path(__file__).parent
    
    # Forward all arguments to generate_images.py
    gen_script = scripts_dir / "generate_images.py"
    
    cmd = [sys.executable, str(gen_script)] + sys.argv[1:]
    
    try:
        sys.exit(subprocess.call(cmd))
    except KeyboardInterrupt:
        print("\nReconstruction interrupted by user")
        sys.exit(130)
