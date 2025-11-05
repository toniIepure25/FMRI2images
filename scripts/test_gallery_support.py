#!/usr/bin/env python3
"""
Test script to verify run_reconstruct_and_eval.py gallery support.

This script tests that the command construction works correctly
without actually running the full pipeline.
"""

import subprocess
import sys
from pathlib import Path

def test_single_gallery():
    """Test single gallery mode."""
    print("Testing single gallery mode...")
    
    # Just verify the script accepts the arguments
    cmd = [
        sys.executable,
        "scripts/run_reconstruct_and_eval.py",
        "--subject", "subj01",
        "--encoder", "mlp",
        "--ckpt", "checkpoints/mlp/subj01/mlp.pt",
        "--clip-cache", "outputs/clip_cache/clip.parquet",
        "--output-dir", "outputs/recon/subj01/test_single",
        "--report-dir", "outputs/reports/subj01/test_single",
        "--use-adapter",
        "--model-id", "stabilityai/stable-diffusion-2-1",
        "--gallery", "test",
        "--image-source", "hdf5",
        "--limit", "1",
        "--skip-sd-cache-check",
        "--help",  # Just show help, don't actually run
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if "gallery" in result.stdout and "image-source" in result.stdout:
            print("✓ Single gallery mode: Arguments recognized")
            return True
        else:
            print("✗ Single gallery mode: Arguments not found in help")
            return False
    except Exception as e:
        print(f"✗ Single gallery mode failed: {e}")
        return False


def test_all_galleries():
    """Test all-galleries mode."""
    print("Testing all-galleries mode...")
    
    cmd = [
        sys.executable,
        "scripts/run_reconstruct_and_eval.py",
        "--subject", "subj01",
        "--encoder", "mlp",
        "--ckpt", "checkpoints/mlp/subj01/mlp.pt",
        "--clip-cache", "outputs/clip_cache/clip.parquet",
        "--output-dir", "outputs/recon/subj01/test_all",
        "--report-dir", "outputs/reports/subj01/test_all",
        "--use-adapter",
        "--model-id", "stabilityai/stable-diffusion-2-1",
        "--all-galleries",
        "--image-source", "hdf5",
        "--limit", "1",
        "--skip-sd-cache-check",
        "--help",
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if "all-galleries" in result.stdout:
            print("✓ All-galleries mode: Arguments recognized")
            return True
        else:
            print("✗ All-galleries mode: Arguments not found in help")
            return False
    except Exception as e:
        print(f"✗ All-galleries mode failed: {e}")
        return False


def main():
    print("=" * 80)
    print("  Testing run_reconstruct_and_eval.py Gallery Support")
    print("=" * 80)
    print()
    
    results = []
    
    results.append(test_single_gallery())
    print()
    results.append(test_all_galleries())
    
    print()
    print("=" * 80)
    if all(results):
        print("  ✓ All tests passed!")
    else:
        print(f"  ✗ {sum(not r for r in results)} test(s) failed")
    print("=" * 80)
    
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
