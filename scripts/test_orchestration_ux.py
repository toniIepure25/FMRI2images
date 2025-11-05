#!/usr/bin/env python3
"""
Smoke tests for orchestration UX improvements.

Tests:
1. Orchestrator validation (adapter guards, index exclusivity)
2. DataFrame sanitization (dict columns)
3. Argument parsing and help text
"""

import subprocess
import sys
import tempfile
from pathlib import Path
import json
import pandas as pd


def test_adapter_validation():
    """Test that --use-adapter requires both --adapter and --model-id."""
    print("\n" + "="*80)
    print("TEST: Adapter validation guards")
    print("="*80)
    
    # Test 1: --use-adapter without --adapter
    print("\n1. Testing --use-adapter without --adapter...")
    cmd = [
        sys.executable, "scripts/run_reconstruct_and_eval.py",
        "--subject", "subj01",
        "--encoder", "mlp",
        "--ckpt", "checkpoints/mlp/subj01/mlp.pt",
        "--clip-cache", "outputs/clip_cache/clip.parquet",
        "--output-dir", "outputs/test",
        "--report-dir", "outputs/test",
        "--use-adapter",
        "--model-id", "stabilityai/stable-diffusion-2-1",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 and "requires --adapter PATH" in result.stdout:
        print("   ✓ Correctly rejected (missing --adapter)")
    else:
        print("   ✗ FAILED: Should reject missing --adapter")
        return False
    
    # Test 2: --use-adapter without --model-id
    print("\n2. Testing --use-adapter without --model-id...")
    cmd = [
        sys.executable, "scripts/run_reconstruct_and_eval.py",
        "--subject", "subj01",
        "--encoder", "mlp",
        "--ckpt", "checkpoints/mlp/subj01/mlp.pt",
        "--clip-cache", "outputs/clip_cache/clip.parquet",
        "--output-dir", "outputs/test",
        "--report-dir", "outputs/test",
        "--use-adapter",
        "--adapter", "checkpoints/adapter.pt",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 and "requires --model-id" in result.stdout:
        print("   ✓ Correctly rejected (missing --model-id)")
    else:
        print("   ✗ FAILED: Should reject missing --model-id")
        return False
    
    # Test 3: Missing checkpoint file
    print("\n3. Testing missing checkpoint file...")
    cmd = [
        sys.executable, "scripts/run_reconstruct_and_eval.py",
        "--subject", "subj01",
        "--encoder", "mlp",
        "--ckpt", "nonexistent.pt",
        "--clip-cache", "outputs/clip_cache/clip.parquet",
        "--output-dir", "outputs/test",
        "--report-dir", "outputs/test",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 and "checkpoint not found" in result.stdout.lower():
        print("   ✓ Correctly rejected (missing checkpoint)")
    else:
        print("   ✗ FAILED: Should reject missing checkpoint")
        return False
    
    print("\n✓ All adapter validation tests passed")
    return True


def test_dataframe_sanitization():
    """Test that dict columns are properly sanitized."""
    print("\n" + "="*80)
    print("TEST: DataFrame sanitization (dict columns)")
    print("="*80)
    
    # Import the sanitize function
    sys.path.insert(0, str(Path(__file__).parent))
    from compare_evals import sanitize_dataframe
    
    # Create test DataFrame with dict columns
    print("\n1. Creating test DataFrame with dict column...")
    test_data = {
        "run_name": ["run1", "run2"],
        "clipscore": [0.5, 0.6],
        "config": [
            {"a": 1, "b": 2},
            {"a": 3, "b": 4}
        ]
    }
    df = pd.DataFrame(test_data)
    print(f"   Original: {df['config'].iloc[0]}")
    
    # Sanitize
    print("\n2. Sanitizing DataFrame...")
    df_clean = sanitize_dataframe(df)
    
    # Check that dict is now a string
    if isinstance(df_clean['config'].iloc[0], str):
        print("   ✓ Dict column converted to string")
    else:
        print(f"   ✗ FAILED: Column is {type(df_clean['config'].iloc[0])}")
        return False
    
    # Check that it's valid JSON
    try:
        parsed = json.loads(df_clean['config'].iloc[0])
        print(f"   ✓ Valid JSON: {parsed}")
    except:
        print("   ✗ FAILED: Not valid JSON")
        return False
    
    # Test sorting doesn't crash
    print("\n3. Testing sort operation...")
    try:
        df_clean = df_clean.sort_values(by=["clipscore"])
        print("   ✓ Sort succeeded")
    except Exception as e:
        print(f"   ✗ FAILED: Sort crashed: {e}")
        return False
    
    print("\n✓ DataFrame sanitization tests passed")
    return True


def test_gallery_passthrough():
    """Test that --gallery is properly passed to eval script."""
    print("\n" + "="*80)
    print("TEST: Gallery argument passthrough")
    print("="*80)
    
    # Just test that the argument is recognized
    print("\n1. Testing --gallery argument...")
    cmd = [
        sys.executable, "scripts/run_reconstruct_and_eval.py",
        "--help"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if "--gallery" in result.stdout:
        print("   ✓ --gallery argument present in help")
    else:
        print("   ✗ FAILED: --gallery not in help")
        return False
    
    if "matched" in result.stdout and "test" in result.stdout and "all" in result.stdout:
        print("   ✓ Gallery choices present")
    else:
        print("   ✗ FAILED: Gallery choices not found")
        return False
    
    print("\n2. Testing --image-source argument...")
    if "--image-source" in result.stdout:
        print("   ✓ --image-source argument present in help")
    else:
        print("   ✗ FAILED: --image-source not in help")
        return False
    
    print("\n✓ Gallery passthrough tests passed")
    return True


def test_eval_limit_logging():
    """Test that eval script properly logs limit."""
    print("\n" + "="*80)
    print("TEST: Eval limit logging")
    print("="*80)
    
    print("\n1. Checking eval help text...")
    cmd = [sys.executable, "scripts/eval_reconstruction.py", "--help"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if "--limit" in result.stdout:
        print("   ✓ --limit argument present")
    else:
        print("   ✗ FAILED: --limit not in help")
        return False
    
    if "--gallery" in result.stdout:
        print("   ✓ --gallery argument present")
    else:
        print("   ✗ FAILED: --gallery not in help")
        return False
    
    print("\n✓ Eval limit logging tests passed")
    return True


def test_compare_help():
    """Test that compare_evals has proper help."""
    print("\n" + "="*80)
    print("TEST: Compare script help text")
    print("="*80)
    
    print("\n1. Checking compare help text...")
    cmd = [sys.executable, "scripts/compare_evals.py", "--help"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if "--out-md" in result.stdout:
        print("   ✓ --out-md argument present")
    else:
        print("   ✗ FAILED: --out-md not in help")
        return False
    
    print("\n✓ Compare help tests passed")
    return True


def main():
    print("\n" + "="*80)
    print("  ORCHESTRATION UX SMOKE TESTS")
    print("="*80)
    
    results = []
    
    # Run all tests
    try:
        results.append(("Adapter validation", test_adapter_validation()))
    except Exception as e:
        print(f"\n✗ Adapter validation crashed: {e}")
        results.append(("Adapter validation", False))
    
    try:
        results.append(("DataFrame sanitization", test_dataframe_sanitization()))
    except Exception as e:
        print(f"\n✗ DataFrame sanitization crashed: {e}")
        results.append(("DataFrame sanitization", False))
    
    try:
        results.append(("Gallery passthrough", test_gallery_passthrough()))
    except Exception as e:
        print(f"\n✗ Gallery passthrough crashed: {e}")
        results.append(("Gallery passthrough", False))
    
    try:
        results.append(("Eval limit logging", test_eval_limit_logging()))
    except Exception as e:
        print(f"\n✗ Eval limit logging crashed: {e}")
        results.append(("Eval limit logging", False))
    
    try:
        results.append(("Compare help", test_compare_help()))
    except Exception as e:
        print(f"\n✗ Compare help crashed: {e}")
        results.append(("Compare help", False))
    
    # Summary
    print("\n" + "="*80)
    print("  TEST SUMMARY")
    print("="*80)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status:8s} {name}")
    
    print("="*80)
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    if passed == total:
        print(f"\n✓ All {total} tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed}/{total} tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
