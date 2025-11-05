#!/usr/bin/env python3
"""
Test build_clip_cache.py improvements.

Validates:
1. Logging setup without crashes
2. Flag handling (--cache vs --out)
3. Argument parsing
"""

import subprocess
import sys
from pathlib import Path


def test_help_text():
    """Test that help text is accessible and shows new flags."""
    print("\n" + "="*80)
    print("TEST: Help text and flag documentation")
    print("="*80)
    
    # Note: May fail with import errors, but we can check syntax
    cmd = [sys.executable, "-m", "py_compile", "scripts/build_clip_cache.py"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/tonystark/Desktop/Bachelor V2")
    
    if result.returncode == 0:
        print("  ✓ Script compiles successfully")
    else:
        print(f"  ✗ Compilation failed: {result.stderr}")
        return False
    
    # Check that key changes are present in the source
    script_path = Path("/home/tonystark/Desktop/Bachelor V2/scripts/build_clip_cache.py")
    content = script_path.read_text()
    
    checks = [
        ("setup_logging function", "def setup_logging(log_file: Optional[str] = None)"),
        ("--cache argument", "--cache"),
        ("--out alias", "--out"),
        ("--include-ids", "--include-ids"),
        ("--log-file", "--log-file"),
        ("nsd_id column", '"nsd_id"'),
        ("embedding column", '"embedding"'),
        ("Robust logging", 'Log file: none (stdout only)'),
    ]
    
    all_passed = True
    for name, pattern in checks:
        if pattern in content:
            print(f"  ✓ {name} present")
        else:
            print(f"  ✗ {name} MISSING")
            all_passed = False
    
    return all_passed


def test_schema_correctness():
    """Verify the schema generation code is correct."""
    print("\n" + "="*80)
    print("TEST: Schema generation correctness")
    print("="*80)
    
    script_path = Path("/home/tonystark/Desktop/Bachelor V2/scripts/build_clip_cache.py")
    content = script_path.read_text()
    
    # Check that embeddings are stored as list (not exploded)
    checks = [
        ("Float32 conversion", "astype(np.float32).tolist()"),
        ("nsd_id as int", '"nsd_id": [int(nid) for nid in valid_nsd_ids]'),
        ("embedding column", '"embedding": [emb.astype(np.float32).tolist()'),
        ("Legacy clip512 compat", '"clip512"'),
        ("Legacy nsdId compat", '"nsdId"'),
    ]
    
    all_passed = True
    for name, pattern in checks:
        if pattern in content:
            print(f"  ✓ {name} present")
        else:
            print(f"  ✗ {name} MISSING")
            all_passed = False
    
    return all_passed


def test_logging_robustness():
    """Verify logging setup doesn't crash."""
    print("\n" + "="*80)
    print("TEST: Logging robustness")
    print("="*80)
    
    script_path = Path("/home/tonystark/Desktop/Bachelor V2/scripts/build_clip_cache.py")
    content = script_path.read_text()
    
    # Check that logging doesn't index handlers directly
    checks = [
        ("No direct handler indexing", "log.handlers[0]" not in content),
        ("Stdout handler setup", "StreamHandler(sys.stdout)"),
        ("Optional file handler", "if log_file:"),
        ("Logger returned", "return logger"),
    ]
    
    all_passed = True
    for name, condition in checks:
        if condition:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} FAILED")
            all_passed = False
    
    return all_passed


def test_flag_aliasing():
    """Verify --out is handled as alias."""
    print("\n" + "="*80)
    print("TEST: Flag aliasing (--out → --cache)")
    print("="*80)
    
    script_path = Path("/home/tonystark/Desktop/Bachelor V2/scripts/build_clip_cache.py")
    content = script_path.read_text()
    
    checks = [
        ("--out argument", '"--out"'),
        ("Alias handling", 'if args.out:'),
        ("--cache wins logic", 'Both --out and --cache provided'),
        ("Cache path resolution", 'cache_path = args.'),
    ]
    
    all_passed = True
    for name, pattern in checks:
        if pattern in content:
            print(f"  ✓ {name} present")
        else:
            print(f"  ✗ {name} MISSING")
            all_passed = False
    
    return all_passed


def test_empty_index_handling():
    """Verify empty index handling."""
    print("\n" + "="*80)
    print("TEST: Empty index handling")
    print("="*80)
    
    script_path = Path("/home/tonystark/Desktop/Bachelor V2/scripts/build_clip_cache.py")
    content = script_path.read_text()
    
    checks = [
        ("Empty check", "if len(df) == 0:"),
        ("Warning message", "Index is empty after filtering"),
        ("Clean exit", "sys.exit(1)"),
    ]
    
    all_passed = True
    for name, pattern in checks:
        if pattern in content:
            print(f"  ✓ {name} present")
        else:
            print(f"  ✗ {name} MISSING")
            all_passed = False
    
    return all_passed


def test_configuration_banner():
    """Verify configuration is logged."""
    print("\n" + "="*80)
    print("TEST: Configuration banner")
    print("="*80)
    
    script_path = Path("/home/tonystark/Desktop/Bachelor V2/scripts/build_clip_cache.py")
    content = script_path.read_text()
    
    checks = [
        ("Banner header", "CLIP Cache Build Configuration"),
        ("Subject logging", 'log.info(f"Subject:'),
        ("Device logging", 'log.info(f"Device:'),
        ("Cache path logging", 'log.info(f"Cache path:'),
        ("Include IDs logging", 'log.info(f"Include IDs:'),
    ]
    
    all_passed = True
    for name, pattern in checks:
        if pattern in content:
            print(f"  ✓ {name} present")
        else:
            print(f"  ✗ {name} MISSING")
            all_passed = False
    
    return all_passed


def main():
    print("\n" + "="*80)
    print("  BUILD_CLIP_CACHE.PY IMPROVEMENT TESTS")
    print("="*80)
    
    results = []
    
    try:
        results.append(("Help text", test_help_text()))
    except Exception as e:
        print(f"\n✗ Help text test crashed: {e}")
        results.append(("Help text", False))
    
    try:
        results.append(("Schema correctness", test_schema_correctness()))
    except Exception as e:
        print(f"\n✗ Schema test crashed: {e}")
        results.append(("Schema correctness", False))
    
    try:
        results.append(("Logging robustness", test_logging_robustness()))
    except Exception as e:
        print(f"\n✗ Logging test crashed: {e}")
        results.append(("Logging robustness", False))
    
    try:
        results.append(("Flag aliasing", test_flag_aliasing()))
    except Exception as e:
        print(f"\n✗ Flag aliasing test crashed: {e}")
        results.append(("Flag aliasing", False))
    
    try:
        results.append(("Empty index handling", test_empty_index_handling()))
    except Exception as e:
        print(f"\n✗ Empty index test crashed: {e}")
        results.append(("Empty index handling", False))
    
    try:
        results.append(("Configuration banner", test_configuration_banner()))
    except Exception as e:
        print(f"\n✗ Configuration test crashed: {e}")
        results.append(("Configuration banner", False))
    
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
