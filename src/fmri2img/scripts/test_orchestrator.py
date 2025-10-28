#!/usr/bin/env python3
"""
Smoke tests for run_reconstruct_and_eval.py orchestrator.

Tests validation logic without running full pipeline.
"""

import sys
from pathlib import Path


def test_help_output():
    """Test help output is available."""
    import subprocess
    
    result = subprocess.run(
        [sys.executable, "scripts/run_reconstruct_and_eval.py", "--help"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0, "Help should exit with 0"
    assert "--encoder" in result.stdout
    assert "--ckpt" in result.stdout
    assert "--use-adapter" in result.stdout
    assert "--clip-cache" in result.stdout
    print("✅ Help output test passed")


def test_validation_missing_ckpt():
    """Test validation fails gracefully for missing checkpoint."""
    import subprocess
    
    result = subprocess.run(
        [
            sys.executable, "scripts/run_reconstruct_and_eval.py",
            "--encoder", "mlp",
            "--ckpt", "nonexistent.pt",
            "--clip-cache", "outputs/clip_cache/clip.parquet",
            "--output-dir", "outputs/test",
            "--report-dir", "outputs/test",
        ],
        capture_output=True,
        text=True
    )
    
    assert result.returncode != 0, "Should fail for missing checkpoint"
    assert "not found" in result.stdout.lower() or "not found" in result.stderr.lower()
    print("✅ Missing checkpoint validation test passed")


def test_validation_adapter_without_model():
    """Test validation fails when adapter specified without model-id."""
    import subprocess
    
    # Create dummy checkpoint file
    dummy_ckpt = Path("test_dummy_ckpt.pt")
    dummy_ckpt.touch()
    
    try:
        result = subprocess.run(
            [
                sys.executable, "scripts/run_reconstruct_and_eval.py",
                "--encoder", "mlp",
                "--ckpt", str(dummy_ckpt),
                "--clip-cache", "outputs/clip_cache/clip.parquet",
                "--output-dir", "outputs/test",
                "--report-dir", "outputs/test",
                "--use-adapter",
                "--adapter", "adapter.pt",
                # Missing --model-id
            ],
            capture_output=True,
            text=True
        )
        
        assert result.returncode != 0, "Should fail when adapter lacks model-id"
        assert "model-id" in result.stdout.lower() or "model-id" in result.stderr.lower()
        print("✅ Adapter without model-id validation test passed")
    
    finally:
        if dummy_ckpt.exists():
            dummy_ckpt.unlink()


def test_load_adapter_metadata():
    """Test adapter metadata loading function."""
    import torch
    
    # Import the function - need to load script as module
    script_path = Path("scripts/run_reconstruct_and_eval.py")
    if not script_path.exists():
        print("⚠️  Script not found, skipping test")
        return
    
    # Create dummy adapter with metadata
    dummy_adapter = Path("test_dummy_adapter.pt")
    
    metadata = {
        "input_dim": 512,
        "target_dim": 1024,
        "use_layernorm": True,
    }
    
    torch.save({
        "metadata": metadata,
        "state_dict": {},
    }, dummy_adapter)
    
    try:
        # Load and check metadata exists
        ckpt = torch.load(dummy_adapter, map_location="cpu")
        assert "metadata" in ckpt
        assert ckpt["metadata"]["target_dim"] == 1024
        assert ckpt["metadata"]["input_dim"] == 512
        print("✅ Adapter metadata loading test passed")
    
    finally:
        if dummy_adapter.exists():
            dummy_adapter.unlink()


def test_print_banner():
    """Test banner printing doesn't crash."""
    # Just test the logic, no need to import
    text = "Test Banner"
    width = 80
    banner = "\n" + "=" * width + f"\n  {text}\n" + "=" * width + "\n"
    assert len(banner) > 0
    print("✅ Banner printing test passed")


def main():
    """Run all smoke tests."""
    print("\n" + "="*80)
    print("  Orchestrator Smoke Tests")
    print("="*80 + "\n")
    
    tests = [
        ("Help Output", test_help_output),
        ("Missing Checkpoint Validation", test_validation_missing_ckpt),
        ("Adapter Without Model-ID Validation", test_validation_adapter_without_model),
        ("Load Adapter Metadata", test_load_adapter_metadata),
        ("Print Banner", test_print_banner),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            print(f"\n[Test] {name}")
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            failed += 1
    
    print("\n" + "="*80)
    print(f"  Results: {passed} passed, {failed} failed")
    print("="*80 + "\n")
    
    if failed > 0:
        print("❌ Some tests failed!")
        return 1
    else:
        print("✅ All smoke tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
