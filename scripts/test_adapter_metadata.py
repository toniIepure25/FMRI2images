#!/usr/bin/env python3
"""
Test adapter metadata saving and loading with robust fallback handling.
"""

import sys
import tempfile
import torch
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fmri2img.models.clip_adapter import CLIPAdapter, load_adapter


def test_save_with_metadata():
    """Test saving adapter with full metadata."""
    print("=" * 80)
    print("TEST 1: Save adapter with metadata")
    print("=" * 80)
    
    adapter = CLIPAdapter(in_dim=512, out_dim=1024, use_layernorm=True)
    
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        temp_path = f.name
    
    try:
        metadata = {
            "subject": "subj01",
            "model_id": "stabilityai/stable-diffusion-2-1",
            "input_dim": 512,
            "target_dim": 1024,
            "created_at": datetime.now().isoformat(),
            "repo_version": "0.1.0",
            "test_cosine": 0.85,
        }
        
        adapter.save(temp_path, metadata)
        print(f"✅ Saved adapter to {temp_path}")
        
        # Load and verify
        loaded_adapter, loaded_metadata = load_adapter(temp_path)
        print(f"✅ Loaded adapter successfully")
        
        # Check required fields
        assert loaded_metadata["subject"] == "subj01", f"subject mismatch: {loaded_metadata['subject']}"
        assert loaded_metadata["model_id"] == "stabilityai/stable-diffusion-2-1", f"model_id mismatch"
        assert loaded_metadata["input_dim"] == 512, f"input_dim mismatch"
        assert loaded_metadata["target_dim"] == 1024, f"target_dim mismatch"
        assert "created_at" in loaded_metadata, "missing created_at"
        assert "repo_version" in loaded_metadata, "missing repo_version"
        
        print(f"✅ All metadata fields verified:")
        for key in ["subject", "model_id", "input_dim", "target_dim", "created_at", "repo_version"]:
            print(f"   {key}: {loaded_metadata[key]}")
        
        # Test forward pass
        x = torch.randn(4, 512)
        with torch.no_grad():
            y = loaded_adapter(x)
        assert y.shape == (4, 1024), f"output shape mismatch: {y.shape}"
        print(f"✅ Forward pass works: {x.shape} → {y.shape}")
        
        print("✅ TEST 1 PASSED\n")
        return True
        
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_load_legacy_checkpoint():
    """Test loading legacy checkpoint (raw state_dict)."""
    print("=" * 80)
    print("TEST 2: Load legacy checkpoint (raw state_dict)")
    print("=" * 80)
    
    adapter = CLIPAdapter(in_dim=512, out_dim=1024, use_layernorm=False)
    
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        temp_path = f.name
    
    try:
        # Save raw state_dict (legacy format)
        torch.save(adapter.state_dict(), temp_path)
        print(f"✅ Saved legacy checkpoint (raw state_dict)")
        
        # Load with new loader (should auto-repair)
        loaded_adapter, loaded_metadata = load_adapter(temp_path)
        print(f"✅ Loaded legacy checkpoint successfully")
        
        # Check default metadata was added
        assert loaded_metadata["subject"] == "unknown", f"subject should be 'unknown': {loaded_metadata['subject']}"
        assert loaded_metadata["model_id"] == "stabilityai/stable-diffusion-2-1", f"model_id mismatch"
        assert loaded_metadata["input_dim"] == 512, f"input_dim mismatch"
        assert loaded_metadata["target_dim"] == 1024, f"target_dim mismatch"
        
        print(f"✅ Metadata auto-repaired with defaults:")
        for key in ["subject", "model_id", "input_dim", "target_dim"]:
            print(f"   {key}: {loaded_metadata[key]}")
        
        # Test forward pass
        x = torch.randn(4, 512)
        with torch.no_grad():
            y = loaded_adapter(x)
        assert y.shape == (4, 1024), f"output shape mismatch: {y.shape}"
        print(f"✅ Forward pass works: {x.shape} → {y.shape}")
        
        print("✅ TEST 2 PASSED\n")
        return True
        
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_load_missing_file():
    """Test loading non-existent checkpoint."""
    print("=" * 80)
    print("TEST 3: Load non-existent checkpoint")
    print("=" * 80)
    
    try:
        load_adapter("/tmp/nonexistent_adapter.pt")
        print("❌ Should have raised FileNotFoundError")
        return False
    except FileNotFoundError as e:
        print(f"✅ Correctly raised FileNotFoundError:")
        print(f"   {e}")
        print("✅ TEST 3 PASSED\n")
        return True


def test_load_old_meta_format():
    """Test loading checkpoint with 'meta' key instead of 'metadata'."""
    print("=" * 80)
    print("TEST 4: Load checkpoint with legacy 'meta' key")
    print("=" * 80)
    
    adapter = CLIPAdapter(in_dim=512, out_dim=768, use_layernorm=True)
    
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        temp_path = f.name
    
    try:
        # Save with legacy 'meta' key
        checkpoint = {
            "state_dict": adapter.state_dict(),
            "meta": {  # Old key name
                "in_dim": 512,
                "out_dim": 768,
                "use_layernorm": True,
                "subject": "subj02",
                "model_id": "runwayml/stable-diffusion-v1-5",
            }
        }
        torch.save(checkpoint, temp_path)
        print(f"✅ Saved checkpoint with legacy 'meta' key")
        
        # Load (should handle both 'meta' and 'metadata')
        loaded_adapter, loaded_metadata = load_adapter(temp_path)
        print(f"✅ Loaded checkpoint successfully")
        
        # Check metadata was read correctly
        assert loaded_metadata["subject"] == "subj02", f"subject mismatch"
        assert loaded_metadata["model_id"] == "runwayml/stable-diffusion-v1-5", f"model_id mismatch"
        assert loaded_metadata.get("input_dim", loaded_metadata.get("in_dim")) == 512, f"input_dim mismatch"
        assert loaded_metadata.get("target_dim", loaded_metadata.get("out_dim")) == 768, f"target_dim mismatch"
        
        print(f"✅ Metadata read from legacy 'meta' key:")
        for key in ["subject", "model_id"]:
            print(f"   {key}: {loaded_metadata[key]}")
        
        # Test forward pass
        x = torch.randn(2, 512)
        with torch.no_grad():
            y = loaded_adapter(x)
        assert y.shape == (2, 768), f"output shape mismatch: {y.shape}"
        print(f"✅ Forward pass works: {x.shape} → {y.shape}")
        
        print("✅ TEST 4 PASSED\n")
        return True
        
    finally:
        Path(temp_path).unlink(missing_ok=True)


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("ADAPTER METADATA TEST SUITE")
    print("=" * 80 + "\n")
    
    tests = [
        test_save_with_metadata,
        test_load_legacy_checkpoint,
        test_load_missing_file,
        test_load_old_meta_format,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ TEST FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()
    
    print("=" * 80)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    if failed == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"❌ {failed} tests failed")
    print("=" * 80)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
