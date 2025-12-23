"""Tests for reproducibility manifest utilities."""
import json
import pytest
from pathlib import Path

from fmri2img.utils.manifest import (
    gather_env_info,
    hash_file,
    hash_dir_listing,
    write_manifest,
    load_manifest,
    compare_manifests
)


def test_gather_env_info():
    """Test environment information gathering."""
    env = gather_env_info()
    
    # Check required keys
    assert "timestamp" in env
    assert "hostname" in env
    assert "python_version" in env
    assert "python_executable" in env
    assert "platform" in env
    
    # Check package versions
    assert "numpy_version" in env
    assert "pandas_version" in env
    
    # Git info should be present (or marked as not a repo)
    assert "git_commit" in env


def test_hash_file(tmp_path):
    """Test file hashing."""
    # Create test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello, World!")
    
    # Compute hash
    hash1 = hash_file(test_file)
    assert len(hash1) == 64  # SHA256 hex digest length
    
    # Same content should give same hash
    hash2 = hash_file(test_file)
    assert hash1 == hash2
    
    # Different content should give different hash
    test_file.write_text("Different content")
    hash3 = hash_file(test_file)
    assert hash1 != hash3


def test_hash_file_not_found():
    """Test hash_file with non-existent file."""
    with pytest.raises(FileNotFoundError):
        hash_file("nonexistent_file.txt")


def test_hash_dir_listing(tmp_path):
    """Test directory hashing."""
    # Create test files
    (tmp_path / "file1.txt").write_text("Content 1")
    (tmp_path / "file2.txt").write_text("Content 2")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "file3.txt").write_text("Content 3")
    
    # Hash all files
    hashes = hash_dir_listing(tmp_path, "*.txt")
    
    assert "file1.txt" in hashes
    assert "file2.txt" in hashes
    assert "subdir/file3.txt" in hashes
    assert len(hashes) == 3
    
    # All hashes should be valid SHA256
    for hash_val in hashes.values():
        assert len(hash_val) == 64


def test_write_and_load_manifest(tmp_path):
    """Test writing and loading manifests."""
    manifest_path = tmp_path / "manifest.json"
    
    config = {
        "subject": "subj01",
        "encoder": "mlp",
        "learning_rate": 0.001
    }
    
    cli_args = ["python", "train.py", "--subject", "subj01"]
    
    env_info = gather_env_info()
    
    input_hashes = {
        "checkpoint": "abc123def456",
        "data": "789ghi012jkl"
    }
    
    # Write manifest
    write_manifest(
        manifest_path,
        config_dict=config,
        cli_args=cli_args,
        env_info=env_info,
        input_hashes=input_hashes
    )
    
    assert manifest_path.exists()
    
    # Load and verify
    manifest = load_manifest(manifest_path)
    
    assert manifest["config"] == config
    assert manifest["cli_args"] == cli_args
    assert manifest["environment"]["python_version"] == env_info["python_version"]
    assert manifest["input_hashes"] == input_hashes


def test_manifest_without_optional_fields(tmp_path):
    """Test manifest creation with minimal fields."""
    manifest_path = tmp_path / "minimal_manifest.json"
    
    config = {"test": "value"}
    
    # Write with only config (no cli_args, no input_hashes)
    write_manifest(manifest_path, config_dict=config)
    
    assert manifest_path.exists()
    
    # Should still have environment info (auto-gathered)
    manifest = load_manifest(manifest_path)
    assert manifest["config"] == config
    assert "environment" in manifest


def test_compare_manifests(tmp_path):
    """Test manifest comparison."""
    # Create two manifests with differences
    manifest1_path = tmp_path / "manifest1.json"
    manifest2_path = tmp_path / "manifest2.json"
    
    env_info = gather_env_info()
    
    config1 = {"subject": "subj01", "lr": 0.001}
    config2 = {"subject": "subj02", "lr": 0.001}
    
    write_manifest(manifest1_path, config_dict=config1, env_info=env_info)
    write_manifest(manifest2_path, config_dict=config2, env_info=env_info)
    
    # Compare
    diff = compare_manifests(manifest1_path, manifest2_path)
    
    assert diff["same_git_commit"] == True  # Same env
    assert diff["same_python_version"] == True
    assert diff["same_torch_version"] == True
    
    # Config should differ
    assert "subject" in diff["config_differences"]
    assert diff["config_differences"]["subject"]["manifest1"] == "subj01"
    assert diff["config_differences"]["subject"]["manifest2"] == "subj02"


def test_load_manifest_not_found():
    """Test loading non-existent manifest."""
    with pytest.raises(FileNotFoundError):
        load_manifest("nonexistent_manifest.json")
