"""
Reproducibility Manifests for fMRI Reconstruction Experiments
============================================================

Track environment, configuration, and input data for full reproducibility.
Every evaluation run should produce a manifest.json capturing:
- Python, PyTorch, CUDA versions
- Git commit hash
- Configuration parameters
- Input data checksums
- Execution timestamp

Scientific Rationale:
- Paper-grade experiments require full reproducibility
- Enables debugging "runs that worked 2 weeks ago"
- Essential for artifact submission with published papers

Usage:
    >>> from fmri2img.utils.manifest import gather_env_info, write_manifest
    >>> 
    >>> # At start of experiment
    >>> env_info = gather_env_info()
    >>> config = {"subject": "subj01", "encoder": "mlp"}
    >>> cli_args = sys.argv
    >>> 
    >>> # After run completes
    >>> write_manifest(
    ...     output_dir / "manifest.json",
    ...     config_dict=config,
    ...     cli_args=cli_args,
    ...     env_info=env_info
    ... )
"""

import hashlib
import json
import logging
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


def gather_env_info() -> Dict[str, Any]:
    """
    Gather comprehensive environment information for reproducibility.
    
    Collects:
    - Python version and executable path
    - PyTorch version and CUDA availability
    - Key package versions (numpy, pandas, PIL, etc.)
    - Git commit hash (if in a git repo)
    - Hostname and timestamp
    - System platform info
    
    Returns:
        Dictionary with environment information
        
    Example:
        >>> env = gather_env_info()
        >>> print(f"Python: {env['python_version']}")
        >>> print(f"Git commit: {env['git_commit']}")
        Python: 3.10.12
        Git commit: a1b2c3d4e5f6
    """
    env_info = {
        "timestamp": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "user": os.environ.get("USER", "unknown"),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
    }
    
    # Python packages
    try:
        import numpy as np
        env_info["numpy_version"] = np.__version__
    except ImportError:
        env_info["numpy_version"] = "not installed"
    
    try:
        import pandas as pd
        env_info["pandas_version"] = pd.__version__
    except ImportError:
        env_info["pandas_version"] = "not installed"
    
    try:
        import torch
        env_info["torch_version"] = torch.__version__
        env_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env_info["cuda_version"] = torch.version.cuda
            env_info["cudnn_version"] = torch.backends.cudnn.version()
            env_info["gpu_count"] = torch.cuda.device_count()
            env_info["gpu_names"] = [
                torch.cuda.get_device_name(i) 
                for i in range(torch.cuda.device_count())
            ]
        else:
            env_info["cuda_version"] = None
            env_info["gpu_count"] = 0
    except ImportError:
        env_info["torch_version"] = "not installed"
        env_info["cuda_available"] = False
    
    try:
        from PIL import Image
        import PIL
        env_info["pillow_version"] = PIL.__version__
    except ImportError:
        env_info["pillow_version"] = "not installed"
    
    try:
        import scipy
        env_info["scipy_version"] = scipy.__version__
    except ImportError:
        env_info["scipy_version"] = "not installed"
    
    # Git information
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        env_info["git_commit"] = git_commit
        
        # Check for uncommitted changes
        git_status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        env_info["git_dirty"] = len(git_status) > 0
        
        # Get current branch
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        env_info["git_branch"] = git_branch
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        env_info["git_commit"] = "not a git repo"
        env_info["git_dirty"] = None
        env_info["git_branch"] = None
    
    return env_info


def hash_file(file_path: Union[str, Path], algorithm: str = "sha256") -> str:
    """
    Compute hash of a file for data provenance tracking.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm (default: sha256)
        
    Returns:
        Hex digest of file hash
        
    Example:
        >>> hash_val = hash_file("data/subject01.nii.gz")
        >>> print(f"Data hash: {hash_val[:16]}...")
        Data hash: a1b2c3d4e5f6g7h8...
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    hash_obj = hashlib.new(algorithm)
    
    # Read in chunks for large files
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def hash_dir_listing(
    dir_path: Union[str, Path],
    pattern: str = "*",
    algorithm: str = "sha256"
) -> Dict[str, str]:
    """
    Compute hashes for all files matching pattern in directory.
    
    Useful for tracking input datasets and cache directories.
    
    Args:
        dir_path: Path to directory
        pattern: Glob pattern for files (default: "*")
        algorithm: Hash algorithm (default: sha256)
        
    Returns:
        Dictionary mapping relative paths to hashes
        
    Example:
        >>> hashes = hash_dir_listing("checkpoints/mlp/subj01", "*.pt")
        >>> for path, hash_val in hashes.items():
        ...     print(f"{path}: {hash_val[:16]}...")
    """
    dir_path = Path(dir_path)
    
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    
    hashes = {}
    
    for file_path in sorted(dir_path.rglob(pattern)):
        if file_path.is_file():
            rel_path = file_path.relative_to(dir_path)
            try:
                hashes[str(rel_path)] = hash_file(file_path, algorithm)
            except Exception as e:
                logger.warning(f"Failed to hash {rel_path}: {e}")
                hashes[str(rel_path)] = f"error: {e}"
    
    return hashes


def write_manifest(
    output_path: Union[str, Path],
    config_dict: Dict[str, Any],
    cli_args: Optional[List[str]] = None,
    env_info: Optional[Dict[str, Any]] = None,
    input_hashes: Optional[Dict[str, str]] = None,
    additional_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Write a comprehensive manifest file for reproducibility.
    
    The manifest captures everything needed to reproduce an experiment:
    - Configuration parameters
    - CLI arguments
    - Environment (Python, PyTorch, git commit, etc.)
    - Input data hashes
    - Any additional metadata
    
    Args:
        output_path: Where to write manifest.json
        config_dict: Configuration dictionary (e.g., from YAML)
        cli_args: Command-line arguments (sys.argv)
        env_info: Environment info from gather_env_info()
        input_hashes: Dictionary of input file hashes
        additional_info: Any additional metadata
        
    Example:
        >>> write_manifest(
        ...     "outputs/eval/manifest.json",
        ...     config_dict={"subject": "subj01", "encoder": "mlp"},
        ...     cli_args=sys.argv,
        ...     env_info=gather_env_info(),
        ...     input_hashes={"checkpoint": hash_file("ckpts/mlp.pt")}
        ... )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Gather env info if not provided
    if env_info is None:
        env_info = gather_env_info()
    
    manifest = {
        "manifest_version": "1.0",
        "created": datetime.now().isoformat(),
        "config": config_dict,
        "environment": env_info,
    }
    
    if cli_args is not None:
        manifest["cli_args"] = cli_args
    
    if input_hashes is not None:
        manifest["input_hashes"] = input_hashes
    
    if additional_info is not None:
        manifest["additional_info"] = additional_info
    
    # Write with pretty formatting
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    
    logger.info(f"Wrote manifest to {output_path}")


def load_manifest(manifest_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a manifest file.
    
    Args:
        manifest_path: Path to manifest.json
        
    Returns:
        Manifest dictionary
        
    Example:
        >>> manifest = load_manifest("outputs/eval/manifest.json")
        >>> print(f"Run on: {manifest['environment']['hostname']}")
        >>> print(f"Git commit: {manifest['environment']['git_commit'][:8]}")
    """
    manifest_path = Path(manifest_path)
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    with open(manifest_path) as f:
        return json.load(f)


def compare_manifests(
    manifest1_path: Union[str, Path],
    manifest2_path: Union[str, Path]
) -> Dict[str, Any]:
    """
    Compare two manifests and report differences.
    
    Useful for understanding why two runs produced different results.
    
    Args:
        manifest1_path: Path to first manifest
        manifest2_path: Path to second manifest
        
    Returns:
        Dictionary with comparison results:
        - same_git_commit: bool
        - same_python_version: bool
        - same_torch_version: bool
        - config_differences: dict
        - input_hash_differences: dict
        
    Example:
        >>> diff = compare_manifests("run1/manifest.json", "run2/manifest.json")
        >>> if not diff["same_git_commit"]:
        ...     print("Warning: Different git commits!")
        >>> if diff["config_differences"]:
        ...     print(f"Config differs: {diff['config_differences']}")
    """
    m1 = load_manifest(manifest1_path)
    m2 = load_manifest(manifest2_path)
    
    comparison = {
        "same_git_commit": (
            m1["environment"].get("git_commit") == m2["environment"].get("git_commit")
        ),
        "same_python_version": (
            m1["environment"].get("python_version") == m2["environment"].get("python_version")
        ),
        "same_torch_version": (
            m1["environment"].get("torch_version") == m2["environment"].get("torch_version")
        ),
        "same_hostname": (
            m1["environment"].get("hostname") == m2["environment"].get("hostname")
        ),
    }
    
    # Compare configs
    config_diff = {}
    all_keys = set(m1.get("config", {}).keys()) | set(m2.get("config", {}).keys())
    for key in all_keys:
        val1 = m1.get("config", {}).get(key)
        val2 = m2.get("config", {}).get(key)
        if val1 != val2:
            config_diff[key] = {"manifest1": val1, "manifest2": val2}
    comparison["config_differences"] = config_diff
    
    # Compare input hashes
    hash_diff = {}
    all_inputs = set(m1.get("input_hashes", {}).keys()) | set(m2.get("input_hashes", {}).keys())
    for key in all_inputs:
        hash1 = m1.get("input_hashes", {}).get(key)
        hash2 = m2.get("input_hashes", {}).get(key)
        if hash1 != hash2:
            hash_diff[key] = {"manifest1": hash1, "manifest2": hash2}
    comparison["input_hash_differences"] = hash_diff
    
    return comparison
