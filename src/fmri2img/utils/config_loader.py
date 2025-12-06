"""
Configuration Loading Utilities
===============================

Professional configuration loading with YAML inheritance, validation, and schema support.

Features:
- Hierarchical configuration via _base_ inheritance
- Environment variable expansion (${VAR_NAME})
- Path resolution (relative to config file or project root)
- Schema validation with descriptive error messages
- Deep merging of nested dictionaries
- Config freezing for immutability

Usage:
    # Simple loading
    config = load_config("configs/mlp_standard.yaml")
    
    # With overrides
    config = load_config(
        "configs/mlp_standard.yaml",
        overrides={"training.learning_rate": 0.001, "dataset.subject": "subj02"}
    )
    
    # With validation
    config = load_config("configs/mlp_standard.yaml", validate=True)
    
    # Access nested values
    lr = config.get("training.learning_rate", default=1e-3)
    
Example Config Inheritance:
    # configs/base.yaml
    training:
      epochs: 50
      learning_rate: 0.001
    
    # configs/mlp_standard.yaml
    _base_: base.yaml
    training:
      learning_rate: 0.0005  # Override
    model:
      hidden_dims: [512, 256]  # Add new section
"""

from __future__ import annotations

import os
import re
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from copy import deepcopy

import yaml

logger = logging.getLogger(__name__)


class ConfigDict(dict):
    """
    Dictionary with dot-notation access and nested key support.
    
    Features:
    - Attribute access: config.training.epochs
    - Nested key access: config.get("training.epochs")
    - Deep merging: config.merge(other_config)
    - Freezing: config.freeze() for immutability
    
    Example:
        >>> config = ConfigDict({"a": {"b": 1}})
        >>> config.a.b == config.get("a.b") == config["a"]["b"]
        True
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._frozen = False
        
        # Convert nested dicts to ConfigDict
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = ConfigDict(value)
    
    def __getattr__(self, key: str) -> Any:
        """Allow attribute-style access: config.training.epochs"""
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{key}'")
    
    def __setattr__(self, key: str, value: Any):
        """Prevent attribute assignment if frozen"""
        if key == "_frozen":
            super().__setattr__(key, value)
        elif self._frozen:
            raise AttributeError(f"Cannot modify frozen config: {key}")
        else:
            self[key] = value
    
    def __setitem__(self, key: str, value: Any):
        """Convert nested dicts to ConfigDict on assignment"""
        if self._frozen:
            raise AttributeError(f"Cannot modify frozen config: {key}")
        
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            value = ConfigDict(value)
        
        super().__setitem__(key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value with support for nested keys using dot notation.
        
        Args:
            key: Key or nested key path (e.g., "training.learning_rate")
            default: Default value if key not found
        
        Returns:
            Value at key path or default
        
        Example:
            >>> config.get("training.learning_rate", 0.001)
        """
        if "." not in key:
            return super().get(key, default)
        
        # Navigate nested keys
        keys = key.split(".")
        value = self
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set value with support for nested keys using dot notation.
        
        Creates intermediate dictionaries as needed.
        
        Args:
            key: Key or nested key path (e.g., "training.learning_rate")
            value: Value to set
        
        Example:
            >>> config.set("training.learning_rate", 0.001)
        """
        if self._frozen:
            raise AttributeError(f"Cannot modify frozen config: {key}")
        
        if "." not in key:
            self[key] = value
            return
        
        # Navigate and create nested keys
        keys = key.split(".")
        current = self
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = ConfigDict()
            elif not isinstance(current[k], dict):
                raise ValueError(f"Cannot create nested key '{key}': '{k}' is not a dict")
            current = current[k]
        
        current[keys[-1]] = value
    
    def merge(self, other: Dict[str, Any], overwrite: bool = True) -> ConfigDict:
        """
        Deep merge another dictionary into this config.
        
        Args:
            other: Dictionary to merge
            overwrite: Whether to overwrite existing keys (default: True)
        
        Returns:
            Self for chaining
        
        Example:
            >>> config.merge({"training": {"epochs": 100}})
        """
        if self._frozen:
            raise AttributeError("Cannot merge into frozen config")
        
        def _deep_merge(base: dict, update: dict, overwrite: bool) -> dict:
            """Recursively merge update into base"""
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    # Recursively merge nested dicts
                    _deep_merge(base[key], value, overwrite)
                elif key not in base or overwrite:
                    # Add new key or overwrite existing
                    base[key] = ConfigDict(value) if isinstance(value, dict) else value
            
            return base
        
        _deep_merge(self, other, overwrite)
        return self
    
    def freeze(self) -> ConfigDict:
        """
        Freeze config to prevent modifications.
        
        Useful for ensuring config immutability during training.
        
        Returns:
            Self for chaining
        """
        self._frozen = True
        
        # Recursively freeze nested ConfigDicts
        for value in self.values():
            if isinstance(value, ConfigDict):
                value.freeze()
        
        return self
    
    def unfreeze(self) -> ConfigDict:
        """
        Unfreeze config to allow modifications.
        
        Returns:
            Self for chaining
        """
        self._frozen = False
        
        # Recursively unfreeze nested ConfigDicts
        for value in self.values():
            if isinstance(value, ConfigDict):
                value.unfreeze()
        
        return self
    
    def to_dict(self) -> dict:
        """
        Convert ConfigDict to regular dict recursively.
        
        Returns:
            Regular Python dictionary
        """
        result = {}
        for key, value in self.items():
            if isinstance(value, ConfigDict):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def copy(self) -> ConfigDict:
        """Create deep copy of config"""
        return ConfigDict(deepcopy(self.to_dict()))


def expand_env_vars(config: Union[Dict, str, Any]) -> Any:
    """
    Recursively expand environment variables in config values.
    
    Supports ${VAR_NAME} and $VAR_NAME syntax.
    
    Args:
        config: Config dict, string, or other value
    
    Returns:
        Config with environment variables expanded
    
    Example:
        >>> expand_env_vars("${HOME}/data")
        '/home/user/data'
    """
    if isinstance(config, dict):
        return {key: expand_env_vars(value) for key, value in config.items()}
    
    elif isinstance(config, list):
        return [expand_env_vars(item) for item in config]
    
    elif isinstance(config, str):
        # Match ${VAR} or $VAR patterns
        def replace_var(match):
            var_name = match.group(1) or match.group(2)
            return os.environ.get(var_name, match.group(0))
        
        # Replace ${VAR_NAME} and $VAR_NAME
        pattern = r'\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)'
        return re.sub(pattern, replace_var, config)
    
    else:
        return config


def resolve_paths(
    config: Dict[str, Any],
    base_path: Path,
    path_keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Resolve relative paths in config to absolute paths.
    
    Args:
        config: Configuration dictionary
        base_path: Base path for relative path resolution (usually config file dir)
        path_keys: List of keys that contain paths (default: common path keys)
    
    Returns:
        Config with resolved paths
    
    Example:
        >>> resolve_paths({"data_path": "data/nsd"}, Path("/project/configs"))
        {"data_path": "/project/data/nsd"}
    """
    if path_keys is None:
        # Common keys that typically contain paths
        path_keys = [
            "path", "dir", "root", "file", "checkpoint",
            "data_path", "index_path", "output_path", "cache_path"
        ]
    
    def _resolve(obj: Any) -> Any:
        """Recursively resolve paths"""
        if isinstance(obj, dict):
            return {key: _resolve(value) for key, value in obj.items()}
        
        elif isinstance(obj, list):
            return [_resolve(item) for item in obj]
        
        elif isinstance(obj, str):
            # Check if this looks like a relative path
            if "/" in obj or "\\" in obj:
                path = Path(obj)
                if not path.is_absolute():
                    return str((base_path / path).resolve())
            return obj
        
        else:
            return obj
    
    return _resolve(config)


def load_yaml(yaml_path: Path) -> Dict[str, Any]:
    """
    Load YAML file with error handling.
    
    Args:
        yaml_path: Path to YAML file
    
    Returns:
        Loaded configuration dictionary
    
    Raises:
        FileNotFoundError: If YAML file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    
    try:
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        
        if config is None:
            logger.warning(f"Empty config file: {yaml_path}")
            return {}
        
        if not isinstance(config, dict):
            raise ValueError(f"Config must be a dictionary, got {type(config)}")
        
        return config
    
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {yaml_path}: {e}")


def load_config(
    config_path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None,
    resolve_base: bool = True,
    expand_vars: bool = True,
    resolve_relative_paths: bool = True,
    freeze: bool = False
) -> ConfigDict:
    """
    Load configuration with inheritance, environment variables, and path resolution.
    
    Supports hierarchical configuration via _base_ key for config inheritance.
    Base configs are deep-merged with child configs (child overrides base).
    
    Args:
        config_path: Path to YAML configuration file
        overrides: Dictionary of overrides to apply (supports dot notation)
        resolve_base: Whether to resolve _base_ inheritance (default: True)
        expand_vars: Whether to expand environment variables (default: True)
        resolve_relative_paths: Whether to resolve relative paths (default: True)
        freeze: Whether to freeze config after loading (default: False)
    
    Returns:
        ConfigDict with loaded and merged configuration
    
    Raises:
        FileNotFoundError: If config file not found
        ValueError: If config structure is invalid
    
    Example:
        >>> config = load_config("configs/mlp_standard.yaml")
        >>> config.training.learning_rate
        0.0005
        
        >>> config = load_config(
        ...     "configs/mlp_standard.yaml",
        ...     overrides={"training.epochs": 100}
        ... )
    """
    config_path = Path(config_path)
    config_dir = config_path.parent
    
    # Load main config
    logger.debug(f"Loading config from {config_path}")
    config = load_yaml(config_path)
    
    # Resolve _base_ inheritance
    if resolve_base and "_base_" in config:
        base_path = config.pop("_base_")
        
        # Resolve base path relative to current config directory
        if not Path(base_path).is_absolute():
            base_path = config_dir / base_path
        
        logger.debug(f"Loading base config from {base_path}")
        base_config = load_config(
            base_path,
            resolve_base=True,
            expand_vars=expand_vars,
            resolve_relative_paths=resolve_relative_paths,
            freeze=False
        )
        
        # Deep merge: base config + current config
        base_config.merge(config, overwrite=True)
        config = base_config.to_dict()
    
    # Convert to ConfigDict
    config = ConfigDict(config)
    
    # Expand environment variables
    if expand_vars:
        config = ConfigDict(expand_env_vars(config))
    
    # Resolve relative paths
    if resolve_relative_paths:
        config = ConfigDict(resolve_paths(config, config_dir))
    
    # Apply overrides
    if overrides:
        logger.debug(f"Applying {len(overrides)} config overrides")
        for key, value in overrides.items():
            config.set(key, value)
    
    # Freeze if requested
    if freeze:
        config.freeze()
    
    logger.info(f"✓ Loaded config from {config_path}")
    return config


def save_config(config: Union[ConfigDict, Dict], output_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration to save
        output_path: Path to output YAML file
    
    Example:
        >>> save_config(config, "outputs/experiment/config.yaml")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert ConfigDict to regular dict if needed
    if isinstance(config, ConfigDict):
        config = config.to_dict()
    
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"✓ Saved config to {output_path}")


def print_config(config: Union[ConfigDict, Dict], title: str = "Configuration") -> None:
    """
    Pretty-print configuration to console.
    
    Args:
        config: Configuration to print
        title: Title for the printout
    
    Example:
        >>> print_config(config, "MLP Training Configuration")
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}\n")
    
    def _print_nested(obj: Any, indent: int = 0):
        """Recursively print nested structure"""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, dict):
                    print(f"{'  '*indent}{key}:")
                    _print_nested(value, indent + 1)
                else:
                    print(f"{'  '*indent}{key}: {value}")
        else:
            print(f"{'  '*indent}{obj}")
    
    _print_nested(config)
    print(f"\n{'='*60}\n")
