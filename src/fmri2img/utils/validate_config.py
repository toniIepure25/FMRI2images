#!/usr/bin/env python3
"""
Configuration Validation Script
===============================

Validates that the optimal production configuration is correctly set up.

Usage:
    python scripts/validate_config.py [--config configs/production_optimal.yaml]
"""

import argparse
import sys
from pathlib import Path

import yaml


def print_header(text):
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def print_check(text, status):
    symbol = "✅" if status else "❌"
    print(f"{symbol} {text}")
    return status


def validate_config(config_path: Path) -> bool:
    """Validate configuration file."""
    print_header("CONFIGURATION VALIDATION")
    print(f"Config file: {config_path}")
    
    all_valid = True
    
    # Check file exists
    if not print_check(f"Configuration file exists: {config_path}", config_path.exists()):
        return False
    
    # Load config
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        print_check("Configuration file is valid YAML", True)
    except Exception as e:
        print_check(f"Configuration file parse error: {e}", False)
        return False
    
    # Validate structure
    required_sections = [
        'experiment', 'dataset', 'preprocessing', 'mlp_encoder',
        'clip_adapter', 'diffusion', 'paths', 'compute', 'reproducibility'
    ]
    
    for section in required_sections:
        all_valid &= print_check(f"Section '{section}' exists", section in config)
    
    if not all_valid:
        return False
    
    print_header("DATASET CONFIGURATION")
    
    # Validate dataset config
    ds = config['dataset']
    all_valid &= print_check(
        f"Subject: {ds.get('subject', 'MISSING')}",
        'subject' in ds and ds['subject'].startswith('subj')
    )
    all_valid &= print_check(
        f"Max trials: {ds.get('max_trials', 'MISSING')} (expected: 750)",
        'max_trials' in ds and ds['max_trials'] == 750
    )
    all_valid &= print_check(
        f"Train samples: {ds.get('train_samples', 'MISSING')}",
        'train_samples' in ds and ds['train_samples'] == 600
    )
    all_valid &= print_check(
        f"Random seed: {ds.get('random_seed', 'MISSING')}",
        'random_seed' in ds
    )
    
    print_header("PREPROCESSING CONFIGURATION")
    
    pp = config['preprocessing']
    all_valid &= print_check(
        f"Reliability threshold: {pp.get('reliability_threshold', 'MISSING')}",
        'reliability_threshold' in pp and pp['reliability_threshold'] == 0.10
    )
    all_valid &= print_check(
        f"PCA components: {pp.get('tier2', {}).get('n_components', 'MISSING')}",
        'tier2' in pp and 'n_components' in pp['tier2']
    )
    
    print_header("MLP ENCODER CONFIGURATION")
    
    mlp = config['mlp_encoder']
    all_valid &= print_check(
        f"Hidden dims: {mlp.get('hidden_dims', 'MISSING')}",
        'hidden_dims' in mlp and len(mlp['hidden_dims']) >= 2
    )
    all_valid &= print_check(
        f"Dropout: {mlp.get('dropout', 'MISSING')}",
        'dropout' in mlp and 0 < mlp['dropout'] < 1
    )
    all_valid &= print_check(
        f"Learning rate: {mlp.get('training', {}).get('learning_rate', 'MISSING')}",
        'training' in mlp and 'learning_rate' in mlp['training']
    )
    all_valid &= print_check(
        f"Batch size: {mlp.get('training', {}).get('batch_size', 'MISSING')}",
        'training' in mlp and 'batch_size' in mlp['training']
    )
    
    # Validate loss configuration
    loss = mlp.get('loss', {})
    all_valid &= print_check(
        f"Cosine weight: {loss.get('cosine_weight', 'MISSING')}",
        'cosine_weight' in loss
    )
    all_valid &= print_check(
        f"MSE weight: {loss.get('mse_weight', 'MISSING')}",
        'mse_weight' in loss
    )
    all_valid &= print_check(
        f"Triplet weight: {loss.get('triplet_weight', 'MISSING')}",
        'triplet_weight' in loss
    )
    
    print_header("CLIP ADAPTER CONFIGURATION")
    
    ada = config['clip_adapter']
    all_valid &= print_check(
        f"Input dim: {ada.get('input_dim', 'MISSING')} (expected: 512)",
        'input_dim' in ada and ada['input_dim'] == 512
    )
    all_valid &= print_check(
        f"Output dim: {ada.get('output_dim', 'MISSING')} (expected: 1024)",
        'output_dim' in ada and ada['output_dim'] == 1024
    )
    
    print_header("DIFFUSION CONFIGURATION")
    
    dif = config['diffusion']
    all_valid &= print_check(
        f"Model: {dif.get('model_id', 'MISSING')}",
        'model_id' in dif and 'stable-diffusion' in dif['model_id'].lower()
    )
    
    inf = dif.get('inference', {})
    all_valid &= print_check(
        f"Steps: {inf.get('num_steps', 'MISSING')} (optimal: 150)",
        'num_steps' in inf and inf['num_steps'] >= 100
    )
    all_valid &= print_check(
        f"Guidance scale: {inf.get('guidance_scale', 'MISSING')} (optimal: 10-12)",
        'guidance_scale' in inf and 10 <= inf['guidance_scale'] <= 12
    )
    all_valid &= print_check(
        f"Scheduler: {inf.get('scheduler', 'MISSING')}",
        'scheduler' in inf and inf['scheduler'] in ['ddim', 'dpm', 'euler', 'pndm']
    )
    
    print_header("COMPUTE CONFIGURATION")
    
    cmp = config['compute']
    all_valid &= print_check(
        f"Device: {cmp.get('device', 'MISSING')}",
        'device' in cmp and cmp['device'] in ['cuda', 'cpu']
    )
    
    print_header("PATHS VALIDATION")
    
    # Check critical directories exist
    base_dir = Path('.')
    paths_to_check = [
        ('configs', True),
        ('scripts', True),
        ('src/fmri2img', True),
        ('outputs', False),
        ('checkpoints', False),
        ('logs', False),
    ]
    
    for path_name, required in paths_to_check:
        path = base_dir / path_name
        exists = path.exists()
        if required:
            all_valid &= print_check(f"Directory exists: {path_name}", exists)
        else:
            print_check(f"Directory exists: {path_name} (will be created)", exists or not required)
    
    print_header("VALIDATION SUMMARY")
    
    if all_valid:
        print("\n✅ Configuration is VALID and ready for use!")
        print(f"\nTo run the pipeline:")
        print(f"  bash scripts/run_production.sh")
        return True
    else:
        print("\n❌ Configuration has ERRORS that need to be fixed")
        print(f"\nPlease review the errors above and:")
        print(f"  1. Edit: {config_path}")
        print(f"  2. Fix the marked issues")
        print(f"  3. Re-run: python scripts/validate_config.py")
        return False


def validate_environment():
    """Validate Python environment."""
    print_header("ENVIRONMENT VALIDATION")
    
    all_valid = True
    
    # Check Python version
    import sys
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    all_valid &= print_check(
        f"Python version: {py_version} (expected: 3.8+)",
        sys.version_info >= (3, 8)
    )
    
    # Check required packages
    required_packages = [
        'torch', 'numpy', 'pandas', 'yaml', 'nibabel',
        'transformers', 'diffusers', 'PIL', 'matplotlib'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print_check(f"Package installed: {package}", True)
        except ImportError:
            all_valid &= print_check(f"Package installed: {package}", False)
    
    # Check CUDA availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            print_check(f"CUDA available: {device_name}", True)
        else:
            print_check("CUDA available: False (will use CPU)", False)
    except:
        print_check("CUDA check failed", False)
    
    return all_valid


def main():
    parser = argparse.ArgumentParser(description="Validate optimal production configuration")
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('configs/production_optimal.yaml'),
        help='Path to configuration file'
    )
    parser.add_argument(
        '--skip-env',
        action='store_true',
        help='Skip environment validation'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("  OPTIMAL PRODUCTION CONFIGURATION VALIDATOR")
    print("=" * 80)
    print(f"\nValidating: {args.config}")
    
    # Validate environment
    if not args.skip_env:
        env_valid = validate_environment()
        if not env_valid:
            print("\n⚠️  Environment validation failed!")
            print("   Please install missing packages:")
            print("   pip install -r requirements.txt")
            print("\n   Or skip with: --skip-env")
    
    # Validate configuration
    config_valid = validate_config(args.config)
    
    # Final status
    print("\n" + "=" * 80)
    if config_valid:
        print("  ✅ ALL CHECKS PASSED - SYSTEM READY")
        print("=" * 80)
        print("\n🚀 Next step:")
        print("   bash scripts/run_production.sh")
        print("\n📖 Documentation:")
        print("   • Configuration: configs/production_optimal.yaml")
        print("   • Full Guide: docs/OPTIMAL_CONFIGURATION_GUIDE.md")
        print("   • Quick Start: docs/PRODUCTION_READY_SUMMARY.md")
        return 0
    else:
        print("  ❌ VALIDATION FAILED - FIX ERRORS ABOVE")
        print("=" * 80)
        return 1


if __name__ == '__main__':
    sys.exit(main())
