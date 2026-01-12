#!/usr/bin/env python3
"""
train.py - Main training entrypoint for experiments
===================================================

Professional training script with:
- Config-driven experiments
- Reproducible seeds and determinism
- Automatic logging and checkpointing
- Run directory with full provenance
- Resume from checkpoint support
- Progress tracking and metrics logging

Usage:
    python -m src.train --config configs/experiment.yaml
    python -m src.train --config configs/experiment.yaml --resume
    python -m src.train --config configs/experiment.yaml --seed 123
"""

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = True):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Enable deterministic operations
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Set random seed to {seed} (deterministic={deterministic})")


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    logger.info(f"Loading config from: {config_path}")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a dictionary, got {type(config)}")
    
    return config


def save_config(config: Dict[str, Any], output_path: Path):
    """Save configuration to file."""
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Saved config to: {output_path}")


def get_git_info(repo_path: Path) -> Dict[str, str]:
    """Get git repository information."""
    git_info = {
        'commit_hash': 'unknown',
        'branch': 'unknown',
        'has_uncommitted_changes': 'unknown'
    }
    
    try:
        # Check if git is available and we're in a repo
        result = subprocess.run(
            ['git', 'rev-parse', '--git-dir'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            # Get commit hash
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            git_info['commit_hash'] = result.stdout.strip()
            
            # Get branch
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            git_info['branch'] = result.stdout.strip()
            
            # Check for uncommitted changes
            result = subprocess.run(
                ['git', 'diff-index', '--quiet', 'HEAD', '--'],
                cwd=repo_path,
                check=False
            )
            git_info['has_uncommitted_changes'] = 'yes' if result.returncode != 0 else 'no'
    except Exception as e:
        logger.warning(f"Could not get git info: {e}")
    
    return git_info


def create_run_directory(
    base_dir: Path,
    exp_name: str,
    config: Dict[str, Any],
    git_info: Dict[str, str]
) -> Path:
    """Create run directory with full provenance."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{timestamp}_{exp_name}"
    run_dir = base_dir / run_name
    
    # Create directory structure
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'checkpoints').mkdir(exist_ok=True)
    (run_dir / 'logs').mkdir(exist_ok=True)
    (run_dir / 'metrics').mkdir(exist_ok=True)
    (run_dir / 'outputs').mkdir(exist_ok=True)
    
    logger.info(f"Created run directory: {run_dir}")
    
    # Save configuration
    save_config(config, run_dir / 'config.yaml')
    
    # Save git info
    with open(run_dir / 'git_info.json', 'w') as f:
        json.dump(git_info, f, indent=2)
    
    # Save environment snapshot
    snapshot_script = Path(__file__).parent.parent / 'scripts' / 'snapshot_env.sh'
    if snapshot_script.exists():
        try:
            subprocess.run(
                ['bash', str(snapshot_script), str(run_dir / 'environment'), '--format', 'json'],
                check=True,
                capture_output=True
            )
            logger.info("Saved environment snapshot")
        except Exception as e:
            logger.warning(f"Could not save environment snapshot: {e}")
    
    return run_dir


def setup_logging(run_dir: Path, log_level: str = 'INFO'):
    """Setup file and console logging."""
    log_file = run_dir / 'logs' / 'train.log'
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Add to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    logger.info(f"Logging to: {log_file}")


class MetricsLogger:
    """Logger for training metrics."""
    
    def __init__(self, metrics_dir: Path):
        self.metrics_dir = metrics_dir
        self.metrics_file = metrics_dir / 'metrics.jsonl'
        self.summary_file = metrics_dir / 'summary.json'
        self.metrics_history = []
    
    def log(self, step: int, metrics: Dict[str, Any]):
        """Log metrics for a step."""
        entry = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        # Append to JSONL file
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        
        self.metrics_history.append(entry)
    
    def save_summary(self, final_metrics: Dict[str, Any]):
        """Save final summary."""
        summary = {
            'final_metrics': final_metrics,
            'total_steps': len(self.metrics_history),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved metrics summary to: {self.summary_file}")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    metrics: Dict[str, Any],
    checkpoint_path: Path,
    is_best: bool = False
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to: {checkpoint_path}")
    
    if is_best:
        best_path = checkpoint_path.parent / 'best_model.pt'
        shutil.copy(checkpoint_path, best_path)
        logger.info(f"Saved best model to: {best_path}")


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Dict[str, Any]:
    """Load checkpoint."""
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    return checkpoint


def dummy_training_loop(
    config: Dict[str, Any],
    run_dir: Path,
    resume_from: Optional[Path] = None
):
    """
    Dummy training loop for demonstration.
    Replace this with your actual training logic.
    """
    # Get config parameters
    num_epochs = config.get('num_epochs', 10)
    batch_size = config.get('batch_size', 32)
    learning_rate = config.get('learning_rate', 1e-4)
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Training config: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}")
    logger.info(f"Device: {device}")
    
    # Create dummy model and optimizer
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 1)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_from is not None:
        checkpoint = load_checkpoint(resume_from, model, optimizer)
        start_epoch = checkpoint.get('epoch', 0) + 1
    
    # Setup metrics logger
    metrics_logger = MetricsLogger(run_dir / 'metrics')
    
    # Training loop
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Dummy training steps
        epoch_loss = 0.0
        num_batches = 100  # Dummy number
        
        with tqdm(total=num_batches, desc=f"Epoch {epoch + 1}") as pbar:
            for batch_idx in range(num_batches):
                # Dummy forward pass
                dummy_input = torch.randn(batch_size, 10).to(device)
                dummy_target = torch.randn(batch_size, 1).to(device)
                
                optimizer.zero_grad()
                output = model(dummy_input)
                loss = torch.nn.functional.mse_loss(output, dummy_target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                global_step += 1
                
                # Log metrics periodically
                if global_step % 10 == 0:
                    metrics_logger.log(global_step, {
                        'epoch': epoch + 1,
                        'batch': batch_idx,
                        'loss': loss.item(),
                        'learning_rate': learning_rate
                    })
                
                pbar.update(1)
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Epoch summary
        avg_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
        
        checkpoint_path = run_dir / 'checkpoints' / f'checkpoint_epoch_{epoch + 1}.pt'
        save_checkpoint(
            model, optimizer, epoch, global_step,
            {'loss': avg_loss},
            checkpoint_path,
            is_best=is_best
        )
    
    # Save final summary
    metrics_logger.save_summary({
        'final_loss': avg_loss,
        'best_loss': best_loss,
        'total_epochs': num_epochs
    })
    
    logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(
        description='Train model with configuration file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config', '-c',
        type=Path,
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from latest checkpoint'
    )
    parser.add_argument(
        '--resume-from',
        type=Path,
        help='Resume from specific checkpoint path'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed (overrides config)'
    )
    parser.add_argument(
        '--runs-dir',
        type=Path,
        help='Base directory for runs (overrides config)'
    )
    parser.add_argument(
        '--name',
        type=str,
        help='Experiment name (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)
    
    # Override config with command-line arguments
    if args.seed is not None:
        config['seed'] = args.seed
    if args.runs_dir is not None:
        config['runs_dir'] = str(args.runs_dir)
    if args.name is not None:
        config['experiment_name'] = args.name
    
    # Get experiment parameters
    seed = config.get('seed', int(os.environ.get('DEFAULT_SEED', 42)))
    exp_name = config.get('experiment_name', args.config.stem)
    runs_dir = Path(config.get('runs_dir', os.environ.get('RUNS_DIR', './runs')))
    
    # Set random seed
    set_seed(seed)
    
    # Get git information
    project_root = Path(__file__).parent.parent
    git_info = get_git_info(project_root)
    
    if git_info['has_uncommitted_changes'] == 'yes':
        logger.warning("You have uncommitted changes! Results may not be fully reproducible.")
    
    # Create run directory
    run_dir = create_run_directory(runs_dir, exp_name, config, git_info)
    
    # Setup logging
    setup_logging(run_dir, config.get('log_level', 'INFO'))
    
    # Log run information
    logger.info("=" * 80)
    logger.info(f"Starting experiment: {exp_name}")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Git commit: {git_info['commit_hash'][:8]}")
    logger.info(f"Git branch: {git_info['branch']}")
    logger.info("=" * 80)
    
    # Determine resume checkpoint
    resume_checkpoint = None
    if args.resume_from:
        resume_checkpoint = args.resume_from
    elif args.resume:
        # Find latest checkpoint in run directory
        checkpoints = sorted((run_dir / 'checkpoints').glob('checkpoint_epoch_*.pt'))
        if checkpoints:
            resume_checkpoint = checkpoints[-1]
            logger.info(f"Resuming from: {resume_checkpoint}")
        else:
            logger.warning("No checkpoints found to resume from")
    
    # Run training
    try:
        dummy_training_loop(config, run_dir, resume_checkpoint)
        
        # Write success marker
        with open(run_dir / 'SUCCESS', 'w') as f:
            f.write(f"Training completed successfully at {datetime.now().isoformat()}\n")
        
        logger.info("✓ Training completed successfully!")
        logger.info(f"Results saved to: {run_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        
        # Write failure marker
        with open(run_dir / 'FAILED', 'w') as f:
            f.write(f"Training failed at {datetime.now().isoformat()}\n")
            f.write(f"Error: {str(e)}\n")
        
        sys.exit(1)


if __name__ == '__main__':
    main()
