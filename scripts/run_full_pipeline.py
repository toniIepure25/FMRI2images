#!/usr/bin/env python3
"""
Production-grade end-to-end pipeline for fMRI-to-Image reconstruction with novel contributions.

This script orchestrates the complete workflow from data preparation to evaluation,
with intelligent checks to avoid redundant computation and automatic recovery from failures.

Novel Contributions:
1. Soft Reliability Weighting (continuous voxel importance)
2. InfoNCE Contrastive Loss (ranking optimization)
3. MC Dropout Uncertainty Estimation (confidence calibration)

Usage:
    # Full pipeline with all novel contributions
    python scripts/run_full_pipeline.py --subject subj01 --mode novel
    
    # Baseline only (for comparison)
    python scripts/run_full_pipeline.py --subject subj01 --mode baseline
    
    # Full ablation study (4 experiments)
    python scripts/run_full_pipeline.py --subject subj01 --mode ablation
    
    # Resume from specific step
    python scripts/run_full_pipeline.py --subject subj01 --mode novel --resume-from train

Author: Generated for Bachelor Thesis V2
Date: December 2025
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib

import numpy as np
import pandas as pd
import torch


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class PipelineOrchestrator:
    """Orchestrates the full fMRI-to-Image pipeline with smart caching and validation."""
    
    def __init__(
        self,
        subject: str,
        mode: str,
        root_dir: Path,
        force_rebuild: bool = False,
        dry_run: bool = False,
        skip_eval: bool = False
    ):
        self.subject = subject
        self.mode = mode  # 'baseline', 'novel', 'ablation'
        self.root = Path(root_dir)
        self.force_rebuild = force_rebuild
        self.dry_run = dry_run
        self.skip_eval = skip_eval
        
        # Define directory structure
        self.data_dir = self.root / "data"
        self.cache_dir = self.root / "cache"
        self.outputs_dir = self.root / "outputs"
        self.checkpoints_dir = self.root / "checkpoints"
        self.scripts_dir = self.root / "scripts"
        
        # Pipeline state
        self.state_file = self.root / f".pipeline_state_{subject}_{mode}.json"
        self.state = self._load_state()
        
        # Configuration
        self.configs = self._get_configs()
        
    def _load_state(self) -> Dict:
        """Load pipeline state from disk."""
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {
            "completed_steps": [],
            "last_run": None,
            "artifacts": {}
        }
    
    def _save_state(self):
        """Save pipeline state to disk."""
        self.state["last_run"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)
    
    def _mark_step_complete(self, step: str, artifacts: Dict = None):
        """Mark a pipeline step as completed."""
        if step not in self.state["completed_steps"]:
            self.state["completed_steps"].append(step)
        if artifacts:
            self.state["artifacts"][step] = artifacts
        self._save_state()
    
    def _is_step_complete(self, step: str) -> bool:
        """Check if a step has been completed."""
        return step in self.state["completed_steps"]
    
    def _get_configs(self) -> Dict:
        """Get configuration for each experiment mode."""
        configs = {
            "baseline": {
                "name": "Baseline",
                "preproc": {
                    "reliability_mode": "hard_threshold",
                    "reliability_threshold": 0.1,
                    "n_components": 3072
                },
                "training": {
                    "cosine_weight": 1.0,
                    "mse_weight": 0.0,
                    "infonce_weight": 0.0,
                    "epochs": 150,
                    "batch_size": 64,
                    "learning_rate": 1e-4
                }
            },
            "soft_only": {
                "name": "Soft Reliability Only",
                "preproc": {
                    "reliability_mode": "soft_weight",
                    "reliability_threshold": 0.1,
                    "reliability_curve": "sigmoid",
                    "reliability_temperature": 0.1,
                    "n_components": 3072
                },
                "training": {
                    "cosine_weight": 1.0,
                    "mse_weight": 0.0,
                    "infonce_weight": 0.0,
                    "epochs": 150,
                    "batch_size": 64,
                    "learning_rate": 1e-4
                }
            },
            "infonce_only": {
                "name": "InfoNCE Only",
                "preproc": {
                    "reliability_mode": "hard_threshold",
                    "reliability_threshold": 0.1,
                    "n_components": 3072
                },
                "training": {
                    "cosine_weight": 1.0,
                    "mse_weight": 0.0,
                    "infonce_weight": 0.3,
                    "temperature": 0.07,
                    "epochs": 150,
                    "batch_size": 64,
                    "learning_rate": 1e-4
                }
            },
            "novel": {
                "name": "Full Novel (Both)",
                "preproc": {
                    "reliability_mode": "soft_weight",
                    "reliability_threshold": 0.1,
                    "reliability_curve": "sigmoid",
                    "reliability_temperature": 0.1,
                    "n_components": 3072
                },
                "training": {
                    "cosine_weight": 1.0,
                    "mse_weight": 0.0,
                    "infonce_weight": 0.3,
                    "temperature": 0.07,
                    "epochs": 150,
                    "batch_size": 64,
                    "learning_rate": 1e-4
                }
            }
        }
        return configs
    
    def print_header(self, text: str):
        """Print formatted section header."""
        print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")
    
    def print_step(self, text: str):
        """Print formatted step."""
        print(f"{Colors.OKBLUE}{Colors.BOLD}▶ {text}{Colors.ENDC}")
    
    def print_success(self, text: str):
        """Print success message."""
        print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")
    
    def print_warning(self, text: str):
        """Print warning message."""
        print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")
    
    def print_error(self, text: str):
        """Print error message."""
        print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")
    
    def print_info(self, text: str):
        """Print info message."""
        print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")
    
    def run_command(self, cmd: List[str], description: str) -> Tuple[bool, str]:
        """Run a shell command with error handling."""
        if self.dry_run:
            print(f"{Colors.OKCYAN}[DRY RUN] Would execute: {' '.join(cmd)}{Colors.ENDC}")
            return True, ""
        
        self.print_step(description)
        print(f"{Colors.OKCYAN}Command: {' '.join(cmd)}{Colors.ENDC}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=self.root
            )
            self.print_success(f"{description} - Completed")
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            self.print_error(f"{description} - Failed")
            print(f"{Colors.FAIL}Error: {e.stderr}{Colors.ENDC}")
            return False, e.stderr
    
    # ============================================================================
    # STEP 1: Environment Validation
    # ============================================================================
    
    def validate_environment(self) -> bool:
        """Validate that the environment is properly set up."""
        self.print_header("STEP 0: Environment Validation")
        
        checks = []
        
        # Check Python version
        self.print_step("Checking Python version...")
        py_version = sys.version_info
        if py_version.major >= 3 and py_version.minor >= 10:
            self.print_success(f"Python {py_version.major}.{py_version.minor}.{py_version.micro}")
            checks.append(True)
        else:
            self.print_error(f"Python 3.10+ required, found {py_version.major}.{py_version.minor}")
            checks.append(False)
        
        # Check CUDA availability
        self.print_step("Checking CUDA availability...")
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            self.print_success(f"CUDA available: {device_name}")
            checks.append(True)
        else:
            self.print_warning("CUDA not available - will use CPU (much slower!)")
            checks.append(True)  # Don't fail, just warn
        
        # Check if package is installed
        self.print_step("Checking if fmri2img package is installed...")
        try:
            import fmri2img
            self.print_success("fmri2img package found")
            checks.append(True)
        except ImportError:
            self.print_error("fmri2img package not installed - run 'pip install -e .'")
            checks.append(False)
        
        # Check directory structure
        self.print_step("Checking directory structure...")
        required_dirs = [self.data_dir, self.cache_dir, self.scripts_dir]
        for d in required_dirs:
            if d.exists():
                self.print_success(f"Found: {d}")
                checks.append(True)
            else:
                self.print_error(f"Missing: {d}")
                checks.append(False)
        
        # Check disk space
        self.print_step("Checking disk space...")
        import shutil
        stat = shutil.disk_usage(self.root)
        free_gb = stat.free / (1024**3)
        if free_gb > 100:
            self.print_success(f"Available: {free_gb:.1f} GB")
            checks.append(True)
        else:
            self.print_warning(f"Low disk space: {free_gb:.1f} GB (recommend 100+ GB)")
            checks.append(True)  # Don't fail, just warn
        
        success = all(checks)
        if success:
            self.print_success("Environment validation passed")
        else:
            self.print_error("Environment validation failed - fix errors above")
        
        return success
    
    # ============================================================================
    # STEP 2: Build NSD Index
    # ============================================================================
    
    def build_index(self) -> bool:
        """Build NSD trial index with train/val/test splits."""
        step_name = "build_index"
        
        if self._is_step_complete(step_name) and not self.force_rebuild:
            self.print_header("STEP 1: Build NSD Index [CACHED]")
            index_file = self._get_index_path()
            if index_file.exists():
                self.print_success(f"Using cached index: {index_file}")
                return True
            else:
                self.print_warning("Index marked complete but file missing - rebuilding")
        else:
            self.print_header("STEP 1: Build NSD Index")
        
        index_file = self._get_index_path()
        index_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if index already exists with correct format
        if index_file.exists() and not self.force_rebuild:
            self.print_step("Validating existing index...")
            try:
                df = pd.read_parquet(index_file)
                required_cols = ['trial_id', 'subject', 'nsd_id', 'split', 'beta_file', 'beta_index']
                
                if all(col in df.columns for col in required_cols):
                    n_trials = len(df)
                    n_train = len(df[df['split'] == 'train'])
                    n_val = len(df[df['split'] == 'val'])
                    n_test = len(df[df['split'] == 'test'])
                    
                    self.print_success(f"Valid index found: {n_trials} trials")
                    self.print_info(f"  Train: {n_train} | Val: {n_val} | Test: {n_test}")
                    
                    self._mark_step_complete(step_name, {"index_file": str(index_file), "n_trials": n_trials})
                    return True
            except Exception as e:
                self.print_warning(f"Index validation failed: {e}")
        
        # Build index using Makefile
        cmd = ["make", "index", f"SUBJECTS={self.subject}"]
        success, output = self.run_command(cmd, "Building NSD index")
        
        if success and index_file.exists():
            df = pd.read_parquet(index_file)
            self._mark_step_complete(step_name, {"index_file": str(index_file), "n_trials": len(df)})
            return True
        
        return False
    
    def _get_index_path(self) -> Path:
        """Get path to index file."""
        return self.data_dir / "indices" / "nsd_index" / f"subject={self.subject}" / "index.parquet"
    
    # ============================================================================
    # STEP 3: Build CLIP Cache
    # ============================================================================
    
    def build_clip_cache(self) -> bool:
        """Build CLIP embeddings cache for all NSD images."""
        step_name = "build_clip_cache"
        
        if self._is_step_complete(step_name) and not self.force_rebuild:
            self.print_header("STEP 2: Build CLIP Cache [CACHED]")
            clip_cache = self._get_clip_cache_path()
            if clip_cache.exists():
                self.print_success(f"Using cached CLIP embeddings: {clip_cache}")
                return True
            else:
                self.print_warning("CLIP cache marked complete but file missing - rebuilding")
        else:
            self.print_header("STEP 2: Build CLIP Cache")
        
        clip_cache = self._get_clip_cache_path()
        clip_cache.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if cache exists with correct size
        if clip_cache.exists() and not self.force_rebuild:
            self.print_step("Validating existing CLIP cache...")
            try:
                df = pd.read_parquet(clip_cache)
                
                # Expected: 73,000 images in NSD
                expected_min = 70000
                if len(df) >= expected_min and 'clip_embedding' in df.columns:
                    # Check embedding dimension
                    first_emb = df['clip_embedding'].iloc[0]
                    emb_dim = len(first_emb) if isinstance(first_emb, (list, np.ndarray)) else first_emb.shape[0]
                    
                    if emb_dim == 512:
                        self.print_success(f"Valid CLIP cache found: {len(df)} images, dim={emb_dim}")
                        self._mark_step_complete(step_name, {
                            "clip_cache": str(clip_cache),
                            "n_images": len(df),
                            "embedding_dim": emb_dim
                        })
                        return True
                    else:
                        self.print_warning(f"Unexpected embedding dimension: {emb_dim} (expected 512)")
                else:
                    self.print_warning(f"Incomplete CLIP cache: {len(df)} images (expected {expected_min}+)")
            except Exception as e:
                self.print_warning(f"CLIP cache validation failed: {e}")
        
        # Build cache using Makefile
        self.print_warning("Building CLIP cache will take 2-3 hours...")
        cmd = ["make", "build-clip-cache", f"CACHE={clip_cache}", "BATCH=256"]
        success, output = self.run_command(cmd, "Building CLIP embeddings cache")
        
        if success and clip_cache.exists():
            df = pd.read_parquet(clip_cache)
            self._mark_step_complete(step_name, {
                "clip_cache": str(clip_cache),
                "n_images": len(df)
            })
            return True
        
        return False
    
    def _get_clip_cache_path(self) -> Path:
        """Get path to CLIP cache."""
        return self.outputs_dir / "clip_cache" / "clip.parquet"
    
    # ============================================================================
    # STEP 4: Preprocessing
    # ============================================================================
    
    def run_preprocessing(self, config: Dict) -> bool:
        """Run preprocessing with specified configuration."""
        config_name = config["name"]
        step_name = f"preproc_{config_name.lower().replace(' ', '_')}"
        
        if self._is_step_complete(step_name) and not self.force_rebuild:
            self.print_header(f"STEP 3: Preprocessing ({config_name}) [CACHED]")
            preproc_dir = self._get_preproc_dir(config_name)
            if self._validate_preprocessing(preproc_dir, config):
                self.print_success(f"Using cached preprocessing: {preproc_dir}")
                return True
            else:
                self.print_warning("Preprocessing marked complete but validation failed - rebuilding")
        else:
            self.print_header(f"STEP 3: Preprocessing ({config_name})")
        
        preproc_dir = self._get_preproc_dir(config_name)
        preproc_dir.mkdir(parents=True, exist_ok=True)
        
        # Build command
        cmd = [
            "python", "scripts/fit_preprocessing.py",
            "--subject", self.subject,
            "--index-file", str(self._get_index_path()),
            "--output-dir", str(preproc_dir),
            "--reliability-mode", config["preproc"]["reliability_mode"],
            "--reliability-threshold", str(config["preproc"]["reliability_threshold"]),
            "--n-components", str(config["preproc"]["n_components"])
        ]
        
        # Add optional parameters
        if "reliability_curve" in config["preproc"]:
            cmd.extend(["--reliability-curve", config["preproc"]["reliability_curve"]])
        if "reliability_temperature" in config["preproc"]:
            cmd.extend(["--reliability-temperature", str(config["preproc"]["reliability_temperature"])])
        
        success, output = self.run_command(cmd, f"Preprocessing ({config_name})")
        
        if success and self._validate_preprocessing(preproc_dir, config):
            self._mark_step_complete(step_name, {"preproc_dir": str(preproc_dir)})
            return True
        
        return False
    
    def _validate_preprocessing(self, preproc_dir: Path, config: Dict) -> bool:
        """Validate preprocessing artifacts."""
        subject_dir = preproc_dir / self.subject
        
        # Check required files
        required_files = [
            "scaler_mean.npy",
            "scaler_std.npy",
            "reliability_mask.npy",
            "pca_components.npy",
            "meta.json"
        ]
        
        for fname in required_files:
            fpath = subject_dir / fname
            if not fpath.exists():
                self.print_warning(f"Missing: {fname}")
                return False
        
        # For soft weighting, check that weights file exists
        if config["preproc"]["reliability_mode"] == "soft_weight":
            weights_file = subject_dir / "reliability_weights.npy"
            if not weights_file.exists():
                self.print_warning("Missing: reliability_weights.npy")
                return False
            
            # Verify weights are continuous
            weights = np.load(weights_file)
            mask = np.load(subject_dir / "reliability_mask.npy")
            
            if np.allclose(weights, mask.astype(float)):
                self.print_warning("Weights are binary (should be continuous)")
                return False
        
        self.print_success(f"Preprocessing validation passed: {subject_dir}")
        return True
    
    def _get_preproc_dir(self, config_name: str) -> Path:
        """Get preprocessing output directory."""
        safe_name = config_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        return self.outputs_dir / "preproc" / safe_name
    
    # ============================================================================
    # STEP 5: Training
    # ============================================================================
    
    def run_training(self, config: Dict) -> bool:
        """Run training with specified configuration."""
        config_name = config["name"]
        step_name = f"train_{config_name.lower().replace(' ', '_')}"
        
        if self._is_step_complete(step_name) and not self.force_rebuild:
            self.print_header(f"STEP 4: Training ({config_name}) [CACHED]")
            checkpoint = self._get_checkpoint_path(config_name)
            if checkpoint.exists():
                self.print_success(f"Using cached checkpoint: {checkpoint}")
                return True
            else:
                self.print_warning("Training marked complete but checkpoint missing - retraining")
        else:
            self.print_header(f"STEP 4: Training ({config_name})")
        
        checkpoint_dir = self._get_checkpoint_dir(config_name)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        preproc_dir = self._get_preproc_dir(config_name)
        
        # Build command
        cmd = [
            "python", "scripts/train_mlp.py",
            "--subject", self.subject,
            "--index-file", str(self._get_index_path()),
            "--clip-cache", str(self._get_clip_cache_path()),
            "--preproc-dir", str(preproc_dir),
            "--output-dir", str(checkpoint_dir),
            "--epochs", str(config["training"]["epochs"]),
            "--batch-size", str(config["training"]["batch_size"]),
            "--learning-rate", str(config["training"]["learning_rate"]),
            "--cosine-weight", str(config["training"]["cosine_weight"]),
            "--mse-weight", str(config["training"]["mse_weight"]),
            "--infonce-weight", str(config["training"]["infonce_weight"]),
            "--save-every", "10"
        ]
        
        # Add optional parameters
        if "temperature" in config["training"]:
            cmd.extend(["--temperature", str(config["training"]["temperature"])])
        
        self.print_warning(f"Training will take ~2 hours...")
        success, output = self.run_command(cmd, f"Training ({config_name})")
        
        if success:
            checkpoint = self._get_checkpoint_path(config_name)
            if checkpoint.exists():
                self._mark_step_complete(step_name, {"checkpoint": str(checkpoint)})
                return True
        
        return False
    
    def _get_checkpoint_dir(self, config_name: str) -> Path:
        """Get checkpoint directory."""
        safe_name = config_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        return self.checkpoints_dir / "mlp" / safe_name / self.subject
    
    def _get_checkpoint_path(self, config_name: str) -> Path:
        """Get best checkpoint path."""
        return self._get_checkpoint_dir(config_name) / "best_model.pt"
    
    # ============================================================================
    # STEP 6: Standard Evaluation
    # ============================================================================
    
    def run_evaluation(self, config: Dict) -> bool:
        """Run standard evaluation (retrieval + similarity)."""
        if self.skip_eval:
            self.print_header(f"STEP 5: Evaluation ({config['name']}) [SKIPPED]")
            return True
        
        config_name = config["name"]
        step_name = f"eval_{config_name.lower().replace(' ', '_')}"
        
        if self._is_step_complete(step_name) and not self.force_rebuild:
            self.print_header(f"STEP 5: Evaluation ({config_name}) [CACHED]")
            eval_dir = self._get_eval_dir(config_name)
            metrics_file = eval_dir / "metrics.json"
            if metrics_file.exists():
                self.print_success(f"Using cached evaluation: {eval_dir}")
                return True
        else:
            self.print_header(f"STEP 5: Evaluation ({config_name})")
        
        eval_dir = self._get_eval_dir(config_name)
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = self._get_checkpoint_path(config_name)
        
        # Build command
        cmd = [
            "python", "scripts/run_reconstruct_and_eval.py",
            "--subject", self.subject,
            "--encoder-checkpoint", str(checkpoint),
            "--encoder-type", "mlp",
            "--clip-cache", str(self._get_clip_cache_path()),
            "--output-dir", str(eval_dir),
            "--split", "test"
        ]
        
        success, output = self.run_command(cmd, f"Standard evaluation ({config_name})")
        
        if success:
            metrics_file = eval_dir / "metrics.json"
            if metrics_file.exists():
                # Print key metrics
                with open(metrics_file) as f:
                    metrics = json.load(f)
                self.print_info(f"  Cosine Similarity: {metrics.get('cosine_similarity', 'N/A'):.4f}")
                self.print_info(f"  Retrieval@1: {metrics.get('retrieval_top1', 'N/A'):.2%}")
                self.print_info(f"  Retrieval@5: {metrics.get('retrieval_top5', 'N/A'):.2%}")
                
                self._mark_step_complete(step_name, {
                    "eval_dir": str(eval_dir),
                    "metrics": metrics
                })
                return True
        
        return False
    
    def _get_eval_dir(self, config_name: str) -> Path:
        """Get evaluation output directory."""
        safe_name = config_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        return self.outputs_dir / "eval" / safe_name
    
    # ============================================================================
    # STEP 7: Uncertainty Evaluation
    # ============================================================================
    
    def run_uncertainty_eval(self, config: Dict) -> bool:
        """Run MC dropout uncertainty evaluation."""
        if self.skip_eval:
            self.print_header(f"STEP 6: Uncertainty Evaluation ({config['name']}) [SKIPPED]")
            return True
        
        config_name = config["name"]
        step_name = f"uncertainty_{config_name.lower().replace(' ', '_')}"
        
        if self._is_step_complete(step_name) and not self.force_rebuild:
            self.print_header(f"STEP 6: Uncertainty Evaluation ({config_name}) [CACHED]")
            uncertainty_dir = self._get_uncertainty_dir(config_name)
            summary_file = uncertainty_dir / "uncertainty_summary.json"
            if summary_file.exists():
                self.print_success(f"Using cached uncertainty eval: {uncertainty_dir}")
                return True
        else:
            self.print_header(f"STEP 6: Uncertainty Evaluation ({config_name})")
        
        # Check if eval script exists
        eval_script = self.scripts_dir / "eval_uncertainty.py"
        if not eval_script.exists():
            self.print_warning("eval_uncertainty.py not found - creating it...")
            if not self._create_uncertainty_eval_script():
                return False
        
        uncertainty_dir = self._get_uncertainty_dir(config_name)
        uncertainty_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = self._get_checkpoint_path(config_name)
        preproc_dir = self._get_preproc_dir(config_name)
        
        # Build command
        cmd = [
            "python", "scripts/eval_uncertainty.py",
            "--subject", self.subject,
            "--checkpoint", str(checkpoint),
            "--index-file", str(self._get_index_path()),
            "--clip-cache", str(self._get_clip_cache_path()),
            "--preproc-dir", str(preproc_dir),
            "--output-dir", str(uncertainty_dir),
            "--n-samples", "20",
            "--split", "test",
            "--limit", "500"
        ]
        
        success, output = self.run_command(cmd, f"Uncertainty evaluation ({config_name})")
        
        if success:
            summary_file = uncertainty_dir / "uncertainty_summary.json"
            if summary_file.exists():
                # Print key metrics
                with open(summary_file) as f:
                    summary = json.load(f)
                self.print_info(f"  Uncertainty-Error Correlation: {summary.get('correlation_pearson', 'N/A'):.4f}")
                self.print_info(f"  Mean Uncertainty: {summary.get('mean_uncertainty', 'N/A'):.4f}")
                
                self._mark_step_complete(step_name, {
                    "uncertainty_dir": str(uncertainty_dir),
                    "summary": summary
                })
                return True
        
        return False
    
    def _get_uncertainty_dir(self, config_name: str) -> Path:
        """Get uncertainty evaluation directory."""
        safe_name = config_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        return self.outputs_dir / "eval" / f"{safe_name}_uncertainty"
    
    def _create_uncertainty_eval_script(self) -> bool:
        """Create eval_uncertainty.py script if it doesn't exist."""
        script_path = self.scripts_dir / "eval_uncertainty.py"
        
        script_content = '''#!/usr/bin/env python3
"""Evaluate model with MC dropout uncertainty estimation."""

import argparse
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from fmri2img.models.mlp import MLPEncoder
from fmri2img.data.datasets import NSDDataset
from fmri2img.eval.uncertainty import (
    predict_with_mc_dropout,
    compute_uncertainty_error_correlation,
    plot_calibration_curve
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=str, default='subj01')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--index-file', type=str, required=True)
    parser.add_argument('--clip-cache', type=str, required=True)
    parser.add_argument('--preproc-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--n-samples', type=int, default=20)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = MLPEncoder(
        input_dim=checkpoint['config']['input_dim'],
        hidden_dims=checkpoint['config']['hidden_dims'],
        output_dim=checkpoint['config']['output_dim'],
        dropout=checkpoint['config'].get('dropout', 0.3)
    ).to(args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load dataset
    print(f"Loading dataset: {args.split}")
    dataset = NSDDataset(
        index_file=args.index_file,
        clip_cache=args.clip_cache,
        preproc_dir=args.preproc_dir,
        subject=args.subject,
        split=args.split
    )
    if args.limit:
        dataset = torch.utils.data.Subset(dataset, range(min(args.limit, len(dataset))))

    # Run MC dropout inference
    print(f"Running MC dropout inference (n={args.n_samples} samples)")
    results = []
    
    for i in tqdm(range(len(dataset))):
        fmri, clip_target = dataset[i]
        fmri = fmri.unsqueeze(0).to(args.device)
        clip_target = clip_target.unsqueeze(0).to(args.device)
        
        # MC dropout prediction
        mc_result = predict_with_mc_dropout(
            model, fmri, n_samples=args.n_samples, device=args.device
        )
        
        # Compute error
        pred_mean = torch.from_numpy(mc_result['mean']).to(args.device)
        error = 1.0 - torch.nn.functional.cosine_similarity(
            pred_mean, clip_target, dim=-1
        ).item()
        
        results.append({
            'trial_idx': i,
            'uncertainty': mc_result['uncertainty'],
            'error': error,
            'std_norm': mc_result['std_norm']
        })
    
    # Compute correlation
    uncertainties = np.array([r['uncertainty'] for r in results])
    errors = np.array([r['error'] for r in results])
    
    correlation = compute_uncertainty_error_correlation(
        uncertainties, errors, method='pearson'
    )
    
    print(f"\\n{'='*60}")
    print(f"Uncertainty-Error Correlation: {correlation:.4f}")
    print(f"Mean uncertainty: {uncertainties.mean():.4f} ± {uncertainties.std():.4f}")
    print(f"Mean error: {errors.mean():.4f} ± {errors.std():.4f}")
    print(f"{'='*60}\\n")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'uncertainty_results.csv', index=False)
    
    summary = {
        'correlation_pearson': float(correlation),
        'mean_uncertainty': float(uncertainties.mean()),
        'std_uncertainty': float(uncertainties.std()),
        'mean_error': float(errors.mean()),
        'std_error': float(errors.std()),
        'n_samples': len(results),
        'mc_dropout_samples': args.n_samples
    }
    
    with open(output_dir / 'uncertainty_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Plot calibration curve
    print("Generating calibration plot...")
    plot_calibration_curve(
        uncertainties, errors,
        save_path=output_dir / 'calibration_curve.png'
    )
    
    print(f"\\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
'''
        
        try:
            with open(script_path, 'w') as f:
                f.write(script_content)
            script_path.chmod(0o755)
            self.print_success(f"Created eval_uncertainty.py")
            return True
        except Exception as e:
            self.print_error(f"Failed to create eval_uncertainty.py: {e}")
            return False
    
    # ============================================================================
    # STEP 8: Generate Report
    # ============================================================================
    
    def generate_report(self, configs: List[Dict]) -> bool:
        """Generate comparison report across all experiments."""
        self.print_header("STEP 7: Generate Comparison Report")
        
        report_dir = self.outputs_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect all metrics
        all_metrics = []
        for config in configs:
            config_name = config["name"]
            eval_dir = self._get_eval_dir(config_name)
            metrics_file = eval_dir / "metrics.json"
            
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)
                metrics["config"] = config_name
                all_metrics.append(metrics)
        
        if not all_metrics:
            self.print_warning("No evaluation metrics found")
            return False
        
        # Create comparison table
        df = pd.DataFrame(all_metrics)
        
        # Save as CSV
        csv_path = report_dir / f"comparison_{self.subject}_{self.mode}.csv"
        df.to_csv(csv_path, index=False)
        self.print_success(f"Saved comparison table: {csv_path}")
        
        # Print summary table
        print(f"\n{Colors.BOLD}Comparison Summary:{Colors.ENDC}\n")
        
        key_metrics = ['config', 'cosine_similarity', 'retrieval_top1', 'retrieval_top5']
        available_metrics = [m for m in key_metrics if m in df.columns]
        
        print(df[available_metrics].to_string(index=False))
        print()
        
        return True
    
    # ============================================================================
    # Main Pipeline Execution
    # ============================================================================
    
    def run(self, resume_from: Optional[str] = None) -> bool:
        """Run the complete pipeline."""
        start_time = time.time()
        
        self.print_header(f"fMRI-to-Image Pipeline - Mode: {self.mode.upper()}")
        self.print_info(f"Subject: {self.subject}")
        self.print_info(f"Root: {self.root}")
        
        if self.dry_run:
            self.print_warning("DRY RUN MODE - No commands will be executed")
        
        if self.force_rebuild:
            self.print_warning("FORCE REBUILD - All steps will be recomputed")
        
        # Step 0: Validate environment
        if not self.validate_environment():
            return False
        
        # Determine which experiments to run
        if self.mode == "ablation":
            experiments = ["baseline", "soft_only", "infonce_only", "novel"]
        else:
            experiments = [self.mode]
        
        configs_to_run = [self.configs[exp] for exp in experiments]
        
        # Step 1: Build index (shared across all experiments)
        if not resume_from or resume_from == "index":
            if not self.build_index():
                return False
        
        # Step 2: Build CLIP cache (shared across all experiments)
        if not resume_from or resume_from in ["index", "clip"]:
            if not self.build_clip_cache():
                return False
        
        # Run each experiment
        for config in configs_to_run:
            config_name = config["name"]
            self.print_header(f"Running Experiment: {config_name}")
            
            # Step 3: Preprocessing
            if not resume_from or resume_from in ["index", "clip", "preproc"]:
                if not self.run_preprocessing(config):
                    self.print_error(f"Preprocessing failed for {config_name}")
                    continue
            
            # Step 4: Training
            if not resume_from or resume_from in ["index", "clip", "preproc", "train"]:
                if not self.run_training(config):
                    self.print_error(f"Training failed for {config_name}")
                    continue
            
            # Step 5: Standard evaluation
            if not resume_from or resume_from in ["index", "clip", "preproc", "train", "eval"]:
                if not self.run_evaluation(config):
                    self.print_error(f"Evaluation failed for {config_name}")
                    continue
            
            # Step 6: Uncertainty evaluation
            if not resume_from or resume_from in ["index", "clip", "preproc", "train", "eval", "uncertainty"]:
                if not self.run_uncertainty_eval(config):
                    self.print_error(f"Uncertainty evaluation failed for {config_name}")
                    continue
        
        # Step 7: Generate comparison report
        if len(configs_to_run) > 1:
            self.generate_report(configs_to_run)
        
        elapsed = time.time() - start_time
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        self.print_header("Pipeline Complete!")
        self.print_success(f"Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end fMRI-to-Image pipeline with novel contributions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with novel contributions
  python scripts/run_full_pipeline.py --subject subj01 --mode novel
  
  # Baseline only
  python scripts/run_full_pipeline.py --subject subj01 --mode baseline
  
  # Full ablation study (4 experiments)
  python scripts/run_full_pipeline.py --subject subj01 --mode ablation
  
  # Resume from training step
  python scripts/run_full_pipeline.py --subject subj01 --mode novel --resume-from train
  
  # Dry run (preview commands without executing)
  python scripts/run_full_pipeline.py --subject subj01 --mode novel --dry-run
        """
    )
    
    parser.add_argument(
        "--subject",
        type=str,
        required=True,
        help="NSD subject ID (e.g., subj01)"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["baseline", "novel", "ablation"],
        help="Pipeline mode: baseline (hard threshold, no InfoNCE), "
             "novel (soft weights + InfoNCE), ablation (all 4 experiments)"
    )
    
    parser.add_argument(
        "--root-dir",
        type=str,
        default=".",
        help="Root directory of the project"
    )
    
    parser.add_argument(
        "--resume-from",
        type=str,
        choices=["index", "clip", "preproc", "train", "eval", "uncertainty"],
        help="Resume pipeline from specific step"
    )
    
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild all steps (ignore cache)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview commands without executing"
    )
    
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation steps (useful for testing training only)"
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    orchestrator = PipelineOrchestrator(
        subject=args.subject,
        mode=args.mode,
        root_dir=args.root_dir,
        force_rebuild=args.force_rebuild,
        dry_run=args.dry_run,
        skip_eval=args.skip_eval
    )
    
    success = orchestrator.run(resume_from=args.resume_from)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
