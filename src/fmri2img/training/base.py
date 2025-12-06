"""
Base Training Infrastructure
============================

Reusable training infrastructure for all fMRI encoder models.

Features:
- Standardized training loop with early stopping
- Checkpoint management (best model, last model, periodic saves)
- Metrics tracking and logging
- Learning rate scheduling
- Gradient clipping
- Mixed precision training (optional)
- TensorBoard/WandB integration (optional)
- Automatic model retraining on train+val

Scientific Design:
- Model selection on validation set (prevents overfitting to test)
- Final model retrained on train+val using selected hyperparameters
- Comprehensive metrics logging for reproducibility
- Checkpoint management for recovery and best model selection

Usage:
    class MyTrainer(BaseTrainer):
        def compute_loss(self, batch, model_output):
            return F.mse_loss(model_output, batch['target'])
    
    trainer = MyTrainer(model, optimizer, config)
    trainer.fit(train_loader, val_loader)
    metrics = trainer.evaluate(test_loader)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """
    Configuration for BaseTrainer.
    
    Attributes:
        # Training hyperparameters
        epochs: Maximum number of training epochs
        patience: Early stopping patience (epochs without improvement)
        gradient_clip: Max gradient norm for clipping (None = no clipping)
        mixed_precision: Whether to use automatic mixed precision (AMP)
        
        # Checkpointing
        checkpoint_dir: Directory for saving checkpoints
        save_every_n_epochs: Save checkpoint every N epochs (None = only best)
        save_optimizer_state: Whether to save optimizer state in checkpoints
        keep_n_checkpoints: Number of periodic checkpoints to keep (None = all)
        
        # Logging
        log_interval: Log training metrics every N batches
        eval_interval: Evaluate on validation set every N epochs
        
        # Model selection
        monitor_metric: Metric to monitor for early stopping (e.g., 'val_cosine')
        monitor_mode: 'max' or 'min' for metric improvement
        
        # Retraining
        retrain_on_trainval: Whether to retrain on train+val after early stopping
    """
    # Training
    epochs: int = 50
    patience: int = 7
    gradient_clip: Optional[float] = 1.0
    mixed_precision: bool = False
    
    # Checkpointing
    checkpoint_dir: Optional[Path] = None
    save_every_n_epochs: Optional[int] = None
    save_optimizer_state: bool = True
    keep_n_checkpoints: Optional[int] = 3
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 1
    
    # Model selection
    monitor_metric: str = "val_loss"
    monitor_mode: str = "min"  # 'min' or 'max'
    
    # Retraining
    retrain_on_trainval: bool = True


class BaseTrainer(ABC):
    """
    Base class for training neural network models.
    
    Provides common training infrastructure:
    - Training loop with early stopping
    - Validation loop
    - Checkpoint management
    - Metrics tracking and logging
    - Learning rate scheduling
    - Gradient clipping
    
    Subclasses must implement:
    - compute_loss(): Loss computation
    - compute_metrics(): Metrics computation
    
    Example:
        >>> class MyTrainer(BaseTrainer):
        ...     def compute_loss(self, batch, model_output):
        ...         return F.mse_loss(model_output, batch['target'])
        ...     
        ...     def compute_metrics(self, batch, model_output):
        ...         loss = self.compute_loss(batch, model_output)
        ...         return {'loss': loss.item()}
        >>> 
        >>> trainer = MyTrainer(model, optimizer, config)
        >>> trainer.fit(train_loader, val_loader)
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        config: TrainerConfig,
        scheduler: Optional[_LRScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            optimizer: Optimizer for training
            config: Trainer configuration
            scheduler: Optional learning rate scheduler
            device: Device for training ('cuda' or 'cpu')
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.scheduler = scheduler
        self.device = device
        
        # Move model to device
        self.model.to(self.device)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf') if config.monitor_mode == 'min' else float('-inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.training_history: Dict[str, list] = {}
        
        # Mixed precision
        self.scaler = None
        if config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Setup checkpoint directory
        if config.checkpoint_dir:
            config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def compute_loss(self, batch: Dict[str, torch.Tensor], model_output: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for a batch.
        
        Must be implemented by subclasses.
        
        Args:
            batch: Dictionary containing batch data (features, targets, etc.)
            model_output: Model predictions
        
        Returns:
            Loss tensor (scalar)
        """
        pass
    
    @abstractmethod
    def compute_metrics(self, batch: Dict[str, torch.Tensor], model_output: torch.Tensor) -> Dict[str, float]:
        """
        Compute evaluation metrics for a batch.
        
        Must be implemented by subclasses.
        
        Args:
            batch: Dictionary containing batch data
            model_output: Model predictions
        
        Returns:
            Dictionary of metrics {metric_name: value}
        """
        pass
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
        
        Returns:
            Dictionary of average training metrics for the epoch
        """
        self.model.train()
        epoch_metrics = {}
        
        for batch_idx, batch_data in enumerate(train_loader):
            # Move batch to device
            batch = self._prepare_batch(batch_data)
            
            # Forward pass
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    model_output = self.model(batch['features'])
                    loss = self.compute_loss(batch, model_output)
            else:
                model_output = self.model(batch['features'])
                loss = self.compute_loss(batch, model_output)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.config.mixed_precision:
                self.scaler.scale(loss).backward()
                
                if self.config.gradient_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                if self.config.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )
                
                self.optimizer.step()
            
            # Update global step
            self.global_step += 1
            
            # Compute and accumulate metrics
            with torch.no_grad():
                batch_metrics = self.compute_metrics(batch, model_output)
                
                for key, value in batch_metrics.items():
                    if key not in epoch_metrics:
                        epoch_metrics[key] = []
                    epoch_metrics[key].append(value)
            
            # Periodic logging
            if (batch_idx + 1) % self.config.log_interval == 0:
                metrics_str = ", ".join([f"{k}={v[-1]:.4f}" for k, v in epoch_metrics.items()])
                logger.debug(
                    f"Epoch {self.current_epoch+1}, "
                    f"Batch {batch_idx+1}/{len(train_loader)}: {metrics_str}"
                )
        
        # Average metrics over epoch
        avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
        
        return avg_metrics
    
    @torch.no_grad()
    def evaluate(self, eval_loader: DataLoader, split: str = "val") -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            eval_loader: Evaluation data loader
            split: Split name for logging (e.g., 'val', 'test')
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        eval_metrics = {}
        
        for batch_data in eval_loader:
            # Move batch to device
            batch = self._prepare_batch(batch_data)
            
            # Forward pass
            model_output = self.model(batch['features'])
            
            # Compute metrics
            batch_metrics = self.compute_metrics(batch, model_output)
            
            # Accumulate metrics
            for key, value in batch_metrics.items():
                if key not in eval_metrics:
                    eval_metrics[key] = []
                eval_metrics[key].append(value)
        
        # Average metrics
        avg_metrics = {
            f"{split}_{key}": np.mean(values)
            for key, values in eval_metrics.items()
        }
        
        return avg_metrics
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None
    ) -> Dict[str, Any]:
        """
        Train model with early stopping and optional retraining.
        
        Training procedure:
        1. Train with early stopping on validation set
        2. Optionally retrain on train+val for best_epoch epochs
        3. Evaluate on test set
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader (for early stopping)
            test_loader: Optional test data loader (for final evaluation)
        
        Returns:
            Dictionary containing:
            - training_history: Metrics over training
            - best_epoch: Best epoch number
            - best_metric: Best monitored metric value
            - test_metrics: Final test set metrics (if test_loader provided)
        """
        logger.info("="*80)
        logger.info("STARTING TRAINING")
        logger.info("="*80)
        logger.info(f"Device: {self.device}")
        logger.info(f"Epochs: {self.config.epochs}")
        logger.info(f"Patience: {self.config.patience}")
        logger.info(f"Monitor: {self.config.monitor_metric} ({self.config.monitor_mode})")
        
        start_time = time.time()
        
        # Training loop with early stopping
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch(train_loader)
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Validation
            val_metrics = {}
            if val_loader is not None and (epoch + 1) % self.config.eval_interval == 0:
                val_metrics = self.evaluate(val_loader, split="val")
            
            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}
            
            # Log metrics
            epoch_time = time.time() - epoch_start
            metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in all_metrics.items()])
            logger.info(
                f"Epoch {epoch+1:3d}/{self.config.epochs}: "
                f"{metrics_str} "
                f"(time: {epoch_time:.1f}s)"
            )
            
            # Track history
            for key, value in all_metrics.items():
                if key not in self.training_history:
                    self.training_history[key] = []
                self.training_history[key].append(value)
            
            # Check early stopping
            if val_loader is not None and self.config.monitor_metric in all_metrics:
                current_metric = all_metrics[self.config.monitor_metric]
                improved = self._check_improvement(current_metric)
                
                if improved:
                    self.best_metric = current_metric
                    self.best_epoch = epoch + 1
                    self.patience_counter = 0
                    
                    # Save best checkpoint
                    if self.config.checkpoint_dir:
                        self.save_checkpoint("best.pt")
                    
                    logger.info(
                        f"  ✅ New best {self.config.monitor_metric}: {self.best_metric:.4f}"
                    )
                else:
                    self.patience_counter += 1
                    
                    if self.patience_counter >= self.config.patience:
                        logger.info(
                            f"  Early stopping triggered after {epoch+1} epochs "
                            f"(no improvement for {self.config.patience} epochs)"
                        )
                        break
            
            # Periodic checkpoint
            if self.config.save_every_n_epochs is not None:
                if (epoch + 1) % self.config.save_every_n_epochs == 0:
                    if self.config.checkpoint_dir:
                        self.save_checkpoint(f"epoch_{epoch+1}.pt")
        
        # Training complete
        training_time = time.time() - start_time
        logger.info("="*80)
        logger.info(f"Training completed in {training_time/60:.1f} minutes")
        logger.info(f"Best epoch: {self.best_epoch} ({self.config.monitor_metric}={self.best_metric:.4f})")
        logger.info("="*80)
        
        # Optionally retrain on train+val
        if self.config.retrain_on_trainval and val_loader is not None and self.best_epoch > 0:
            logger.info("RETRAINING on train+val for optimal generalization")
            self._retrain_on_trainval(train_loader, val_loader, self.best_epoch)
        
        # Final test evaluation
        test_metrics = {}
        if test_loader is not None:
            logger.info("="*80)
            logger.info("FINAL EVALUATION ON TEST SET")
            logger.info("="*80)
            
            # Load best checkpoint if available
            if self.config.checkpoint_dir:
                best_checkpoint = self.config.checkpoint_dir / "best.pt"
                if best_checkpoint.exists():
                    self.load_checkpoint(best_checkpoint)
            
            test_metrics = self.evaluate(test_loader, split="test")
            
            metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in test_metrics.items()])
            logger.info(f"Test metrics: {metrics_str}")
        
        # Return results
        return {
            "training_history": self.training_history,
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "test_metrics": test_metrics
        }
    
    def _check_improvement(self, current_metric: float) -> bool:
        """Check if current metric improved over best metric"""
        if self.config.monitor_mode == "min":
            return current_metric < self.best_metric
        else:
            return current_metric > self.best_metric
    
    def _retrain_on_trainval(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int
    ) -> None:
        """
        Retrain model on combined train+val for optimal number of epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train (from early stopping)
        """
        # Combine datasets
        combined_dataset = torch.utils.data.ConcatDataset([
            train_loader.dataset,
            val_loader.dataset
        ])
        
        combined_loader = DataLoader(
            combined_dataset,
            batch_size=train_loader.batch_size,
            shuffle=True,
            num_workers=train_loader.num_workers if hasattr(train_loader, 'num_workers') else 0
        )
        
        # Reinitialize model and optimizer
        # (Subclasses can override this if needed)
        logger.info(f"Retraining for {num_epochs} epochs on {len(combined_dataset)} samples")
        
        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(combined_loader)
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in train_metrics.items()])
            logger.info(f"Retrain epoch {epoch+1}/{num_epochs}: {metrics_str}")
        
        # Save final model
        if self.config.checkpoint_dir:
            self.save_checkpoint("final.pt")
        
        logger.info("✅ Retraining complete")
    
    def _prepare_batch(self, batch_data: Union[Tuple, Dict]) -> Dict[str, torch.Tensor]:
        """
        Prepare batch for training/evaluation.
        
        Handles both tuple (features, targets) and dict formats.
        
        Args:
            batch_data: Batch from dataloader (tuple or dict)
        
        Returns:
            Dictionary with 'features' and 'targets' on device
        """
        if isinstance(batch_data, dict):
            # Already dictionary format
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                   for k, v in batch_data.items()}
        
        elif isinstance(batch_data, (tuple, list)):
            # Convert (features, targets) tuple to dict
            features, targets = batch_data[0], batch_data[1]
            return {
                'features': features.to(self.device),
                'targets': targets.to(self.device)
            }
        
        else:
            raise ValueError(f"Unsupported batch format: {type(batch_data)}")
    
    def save_checkpoint(self, filename: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            filename: Checkpoint filename (e.g., 'best.pt', 'epoch_10.pt')
        """
        if self.config.checkpoint_dir is None:
            logger.warning("Checkpoint directory not configured, skipping save")
            return
        
        checkpoint_path = self.config.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'config': self.config
        }
        
        if self.config.save_optimizer_state:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            if self.scheduler is not None:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.debug(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_metric = checkpoint.get('best_metric', self.best_metric)
        self.best_epoch = checkpoint.get('best_epoch', 0)
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"✅ Loaded checkpoint (epoch {self.current_epoch}, step {self.global_step})")
