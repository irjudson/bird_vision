"""Training loop and utilities."""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import mlflow
from omegaconf import DictConfig
from loguru import logger

from bird_vision.utils.metrics import MetricsCalculator
from bird_vision.utils.checkpoint import CheckpointManager


class Trainer:
    """Training class for bird vision models."""
    
    def __init__(
        self,
        model: nn.Module,
        cfg: DictConfig,
        device: torch.device,
    ):
        self.model = model
        self.cfg = cfg
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Initialize training components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = self._create_criterion()
        self.scaler = GradScaler() if cfg.training.mixed_precision.enabled else None
        
        # Metrics and logging
        self.metrics_calculator = MetricsCalculator(cfg.model.metrics)
        self.checkpoint_manager = CheckpointManager(cfg)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = 0.0
        self.patience_counter = 0
        
        # Logging
        self.writer = None
        if cfg.logging.get("use_tensorboard", True):
            log_dir = Path(cfg.paths.logs_dir) / "tensorboard"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        optimizer_cfg = self.cfg.model.optimizer
        optimizer_class = getattr(torch.optim, optimizer_cfg._target_.split('.')[-1])
        
        # Remove _target_ from params
        params = {k: v for k, v in optimizer_cfg.items() if k != '_target_'}
        return optimizer_class(self.model.parameters(), **params)
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler from config."""
        if not hasattr(self.cfg.model, 'scheduler'):
            return None
        
        scheduler_cfg = self.cfg.model.scheduler
        scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_cfg._target_.split('.')[-1])
        
        # Remove _target_ from params
        params = {k: v for k, v in scheduler_cfg.items() if k != '_target_'}
        return scheduler_class(self.optimizer, **params)
    
    def _create_criterion(self) -> nn.Module:
        """Create loss function from config."""
        loss_cfg = self.cfg.model.loss
        loss_class = getattr(nn, loss_cfg._target_.split('.')[-1])
        
        # Remove _target_ from params
        params = {k: v for k, v in loss_cfg.items() if k != '_target_'}
        return loss_class(**params)
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = []
        epoch_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.scaler:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.cfg.training.gradient_clipping.enabled:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.training.gradient_clipping.max_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.cfg.training.gradient_clipping.enabled:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.training.gradient_clipping.max_norm
                    )
                
                self.optimizer.step()
            
            # Calculate metrics
            batch_metrics = self.metrics_calculator.calculate(outputs, targets)
            epoch_metrics.append(batch_metrics)
            epoch_loss += loss.item()
            
            # Logging
            if batch_idx % self.cfg.training.logging.log_every_n_steps == 0:
                lr = self.optimizer.param_groups[0]['lr']
                logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}, LR: {lr:.6f}"
                )
                
                if self.writer:
                    self.writer.add_scalar("train/batch_loss", loss.item(), self.global_step)
                    self.writer.add_scalar("train/learning_rate", lr, self.global_step)
            
            self.global_step += 1
        
        # Average metrics over epoch
        avg_metrics = self.metrics_calculator.average_metrics(epoch_metrics)
        avg_metrics["loss"] = epoch_loss / len(train_loader)
        
        return avg_metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_metrics = []
        epoch_loss = 0.0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                
                # Forward pass
                if self.scaler:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                
                # Calculate metrics
                batch_metrics = self.metrics_calculator.calculate(outputs, targets)
                epoch_metrics.append(batch_metrics)
                epoch_loss += loss.item()
        
        # Average metrics over epoch
        avg_metrics = self.metrics_calculator.average_metrics(epoch_metrics)
        avg_metrics["loss"] = epoch_loss / len(val_loader)
        
        return avg_metrics
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Main training loop."""
        logger.info("Starting training...")
        start_time = time.time()
        
        for epoch in range(self.cfg.training.max_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate_epoch(val_loader)
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
            
            # Logging
            epoch_time = time.time() - epoch_start_time
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics.get('accuracy', 0):.4f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # TensorBoard logging
            if self.writer:
                for metric, value in train_metrics.items():
                    self.writer.add_scalar(f"train/{metric}", value, epoch)
                for metric, value in val_metrics.items():
                    self.writer.add_scalar(f"val/{metric}", value, epoch)
            
            # MLflow logging
            if self.cfg.logging.get("use_mlflow", False):
                mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()}, step=epoch)
                mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()}, step=epoch)
            
            # Checkpointing
            current_val_metric = val_metrics.get(
                self.cfg.training.checkpointing.monitor.replace("val_", ""),
                val_metrics.get("accuracy", 0)
            )
            
            is_best = current_val_metric > self.best_val_metric
            if is_best:
                self.best_val_metric = current_val_metric
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.checkpoint_manager.save_checkpoint({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'best_val_metric': self.best_val_metric,
                'val_metrics': val_metrics,
                'train_metrics': train_metrics,
            }, is_best, epoch, current_val_metric)
            
            # Early stopping
            if self.patience_counter >= self.cfg.training.early_stopping.patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")
        
        if self.writer:
            self.writer.close()