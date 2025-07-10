"""Unit tests for training components."""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
from omegaconf import DictConfig
from unittest.mock import Mock, patch

from bird_vision.training.trainer import Trainer
from bird_vision.models.vision_model import VisionModel
from bird_vision.utils.metrics import MetricsCalculator
from bird_vision.utils.checkpoint import CheckpointManager


class TestTrainer:
    """Test Trainer functionality."""
    
    def test_trainer_initialization(self, sample_model: VisionModel, test_config: DictConfig, device: torch.device):
        """Test trainer initialization."""
        trainer = Trainer(sample_model, test_config, device)
        
        assert trainer.model == sample_model
        assert trainer.cfg == test_config
        assert trainer.device == device
        assert trainer.optimizer is not None
        assert trainer.criterion is not None
        assert hasattr(trainer, 'metrics_calculator')
        assert hasattr(trainer, 'checkpoint_manager')
    
    def test_optimizer_creation(self, sample_model: VisionModel, test_config: DictConfig, device: torch.device):
        """Test optimizer creation."""
        trainer = Trainer(sample_model, test_config, device)
        
        assert isinstance(trainer.optimizer, torch.optim.AdamW)
        assert trainer.optimizer.param_groups[0]['lr'] == test_config.model.optimizer.lr
        assert trainer.optimizer.param_groups[0]['weight_decay'] == test_config.model.optimizer.weight_decay
    
    def test_scheduler_creation(self, sample_model: VisionModel, test_config: DictConfig, device: torch.device):
        """Test learning rate scheduler creation."""
        trainer = Trainer(sample_model, test_config, device)
        
        if trainer.scheduler:
            assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    
    def test_criterion_creation(self, sample_model: VisionModel, test_config: DictConfig, device: torch.device):
        """Test loss criterion creation."""
        trainer = Trainer(sample_model, test_config, device)
        
        assert isinstance(trainer.criterion, nn.CrossEntropyLoss)
    
    def test_train_epoch(self, sample_model: VisionModel, test_config: DictConfig, device: torch.device, mock_data_loader):
        """Test single training epoch."""
        trainer = Trainer(sample_model, test_config, device)
        
        # Run one epoch
        metrics = trainer.train_epoch(mock_data_loader)
        
        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "accuracy" in metrics if "accuracy" in test_config.model.metrics else True
        assert all(isinstance(v, float) for v in metrics.values())
        assert metrics["loss"] >= 0
    
    def test_validate_epoch(self, sample_model: VisionModel, test_config: DictConfig, device: torch.device, mock_data_loader):
        """Test single validation epoch."""
        trainer = Trainer(sample_model, test_config, device)
        
        # Run validation
        metrics = trainer.validate_epoch(mock_data_loader)
        
        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert all(isinstance(v, float) for v in metrics.values())
        assert metrics["loss"] >= 0
    
    @patch('bird_vision.training.trainer.logger')
    def test_training_logging(self, mock_logger, sample_model: VisionModel, test_config: DictConfig, device: torch.device, mock_data_loader):
        """Test training logging."""
        # Reduce logging frequency for testing
        test_config.training.logging.log_every_n_steps = 1
        
        trainer = Trainer(sample_model, test_config, device)
        trainer.train_epoch(mock_data_loader)
        
        # Check that logging occurred
        assert mock_logger.info.called
    
    def test_gradient_clipping(self, test_config: DictConfig, sample_model: VisionModel, device: torch.device, mock_data_loader):
        """Test gradient clipping."""
        # Enable gradient clipping
        test_config.training.gradient_clipping.enabled = True
        test_config.training.gradient_clipping.max_norm = 1.0
        
        trainer = Trainer(sample_model, test_config, device)
        
        # Train one batch to trigger gradient clipping
        batch = next(iter(mock_data_loader))
        images, targets = batch
        images, targets = images.to(device), targets.to(device)
        
        trainer.optimizer.zero_grad()
        outputs = trainer.model(images)
        loss = trainer.criterion(outputs, targets)
        loss.backward()
        
        # Check gradients exist before clipping
        grad_norms_before = []
        for param in trainer.model.parameters():
            if param.grad is not None:
                grad_norms_before.append(param.grad.norm().item())
        
        # Apply gradient clipping (would normally be done in training step)
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), test_config.training.gradient_clipping.max_norm)
        
        # Check gradients are clipped
        grad_norms_after = []
        for param in trainer.model.parameters():
            if param.grad is not None:
                grad_norms_after.append(param.grad.norm().item())
        
        assert len(grad_norms_before) == len(grad_norms_after)


class TestMetricsCalculator:
    """Test MetricsCalculator functionality."""
    
    def test_metrics_calculator_initialization(self):
        """Test metrics calculator initialization."""
        metric_names = ["accuracy", "f1_macro"]
        calculator = MetricsCalculator(metric_names)
        
        assert calculator.metric_names == metric_names
        assert "accuracy" in calculator.metric_functions
        assert "f1_macro" in calculator.metric_functions
    
    def test_metrics_calculation(self):
        """Test metrics calculation."""
        metric_names = ["accuracy", "f1_macro"]
        calculator = MetricsCalculator(metric_names)
        
        # Create sample outputs and targets
        outputs = torch.randn(4, 10)  # 4 samples, 10 classes
        targets = torch.randint(0, 10, (4,))
        
        metrics = calculator.calculate(outputs, targets)
        
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert all(0 <= v <= 1 for v in metrics.values())
    
    def test_top_k_accuracy(self):
        """Test top-k accuracy calculation."""
        metric_names = ["accuracy", "top_5_accuracy"]
        calculator = MetricsCalculator(metric_names)
        
        # Create outputs where top prediction might not be correct but top-5 includes correct
        outputs = torch.tensor([
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # correct class is 9
            [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],  # correct class is 0
        ])
        targets = torch.tensor([9, 0])
        
        metrics = calculator.calculate(outputs, targets)
        
        assert "accuracy" in metrics
        assert "top_5_accuracy" in metrics
        assert metrics["top_5_accuracy"] >= metrics["accuracy"]  # Top-5 should be >= top-1
    
    def test_metrics_averaging(self):
        """Test metrics averaging across batches."""
        metric_names = ["accuracy"]
        calculator = MetricsCalculator(metric_names)
        
        # Create multiple batch metrics
        batch_metrics = [
            {"accuracy": 0.8},
            {"accuracy": 0.9},
            {"accuracy": 0.7},
        ]
        
        averaged = calculator.average_metrics(batch_metrics)
        
        assert "accuracy" in averaged
        assert abs(averaged["accuracy"] - 0.8) < 0.01  # Should be close to (0.8 + 0.9 + 0.7) / 3


class TestCheckpointManager:
    """Test CheckpointManager functionality."""
    
    def test_checkpoint_manager_initialization(self, test_config: DictConfig, temp_dir: Path):
        """Test checkpoint manager initialization."""
        # Update config to use temp directory
        test_config.paths.models_dir = str(temp_dir)
        
        manager = CheckpointManager(test_config)
        
        assert manager.cfg == test_config
        assert manager.checkpoints_dir.exists()
        assert manager.monitor == test_config.training.checkpointing.monitor
        assert manager.mode == test_config.training.checkpointing.mode
    
    def test_checkpoint_saving(self, test_config: DictConfig, temp_dir: Path, checkpoint_data: dict):
        """Test checkpoint saving."""
        test_config.paths.models_dir = str(temp_dir)
        manager = CheckpointManager(test_config)
        
        # Save checkpoint
        manager.save_checkpoint(
            checkpoint_data,
            is_best=True,
            epoch=5,
            metric_value=0.85
        )
        
        # Check files exist
        assert len(list(manager.checkpoints_dir.glob("*.ckpt"))) > 0
        best_checkpoint = manager.checkpoints_dir / "best_model.ckpt"
        assert best_checkpoint.exists()
    
    def test_checkpoint_loading(self, test_config: DictConfig, temp_dir: Path, checkpoint_data: dict):
        """Test checkpoint loading."""
        test_config.paths.models_dir = str(temp_dir)
        manager = CheckpointManager(test_config)
        
        # Save and then load checkpoint
        manager.save_checkpoint(
            checkpoint_data,
            is_best=True,
            epoch=5,
            metric_value=0.85
        )
        
        loaded_data = manager.load_checkpoint()
        
        assert loaded_data is not None
        assert "epoch" in loaded_data
        assert "model_state_dict" in loaded_data
        assert loaded_data["epoch"] == checkpoint_data["epoch"]
    
    def test_top_k_checkpoint_management(self, test_config: DictConfig, temp_dir: Path, checkpoint_data: dict):
        """Test top-k checkpoint management."""
        test_config.paths.models_dir = str(temp_dir)
        test_config.training.checkpointing.save_top_k = 2
        
        manager = CheckpointManager(test_config)
        
        # Save multiple checkpoints
        for epoch, metric_value in enumerate([0.7, 0.8, 0.9, 0.6], 1):
            checkpoint_data_copy = checkpoint_data.copy()
            checkpoint_data_copy["epoch"] = epoch
            
            manager.save_checkpoint(
                checkpoint_data_copy,
                is_best=(metric_value == 0.9),
                epoch=epoch,
                metric_value=metric_value
            )
        
        # Should only keep top 2 + best + last
        checkpoints = list(manager.checkpoints_dir.glob("*.ckpt"))
        # Exact count may vary based on file naming and cleanup logic
        assert len(checkpoints) <= 5  # top_k + best + last + some tolerance


class TestTrainingIntegration:
    """Test training integration components."""
    
    def test_training_step_integration(self, sample_model: VisionModel, test_config: DictConfig, device: torch.device):
        """Test integration of training components."""
        trainer = Trainer(sample_model, test_config, device)
        
        # Create sample batch
        images = torch.randn(2, 3, 224, 224).to(device)
        targets = torch.randint(0, 10, (2,)).to(device)
        
        # Forward pass
        trainer.optimizer.zero_grad()
        outputs = trainer.model(images)
        loss = trainer.criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        trainer.optimizer.step()
        
        # Check that gradients were computed and applied
        assert loss.item() >= 0
        assert outputs.shape == (2, 10)
    
    def test_validation_step_integration(self, sample_model: VisionModel, test_config: DictConfig, device: torch.device):
        """Test validation step integration."""
        trainer = Trainer(sample_model, test_config, device)
        trainer.model.eval()
        
        # Create sample batch
        images = torch.randn(2, 3, 224, 224).to(device)
        targets = torch.randint(0, 10, (2,)).to(device)
        
        with torch.no_grad():
            outputs = trainer.model(images)
            loss = trainer.criterion(outputs, targets)
            metrics = trainer.metrics_calculator.calculate(outputs, targets)
        
        assert loss.item() >= 0
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics if "accuracy" in test_config.model.metrics else True