"""Checkpoint management utilities."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from omegaconf import DictConfig
from loguru import logger


class CheckpointManager:
    """Manage model checkpoints with configurable saving strategy."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.checkpoints_dir = Path(cfg.paths.models_dir) / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        self.monitor = cfg.training.checkpointing.monitor
        self.mode = cfg.training.checkpointing.mode
        self.save_top_k = cfg.training.checkpointing.save_top_k
        self.save_last = cfg.training.checkpointing.save_last
        self.filename_template = cfg.training.checkpointing.filename
        
        self.best_checkpoints = []  # List of (score, filepath) tuples
        self.last_checkpoint_path = None
    
    def save_checkpoint(
        self,
        checkpoint_dict: Dict[str, Any],
        is_best: bool,
        epoch: int,
        metric_value: float,
    ) -> None:
        """Save checkpoint with configured strategy."""
        # Create filename
        filename = self.filename_template.format(
            epoch=epoch,
            **{self.monitor.replace("val_", ""): metric_value}
        )
        checkpoint_path = self.checkpoints_dir / f"{filename}.ckpt"
        
        # Save checkpoint
        torch.save(checkpoint_dict, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Update last checkpoint
        if self.save_last:
            self.last_checkpoint_path = checkpoint_path
        
        # Manage top-k checkpoints
        if self.save_top_k > 0:
            self._update_top_k_checkpoints(metric_value, checkpoint_path)
        
        # Save best checkpoint separately
        if is_best:
            best_path = self.checkpoints_dir / "best_model.ckpt"
            torch.save(checkpoint_dict, best_path)
            logger.info(f"Saved best checkpoint: {best_path}")
    
    def _update_top_k_checkpoints(self, metric_value: float, checkpoint_path: Path) -> None:
        """Update the list of top-k checkpoints."""
        # Add current checkpoint
        self.best_checkpoints.append((metric_value, checkpoint_path))
        
        # Sort based on mode (max or min)
        reverse_sort = self.mode == "max"
        self.best_checkpoints.sort(key=lambda x: x[0], reverse=reverse_sort)
        
        # Keep only top-k checkpoints
        if len(self.best_checkpoints) > self.save_top_k:
            # Remove worst checkpoint file
            _, worst_path = self.best_checkpoints.pop()
            if worst_path.exists() and worst_path != self.last_checkpoint_path:
                os.remove(worst_path)
                logger.info(f"Removed checkpoint: {worst_path}")
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """Load checkpoint from path or best checkpoint."""
        if checkpoint_path is None:
            checkpoint_path = self.checkpoints_dir / "best_model.ckpt"
        else:
            checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        return torch.load(checkpoint_path, map_location="cpu")
    
    def get_best_checkpoint_path(self) -> Optional[Path]:
        """Get path to the best checkpoint."""
        best_path = self.checkpoints_dir / "best_model.ckpt"
        return best_path if best_path.exists() else None
    
    def get_last_checkpoint_path(self) -> Optional[Path]:
        """Get path to the last checkpoint."""
        return self.last_checkpoint_path
    
    def list_checkpoints(self) -> List[Path]:
        """List all available checkpoints."""
        return list(self.checkpoints_dir.glob("*.ckpt"))