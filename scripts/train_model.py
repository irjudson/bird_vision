#!/usr/bin/env python3
"""Training script for bird vision models."""

import hydra
from omegaconf import DictConfig
import torch
from loguru import logger

from bird_vision.data.nabirds_dataset import NABirdsDataModule
from bird_vision.models.vision_model import VisionModel
from bird_vision.training.trainer import Trainer


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main training function."""
    logger.info("Starting bird vision model training")
    
    # Set random seed
    torch.manual_seed(cfg.seed)
    
    # Setup device
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    
    logger.info(f"Using device: {device}")
    
    # Setup data
    data_module = NABirdsDataModule(cfg.data)
    data_module.setup()
    
    # Setup model
    model = VisionModel(cfg.model)
    logger.info(f"Model info: {model.get_model_info()}")
    
    # Setup trainer
    trainer = Trainer(model, cfg, device)
    
    # Train
    trainer.fit(data_module.train_dataloader(), data_module.val_dataloader())
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()