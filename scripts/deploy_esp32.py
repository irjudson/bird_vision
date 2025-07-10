#!/usr/bin/env python3
"""ESP32-P4-Eye deployment script."""

import hydra
from omegaconf import DictConfig
import torch
from pathlib import Path
from loguru import logger

from bird_vision.models.vision_model import VisionModel
from bird_vision.deployment.esp32_deployer import ESP32Deployer


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Deploy model to ESP32-P4-Eye."""
    logger.info("Starting ESP32-P4-Eye deployment")
    
    # Override config for ESP32
    cfg.deployment = hydra.compose(config_name="deployment/esp32")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model (you would specify the actual model path here)
    model_path = "models/checkpoints/best_model.ckpt"  # Update this path
    
    if not Path(model_path).exists():
        logger.error(f"Model checkpoint not found: {model_path}")
        logger.info("Please train a model first or provide a valid checkpoint path")
        return
    
    # Load model
    model = VisionModel(cfg.model)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded: {model.get_model_info()}")
    
    # Create sample input
    sample_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Deploy to ESP32
    deployer = ESP32Deployer(cfg)
    results = deployer.prepare_for_esp32(model, sample_input, "bird_classifier_esp32")
    
    # Display results
    if results.get("esp_dl_model", {}).get("success"):
        logger.success("✅ ESP-DL model conversion successful")
    else:
        logger.error("❌ ESP-DL model conversion failed")
    
    if results.get("firmware", {}).get("success"):
        firmware_dir = results["firmware"]["firmware_dir"]
        logger.success("✅ Firmware generation successful")
        logger.info(f"📁 Firmware location: {firmware_dir}")
        logger.info("🔧 To build and flash:")
        logger.info(f"   cd {firmware_dir}")
        logger.info("   ./build.sh")
        logger.info("   ./flash.sh")
    else:
        logger.error("❌ Firmware generation failed")
    
    if results.get("package", {}).get("success"):
        package_dir = results["package"]["package_dir"]
        logger.success(f"📦 Complete package: {package_dir}")
    
    logger.success("ESP32-P4-Eye deployment completed!")


if __name__ == "__main__":
    main()