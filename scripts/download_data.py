#!/usr/bin/env python3
"""Script to download and prepare NABirds dataset."""

import deeplake
from pathlib import Path
from loguru import logger


def download_nabirds_dataset(data_dir: str = "./data"):
    """Download NABirds dataset using Deep Lake."""
    logger.info("Downloading NABirds dataset...")
    
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset from Deep Lake Hub
    dataset = deeplake.load("hub://activeloop/nabirds")
    
    logger.info(f"Dataset loaded successfully!")
    logger.info(f"Total samples: {len(dataset)}")
    logger.info(f"Dataset structure: {dataset.summary()}")
    
    return dataset


if __name__ == "__main__":
    dataset = download_nabirds_dataset()
    print("NABirds dataset ready for training!")