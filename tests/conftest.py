"""Pytest configuration and shared fixtures for Bird Vision testing."""

import os
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any
import pytest
import torch
import numpy as np
from PIL import Image
from omegaconf import DictConfig, OmegaConf
import json

from bird_vision.models.vision_model import VisionModel
from bird_vision.data.nabirds_dataset import NABirdsDataModule


@pytest.fixture(scope="session")
def test_config() -> DictConfig:
    """Create test configuration."""
    config = {
        "project": {
            "name": "bird_vision_test",
            "version": "0.1.0",
        },
        "data": {
            "dataset_name": "test_dataset",
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "image_size": 224,
            "num_classes": 10,  # Reduced for testing
            "augmentation": {
                "train": [
                    {"Resize": {"size": 256}},
                    {"CenterCrop": {"size": 224}},
                    {"Normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}},
                ],
                "val": [
                    {"Resize": {"size": 256}},
                    {"CenterCrop": {"size": 224}},
                    {"Normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}},
                ],
            },
        },
        "model": {
            "backbone": {
                "name": "efficientnet_b0",  # Smaller for testing
                "pretrained": False,
                "features_only": False,
                "drop_rate": 0.1,
                "drop_path_rate": 0.1,
            },
            "head": {
                "type": "classification",
                "num_classes": 10,
                "dropout": 0.2,
                "hidden_dim": 128,
            },
            "optimizer": {
                "_target_": "torch.optim.AdamW",
                "lr": 0.001,
                "weight_decay": 0.01,
            },
            "scheduler": {
                "_target_": "torch.optim.lr_scheduler.CosineAnnealingLR",
                "T_max": 10,
            },
            "loss": {
                "_target_": "torch.nn.CrossEntropyLoss",
                "label_smoothing": 0.1,
            },
            "metrics": ["accuracy", "f1_macro"],
        },
        "training": {
            "max_epochs": 2,  # Minimal for testing
            "patience": 5,
            "checkpointing": {
                "monitor": "val_accuracy",
                "mode": "max",
                "save_top_k": 1,
                "save_last": True,
                "filename": "test_epoch_{epoch:02d}",
            },
            "early_stopping": {
                "monitor": "val_accuracy",
                "mode": "max",
                "patience": 5,
            },
            "gradient_clipping": {
                "enabled": False,
                "max_norm": 1.0,
            },
            "mixed_precision": {
                "enabled": False,
                "precision": 16,
            },
            "logging": {
                "log_every_n_steps": 1,
            },
        },
        "compression": {
            "quantization": {"enabled": True, "backend": "fbgemm"},
            "pruning": {"enabled": False, "sparsity": 0.3},
            "distillation": {"enabled": False},
            "onnx_export": {"enabled": True, "opset_version": 11},
            "mobile_export": {
                "torchscript": True,
                "coreml": False,  # Disabled for CI
                "tflite": False,
                "esp_dl": False,
            },
        },
        "deployment": {
            "target_platform": "mobile",
            "preprocessing": {
                "input_size": [224, 224],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "format": "RGB",
            },
            "postprocessing": {
                "output_format": "probabilities",
                "top_k": 5,
                "confidence_threshold": 0.1,
            },
        },
        "paths": {
            "data_dir": "./test_data",
            "models_dir": "./test_models",
            "logs_dir": "./test_logs",
            "artifacts_dir": "./test_artifacts",
        },
        "device": "cpu",
        "seed": 42,
        "logging": {
            "level": "INFO",
            "use_wandb": False,
            "use_mlflow": False,
        },
    }
    
    return OmegaConf.create(config)


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture(scope="session")
def mock_dataset(temp_dir: Path, test_config: DictConfig) -> Path:
    """Create mock dataset for testing."""
    dataset_dir = temp_dir / "mock_dataset"
    dataset_dir.mkdir(exist_ok=True)
    
    # Create mock images
    num_classes = test_config.data.num_classes
    samples_per_class = 20
    
    images_dir = dataset_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    labels = []
    image_paths = []
    
    for class_id in range(num_classes):
        class_dir = images_dir / f"class_{class_id}"
        class_dir.mkdir(exist_ok=True)
        
        for sample_id in range(samples_per_class):
            # Create random image
            image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            image_pil = Image.fromarray(image)
            
            image_path = class_dir / f"sample_{sample_id}.jpg"
            image_pil.save(image_path)
            
            image_paths.append(str(image_path))
            labels.append(class_id)
    
    # Create metadata
    metadata = {
        "num_classes": num_classes,
        "num_samples": len(image_paths),
        "image_paths": image_paths,
        "labels": labels,
        "class_names": [f"Bird_Species_{i}" for i in range(num_classes)],
    }
    
    metadata_path = dataset_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return dataset_dir


@pytest.fixture
def sample_model(test_config: DictConfig) -> VisionModel:
    """Create sample model for testing."""
    torch.manual_seed(42)
    model = VisionModel(test_config.model)
    model.eval()
    return model


@pytest.fixture
def sample_input() -> torch.Tensor:
    """Create sample input tensor."""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def sample_batch() -> Dict[str, torch.Tensor]:
    """Create sample batch for testing."""
    return {
        "images": torch.randn(4, 3, 224, 224),
        "targets": torch.randint(0, 10, (4,)),
    }


@pytest.fixture
def device() -> torch.device:
    """Get test device (CPU for CI compatibility)."""
    return torch.device("cpu")


@pytest.fixture(scope="session")
def class_labels() -> list:
    """Sample class labels for testing."""
    return [
        "American Robin",
        "Blue Jay",
        "Cardinal",
        "Sparrow",
        "Eagle",
        "Hawk",
        "Owl",
        "Woodpecker",
        "Hummingbird",
        "Crow",
    ]


@pytest.fixture
def checkpoint_data(sample_model: VisionModel, test_config: DictConfig) -> Dict[str, Any]:
    """Create sample checkpoint data."""
    return {
        "epoch": 5,
        "model_state_dict": sample_model.state_dict(),
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
        "best_val_metric": 0.85,
        "val_metrics": {
            "accuracy": 0.85,
            "f1_macro": 0.82,
            "loss": 0.45,
        },
        "train_metrics": {
            "accuracy": 0.90,
            "f1_macro": 0.88,
            "loss": 0.35,
        },
        "config": OmegaConf.to_yaml(test_config),
    }


@pytest.fixture(autouse=True)
def setup_test_environment(temp_dir: Path, test_config: DictConfig):
    """Setup test environment before each test."""
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create test directories
    for path_key, path_value in test_config.paths.items():
        test_path = temp_dir / path_value.lstrip("./")
        test_path.mkdir(parents=True, exist_ok=True)
        # Update config to use absolute paths
        test_config.paths[path_key] = str(test_path)
    
    # Set environment variables for testing
    os.environ["TESTING"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU
    
    yield
    
    # Cleanup
    if "TESTING" in os.environ:
        del os.environ["TESTING"]


class MockDataset:
    """Mock dataset for testing without real data."""
    
    def __init__(self, num_samples: int = 100, num_classes: int = 10):
        self.num_samples = num_samples
        self.num_classes = num_classes
        
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> tuple:
        # Generate deterministic but varied data
        np.random.seed(idx)
        image = torch.randn(3, 224, 224)
        label = torch.tensor(idx % self.num_classes, dtype=torch.long)
        return image, label


@pytest.fixture
def mock_data_loader(test_config: DictConfig):
    """Create mock data loader."""
    from torch.utils.data import DataLoader
    
    dataset = MockDataset(
        num_samples=40,
        num_classes=test_config.data.num_classes
    )
    
    return DataLoader(
        dataset,
        batch_size=test_config.data.batch_size,
        shuffle=False,
        num_workers=0,
    )


# Performance testing fixtures
@pytest.fixture
def performance_thresholds() -> Dict[str, Dict[str, float]]:
    """Performance thresholds for different platforms."""
    return {
        "mobile": {
            "max_model_size_mb": 50,
            "max_inference_time_ms": 100,
            "min_accuracy": 0.80,
        },
        "esp32": {
            "max_model_size_mb": 8,
            "max_inference_time_ms": 200,
            "min_accuracy": 0.75,
        },
    }


@pytest.fixture(scope="session")
def benchmark_config() -> Dict[str, Any]:
    """Configuration for benchmark tests."""
    return {
        "num_warmup_runs": 5,
        "num_benchmark_runs": 50,
        "input_sizes": [(1, 3, 224, 224), (1, 3, 512, 512)],
        "batch_sizes": [1, 4, 8],
        "memory_threshold_mb": 1000,
    }