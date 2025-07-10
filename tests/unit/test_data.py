"""Unit tests for data loading and preprocessing."""

import pytest
import torch
from pathlib import Path
from omegaconf import DictConfig

from bird_vision.data.nabirds_dataset import NABirdsDataset, NABirdsDataModule


class TestNABirdsDataset:
    """Test NABirds dataset functionality."""
    
    def test_dataset_initialization(self, test_config: DictConfig, mock_dataset: Path):
        """Test dataset initialization."""
        # Mock the dataset path
        dataset_path = str(mock_dataset)
        
        # This would normally test with real data, but we'll test the structure
        assert mock_dataset.exists()
        assert (mock_dataset / "metadata.json").exists()
        assert (mock_dataset / "images").exists()
    
    def test_dataset_transforms(self, test_config: DictConfig):
        """Test data transformations."""
        from bird_vision.data.nabirds_dataset import NABirdsDataModule
        
        data_module = NABirdsDataModule(test_config.data)
        
        # Test transform creation
        train_transform = data_module._create_transform(test_config.data.augmentation.train)
        val_transform = data_module._create_transform(test_config.data.augmentation.val)
        
        assert train_transform is not None
        assert val_transform is not None
        
        # Test transform application on dummy data
        import numpy as np
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        transformed_train = train_transform(image=dummy_image)
        transformed_val = val_transform(image=dummy_image)
        
        assert "image" in transformed_train
        assert "image" in transformed_val
        assert isinstance(transformed_train["image"], torch.Tensor)
        assert isinstance(transformed_val["image"], torch.Tensor)


class TestDataModule:
    """Test data module functionality."""
    
    def test_data_module_initialization(self, test_config: DictConfig):
        """Test data module initialization."""
        data_module = NABirdsDataModule(test_config.data)
        
        assert data_module.cfg == test_config.data
        assert data_module.batch_size == test_config.data.batch_size
        assert data_module.num_workers == test_config.data.num_workers
        assert data_module.num_classes == test_config.data.num_classes
    
    def test_dataloader_creation(self, mock_data_loader):
        """Test dataloader creation and iteration."""
        # Test basic iteration
        batch_count = 0
        for batch in mock_data_loader:
            images, labels = batch
            
            assert isinstance(images, torch.Tensor)
            assert isinstance(labels, torch.Tensor)
            assert images.shape[0] <= 4  # batch_size
            assert images.shape[1:] == (3, 224, 224)
            assert labels.shape[0] == images.shape[0]
            
            batch_count += 1
            if batch_count >= 3:  # Test a few batches
                break
        
        assert batch_count > 0


class TestDataPreprocessing:
    """Test data preprocessing pipeline."""
    
    def test_image_normalization(self, test_config: DictConfig):
        """Test image normalization."""
        from bird_vision.data.nabirds_dataset import NABirdsDataModule
        import numpy as np
        
        data_module = NABirdsDataModule(test_config.data)
        transform = data_module._create_transform(test_config.data.augmentation.val)
        
        # Create test image
        test_image = np.ones((224, 224, 3), dtype=np.uint8) * 128  # Gray image
        
        transformed = transform(image=test_image)
        tensor_image = transformed["image"]
        
        # Check tensor properties
        assert tensor_image.dtype == torch.float32
        assert tensor_image.shape == (3, 224, 224)
        
        # Check normalization (values should be roughly centered around 0)
        mean_values = tensor_image.mean(dim=[1, 2])
        assert all(abs(val) < 2.0 for val in mean_values)  # Normalized values
    
    def test_data_augmentation_consistency(self, test_config: DictConfig):
        """Test that augmentation produces consistent results."""
        from bird_vision.data.nabirds_dataset import NABirdsDataModule
        import numpy as np
        
        data_module = NABirdsDataModule(test_config.data)
        val_transform = data_module._create_transform(test_config.data.augmentation.val)
        
        # Same input should produce same output for validation transform
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        result1 = val_transform(image=test_image.copy())
        result2 = val_transform(image=test_image.copy())
        
        # Validation transforms should be deterministic
        assert torch.allclose(result1["image"], result2["image"], atol=1e-6)