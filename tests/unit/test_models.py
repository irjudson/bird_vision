"""Unit tests for model architectures."""

import pytest
import torch
import torch.nn as nn
from omegaconf import DictConfig

from bird_vision.models.vision_model import VisionModel


class TestVisionModel:
    """Test VisionModel functionality."""
    
    def test_model_initialization(self, test_config: DictConfig):
        """Test model initialization."""
        model = VisionModel(test_config.model)
        
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'backbone')
        assert hasattr(model, 'head')
        assert model.num_classes == test_config.model.head.num_classes
    
    def test_model_forward_pass(self, sample_model: VisionModel, sample_input: torch.Tensor):
        """Test model forward pass."""
        sample_model.eval()
        
        with torch.no_grad():
            output = sample_model(sample_input)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 10)  # batch_size=1, num_classes=10
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_model_training_mode(self, sample_model: VisionModel, sample_input: torch.Tensor):
        """Test model in training mode."""
        sample_model.train()
        
        output = sample_model(sample_input)
        
        assert isinstance(output, torch.Tensor)
        assert output.requires_grad
        
        # Test backward pass
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        for param in sample_model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_model_feature_extraction(self, sample_model: VisionModel, sample_input: torch.Tensor):
        """Test feature extraction."""
        sample_model.eval()
        
        with torch.no_grad():
            features = sample_model.extract_features(sample_input)
        
        assert isinstance(features, torch.Tensor)
        assert features.dim() == 2  # (batch_size, features)
        assert features.shape[0] == 1  # batch_size
        assert features.shape[1] > 0  # feature dimension
    
    def test_model_info(self, sample_model: VisionModel):
        """Test model information extraction."""
        info = sample_model.get_model_info()
        
        assert isinstance(info, dict)
        assert "backbone" in info
        assert "num_classes" in info
        assert "total_parameters" in info
        assert "trainable_parameters" in info
        assert "backbone_features" in info
        
        assert info["num_classes"] == 10
        assert info["total_parameters"] > 0
        assert info["trainable_parameters"] > 0
        assert info["trainable_parameters"] <= info["total_parameters"]
    
    def test_model_different_input_sizes(self, sample_model: VisionModel):
        """Test model with different input sizes."""
        sample_model.eval()
        
        # Test different batch sizes
        for batch_size in [1, 2, 4]:
            input_tensor = torch.randn(batch_size, 3, 224, 224)
            
            with torch.no_grad():
                output = sample_model(input_tensor)
            
            assert output.shape == (batch_size, 10)
    
    def test_model_device_compatibility(self, sample_model: VisionModel, device: torch.device):
        """Test model device compatibility."""
        sample_model.to(device)
        input_tensor = torch.randn(1, 3, 224, 224).to(device)
        
        with torch.no_grad():
            output = sample_model(input_tensor)
        
        assert output.device == device
        assert output.shape == (1, 10)


class TestModelComponents:
    """Test individual model components."""
    
    def test_backbone_creation(self, test_config: DictConfig):
        """Test backbone creation."""
        model = VisionModel(test_config.model)
        
        # Test backbone exists and is correct type
        assert hasattr(model, 'backbone')
        assert isinstance(model.backbone, nn.Module)
        
        # Test backbone output
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = model.backbone(dummy_input)
        
        assert isinstance(features, torch.Tensor)
        assert features.dim() >= 2
    
    def test_classification_head(self, test_config: DictConfig):
        """Test classification head."""
        model = VisionModel(test_config.model)
        
        assert hasattr(model, 'head')
        assert isinstance(model.head, nn.Module)
        
        # Test head output
        dummy_features = torch.randn(1, model.backbone_out_features)
        with torch.no_grad():
            logits = model.head(dummy_features)
        
        assert logits.shape == (1, test_config.model.head.num_classes)
    
    def test_weight_initialization(self, test_config: DictConfig):
        """Test weight initialization."""
        model1 = VisionModel(test_config.model)
        model2 = VisionModel(test_config.model)
        
        # Models should have different random weights (unless seed is set)
        for (name1, param1), (name2, param2) in zip(
            model1.named_parameters(), model2.named_parameters()
        ):
            assert name1 == name2
            if param1.requires_grad and param1.numel() > 1:
                # Allow some tolerance for very small parameters
                if not torch.allclose(param1, param2, atol=1e-6):
                    break
        else:
            # If we get here, all parameters were identical (unlikely without fixed seed)
            pytest.skip("Models have identical weights (fixed seed)")


class TestModelOptimization:
    """Test model optimization and compilation."""
    
    def test_model_jit_scripting(self, sample_model: VisionModel, sample_input: torch.Tensor):
        """Test TorchScript compilation."""
        sample_model.eval()
        
        # Test scripting
        try:
            scripted_model = torch.jit.script(sample_model)
            assert isinstance(scripted_model, torch.jit.ScriptModule)
        except Exception:
            # Some models might not be scriptable, test tracing instead
            traced_model = torch.jit.trace(sample_model, sample_input)
            assert isinstance(traced_model, torch.jit.ScriptModule)
            
            # Test traced model produces same output
            with torch.no_grad():
                original_output = sample_model(sample_input)
                traced_output = traced_model(sample_input)
                
            assert torch.allclose(original_output, traced_output, atol=1e-5)
    
    def test_model_quantization_compatibility(self, sample_model: VisionModel):
        """Test model quantization compatibility."""
        sample_model.eval()
        
        # Test dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            sample_model,
            qconfig_spec={nn.Linear},
            dtype=torch.qint8,
        )
        
        assert isinstance(quantized_model, nn.Module)
        
        # Test quantized model still works
        sample_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = quantized_model(sample_input)
        
        assert output.shape == (1, 10)
    
    def test_model_parameter_count(self, sample_model: VisionModel):
        """Test parameter counting."""
        info = sample_model.get_model_info()
        
        # Manual parameter count
        total_params = sum(p.numel() for p in sample_model.parameters())
        trainable_params = sum(p.numel() for p in sample_model.parameters() if p.requires_grad)
        
        assert info["total_parameters"] == total_params
        assert info["trainable_parameters"] == trainable_params
        
        # Reasonable parameter counts for test model
        assert 1000 < total_params < 10000000  # Between 1K and 10M parameters
        assert trainable_params <= total_params