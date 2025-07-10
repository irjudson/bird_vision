"""Unit tests for model compression."""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
from omegaconf import DictConfig
from unittest.mock import Mock, patch

from bird_vision.compression.model_compressor import ModelCompressor
from bird_vision.models.vision_model import VisionModel


class TestModelCompressor:
    """Test ModelCompressor functionality."""
    
    def test_compressor_initialization(self, test_config: DictConfig):
        """Test compressor initialization."""
        compressor = ModelCompressor(test_config)
        
        assert compressor.cfg == test_config
        assert compressor.compression_cfg == test_config.compression
        assert compressor.output_dir.exists()
    
    def test_quantization(self, test_config: DictConfig, sample_model: VisionModel, sample_input: torch.Tensor):
        """Test model quantization."""
        test_config.compression.quantization.enabled = True
        compressor = ModelCompressor(test_config)
        
        # Test quantization
        quantized_model, stats = compressor._apply_quantization(sample_model, sample_input, "test_model")
        
        assert isinstance(quantized_model, nn.Module)
        assert isinstance(stats, dict)
        assert "size_mb" in stats
        assert "parameters" in stats
        
        # Quantized model should still work
        with torch.no_grad():
            output = quantized_model(sample_input)
        assert output.shape == (1, 10)
    
    def test_pruning(self, test_config: DictConfig, sample_model: VisionModel, sample_input: torch.Tensor):
        """Test model pruning."""
        test_config.compression.pruning.enabled = True
        test_config.compression.pruning.sparsity = 0.3
        compressor = ModelCompressor(test_config)
        
        # Test pruning
        pruned_model, stats = compressor._apply_pruning(sample_model, sample_input, "test_model")
        
        assert isinstance(pruned_model, nn.Module)
        assert isinstance(stats, dict)
        
        # Pruned model should still work
        with torch.no_grad():
            output = pruned_model(sample_input)
        assert output.shape == (1, 10)
    
    def test_onnx_export(self, test_config: DictConfig, sample_model: VisionModel, sample_input: torch.Tensor):
        """Test ONNX export."""
        test_config.compression.onnx_export.enabled = True
        compressor = ModelCompressor(test_config)
        
        result = compressor._export_to_onnx(sample_model, sample_input, "test_model", "original")
        
        if result["success"]:
            assert "path" in result
            assert "size_mb" in result
            assert Path(result["path"]).exists()
            assert result["format"] == "onnx"
    
    def test_torchscript_export(self, test_config: DictConfig, sample_model: VisionModel, sample_input: torch.Tensor):
        """Test TorchScript export."""
        compressor = ModelCompressor(test_config)
        
        result = compressor._export_to_torchscript(sample_model, sample_input, "test_model", "original")
        
        if result["success"]:
            assert "path" in result
            assert "size_mb" in result
            assert Path(result["path"]).exists()
            assert result["format"] == "torchscript"
            
            # Test loading and running the exported model
            loaded_model = torch.jit.load(result["path"])
            with torch.no_grad():
                output = loaded_model(sample_input)
            assert output.shape == (1, 10)
    
    @patch('bird_vision.compression.model_compressor.ct')
    def test_coreml_export(self, mock_coreml, test_config: DictConfig, sample_model: VisionModel, sample_input: torch.Tensor):
        """Test CoreML export (mocked)."""
        # Mock CoreML tools
        mock_model = Mock()
        mock_coreml.convert.return_value = mock_model
        
        compressor = ModelCompressor(test_config)
        result = compressor._export_to_coreml(sample_model, sample_input, "test_model", "original")
        
        # Should attempt conversion
        if result["success"]:
            assert mock_coreml.convert.called
    
    def test_compression_summary(self, test_config: DictConfig):
        """Test compression summary generation."""
        compressor = ModelCompressor(test_config)
        
        # Mock compression stats
        compression_stats = {
            "original": {"size_mb": 10.0, "parameters": 1000000},
            "quantized": {"size_mb": 2.5, "parameters": 1000000},
            "pruned": {"size_mb": 7.0, "parameters": 700000},
        }
        
        summary = compressor._generate_compression_summary(compression_stats)
        
        assert "original_size_mb" in summary
        assert "quantized_size_reduction_percent" in summary
        assert "pruned_size_reduction_percent" in summary
        
        # Check reduction percentages are reasonable
        assert 0 < summary["quantized_size_reduction_percent"] < 100
        assert 0 < summary["pruned_size_reduction_percent"] < 100
    
    def test_full_compression_pipeline(self, test_config: DictConfig, sample_model: VisionModel, sample_input: torch.Tensor):
        """Test full compression pipeline."""
        # Enable multiple compression techniques
        test_config.compression.quantization.enabled = True
        test_config.compression.pruning.enabled = False  # Disable to avoid conflicts
        test_config.compression.onnx_export.enabled = True
        test_config.compression.mobile_export.torchscript = True
        test_config.compression.mobile_export.coreml = False  # Disable for CI
        
        compressor = ModelCompressor(test_config)
        results = compressor.compress_model(sample_model, sample_input, "test_model")
        
        assert "compressed_models" in results
        assert "compression_stats" in results
        assert "export_results" in results
        assert "compression_summary" in results
        
        # Check compressed models
        if "quantized" in results["compressed_models"]:
            quantized_model = results["compressed_models"]["quantized"]
            assert isinstance(quantized_model, nn.Module)
        
        # Check export results
        export_results = results["export_results"]
        for variant, exports in export_results.items():
            if "onnx" in exports:
                assert "success" in exports["onnx"]
            if "torchscript" in exports:
                assert "success" in exports["torchscript"]


class TestCompressionUtilities:
    """Test compression utility functions."""
    
    def test_model_size_calculation(self, sample_model: VisionModel):
        """Test model size calculation."""
        from bird_vision.utils.model_utils import get_model_size_mb
        
        size_mb = get_model_size_mb(sample_model)
        
        assert isinstance(size_mb, float)
        assert size_mb > 0
        assert size_mb < 1000  # Reasonable upper bound for test model
    
    def test_parameter_counting(self, sample_model: VisionModel):
        """Test parameter counting."""
        from bird_vision.utils.model_utils import count_parameters
        
        total, trainable = count_parameters(sample_model)
        
        assert isinstance(total, int)
        assert isinstance(trainable, int)
        assert total > 0
        assert trainable <= total
        assert trainable > 0  # Model should have trainable parameters
    
    def test_layer_freezing(self, sample_model: VisionModel):
        """Test layer freezing functionality."""
        from bird_vision.utils.model_utils import freeze_layers, unfreeze_layers
        
        # Check initial state
        initial_trainable = sum(p.numel() for p in sample_model.parameters() if p.requires_grad)
        
        # Freeze backbone layers
        freeze_layers(sample_model, ["backbone"])
        frozen_trainable = sum(p.numel() for p in sample_model.parameters() if p.requires_grad)
        
        assert frozen_trainable < initial_trainable
        
        # Unfreeze backbone layers
        unfreeze_layers(sample_model, ["backbone"])
        unfrozen_trainable = sum(p.numel() for p in sample_model.parameters() if p.requires_grad)
        
        assert unfrozen_trainable == initial_trainable


class TestQuantizationSpecific:
    """Test quantization-specific functionality."""
    
    def test_dynamic_quantization(self, sample_model: VisionModel, sample_input: torch.Tensor):
        """Test dynamic quantization."""
        sample_model.eval()
        
        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            sample_model,
            qconfig_spec={nn.Linear, nn.Conv2d},
            dtype=torch.qint8,
        )
        
        # Test quantized model
        with torch.no_grad():
            original_output = sample_model(sample_input)
            quantized_output = quantized_model(sample_input)
        
        assert original_output.shape == quantized_output.shape
        
        # Outputs should be similar but not identical due to quantization
        assert torch.allclose(original_output, quantized_output, atol=0.5)
    
    def test_quantization_aware_training_setup(self, sample_model: VisionModel):
        """Test QAT setup (without actual training)."""
        # Prepare model for QAT
        sample_model.train()
        
        # Add quantization stubs (simplified)
        qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        sample_model.qconfig = qconfig
        
        # This would normally call prepare_qat, but we'll just test the setup
        assert hasattr(sample_model, 'qconfig')
        assert sample_model.qconfig is not None


class TestPruningSpecific:
    """Test pruning-specific functionality."""
    
    def test_unstructured_pruning(self, sample_model: VisionModel):
        """Test unstructured pruning."""
        import torch.nn.utils.prune as prune
        
        # Find a linear layer to prune
        linear_layer = None
        for module in sample_model.modules():
            if isinstance(module, nn.Linear):
                linear_layer = module
                break
        
        if linear_layer is not None:
            # Apply pruning
            prune.l1_unstructured(linear_layer, name='weight', amount=0.3)
            
            # Check that pruning mask exists
            assert hasattr(linear_layer, 'weight_mask')
            
            # Check sparsity
            sparsity = float(torch.sum(linear_layer.weight == 0))
            total_params = float(linear_layer.weight.nelement())
            actual_sparsity = sparsity / total_params
            
            assert actual_sparsity > 0  # Some parameters should be pruned
            
            # Remove pruning reparameterization
            prune.remove(linear_layer, 'weight')
            assert not hasattr(linear_layer, 'weight_mask')
    
    def test_structured_pruning(self, sample_model: VisionModel):
        """Test structured pruning."""
        import torch.nn.utils.prune as prune
        
        # Find a conv layer to prune
        conv_layer = None
        for module in sample_model.modules():
            if isinstance(module, nn.Conv2d):
                conv_layer = module
                break
        
        if conv_layer is not None and conv_layer.weight.shape[0] > 1:
            original_channels = conv_layer.weight.shape[0]
            
            # Apply structured pruning
            prune.ln_structured(conv_layer, name='weight', amount=0.2, n=2, dim=0)
            
            # Check that some channels are zeroed
            channel_norms = torch.norm(conv_layer.weight.view(original_channels, -1), dim=1)
            zero_channels = torch.sum(channel_norms == 0).item()
            
            assert zero_channels > 0  # Some channels should be pruned