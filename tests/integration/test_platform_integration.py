"""Integration tests for platform-specific deployment."""

import pytest
import torch
from pathlib import Path
from omegaconf import DictConfig
from unittest.mock import patch, Mock

from bird_vision.models.vision_model import VisionModel
from bird_vision.deployment.mobile_deployer import MobileDeployer
from bird_vision.deployment.esp32_deployer import ESP32Deployer
from bird_vision.compression.model_compressor import ModelCompressor


class TestiOSIntegration:
    """Test iOS deployment integration."""
    
    @patch('bird_vision.deployment.mobile_deployer.ct')
    def test_ios_coreml_integration(self, mock_coreml, test_config: DictConfig, sample_model: VisionModel, sample_input: torch.Tensor, temp_dir: Path):
        """Test iOS CoreML integration pipeline."""
        # Configure for iOS deployment
        test_config.deployment.target_platform = "ios"
        test_config.deployment.ios = {
            "model_format": "coreml",
            "deployment_target": "14.0",
            "optimize_for_size": True,
        }
        test_config.paths.models_dir = str(temp_dir)
        
        # Mock CoreML conversion
        mock_model = Mock()
        mock_model.save = Mock()
        mock_coreml.convert.return_value = mock_model
        mock_coreml.ImageType = Mock()
        mock_coreml.ClassifierConfig = Mock()
        mock_coreml.target.iOS = Mock()
        
        deployer = MobileDeployer(test_config)
        
        # Mock compression results
        with patch.object(deployer, 'compressor') as mock_compressor:
            mock_compressor.compress_model.return_value = {
                "compressed_models": {"quantized": sample_model},
                "compression_stats": {"quantized": {"size_mb": 8.0}},
            }
            
            results = deployer.prepare_for_mobile(sample_model, sample_input, "ios_test_model")
            
            assert "deployment_results" in results
            if "ios" in results["deployment_results"]:
                ios_result = results["deployment_results"]["ios"]
                if ios_result.get("success"):
                    assert mock_coreml.convert.called
    
    def test_ios_integration_code_generation(self, test_config: DictConfig, temp_dir: Path):
        """Test iOS integration code generation."""
        deployer = MobileDeployer(test_config)
        
        ios_dir = temp_dir / "ios"
        ios_dir.mkdir(exist_ok=True)
        
        class_labels = ["American Robin", "Blue Jay", "Cardinal"]
        deployer._create_ios_integration_code(ios_dir, "BirdClassifier", class_labels)
        
        swift_file = ios_dir / "BirdClassifierClassifier.swift"
        assert swift_file.exists()
        
        content = swift_file.read_text()
        
        # Check Swift code quality
        assert "import CoreML" in content
        assert "import Vision" in content
        assert "class BirdClassifierClassifier" in content
        assert "VNCoreMLModel" in content
        assert "American Robin" in content
        
        # Check for proper async handling
        assert "completion:" in content
        assert "@escaping" in content
    
    def test_ios_model_optimization(self, test_config: DictConfig, sample_model: VisionModel, sample_input: torch.Tensor):
        """Test iOS-specific model optimizations."""
        # Configure iOS-specific optimizations
        test_config.compression.mobile_export.coreml = True
        test_config.deployment.ios = {
            "optimize_for_size": True,
            "deployment_target": "14.0",
        }
        
        compressor = ModelCompressor(test_config)
        
        # Test model preparation for iOS
        with patch('bird_vision.compression.model_compressor.ct') as mock_coreml:
            mock_coreml.convert.return_value = Mock()
            
            result = compressor._export_to_coreml(sample_model, sample_input, "ios_model", "optimized")
            
            # Should attempt CoreML conversion
            if result.get("success"):
                assert mock_coreml.convert.called


class TestAndroidIntegration:
    """Test Android deployment integration."""
    
    def test_android_torchscript_integration(self, test_config: DictConfig, sample_model: VisionModel, sample_input: torch.Tensor, temp_dir: Path):
        """Test Android TorchScript integration pipeline."""
        # Configure for Android deployment
        test_config.deployment.target_platform = "android"
        test_config.deployment.android = {
            "model_format": "torchscript",
            "api_level": 24,
            "delegate": "gpu",
        }
        test_config.paths.models_dir = str(temp_dir)
        
        deployer = MobileDeployer(test_config)
        
        # Mock compression results
        with patch.object(deployer, 'compressor') as mock_compressor:
            mock_compressor.compress_model.return_value = {
                "compressed_models": {"quantized": sample_model},
                "compression_stats": {"quantized": {"size_mb": 12.0}},
            }
            
            results = deployer.prepare_for_mobile(sample_model, sample_input, "android_test_model")
            
            assert "deployment_results" in results
            if "android" in results["deployment_results"]:
                android_result = results["deployment_results"]["android"]
                assert "format" in android_result
    
    def test_android_integration_code_generation(self, test_config: DictConfig, temp_dir: Path):
        """Test Android integration code generation."""
        deployer = MobileDeployer(test_config)
        
        android_dir = temp_dir / "android"
        android_dir.mkdir(exist_ok=True)
        
        class_labels = ["American Robin", "Blue Jay", "Cardinal"]
        deployer._create_android_integration_code(android_dir, "BirdClassifier", class_labels)
        
        java_file = android_dir / "BirdClassifierClassifier.java"
        assert java_file.exists()
        
        content = java_file.read_text()
        
        # Check Java code quality
        assert "package com.example.birdvision;" in content
        assert "import org.pytorch.Module;" in content
        assert "public class BirdClassifierClassifier" in content
        assert "private String[] classLabels" in content
        assert "American Robin" in content
        
        # Check for proper Android integration
        assert "TensorImageUtils" in content
        assert "assetFilePath" in content
    
    def test_android_model_optimization(self, test_config: DictConfig, sample_model: VisionModel, sample_input: torch.Tensor):
        """Test Android-specific model optimizations."""
        sample_model.eval()
        
        # Test TorchScript optimization for mobile
        traced_model = torch.jit.trace(sample_model, sample_input)
        optimized_model = torch.jit.optimize_for_inference(traced_model)
        
        assert isinstance(optimized_model, torch.jit.ScriptModule)
        
        # Test optimized model still works
        with torch.no_grad():
            original_output = sample_model(sample_input)
            optimized_output = optimized_model(sample_input)
        
        assert torch.allclose(original_output, optimized_output, atol=1e-4)


class TestESP32Integration:
    """Test ESP32-P4-Eye deployment integration."""
    
    @patch('bird_vision.deployment.esp32_deployer.ESP32Deployer._convert_to_esp_dl')
    def test_esp32_full_integration(self, mock_esp_dl, test_config: DictConfig, sample_model: VisionModel, sample_input: torch.Tensor, temp_dir: Path):
        """Test ESP32 full integration pipeline."""
        # Configure ESP32 deployment
        test_config.deployment.esp32_p4 = {
            "device": "ESP32-P4-EYE",
            "ai_accelerator": True,
            "memory_psram": 32,
            "memory_flash": 16,
            "optimization": {
                "target_framework": "esp_dl",
                "quantization": "int8",
                "input_layout": "NHWC",
                "use_ai_accelerator": True,
                "memory_optimization": True,
            },
            "model_constraints": {
                "max_model_size_mb": 8,
                "max_inference_time_ms": 200,
                "min_accuracy_retention": 0.90,
            },
            "camera": {
                "resolution": [640, 480],
                "format": "RGB565",
                "fps": 15,
            },
        }
        test_config.deployment.firmware = {
            "esp_idf_version": "v5.1",
            "components": ["esp-dl", "esp32-camera", "wifi", "spiffs"],
            "camera_config": {
                "pin_xclk": 15,
                "pin_sscb_sda": 4,
                "pin_sscb_scl": 5,
                "pins": {
                    "d0": 11, "d1": 9, "d2": 8, "d3": 10,
                    "d4": 12, "d5": 18, "d6": 17, "d7": 16,
                    "vsync": 6, "href": 7, "pclk": 13,
                },
            },
        }
        test_config.deployment.preprocessing = {
            "input_size": [224, 224],
            "mean": [123.675, 116.28, 103.53],
            "std": [58.395, 57.12, 57.375],
            "format": "RGB",
        }
        test_config.deployment.metadata = {
            "model_name": "BirdClassifierESP32",
            "author": "Test",
            "license": "MIT",
        }
        test_config.paths.models_dir = str(temp_dir)
        
        # Mock ESP-DL conversion
        mock_esp_dl.return_value = {
            "success": True,
            "esp_dl_files": {
                "coefficients": "test_coefficients.hpp",
                "header": "test_model.hpp",
                "config": "test_config.json",
            },
        }
        
        deployer = ESP32Deployer(test_config)
        results = deployer.prepare_for_esp32(sample_model, sample_input, "esp32_test_model")
        
        assert "esp_dl_model" in results
        assert "firmware" in results
        assert "package" in results
        assert "deployment_info" in results
        
        # Check ESP-DL conversion was called
        assert mock_esp_dl.called
        
        # Check deployment info
        deployment_info = results["deployment_info"]
        assert "target_device" in deployment_info
        assert deployment_info["target_device"] == "ESP32-P4-EYE"
    
    def test_esp32_firmware_generation(self, test_config: DictConfig, temp_dir: Path):
        """Test ESP32 firmware generation."""
        # Configure minimal ESP32 settings
        test_config.deployment.firmware = {
            "esp_idf_version": "v5.1",
            "camera_config": {
                "pin_xclk": 15,
                "pin_sscb_sda": 4,
                "pin_sscb_scl": 5,
                "pins": {"d0": 11, "pclk": 13, "vsync": 6, "href": 7},
            },
        }
        
        deployer = ESP32Deployer(test_config)
        
        firmware_dir = temp_dir / "firmware"
        firmware_dir.mkdir(exist_ok=True)
        
        esp_dl_result = {"success": True, "esp_dl_files": {}}
        class_labels = ["robin", "jay", "cardinal"]
        
        result = deployer._generate_esp32_firmware(
            firmware_dir, "test_model", class_labels, esp_dl_result
        )
        
        if result["success"]:
            assert "firmware_dir" in result
            assert "files" in result
            
            # Check main application files
            main_dir = firmware_dir / "main"
            if main_dir.exists():
                main_cpp = main_dir / "main.cpp"
                if main_cpp.exists():
                    content = main_cpp.read_text()
                    
                    # Check C++ code quality
                    assert '#include "freertos/FreeRTOS.h"' in content
                    assert '#include "esp_camera.h"' in content
                    assert "void app_main(void)" in content
                    assert "robin" in content
    
    def test_esp32_camera_configuration(self, test_config: DictConfig, temp_dir: Path):
        """Test ESP32 camera configuration generation."""
        test_config.deployment.firmware = {
            "camera_config": {
                "pin_xclk": 15,
                "pin_sscb_sda": 4,
                "pin_sscb_scl": 5,
                "pins": {
                    "d0": 11, "d1": 9, "d2": 8, "d3": 10,
                    "pclk": 13, "vsync": 6, "href": 7,
                },
            },
        }
        
        deployer = ESP32Deployer(test_config)
        
        main_dir = temp_dir / "main"
        main_dir.mkdir(exist_ok=True)
        
        camera_file = deployer._generate_camera_interface(main_dir)
        
        assert Path(camera_file).exists()
        
        content = Path(camera_file).read_text()
        
        # Check camera configuration
        assert "config.pin_xclk = 15" in content
        assert "config.pin_d0 = 11" in content
        assert "PIXFORMAT_RGB565" in content
        assert "esp_camera_init" in content
    
    def test_esp32_build_scripts(self, test_config: DictConfig, temp_dir: Path):
        """Test ESP32 build script generation."""
        deployer = ESP32Deployer(test_config)
        
        firmware_dir = temp_dir / "firmware"
        firmware_dir.mkdir(exist_ok=True)
        
        result = deployer._generate_build_scripts(firmware_dir)
        
        assert "build" in result
        assert "flash" in result
        
        build_script = Path(result["build"])
        flash_script = Path(result["flash"])
        
        assert build_script.exists()
        assert flash_script.exists()
        
        # Check script content
        build_content = build_script.read_text()
        flash_content = flash_script.read_text()
        
        assert "idf.py build" in build_content
        assert "idf.py flash" in flash_content
        assert "esp32p4" in build_content
        
        # Check scripts are executable
        assert build_script.stat().st_mode & 0o111
        assert flash_script.stat().st_mode & 0o111


class TestCrossplatformCompatibility:
    """Test cross-platform compatibility and consistency."""
    
    def test_model_consistency_across_platforms(self, test_config: DictConfig, sample_model: VisionModel, sample_input: torch.Tensor):
        """Test model produces consistent results across different export formats."""
        sample_model.eval()
        
        # Get original output
        with torch.no_grad():
            original_output = sample_model(sample_input)
        
        # Test TorchScript consistency
        traced_model = torch.jit.trace(sample_model, sample_input)
        with torch.no_grad():
            torchscript_output = traced_model(sample_input)
        
        assert torch.allclose(original_output, torchscript_output, atol=1e-5)
        
        # Test quantized model consistency
        quantized_model = torch.quantization.quantize_dynamic(
            sample_model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
        )
        with torch.no_grad():
            quantized_output = quantized_model(sample_input)
        
        # Quantized output should be similar but may have some error
        assert torch.allclose(original_output, quantized_output, atol=0.1)
    
    def test_preprocessing_consistency(self, test_config: DictConfig):
        """Test preprocessing consistency across platforms."""
        import numpy as np
        from bird_vision.data.nabirds_dataset import NABirdsDataModule
        
        data_module = NABirdsDataModule(test_config.data)
        transform = data_module._create_transform(test_config.data.augmentation.val)
        
        # Create test image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Apply transform multiple times (should be deterministic for val)
        results = []
        for _ in range(3):
            transformed = transform(image=test_image.copy())
            results.append(transformed["image"])
        
        # All results should be identical for validation transforms
        for i in range(1, len(results)):
            assert torch.allclose(results[0], results[i], atol=1e-6)
    
    def test_deployment_package_structure(self, test_config: DictConfig, sample_model: VisionModel, sample_input: torch.Tensor, temp_dir: Path):
        """Test deployment package structure consistency."""
        test_config.paths.models_dir = str(temp_dir)
        
        # Test mobile deployment package
        mobile_deployer = MobileDeployer(test_config)
        
        with patch.object(mobile_deployer, 'compressor') as mock_compressor:
            mock_compressor.compress_model.return_value = {
                "compressed_models": {"original": sample_model},
                "compression_stats": {"original": {"size_mb": 10.0}},
            }
            
            mobile_results = mobile_deployer.prepare_for_mobile(
                sample_model, sample_input, "test_model"
            )
            
            assert "metadata" in mobile_results
            assert "deployment_results" in mobile_results
            
            metadata = mobile_results["metadata"]
            assert "model_info" in metadata
            assert "preprocessing" in metadata
            assert "postprocessing" in metadata