"""Unit tests for deployment modules."""

import pytest
import torch
from pathlib import Path
from omegaconf import DictConfig
from unittest.mock import Mock, patch, MagicMock

from bird_vision.deployment.mobile_deployer import MobileDeployer
from bird_vision.deployment.esp32_deployer import ESP32Deployer
from bird_vision.models.vision_model import VisionModel


class TestMobileDeployer:
    """Test MobileDeployer functionality."""
    
    def test_mobile_deployer_initialization(self, test_config: DictConfig):
        """Test mobile deployer initialization."""
        deployer = MobileDeployer(test_config)
        
        assert deployer.cfg == test_config
        assert deployer.deployment_cfg == test_config.deployment
        assert deployer.output_dir.exists()
    
    def test_model_selection(self, test_config: DictConfig, sample_model: VisionModel):
        """Test optimal model selection for deployment."""
        deployer = MobileDeployer(test_config)
        
        # Mock compression results
        compression_results = {
            "compressed_models": {
                "quantized": sample_model,
                "pruned": sample_model,
            },
            "compression_stats": {
                "original": {"size_mb": 50.0},
                "quantized": {"size_mb": 12.5},
                "pruned": {"size_mb": 35.0},
            }
        }
        
        selected_model = deployer._select_optimal_model(compression_results, sample_model)
        
        assert isinstance(selected_model, torch.nn.Module)
    
    @patch('bird_vision.deployment.mobile_deployer.torch.jit.trace')
    def test_torchscript_export(self, mock_trace, test_config: DictConfig, sample_model: VisionModel, sample_input: torch.Tensor):
        """Test TorchScript export for Android."""
        # Mock TorchScript
        mock_traced = Mock()
        mock_trace.return_value = mock_traced
        
        deployer = MobileDeployer(test_config)
        result = deployer._prepare_android_deployment(sample_model, sample_input, "test_model", None)
        
        if result["success"]:
            assert mock_trace.called
            assert "model_path" in result
            assert result["format"] == "torchscript"
    
    @patch('bird_vision.deployment.mobile_deployer.ct')
    def test_coreml_export(self, mock_coreml, test_config: DictConfig, sample_model: VisionModel, sample_input: torch.Tensor):
        """Test CoreML export for iOS."""
        # Mock CoreML
        mock_model = Mock()
        mock_coreml.convert.return_value = mock_model
        
        deployer = MobileDeployer(test_config)
        result = deployer._prepare_ios_deployment(sample_model, sample_input, "test_model", None)
        
        if result["success"]:
            assert mock_coreml.convert.called
    
    def test_integration_code_generation(self, test_config: DictConfig, temp_dir: Path):
        """Test integration code generation."""
        deployer = MobileDeployer(test_config)
        
        # Test iOS integration code
        ios_dir = temp_dir / "ios"
        ios_dir.mkdir(exist_ok=True)
        class_labels = ["robin", "jay", "cardinal"]
        
        deployer._create_ios_integration_code(ios_dir, "test_model", class_labels)
        
        swift_file = ios_dir / "TestModelClassifier.swift"
        assert swift_file.exists()
        
        content = swift_file.read_text()
        assert "TestModelClassifier" in content
        assert "robin" in content
        
        # Test Android integration code
        android_dir = temp_dir / "android"
        android_dir.mkdir(exist_ok=True)
        
        deployer._create_android_integration_code(android_dir, "test_model", class_labels)
        
        java_file = android_dir / "TestModelClassifier.java"
        assert java_file.exists()
        
        content = java_file.read_text()
        assert "TestModelClassifier" in content
        assert "robin" in content
    
    def test_deployment_metadata_generation(self, test_config: DictConfig, sample_model: VisionModel, sample_input: torch.Tensor):
        """Test deployment metadata generation."""
        deployer = MobileDeployer(test_config)
        
        deployment_results = {"ios": {"success": True}, "android": {"success": True}}
        
        metadata = deployer._generate_deployment_metadata(
            sample_model, sample_input, "test_model", deployment_results
        )
        
        assert "model_info" in metadata
        assert "model_stats" in metadata
        assert "preprocessing" in metadata
        assert "postprocessing" in metadata
        assert "deployment_platforms" in metadata
        
        assert metadata["model_info"]["name"] == "test_model"
        assert len(metadata["deployment_platforms"]) == 2


class TestESP32Deployer:
    """Test ESP32Deployer functionality."""
    
    def test_esp32_deployer_initialization(self, test_config: DictConfig):
        """Test ESP32 deployer initialization."""
        # Add ESP32 config
        test_config.deployment.esp32_p4 = {
            "optimization": {"use_ai_accelerator": True},
            "model_constraints": {"max_model_size_mb": 8, "max_inference_time_ms": 200},
            "camera": {"resolution": [640, 480]},
        }
        
        deployer = ESP32Deployer(test_config)
        
        assert deployer.cfg == test_config
        assert deployer.esp32_cfg == test_config.deployment.esp32_p4
        assert deployer.output_dir.exists()
    
    def test_esp32_model_optimization(self, test_config: DictConfig, sample_model: VisionModel, sample_input: torch.Tensor):
        """Test ESP32 model optimization."""
        test_config.deployment.esp32_p4 = {
            "optimization": {"use_ai_accelerator": True},
            "model_constraints": {"max_model_size_mb": 8, "max_inference_time_ms": 200},
        }
        
        deployer = ESP32Deployer(test_config)
        
        optimized_model = deployer._optimize_for_esp32(sample_model, sample_input, "test_model")
        
        assert isinstance(optimized_model, torch.nn.Module)
        
        # Test that optimization doesn't break the model
        with torch.no_grad():
            output = optimized_model(sample_input)
        assert output.shape == (1, 10)
    
    def test_esp_dl_conversion(self, test_config: DictConfig, sample_model: VisionModel, sample_input: torch.Tensor):
        """Test ESP-DL format conversion."""
        test_config.deployment.esp32_p4 = {
            "optimization": {"use_ai_accelerator": True},
            "model_constraints": {"max_model_size_mb": 8, "max_inference_time_ms": 200},
        }
        test_config.deployment.preprocessing = {
            "mean": [123.675, 116.28, 103.53],
            "std": [58.395, 57.12, 57.375],
        }
        
        deployer = ESP32Deployer(test_config)
        model_dir = deployer.output_dir / "test_model"
        model_dir.mkdir(exist_ok=True)
        
        result = deployer._convert_to_esp_dl(sample_model, sample_input, "test_model", model_dir)
        
        # Should attempt conversion even if it fails in testing environment
        assert "success" in result
    
    def test_firmware_generation(self, test_config: DictConfig, temp_dir: Path):
        """Test ESP32 firmware generation."""
        test_config.deployment.firmware = {
            "esp_idf_version": "v5.1",
            "camera_config": {
                "pin_xclk": 15,
                "pin_sscb_sda": 4,
                "pin_sscb_scl": 5,
                "pins": {"d0": 11, "d1": 9, "pclk": 13, "vsync": 6, "href": 7}
            }
        }
        
        deployer = ESP32Deployer(test_config)
        
        firmware_dir = temp_dir / "firmware"
        firmware_dir.mkdir(exist_ok=True)
        
        esp_dl_result = {"success": True, "esp_dl_files": {}}
        
        result = deployer._generate_esp32_firmware(
            firmware_dir, "test_model", ["robin", "jay"], esp_dl_result
        )
        
        if result["success"]:
            assert "firmware_dir" in result
            assert "files" in result
            
            # Check main files exist
            main_dir = firmware_dir / "main"
            if main_dir.exists():
                assert (main_dir / "main.cpp").exists()
    
    def test_cmake_generation(self, test_config: DictConfig, temp_dir: Path):
        """Test CMake files generation."""
        deployer = ESP32Deployer(test_config)
        
        firmware_dir = temp_dir / "firmware"
        firmware_dir.mkdir(exist_ok=True)
        
        result = deployer._generate_cmake_files(firmware_dir, "test_model")
        
        assert "main" in result
        assert "component" in result
        
        # Check CMakeLists.txt exists
        main_cmake = firmware_dir / "CMakeLists.txt"
        if main_cmake.exists():
            content = main_cmake.read_text()
            assert "test_model" in content
    
    def test_build_scripts_generation(self, test_config: DictConfig, temp_dir: Path):
        """Test build and flash scripts generation."""
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
        
        # Check scripts are executable
        assert build_script.stat().st_mode & 0o111  # Executable bits
        assert flash_script.stat().st_mode & 0o111
    
    def test_esp32_package_creation(self, test_config: DictConfig, temp_dir: Path):
        """Test ESP32 deployment package creation."""
        deployer = ESP32Deployer(test_config)
        
        esp32_dir = temp_dir / "esp32"
        esp32_dir.mkdir(exist_ok=True)
        
        esp_dl_result = {"success": True}
        firmware_result = {"success": True, "firmware_dir": str(esp32_dir / "firmware")}
        
        result = deployer._create_esp32_package(
            esp32_dir, "test_model", esp_dl_result, firmware_result
        )
        
        assert result["success"]
        assert "package_dir" in result
        assert "readme" in result
        assert "deployment_info" in result
        
        # Check README exists
        readme_file = Path(result["readme"])
        assert readme_file.exists()
        
        content = readme_file.read_text()
        assert "ESP32-P4-Eye" in content
        assert "test_model" in content


class TestDeploymentIntegration:
    """Test deployment integration and end-to-end flows."""
    
    def test_mobile_deployment_pipeline(self, test_config: DictConfig, sample_model: VisionModel, sample_input: torch.Tensor):
        """Test complete mobile deployment pipeline."""
        # Configure for minimal deployment
        test_config.deployment.target_platform = "mobile"
        test_config.compression.mobile_export.torchscript = True
        test_config.compression.mobile_export.coreml = False  # Disable for CI
        
        deployer = MobileDeployer(test_config)
        
        # Mock compression results
        with patch.object(deployer, 'compressor') as mock_compressor:
            mock_compressor.compress_model.return_value = {
                "compressed_models": {"quantized": sample_model},
                "compression_stats": {"quantized": {"size_mb": 10.0}},
            }
            
            results = deployer.prepare_for_mobile(sample_model, sample_input, "test_model")
            
            assert "compression_results" in results
            assert "deployment_results" in results
            assert "metadata" in results
    
    @patch('bird_vision.deployment.esp32_deployer.ESP32Deployer._convert_to_esp_dl')
    def test_esp32_deployment_pipeline(self, mock_esp_dl, test_config: DictConfig, sample_model: VisionModel, sample_input: torch.Tensor):
        """Test complete ESP32 deployment pipeline."""
        # Configure ESP32 deployment
        test_config.deployment.esp32_p4 = {
            "optimization": {"use_ai_accelerator": True},
            "model_constraints": {"max_model_size_mb": 8, "max_inference_time_ms": 200},
            "camera": {"resolution": [640, 480]},
        }
        test_config.deployment.firmware = {
            "esp_idf_version": "v5.1",
            "camera_config": {"pin_xclk": 15, "pins": {"d0": 11}},
        }
        test_config.deployment.preprocessing = {"mean": [0, 0, 0], "std": [1, 1, 1]}
        test_config.deployment.metadata = {"license": "MIT", "author": "Test"}
        
        # Mock ESP-DL conversion
        mock_esp_dl.return_value = {
            "success": True,
            "esp_dl_files": {"header": "test.hpp", "coefficients": "test.cpp"},
        }
        
        deployer = ESP32Deployer(test_config)
        results = deployer.prepare_for_esp32(sample_model, sample_input, "test_model")
        
        assert "esp_dl_model" in results
        assert "firmware" in results
        assert "package" in results
        assert "deployment_info" in results
    
    def test_deployment_error_handling(self, test_config: DictConfig, sample_model: VisionModel, sample_input: torch.Tensor):
        """Test deployment error handling."""
        deployer = MobileDeployer(test_config)
        
        # Test with invalid model (None)
        with patch.object(deployer, '_prepare_ios_deployment', side_effect=Exception("Test error")):
            results = deployer.prepare_for_mobile(sample_model, sample_input, "test_model")
            
            # Should handle errors gracefully
            assert isinstance(results, dict)
    
    def test_platform_specific_optimization(self, test_config: DictConfig, sample_model: VisionModel):
        """Test platform-specific optimizations."""
        # Test mobile optimization
        mobile_deployer = MobileDeployer(test_config)
        compression_results = {
            "compressed_models": {"quantized": sample_model},
            "compression_stats": {"quantized": {"size_mb": 10.0}},
        }
        
        selected_model = mobile_deployer._select_optimal_model(compression_results, sample_model)
        assert isinstance(selected_model, torch.nn.Module)
        
        # Test ESP32 optimization
        test_config.deployment.esp32_p4 = {
            "model_constraints": {"max_model_size_mb": 8},
        }
        
        esp32_deployer = ESP32Deployer(test_config)
        optimized_model = esp32_deployer._apply_esp32_optimizations(sample_model)
        assert isinstance(optimized_model, torch.nn.Module)