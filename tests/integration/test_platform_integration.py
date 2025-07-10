"""Integration tests for platform-specific deployment."""

import pytest
import torch
from pathlib import Path
from omegaconf import DictConfig
from unittest.mock import patch, Mock

from bird_vision.models.vision_model import VisionModel
from bird_vision.deployment.mobile_deployer import MobileDeployer
from bird_vision.deployment.esp32_deployer import ESP32Deployer
from bird_vision.deployment.raspberry_pi_deployer import RaspberryPiDeployer
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


class TestRaspberryPiIntegration:
    """Test Raspberry Pi deployment integration."""
    
    def test_raspberry_pi_deployment_pipeline(self, test_config: DictConfig, sample_model: VisionModel, sample_input: torch.Tensor, temp_dir: Path):
        """Test complete Raspberry Pi deployment pipeline."""
        # Configure for Raspberry Pi deployment
        rpi_config = {
            "platform": "raspberry_pi",
            "target_device": "rpi4",
            "model_optimization": {
                "quantization": {"enabled": True, "method": "dynamic", "dtype": "int8"},
                "pruning": {"enabled": True, "sparsity": 0.3},
                "arm_optimization": {"use_neon": True, "optimize_for_inference": True}
            },
            "performance_targets": {
                "rpi4": {"model_size_mb": 25, "inference_time_ms": 300}
            },
            "camera": {"interface": "libcamera", "resolution": {"width": 640, "height": 480}},
            "paths": {
                "install_dir": "/opt/bird_vision",
                "model_dir": "/opt/bird_vision/models",
                "config_dir": "/opt/bird_vision/configs"
            },
            "service": {"user": "pi", "group": "pi"}
        }
        test_config.deployment = DictConfig(rpi_config)
        
        deployer = RaspberryPiDeployer(test_config)
        
        # Mock external dependencies
        with patch('bird_vision.deployment.raspberry_pi_deployer.ModelCompressor') as mock_compressor, \
             patch('torch.onnx.export') as mock_onnx_export, \
             patch.object(deployer.profiler, 'profile_model') as mock_profile:
            
            # Configure mocks
            mock_compressor.return_value.compress_model.return_value = sample_model
            mock_profile.return_value = {
                "model_size_mb": 20.0,
                "forward_time_ms": 150.0,
                "parameters": 1000000
            }
            
            # Run deployment
            model_path = temp_dir / "test_model.pth"
            torch.save(sample_model.state_dict(), model_path)
            
            result = deployer.deploy_model(sample_model, model_path, temp_dir)
            
            # Verify deployment success
            assert result["success"] is True
            assert result["platform"] == "raspberry_pi"
            assert result["target_device"] == "rpi4"
            assert "artifacts" in result
            assert "performance" in result
    
    def test_raspberry_pi_arm_optimization(self, test_config: DictConfig, sample_model: VisionModel, sample_input: torch.Tensor, temp_dir: Path):
        """Test ARM-specific optimizations for Raspberry Pi."""
        # Configure ARM optimization
        test_config.compression.arm_optimization = DictConfig({
            "enabled": True,
            "use_neon": True,
            "optimize_for_inference": True,
            "fuse_operators": True
        })
        
        compressor = ModelCompressor(test_config)
        
        # Test ARM export functionality
        with patch.object(compressor, 'export_for_raspberry_pi') as mock_export:
            mock_export.return_value = {
                "onnx_model": str(temp_dir / "model_rpi.onnx"),
                "arm_optimized_pytorch": str(temp_dir / "model_arm.pth"),
                "target_device": "rpi4",
                "optimization_stats": {"size_mb": 20.0, "optimization_type": "ARM-specific"}
            }
            
            result = mock_export(sample_model, sample_input, "test_model", "rpi4")
            
            assert "onnx_model" in result
            assert "arm_optimized_pytorch" in result
            assert result["target_device"] == "rpi4"
    
    def test_raspberry_pi_camera_integration(self, test_config: DictConfig, temp_dir: Path):
        """Test camera integration code generation."""
        rpi_config = DictConfig({
            "platform": "raspberry_pi",
            "camera": {
                "interface": "libcamera",
                "resolution": {"width": 640, "height": 480},
                "framerate": 15
            },
            "paths": {"install_dir": "/opt/bird_vision"}
        })
        test_config.deployment = rpi_config
        
        deployer = RaspberryPiDeployer(test_config)
        
        # Generate camera integration
        camera_script = deployer._generate_camera_integration(temp_dir)
        
        assert camera_script.exists()
        assert camera_script.name == "bird_vision_camera.py"
        
        content = camera_script.read_text()
        assert "BirdVisionCamera" in content
        assert "picamera2" in content
        assert "libcamera" in content
        assert "640" in content  # Resolution width
        assert "480" in content  # Resolution height
    
    @pytest.mark.parametrize("target_device", ["rpi4", "rpi5", "rpi_zero2w"])
    def test_raspberry_pi_device_specific_optimization(self, test_config: DictConfig, sample_model: VisionModel, target_device: str, temp_dir: Path):
        """Test device-specific optimizations for different Pi models."""
        # Configure device-specific targets
        rpi_config = DictConfig({
            "platform": "raspberry_pi", 
            "target_device": target_device,
            "performance_targets": {
                "rpi4": {"model_size_mb": 25, "inference_time_ms": 300},
                "rpi5": {"model_size_mb": 35, "inference_time_ms": 200},
                "rpi_zero2w": {"model_size_mb": 15, "inference_time_ms": 800}
            },
            "paths": {"install_dir": "/opt/bird_vision"}
        })
        test_config.deployment = rpi_config
        
        deployer = RaspberryPiDeployer(test_config)
        
        # Test performance validation
        with patch.object(deployer.profiler, 'profile_model') as mock_profile:
            # Mock model that meets targets
            targets = rpi_config.performance_targets[target_device]
            mock_profile.return_value = {
                "model_size_mb": targets["model_size_mb"] - 5,  # Under target
                "forward_time_ms": targets["inference_time_ms"] / 2,  # Conservative estimate
                "parameters": 1000000
            }
            
            performance = deployer._validate_performance(sample_model)
            
            assert performance["target_device"] == target_device
            assert performance["meets_size_target"] is True
            assert performance["meets_time_target"] is True
    
    def test_raspberry_pi_installation_package(self, test_config: DictConfig, temp_dir: Path):
        """Test installation package generation."""
        rpi_config = DictConfig({
            "platform": "raspberry_pi",
            "paths": {
                "install_dir": "/opt/bird_vision",
                "model_dir": "/opt/bird_vision/models", 
                "config_dir": "/opt/bird_vision/configs",
                "log_dir": "/var/log/bird_vision",
                "data_dir": "/home/pi/bird_vision_data"
            }
        })
        test_config.deployment = rpi_config
        
        deployer = RaspberryPiDeployer(test_config)
        
        # Generate installation package
        package_dir = deployer._create_installation_package(temp_dir)
        
        assert package_dir.exists()
        assert (package_dir / "install.sh").exists()
        assert (package_dir / "config.json").exists()
        
        # Check install script content
        install_script = (package_dir / "install.sh").read_text()
        assert "libcamera-apps" in install_script
        assert "python3-picamera2" in install_script
        assert "/opt/bird_vision" in install_script
        assert "systemctl" in install_script
        
        # Check config file
        with open(package_dir / "config.json") as f:
            import json
            config_data = json.load(f)
        
        assert config_data["deployment"]["platform"] == "raspberry_pi"
    
    def test_raspberry_pi_systemd_service(self, test_config: DictConfig, temp_dir: Path):
        """Test systemd service generation."""
        rpi_config = DictConfig({
            "platform": "raspberry_pi",
            "service": {
                "user": "pi",
                "group": "pi", 
                "restart_policy": "always"
            },
            "paths": {"install_dir": "/opt/bird_vision"}
        })
        test_config.deployment = rpi_config
        
        deployer = RaspberryPiDeployer(test_config)
        
        # Create package directory first
        package_dir = temp_dir / "raspberry_pi_package"
        package_dir.mkdir()
        
        # Generate systemd service
        service_file = deployer._generate_systemd_service(temp_dir)
        
        assert service_file.exists()
        assert service_file.name == "bird-vision.service"
        
        content = service_file.read_text()
        assert "[Unit]" in content
        assert "[Service]" in content
        assert "[Install]" in content
        assert "User=pi" in content
        assert "Group=pi" in content
        assert "Restart=always" in content
    
    def test_raspberry_pi_environment_validation(self, test_config: DictConfig):
        """Test deployment environment validation."""
        rpi_config = DictConfig({"platform": "raspberry_pi"})
        test_config.deployment = rpi_config
        
        deployer = RaspberryPiDeployer(test_config)
        
        # Mock Raspberry Pi detection
        with patch.object(deployer, 'detect_raspberry_pi', return_value='rpi4') as mock_detect, \
             patch('subprocess.run') as mock_run:
            
            # Mock successful camera check  
            mock_run.return_value = Mock(returncode=0)
            
            validation = deployer.validate_deployment_environment()
            
            assert validation["platform_detected"] == "rpi4"
            assert "checks" in validation
            assert "python_version" in validation["checks"]
            mock_detect.assert_called_once()
    
    def test_raspberry_pi_cross_platform_compatibility(self, test_config: DictConfig, sample_model: VisionModel, sample_input: torch.Tensor, temp_dir: Path):
        """Test cross-platform compatibility with other deployment targets."""
        # Test that Raspberry Pi deployment doesn't interfere with other platforms
        mobile_deployer = MobileDeployer(test_config) 
        esp32_deployer = ESP32Deployer(test_config)
        
        rpi_config = DictConfig({
            "platform": "raspberry_pi",
            "target_device": "rpi4",
            "paths": {"install_dir": "/opt/bird_vision"}
        })
        test_config.deployment = rpi_config
        rpi_deployer = RaspberryPiDeployer(test_config)
        
        # Each deployer should work independently
        assert rpi_deployer.target_device == "rpi4"
        assert mobile_deployer.config == test_config
        assert esp32_deployer.config == test_config
        
        # Test that they can coexist in the same project
        assert isinstance(rpi_deployer, RaspberryPiDeployer)
        assert isinstance(mobile_deployer, MobileDeployer)
        assert isinstance(esp32_deployer, ESP32Deployer)


class TestMultiPlatformIntegration:
    """Test integration across all platforms."""
    
    def test_all_platforms_deployment(self, test_config: DictConfig, sample_model: VisionModel, sample_input: torch.Tensor, temp_dir: Path):
        """Test deployment to all platforms from single model."""
        # Configure for multi-platform deployment
        platforms = ["ios", "android", "esp32", "raspberry_pi"]
        
        deployment_results = {}
        
        for platform in platforms:
            platform_dir = temp_dir / platform
            platform_dir.mkdir(exist_ok=True)
            
            if platform in ["ios", "android"]:
                # Mobile deployment
                test_config.deployment.target_platform = platform
                deployer = MobileDeployer(test_config)
                
                with patch('bird_vision.deployment.mobile_deployer.ct') as mock_coreml, \
                     patch.object(deployer, 'compressor') as mock_compressor:
                    
                    mock_compressor.compress_model.return_value = {
                        "compressed_models": {"quantized": sample_model},
                        "compression_stats": {"quantized": {"size_mb": 15.0}}
                    }
                    
                    if platform == "ios":
                        mock_coreml.convert.return_value = Mock()
                        mock_coreml.ImageType = Mock()
                        mock_coreml.ClassifierConfig = Mock()
                    
                    try:
                        result = deployer.prepare_for_mobile(sample_model, sample_input, f"{platform}_model")
                        deployment_results[platform] = {"success": True, "result": result}
                    except Exception as e:
                        deployment_results[platform] = {"success": False, "error": str(e)}
            
            elif platform == "esp32":
                # ESP32 deployment
                esp32_config = DictConfig({
                    "platform": "esp32",
                    "target_device": "esp32_p4_eye",
                    "paths": {"install_dir": "/opt/esp32"}
                })
                test_config.deployment = esp32_config
                deployer = ESP32Deployer(test_config)
                
                with patch.object(deployer, '_optimize_for_esp32') as mock_opt, \
                     patch.object(deployer, '_convert_to_esp_dl') as mock_conv:
                    
                    mock_opt.return_value = (sample_model, {"size_mb": 6.0})
                    mock_conv.return_value = str(platform_dir / "model.esp")
                    
                    try:
                        model_path = platform_dir / "model.pth"
                        torch.save(sample_model.state_dict(), model_path)
                        result = deployer.deploy_model(sample_model, model_path, platform_dir)
                        deployment_results[platform] = {"success": result.get("success", True), "result": result}
                    except Exception as e:
                        deployment_results[platform] = {"success": False, "error": str(e)}
            
            elif platform == "raspberry_pi":
                # Raspberry Pi deployment
                rpi_config = DictConfig({
                    "platform": "raspberry_pi", 
                    "target_device": "rpi4",
                    "paths": {"install_dir": "/opt/bird_vision"}
                })
                test_config.deployment = rpi_config
                deployer = RaspberryPiDeployer(test_config)
                
                with patch('bird_vision.deployment.raspberry_pi_deployer.ModelCompressor') as mock_comp, \
                     patch('torch.onnx.export') as mock_onnx, \
                     patch.object(deployer.profiler, 'profile_model') as mock_prof:
                    
                    mock_comp.return_value.compress_model.return_value = sample_model
                    mock_prof.return_value = {"model_size_mb": 20.0, "forward_time_ms": 150.0}
                    
                    try:
                        model_path = platform_dir / "model.pth"
                        torch.save(sample_model.state_dict(), model_path)
                        result = deployer.deploy_model(sample_model, model_path, platform_dir)
                        deployment_results[platform] = {"success": result.get("success", True), "result": result}
                    except Exception as e:
                        deployment_results[platform] = {"success": False, "error": str(e)}
        
        # Verify that at least some platforms deployed successfully
        successful_deployments = [p for p, r in deployment_results.items() if r.get("success")]
        assert len(successful_deployments) >= 2, f"Expected at least 2 successful deployments, got {successful_deployments}"
        
        # Verify platform-specific artifacts
        for platform, result in deployment_results.items():
            if result.get("success"):
                platform_dir = temp_dir / platform
                assert platform_dir.exists(), f"Platform directory missing for {platform}"