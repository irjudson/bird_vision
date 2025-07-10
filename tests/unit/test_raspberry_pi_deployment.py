"""Unit tests for Raspberry Pi deployment functionality."""

import json
import platform
import tempfile
import unittest.mock as mock
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from bird_vision.deployment.raspberry_pi_deployer import RaspberryPiDeployer
from bird_vision.models.vision_model import VisionModel


class TestRaspberryPiDeployer:
    """Test Raspberry Pi deployment functionality."""

    @pytest.fixture
    def rpi_config(self):
        """Create test configuration for Raspberry Pi deployment."""
        config = {
            "deployment": {
                "platform": "raspberry_pi",
                "target_device": "rpi4",
                "model_optimization": {
                    "quantization": {
                        "enabled": True,
                        "method": "dynamic",
                        "dtype": "int8",
                        "backend": "fbgemm"
                    },
                    "pruning": {
                        "enabled": True,
                        "sparsity": 0.3,
                        "structured": True
                    },
                    "arm_optimization": {
                        "use_neon": True,
                        "optimize_for_inference": True,
                        "use_onnx_runtime": True,
                        "fuse_operators": True
                    }
                },
                "performance_targets": {
                    "rpi4": {
                        "model_size_mb": 25,
                        "inference_time_ms": 300,
                        "memory_usage_mb": 512,
                        "power_consumption_w": 5.0
                    }
                },
                "camera": {
                    "interface": "libcamera",
                    "resolution": {"width": 640, "height": 480},
                    "framerate": 15,
                    "format": "RGB888"
                },
                "preprocessing": {
                    "resize_method": "bilinear",
                    "normalization": {
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225]
                    }
                },
                "runtime": {
                    "backend": "onnxruntime",
                    "num_threads": 4,
                    "inter_op_num_threads": 2,
                    "intra_op_num_threads": 2
                },
                "output": {
                    "format": "json",
                    "confidence_threshold": 0.7,
                    "top_k_predictions": 5
                },
                "integration": {
                    "api_server": {"enabled": True, "port": 8080}
                },
                "paths": {
                    "install_dir": "/opt/bird_vision",
                    "model_dir": "/opt/bird_vision/models",
                    "config_dir": "/opt/bird_vision/configs",
                    "log_dir": "/var/log/bird_vision",
                    "data_dir": "/home/pi/bird_vision_data"
                },
                "service": {
                    "create_systemd_service": True,
                    "service_name": "bird-vision",
                    "user": "pi",
                    "group": "pi",
                    "restart_policy": "always"
                }
            }
        }
        return DictConfig(config)

    @pytest.fixture
    def sample_model(self):
        """Create a sample model for testing."""
        model_config = {
            "model_name": "efficientnet_v2_s",
            "num_classes": 400,
            "pretrained": False
        }
        return VisionModel(DictConfig(model_config))

    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        return torch.randn(1, 3, 224, 224)

    def test_deployer_initialization(self, rpi_config):
        """Test Raspberry Pi deployer initialization."""
        deployer = RaspberryPiDeployer(rpi_config)
        
        assert deployer.config == rpi_config
        assert deployer.target_device == "rpi4"
        assert deployer.deployment_config == rpi_config.deployment

    def test_raspberry_pi_detection(self):
        """Test Raspberry Pi model detection."""
        # Mock /proc/cpuinfo for different Pi models
        test_cases = [
            ("BCM2711", "rpi4"),
            ("BCM2712", "rpi5"),
            ("BCM2837", "rpi_zero2w"),
            ("BCM2835", "rpi_unknown"),
            ("Intel", None)
        ]
        
        for cpu_info, expected in test_cases:
            with patch("builtins.open", mock.mock_open(read_data=f"Hardware\t: {cpu_info}\nRevision\t: c03111\nSerial\t: 00000000deadbeef\nModel\t: Raspberry Pi 4 Model B Rev 1.1")):
                result = RaspberryPiDeployer.detect_raspberry_pi()
                assert result == expected

    def test_raspberry_pi_detection_no_file(self):
        """Test Pi detection when /proc/cpuinfo doesn't exist."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            result = RaspberryPiDeployer.detect_raspberry_pi()
            assert result is None

    @pytest.mark.parametrize("target_device", ["rpi4", "rpi5", "rpi_zero2w"])
    def test_deploy_model_success(self, rpi_config, sample_model, sample_input, target_device):
        """Test successful model deployment to different Pi models."""
        rpi_config.deployment.target_device = target_device
        deployer = RaspberryPiDeployer(rpi_config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Mock the compression and export methods
            with patch.object(deployer, '_optimize_for_arm') as mock_optimize, \
                 patch.object(deployer, '_export_to_onnx') as mock_export, \
                 patch.object(deployer, '_generate_inference_script') as mock_inference, \
                 patch.object(deployer, '_generate_camera_integration') as mock_camera, \
                 patch.object(deployer, '_create_installation_package') as mock_package, \
                 patch.object(deployer, '_generate_systemd_service') as mock_service, \
                 patch.object(deployer, '_validate_performance') as mock_validate, \
                 patch.object(deployer, '_generate_deployment_docs') as mock_docs:
                
                # Set up mock returns
                mock_optimize.return_value = (sample_model, {"size_mb": 20.5, "optimization_type": "ARM"})
                mock_export.return_value = output_dir / "model.onnx"
                mock_inference.return_value = output_dir / "inference.py"
                mock_camera.return_value = output_dir / "camera.py"
                mock_package.return_value = output_dir / "package"
                mock_service.return_value = output_dir / "service.service"
                mock_validate.return_value = {"meets_size_target": True, "meets_time_target": True}
                mock_docs.return_value = output_dir / "docs.md"
                
                result = deployer.deploy_model(
                    sample_model, 
                    output_dir / "model.pth", 
                    output_dir
                )
                
                assert result["success"] is True
                assert result["platform"] == "raspberry_pi"
                assert result["target_device"] == target_device
                assert "artifacts" in result
                assert "performance" in result

    def test_deploy_model_failure(self, rpi_config, sample_model, sample_input):
        """Test model deployment failure handling."""
        deployer = RaspberryPiDeployer(rpi_config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Mock failure in optimization step
            with patch.object(deployer, '_optimize_for_arm', side_effect=Exception("Optimization failed")):
                result = deployer.deploy_model(
                    sample_model,
                    output_dir / "model.pth",
                    output_dir
                )
                
                assert result["success"] is False
                assert len(result["errors"]) > 0
                assert "Optimization failed" in result["errors"][0]

    def test_arm_optimization(self, rpi_config, sample_model, sample_input):
        """Test ARM-specific optimizations."""
        deployer = RaspberryPiDeployer(rpi_config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Mock the compressor
            with patch('bird_vision.deployment.raspberry_pi_deployer.ModelCompressor') as MockCompressor:
                mock_compressor = MockCompressor.return_value
                mock_compressor.compress_model.return_value = sample_model
                
                optimized_model, stats = deployer._optimize_for_arm(
                    sample_model,
                    temp_dir + "/model.pth", 
                    output_dir
                )
                
                assert optimized_model is not None
                assert "optimization_type" in stats
                assert stats["optimization_type"] == "ARM-specific"

    def test_onnx_export(self, rpi_config, sample_model, sample_input):
        """Test ONNX export functionality."""
        deployer = RaspberryPiDeployer(rpi_config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Mock torch.onnx.export
            with patch('torch.onnx.export') as mock_export:
                onnx_path = deployer._export_to_onnx(sample_model, output_dir)
                
                assert onnx_path.name == "model_optimized.onnx"
                mock_export.assert_called_once()

    def test_inference_script_generation(self, rpi_config):
        """Test inference script generation."""
        deployer = RaspberryPiDeployer(rpi_config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            script_path = deployer._generate_inference_script(output_dir)
            
            assert script_path.exists()
            assert script_path.name == "bird_vision_inference.py"
            
            # Check script content
            content = script_path.read_text()
            assert "BirdVisionInference" in content
            assert "onnxruntime" in content
            assert "ARM optimizations" in content

    def test_camera_integration_generation(self, rpi_config):
        """Test camera integration script generation."""
        deployer = RaspberryPiDeployer(rpi_config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            script_path = deployer._generate_camera_integration(output_dir)
            
            assert script_path.exists()
            assert script_path.name == "bird_vision_camera.py"
            
            # Check script content
            content = script_path.read_text()
            assert "BirdVisionCamera" in content
            assert "picamera2" in content
            assert "libcamera" in content

    def test_installation_package_creation(self, rpi_config):
        """Test installation package creation."""
        deployer = RaspberryPiDeployer(rpi_config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            package_dir = deployer._create_installation_package(output_dir)
            
            assert package_dir.exists()
            assert (package_dir / "install.sh").exists()
            assert (package_dir / "config.json").exists()
            
            # Check install script
            install_script = (package_dir / "install.sh").read_text()
            assert "Raspberry Pi" in install_script
            assert "libcamera-apps" in install_script
            assert "python3-picamera2" in install_script

    def test_systemd_service_generation(self, rpi_config):
        """Test systemd service file generation."""
        deployer = RaspberryPiDeployer(rpi_config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            package_dir = output_dir / "raspberry_pi_package"
            package_dir.mkdir()
            
            service_file = deployer._generate_systemd_service(output_dir)
            
            assert service_file.exists()
            assert service_file.name == "bird-vision.service"
            
            # Check service content
            content = service_file.read_text()
            assert "[Unit]" in content
            assert "[Service]" in content
            assert "[Install]" in content
            assert "User=pi" in content

    def test_performance_validation(self, rpi_config, sample_model, sample_input):
        """Test performance validation against targets."""
        deployer = RaspberryPiDeployer(rpi_config)
        
        # Mock the profiler
        with patch.object(deployer.profiler, 'profile_model') as mock_profile:
            mock_profile.return_value = {
                "model_size_mb": 20.0,
                "forward_time_ms": 100.0,
                "parameters": 1000000
            }
            
            performance = deployer._validate_performance(sample_model)
            
            assert "target_device" in performance
            assert "meets_size_target" in performance
            assert "meets_time_target" in performance
            assert performance["target_device"] == "rpi4"
            assert performance["meets_size_target"] is True  # 20MB < 25MB target

    def test_deployment_environment_validation(self, rpi_config):
        """Test deployment environment validation."""
        deployer = RaspberryPiDeployer(rpi_config)
        
        with patch.object(deployer, 'detect_raspberry_pi', return_value='rpi4'), \
             patch('subprocess.run') as mock_run:
            
            # Mock successful camera check
            mock_run.return_value = MagicMock(returncode=0)
            
            validation = deployer.validate_deployment_environment()
            
            assert validation["platform_detected"] == "rpi4"
            assert "checks" in validation
            assert "python_version" in validation["checks"]

    def test_config_file_generation(self, rpi_config):
        """Test configuration file generation."""
        deployer = RaspberryPiDeployer(rpi_config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            package_dir = deployer._create_installation_package(output_dir)
            config_file = package_dir / "config.json"
            
            assert config_file.exists()
            
            # Validate JSON structure
            with open(config_file) as f:
                config_data = json.load(f)
            
            assert "deployment" in config_data
            assert config_data["deployment"]["platform"] == "raspberry_pi"

    @pytest.mark.parametrize("device,expected_size,expected_time", [
        ("rpi4", 25, 300),
        ("rpi5", 35, 200),
        ("rpi_zero2w", 15, 800)
    ])
    def test_device_specific_targets(self, rpi_config, device, expected_size, expected_time):
        """Test device-specific performance targets."""
        rpi_config.deployment.target_device = device
        rpi_config.deployment.performance_targets = {
            "rpi4": {"model_size_mb": 25, "inference_time_ms": 300},
            "rpi5": {"model_size_mb": 35, "inference_time_ms": 200},
            "rpi_zero2w": {"model_size_mb": 15, "inference_time_ms": 800}
        }
        
        deployer = RaspberryPiDeployer(rpi_config)
        
        assert deployer.target_device == device
        targets = deployer.deployment_config.performance_targets[device]
        assert targets["model_size_mb"] == expected_size
        assert targets["inference_time_ms"] == expected_time

    def test_documentation_generation(self, rpi_config):
        """Test deployment documentation generation."""
        deployer = RaspberryPiDeployer(rpi_config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            deployment_result = {
                "success": True,
                "performance": {
                    "model_size_mb": 20.0,
                    "meets_size_target": True,
                    "meets_time_target": True
                },
                "artifacts": {
                    "onnx_model": "model.onnx",
                    "inference_script": "inference.py"
                }
            }
            
            docs_path = deployer._generate_deployment_docs(output_dir, deployment_result)
            
            assert docs_path.exists()
            assert docs_path.name == "RASPBERRY_PI_DEPLOYMENT.md"
            
            content = docs_path.read_text()
            assert "Bird Vision Raspberry Pi Deployment" in content
            assert "Installation Instructions" in content
            assert "Troubleshooting" in content


class TestRaspberryPiCameraIntegration:
    """Test camera integration functionality."""
    
    def test_camera_config_validation(self, rpi_config):
        """Test camera configuration validation."""
        camera_config = rpi_config.deployment.camera
        
        assert camera_config.interface == "libcamera"
        assert camera_config.resolution.width == 640
        assert camera_config.resolution.height == 480
        assert camera_config.framerate == 15

    @pytest.mark.skipif(
        platform.system() != "Linux" or not Path("/proc/cpuinfo").exists(),
        reason="Raspberry Pi specific test"
    )
    def test_pi_camera_detection(self):
        """Test Pi camera detection on actual hardware."""
        # This test only runs on actual Raspberry Pi hardware
        try:
            import subprocess
            result = subprocess.run(
                ['libcamera-hello', '--list-cameras'],
                capture_output=True, text=True, timeout=5
            )
            # If we get here without exception, camera interface is available
            assert result.returncode == 0 or result.returncode == 1  # 1 is "no cameras"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pytest.skip("libcamera tools not available")


class TestRaspberryPiPerformance:
    """Test performance optimization for Raspberry Pi."""
    
    def test_arm_quantization(self, rpi_config, sample_model):
        """Test ARM-specific quantization."""
        deployer = RaspberryPiDeployer(rpi_config)
        
        # Mock torch quantization
        with patch('torch.quantization.quantize_dynamic') as mock_quantize:
            mock_quantize.return_value = sample_model
            
            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir)
                sample_input = torch.randn(1, 3, 224, 224)
                
                optimized_model, stats = deployer._optimize_for_arm(
                    sample_model, temp_dir + "/model.pth", output_dir
                )
                
                # Check that quantization was called with correct parameters
                mock_quantize.assert_called_once()
                args, kwargs = mock_quantize.call_args
                assert kwargs['dtype'] == torch.qint8

    def test_performance_meets_targets(self, rpi_config, sample_model):
        """Test that optimized model meets performance targets."""
        deployer = RaspberryPiDeployer(rpi_config)
        
        # Mock profiler to return values that meet targets
        with patch.object(deployer.profiler, 'profile_model') as mock_profile:
            mock_profile.return_value = {
                "model_size_mb": 20.0,  # Under 25MB target for rpi4
                "forward_time_ms": 150.0,  # Under 300ms target
                "parameters": 1000000
            }
            
            performance = deployer._validate_performance(sample_model)
            
            assert performance["meets_size_target"] is True
            assert performance["meets_time_target"] is True

    def test_performance_exceeds_targets(self, rpi_config, sample_model):
        """Test handling when model exceeds performance targets."""
        deployer = RaspberryPiDeployer(rpi_config)
        
        # Mock profiler to return values that exceed targets
        with patch.object(deployer.profiler, 'profile_model') as mock_profile:
            mock_profile.return_value = {
                "model_size_mb": 30.0,  # Over 25MB target for rpi4
                "forward_time_ms": 400.0,  # Over 300ms target  
                "parameters": 2000000
            }
            
            performance = deployer._validate_performance(sample_model)
            
            assert performance["meets_size_target"] is False
            assert performance["meets_time_target"] is False


@pytest.mark.integration
class TestRaspberryPiIntegration:
    """Integration tests for Raspberry Pi deployment."""
    
    def test_full_deployment_pipeline(self, rpi_config, sample_model):
        """Test the complete deployment pipeline."""
        deployer = RaspberryPiDeployer(rpi_config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            model_path = output_dir / "test_model.pth"
            
            # Save a test model
            torch.save(sample_model.state_dict(), model_path)
            
            # Mock all external dependencies
            with patch('bird_vision.deployment.raspberry_pi_deployer.ModelCompressor'), \
                 patch('torch.onnx.export'), \
                 patch.object(deployer.profiler, 'profile_model') as mock_profile:
                
                mock_profile.return_value = {
                    "model_size_mb": 20.0,
                    "forward_time_ms": 150.0,
                    "parameters": 1000000
                }
                
                result = deployer.deploy_model(sample_model, model_path, output_dir)
                
                # Check that deployment completed successfully
                assert result["success"] is True
                assert "artifacts" in result
                assert "performance" in result
                
                # Check that required files were created
                package_dir = output_dir / "raspberry_pi_package"
                assert package_dir.exists()
                assert (package_dir / "install.sh").exists()
                assert (package_dir / "config.json").exists()

    def test_deployment_with_errors(self, rpi_config, sample_model):
        """Test deployment error handling and recovery."""
        deployer = RaspberryPiDeployer(rpi_config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            model_path = output_dir / "test_model.pth"
            
            # Create an invalid model file
            with open(model_path, 'w') as f:
                f.write("invalid model data")
            
            # This should handle the error gracefully
            result = deployer.deploy_model(sample_model, model_path, output_dir)
            
            # Deployment should fail but handle errors gracefully
            assert result["success"] is False
            assert len(result["errors"]) > 0
            
            # Result file should still be created
            result_file = output_dir / "deployment_result.json"
            assert result_file.exists()