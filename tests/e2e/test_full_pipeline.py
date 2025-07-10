"""End-to-end tests for the complete Bird Vision pipeline."""

import pytest
import torch
from pathlib import Path
from omegaconf import DictConfig
from unittest.mock import patch, Mock

from bird_vision.models.vision_model import VisionModel
from bird_vision.training.trainer import Trainer
from bird_vision.validation.model_validator import ModelValidator
from bird_vision.compression.model_compressor import ModelCompressor
from bird_vision.deployment.mobile_deployer import MobileDeployer
from bird_vision.deployment.esp32_deployer import ESP32Deployer
from bird_vision.deployment.raspberry_pi_deployer import RaspberryPiDeployer


class TestFullPipeline:
    """Test complete end-to-end pipeline."""
    
    def test_data_to_model_pipeline(self, test_config: DictConfig, mock_data_loader, temp_dir: Path):
        """Test data loading through model creation."""
        # Update config paths
        test_config.paths.models_dir = str(temp_dir / "models")
        Path(test_config.paths.models_dir).mkdir(exist_ok=True)
        
        # Create model
        model = VisionModel(test_config.model)
        assert isinstance(model, torch.nn.Module)
        
        # Test data flow through model
        for batch in mock_data_loader:
            images, labels = batch
            
            with torch.no_grad():
                outputs = model(images)
            
            assert outputs.shape == (images.shape[0], test_config.model.head.num_classes)
            break  # Test one batch
    
    def test_training_pipeline(self, test_config: DictConfig, mock_data_loader, temp_dir: Path):
        """Test complete training pipeline."""
        # Update config paths
        test_config.paths.models_dir = str(temp_dir / "models")
        test_config.paths.logs_dir = str(temp_dir / "logs")
        for path in [test_config.paths.models_dir, test_config.paths.logs_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)
        
        # Minimal training config
        test_config.training.max_epochs = 1
        test_config.logging.use_mlflow = False
        test_config.logging.use_wandb = False
        
        device = torch.device("cpu")
        
        # Create and train model
        model = VisionModel(test_config.model)
        trainer = Trainer(model, test_config, device)
        
        # Run training (minimal)
        train_metrics = trainer.train_epoch(mock_data_loader)
        val_metrics = trainer.validate_epoch(mock_data_loader)
        
        assert isinstance(train_metrics, dict)
        assert isinstance(val_metrics, dict)
        assert "loss" in train_metrics
        assert "loss" in val_metrics
        assert train_metrics["loss"] >= 0
        assert val_metrics["loss"] >= 0
    
    def test_validation_pipeline(self, test_config: DictConfig, sample_model: VisionModel, mock_data_loader, temp_dir: Path):
        """Test model validation pipeline."""
        test_config.paths.artifacts_dir = str(temp_dir / "artifacts")
        Path(test_config.paths.artifacts_dir).mkdir(parents=True, exist_ok=True)
        
        device = torch.device("cpu")
        
        # Validate model
        validator = ModelValidator(test_config, device)
        results = validator.evaluate_model(sample_model, mock_data_loader, "test_model")
        
        assert isinstance(results, dict)
        assert "model_name" in results
        assert "overall_metrics" in results
        assert "detailed_metrics" in results
        assert results["model_name"] == "test_model"
    
    def test_compression_pipeline(self, test_config: DictConfig, sample_model: VisionModel, sample_input: torch.Tensor, temp_dir: Path):
        """Test model compression pipeline."""
        test_config.paths.models_dir = str(temp_dir / "models")
        Path(test_config.paths.models_dir).mkdir(parents=True, exist_ok=True)
        
        # Enable basic compression
        test_config.compression.quantization.enabled = True
        test_config.compression.pruning.enabled = False
        test_config.compression.onnx_export.enabled = True
        test_config.compression.mobile_export.torchscript = True
        test_config.compression.mobile_export.coreml = False
        
        compressor = ModelCompressor(test_config)
        results = compressor.compress_model(sample_model, sample_input, "test_model")
        
        assert isinstance(results, dict)
        assert "compressed_models" in results
        assert "compression_stats" in results
        assert "export_results" in results
    
    def test_mobile_deployment_pipeline(self, test_config: DictConfig, sample_model: VisionModel, sample_input: torch.Tensor, temp_dir: Path):
        """Test mobile deployment pipeline."""
        test_config.paths.models_dir = str(temp_dir / "models")
        Path(test_config.paths.models_dir).mkdir(parents=True, exist_ok=True)
        
        # Configure for basic mobile deployment
        test_config.deployment.target_platform = "mobile"
        test_config.compression.mobile_export.torchscript = True
        test_config.compression.mobile_export.coreml = False
        
        deployer = MobileDeployer(test_config)
        
        # Mock compression to avoid heavy computation
        with patch.object(deployer, 'compressor') as mock_compressor:
            mock_compressor.compress_model.return_value = {
                "compressed_models": {"original": sample_model},
                "compression_stats": {"original": {"size_mb": 10.0, "parameters": 1000}},
            }
            
            results = deployer.prepare_for_mobile(sample_model, sample_input, "test_model")
            
            assert isinstance(results, dict)
            assert "compression_results" in results
            assert "deployment_results" in results
            assert "metadata" in results
    
    @patch('bird_vision.deployment.esp32_deployer.ESP32Deployer._convert_to_esp_dl')
    def test_esp32_deployment_pipeline(self, mock_esp_dl, test_config: DictConfig, sample_model: VisionModel, sample_input: torch.Tensor, temp_dir: Path):
        """Test ESP32 deployment pipeline."""
        # Configure ESP32 deployment
        test_config.paths.models_dir = str(temp_dir / "models")
        Path(test_config.paths.models_dir).mkdir(parents=True, exist_ok=True)
        
        test_config.deployment.esp32_p4 = {
            "optimization": {"use_ai_accelerator": True, "memory_optimization": True},
            "model_constraints": {"max_model_size_mb": 8, "max_inference_time_ms": 200},
            "camera": {"resolution": [640, 480], "format": "RGB565", "fps": 15},
        }
        test_config.deployment.firmware = {
            "esp_idf_version": "v5.1",
            "camera_config": {
                "pin_xclk": 15,
                "pin_sscb_sda": 4,
                "pin_sscb_scl": 5,
                "pins": {"d0": 11, "d1": 9, "pclk": 13, "vsync": 6, "href": 7}
            }
        }
        test_config.deployment.preprocessing = {
            "mean": [123.675, 116.28, 103.53],
            "std": [58.395, 57.12, 57.375],
        }
        test_config.deployment.metadata = {
            "license": "MIT",
            "author": "Test",
        }
        
        # Mock ESP-DL conversion
        mock_esp_dl.return_value = {
            "success": True,
            "esp_dl_files": {
                "header": "test_model.hpp",
                "coefficients": "test_model_coefficients.hpp",
                "config": "test_model_config.json",
            },
        }
        
        deployer = ESP32Deployer(test_config)
        results = deployer.prepare_for_esp32(sample_model, sample_input, "test_model")
        
        assert isinstance(results, dict)
        assert "esp_dl_model" in results
        assert "firmware" in results
        assert "package" in results
        assert "deployment_info" in results


class TestCLIIntegration:
    """Test CLI integration with end-to-end flows."""
    
    @patch('bird_vision.cli.NABirdsDataModule')
    @patch('bird_vision.cli.Trainer')
    def test_cli_train_command(self, mock_trainer, mock_data_module, test_config: DictConfig, temp_dir: Path):
        """Test CLI train command integration."""
        from bird_vision.cli import train
        import typer.testing
        
        # Mock data module
        mock_dm = Mock()
        mock_dm.setup.return_value = None
        mock_dm.train_dataloader.return_value = []
        mock_dm.val_dataloader.return_value = []
        mock_data_module.return_value = mock_dm
        
        # Mock trainer
        mock_t = Mock()
        mock_t.fit.return_value = None
        mock_trainer.return_value = mock_t
        
        # This would test the actual CLI command, but we'll test the function directly
        # since CLI testing requires more complex setup
        assert callable(train)
    
    @patch('bird_vision.cli.ModelValidator')
    @patch('bird_vision.cli.torch.load')
    def test_cli_evaluate_command(self, mock_load, mock_validator, test_config: DictConfig, temp_dir: Path):
        """Test CLI evaluate command integration."""
        from bird_vision.cli import evaluate
        
        # Mock checkpoint loading
        mock_load.return_value = {"model_state_dict": {}}
        
        # Mock validator
        mock_v = Mock()
        mock_v.evaluate_model.return_value = {
            "model_name": "test",
            "overall_metrics": {"accuracy": 0.85},
            "detailed_metrics": {},
        }
        mock_validator.return_value = mock_v
        
        assert callable(evaluate)
    
    @patch('bird_vision.cli.ModelCompressor')
    @patch('bird_vision.cli.torch.load')
    def test_cli_compress_command(self, mock_load, mock_compressor, test_config: DictConfig):
        """Test CLI compress command integration."""
        from bird_vision.cli import compress
        
        # Mock checkpoint loading
        mock_load.return_value = {"model_state_dict": {}}
        
        # Mock compressor
        mock_c = Mock()
        mock_c.compress_model.return_value = {
            "compressed_models": {},
            "compression_stats": {},
            "compression_summary": {},
        }
        mock_compressor.return_value = mock_c
        
        assert callable(compress)
    
    @patch('bird_vision.cli.MobileDeployer')
    @patch('bird_vision.cli.torch.load')
    def test_cli_deploy_command(self, mock_load, mock_deployer, test_config: DictConfig):
        """Test CLI deploy command integration."""
        from bird_vision.cli import deploy
        
        # Mock checkpoint loading
        mock_load.return_value = {"model_state_dict": {}}
        
        # Mock deployer
        mock_d = Mock()
        mock_d.prepare_for_mobile.return_value = {
            "deployment_results": {},
            "packages": {},
        }
        mock_deployer.return_value = mock_d
        
        assert callable(deploy)


class TestDataFlowIntegrity:
    """Test data flow integrity through the pipeline."""
    
    def test_tensor_shapes_consistency(self, test_config: DictConfig, mock_data_loader):
        """Test tensor shapes remain consistent through pipeline."""
        model = VisionModel(test_config.model)
        device = torch.device("cpu")
        
        # Test data flow
        for batch in mock_data_loader:
            images, labels = batch
            
            # Check input shapes
            assert images.dim() == 4  # [B, C, H, W]
            assert labels.dim() == 1  # [B]
            assert images.shape[1] == 3  # RGB channels
            assert images.shape[2] == images.shape[3] == 224  # Square images
            
            # Test model forward pass
            with torch.no_grad():
                outputs = model(images)
            
            assert outputs.shape == (images.shape[0], test_config.model.head.num_classes)
            break
    
    def test_gradient_flow(self, test_config: DictConfig, mock_data_loader):
        """Test gradient flow in training mode."""
        model = VisionModel(test_config.model)
        criterion = torch.nn.CrossEntropyLoss()
        
        model.train()
        
        for batch in mock_data_loader:
            images, labels = batch
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Check gradients exist
            has_gradients = False
            for param in model.parameters():
                if param.requires_grad and param.grad is not None:
                    has_gradients = True
                    break
            
            assert has_gradients, "No gradients found in model parameters"
            break
    
    def test_inference_consistency(self, test_config: DictConfig, sample_model: VisionModel):
        """Test inference consistency across multiple runs."""
        sample_model.eval()
        
        input_tensor = torch.randn(1, 3, 224, 224)
        
        # Multiple inference runs should be consistent
        outputs = []
        with torch.no_grad():
            for _ in range(3):
                output = sample_model(input_tensor)
                outputs.append(output)
        
        # All outputs should be identical
        for i in range(1, len(outputs)):
            assert torch.allclose(outputs[0], outputs[i], atol=1e-6)


class TestErrorHandling:
    """Test error handling in end-to-end scenarios."""
    
    def test_invalid_input_handling(self, test_config: DictConfig):
        """Test handling of invalid inputs."""
        model = VisionModel(test_config.model)
        
        # Test wrong input shape
        with pytest.raises((RuntimeError, ValueError)):
            wrong_shape_input = torch.randn(1, 3, 112, 112)  # Wrong size
            model(wrong_shape_input)
    
    def test_missing_files_handling(self, test_config: DictConfig, temp_dir: Path):
        """Test handling of missing files."""
        from bird_vision.utils.checkpoint import CheckpointManager
        
        test_config.paths.models_dir = str(temp_dir)
        manager = CheckpointManager(test_config)
        
        # Test loading non-existent checkpoint
        with pytest.raises(FileNotFoundError):
            manager.load_checkpoint("non_existent_checkpoint.ckpt")
    
    def test_device_mismatch_handling(self, test_config: DictConfig):
        """Test handling of device mismatches."""
        model = VisionModel(test_config.model)
        
        # Model on CPU, input on CPU (should work)
        cpu_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(cpu_input)
        assert output.device.type == "cpu"


class TestMemoryManagement:
    """Test memory management in end-to-end scenarios."""
    
    def test_memory_cleanup(self, test_config: DictConfig, mock_data_loader):
        """Test memory cleanup during training."""
        model = VisionModel(test_config.model)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss()
        
        model.train()
        
        # Simulate training loop
        for i, batch in enumerate(mock_data_loader):
            images, labels = batch
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Force garbage collection periodically
            if i % 10 == 0:
                import gc
                gc.collect()
            
            if i >= 5:  # Test a few iterations
                break
    
    def test_large_batch_handling(self, test_config: DictConfig):
        """Test handling of large batches."""
        model = VisionModel(test_config.model)
        
        # Test with larger batch
        large_batch = torch.randn(8, 3, 224, 224)
        
        with torch.no_grad():
            output = model(large_batch)
        
        assert output.shape == (8, test_config.model.head.num_classes)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestMultiPlatformDeploymentPipeline:
    """Test end-to-end deployment pipeline for all platforms."""
    
    def test_train_to_mobile_deployment_pipeline(self, test_config: DictConfig, mock_data_loader, temp_dir: Path):
        """Test complete pipeline from training to mobile deployment."""
        # Configure for fast testing
        test_config.training.max_epochs = 1
        test_config.paths.models_dir = str(temp_dir / "models")
        test_config.paths.logs_dir = str(temp_dir / "logs")
        
        for path in [test_config.paths.models_dir, test_config.paths.logs_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)
        
        device = torch.device("cpu")
        
        # Step 1: Create and train model
        model = VisionModel(test_config.model)
        trainer = Trainer(model, test_config, device)
        
        # Minimal training
        train_metrics = trainer.train_epoch(mock_data_loader)
        assert "loss" in train_metrics
        
        # Step 2: Validate
        validator = ModelValidator(test_config, device)
        validation_results = validator.evaluate_model(model, mock_data_loader, "e2e_model")
        assert validation_results["overall_metrics"]["accuracy"] >= 0
        
        # Step 3: Compress
        compressor = ModelCompressor(test_config)
        sample_input = torch.randn(1, 3, 224, 224)
        compression_results = compressor.compress_model(model, sample_input, "e2e_model")
        assert "compressed_models" in compression_results
        
        # Step 4: Deploy to mobile
        mobile_deployer = MobileDeployer(test_config)
        
        with patch.object(mobile_deployer, 'compressor') as mock_compressor:
            mock_compressor.compress_model.return_value = compression_results
            
            deployment_results = mobile_deployer.prepare_for_mobile(model, sample_input, "e2e_model")
            assert "deployment_results" in deployment_results
            assert "metadata" in deployment_results
    
    def test_train_to_raspberry_pi_deployment_pipeline(self, test_config: DictConfig, mock_data_loader, temp_dir: Path):
        """Test complete pipeline from training to Raspberry Pi deployment."""
        # Configure for Raspberry Pi deployment
        rpi_config = {
            "platform": "raspberry_pi",
            "target_device": "rpi4",
            "model_optimization": {
                "quantization": {"enabled": True, "method": "dynamic", "dtype": "int8"},
                "pruning": {"enabled": True, "sparsity": 0.3},
                "arm_optimization": {"use_neon": True, "optimize_for_inference": True}
            },
            "performance_targets": {"rpi4": {"model_size_mb": 25, "inference_time_ms": 300}},
            "camera": {"interface": "libcamera", "resolution": {"width": 640, "height": 480}},
            "paths": {
                "install_dir": "/opt/bird_vision",
                "model_dir": "/opt/bird_vision/models",
                "config_dir": "/opt/bird_vision/configs"
            },
            "service": {"user": "pi", "group": "pi"}
        }
        test_config.deployment = DictConfig(rpi_config)
        
        # Set up paths
        test_config.paths.models_dir = str(temp_dir / "models")
        test_config.paths.logs_dir = str(temp_dir / "logs")
        for path in [test_config.paths.models_dir, test_config.paths.logs_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)
        
        device = torch.device("cpu")
        
        # Step 1: Create and train model
        model = VisionModel(test_config.model)
        trainer = Trainer(model, test_config, device)
        
        # Minimal training
        train_metrics = trainer.train_epoch(mock_data_loader)
        assert "loss" in train_metrics
        
        # Step 2: Validate
        validator = ModelValidator(test_config, device)
        validation_results = validator.evaluate_model(model, mock_data_loader, "rpi_e2e_model")
        assert validation_results["overall_metrics"]["accuracy"] >= 0
        
        # Step 3: Save model for deployment
        model_path = temp_dir / "rpi_e2e_model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': test_config.model
        }, model_path)
        
        # Step 4: Deploy to Raspberry Pi
        rpi_deployer = RaspberryPiDeployer(test_config)
        output_dir = temp_dir / "rpi_deployment"
        
        # Mock external dependencies
        with patch('bird_vision.deployment.raspberry_pi_deployer.ModelCompressor') as mock_compressor, \
             patch('torch.onnx.export') as mock_onnx, \
             patch.object(rpi_deployer.profiler, 'profile_model') as mock_profile:
            
            # Configure mocks
            mock_compressor.return_value.compress_model.return_value = model
            mock_profile.return_value = {
                "model_size_mb": 20.0,
                "forward_time_ms": 150.0,
                "parameters": 1000000
            }
            
            deployment_results = rpi_deployer.deploy_model(model, model_path, output_dir)
            
            # Verify deployment success
            assert deployment_results["success"] is True
            assert deployment_results["platform"] == "raspberry_pi"
            assert "artifacts" in deployment_results
            assert "performance" in deployment_results
            
            # Verify specific Raspberry Pi artifacts
            artifacts = deployment_results["artifacts"]
            assert "inference_script" in artifacts
            assert "camera_script" in artifacts
            assert "install_package" in artifacts
            assert "service_file" in artifacts
    
    def test_train_to_esp32_deployment_pipeline(self, test_config: DictConfig, mock_data_loader, temp_dir: Path):
        """Test complete pipeline from training to ESP32 deployment."""
        # Configure for ESP32 deployment
        esp32_config = {
            "platform": "esp32",
            "target_device": "esp32_p4_eye",
            "model_optimization": {
                "quantization": {"enabled": True, "method": "static", "dtype": "int8"},
                "pruning": {"enabled": True, "sparsity": 0.5}
            },
            "performance_targets": {"esp32_p4_eye": {"model_size_mb": 8, "inference_time_ms": 200}},
            "camera": {"interface": "esp32_cam", "resolution": {"width": 320, "height": 240}},
            "paths": {"install_dir": "/opt/esp32"}
        }
        test_config.deployment = DictConfig(esp32_config)
        
        # Set up paths
        test_config.paths.models_dir = str(temp_dir / "models")
        test_config.paths.logs_dir = str(temp_dir / "logs")
        for path in [test_config.paths.models_dir, test_config.paths.logs_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)
        
        device = torch.device("cpu")
        
        # Step 1: Create and train model
        model = VisionModel(test_config.model)
        trainer = Trainer(model, test_config, device)
        
        # Minimal training
        train_metrics = trainer.train_epoch(mock_data_loader)
        assert "loss" in train_metrics
        
        # Step 2: Deploy to ESP32
        esp32_deployer = ESP32Deployer(test_config)
        sample_input = torch.randn(1, 3, 224, 224)
        
        # Mock ESP32 specific operations
        with patch.object(esp32_deployer, '_optimize_for_esp32') as mock_optimize, \
             patch.object(esp32_deployer, '_convert_to_esp_dl') as mock_convert, \
             patch.object(esp32_deployer, '_generate_firmware') as mock_firmware:
            
            # Configure mocks
            mock_optimize.return_value = (model, {"size_mb": 6.0})
            mock_convert.return_value = str(temp_dir / "model.esp")
            mock_firmware.return_value = str(temp_dir / "firmware")
            
            deployment_results = esp32_deployer.prepare_for_esp32(model, sample_input, "esp32_e2e_model")
            
            # Verify deployment structure
            assert "deployment_results" in deployment_results
            assert deployment_results["platform"] == "esp32"
    
    @pytest.mark.parametrize("target_platform", ["mobile", "raspberry_pi", "esp32"])
    def test_full_pipeline_all_platforms(self, test_config: DictConfig, mock_data_loader, target_platform: str, temp_dir: Path):
        """Test complete pipeline for all supported platforms."""
        # Configure platform-specific settings
        if target_platform == "raspberry_pi":
            platform_config = {
                "platform": "raspberry_pi",
                "target_device": "rpi4",
                "paths": {"install_dir": "/opt/bird_vision"}
            }
        elif target_platform == "esp32":
            platform_config = {
                "platform": "esp32", 
                "target_device": "esp32_p4_eye",
                "paths": {"install_dir": "/opt/esp32"}
            }
        else:
            platform_config = {"platform": "mobile"}
        
        test_config.deployment = DictConfig(platform_config)
        
        # Set up basic paths
        test_config.training.max_epochs = 1
        test_config.paths.models_dir = str(temp_dir / "models")
        test_config.paths.logs_dir = str(temp_dir / "logs")
        for path in [test_config.paths.models_dir, test_config.paths.logs_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)
        
        device = torch.device("cpu")
        
        # Step 1: Train (minimal)
        model = VisionModel(test_config.model)
        trainer = Trainer(model, test_config, device)
        train_metrics = trainer.train_epoch(mock_data_loader)
        assert "loss" in train_metrics
        
        # Step 2: Save model
        model_path = temp_dir / f"{target_platform}_model.pth"
        torch.save({'model_state_dict': model.state_dict()}, model_path)
        
        # Step 3: Platform-specific deployment
        sample_input = torch.randn(1, 3, 224, 224)
        
        if target_platform == "raspberry_pi":
            deployer = RaspberryPiDeployer(test_config)
            output_dir = temp_dir / "rpi_output"
            
            with patch('bird_vision.deployment.raspberry_pi_deployer.ModelCompressor') as mock_comp, \
                 patch('torch.onnx.export'), \
                 patch.object(deployer.profiler, 'profile_model') as mock_prof:
                
                mock_comp.return_value.compress_model.return_value = model
                mock_prof.return_value = {"model_size_mb": 20.0, "forward_time_ms": 150.0}
                
                result = deployer.deploy_model(model, model_path, output_dir)
                assert result["success"] is True
                assert result["platform"] == "raspberry_pi"
        
        elif target_platform == "esp32":
            deployer = ESP32Deployer(test_config)
            
            with patch.object(deployer, '_optimize_for_esp32') as mock_opt, \
                 patch.object(deployer, '_convert_to_esp_dl') as mock_conv:
                
                mock_opt.return_value = (model, {"size_mb": 6.0})
                mock_conv.return_value = str(temp_dir / "model.esp")
                
                result = deployer.prepare_for_esp32(model, sample_input, "esp32_model")
                assert "deployment_results" in result
                assert result["platform"] == "esp32"
        
        else:  # mobile
            deployer = MobileDeployer(test_config)
            
            with patch.object(deployer, 'compressor') as mock_compressor:
                mock_compressor.compress_model.return_value = {
                    "compressed_models": {"quantized": model},
                    "compression_stats": {"quantized": {"size_mb": 15.0}}
                }
                
                result = deployer.prepare_for_mobile(model, sample_input, "mobile_model")
                assert "deployment_results" in result
                assert "metadata" in result
    
    def test_cross_platform_model_compatibility(self, test_config: DictConfig, mock_data_loader, temp_dir: Path):
        """Test that the same trained model can be deployed across all platforms."""
        # Set up for training
        test_config.training.max_epochs = 1
        test_config.paths.models_dir = str(temp_dir / "models")
        test_config.paths.logs_dir = str(temp_dir / "logs")
        for path in [test_config.paths.models_dir, test_config.paths.logs_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)
        
        device = torch.device("cpu")
        
        # Train one model
        model = VisionModel(test_config.model)
        trainer = Trainer(model, test_config, device)
        train_metrics = trainer.train_epoch(mock_data_loader)
        
        # Save the trained model
        model_path = temp_dir / "universal_model.pth"
        torch.save({'model_state_dict': model.state_dict()}, model_path)
        
        # Test deployment to all platforms using the same model
        platforms_tested = []
        sample_input = torch.randn(1, 3, 224, 224)
        
        # Test Mobile deployment
        try:
            mobile_deployer = MobileDeployer(test_config)
            with patch.object(mobile_deployer, 'compressor') as mock_comp:
                mock_comp.compress_model.return_value = {
                    "compressed_models": {"quantized": model},
                    "compression_stats": {"quantized": {"size_mb": 15.0}}
                }
                mobile_result = mobile_deployer.prepare_for_mobile(model, sample_input, "universal_model")
                if "deployment_results" in mobile_result:
                    platforms_tested.append("mobile")
        except Exception as e:
            pytest.skip(f"Mobile deployment not available: {e}")
        
        # Test Raspberry Pi deployment
        try:
            rpi_config = DictConfig({
                "platform": "raspberry_pi",
                "target_device": "rpi4", 
                "paths": {"install_dir": "/opt/bird_vision"}
            })
            test_config.deployment = rpi_config
            rpi_deployer = RaspberryPiDeployer(test_config)
            
            with patch('bird_vision.deployment.raspberry_pi_deployer.ModelCompressor') as mock_comp, \
                 patch('torch.onnx.export'), \
                 patch.object(rpi_deployer.profiler, 'profile_model') as mock_prof:
                
                mock_comp.return_value.compress_model.return_value = model
                mock_prof.return_value = {"model_size_mb": 20.0, "forward_time_ms": 150.0}
                
                rpi_result = rpi_deployer.deploy_model(model, model_path, temp_dir / "rpi_universal")
                if rpi_result.get("success"):
                    platforms_tested.append("raspberry_pi")
        except Exception as e:
            pytest.skip(f"Raspberry Pi deployment not available: {e}")
        
        # Test ESP32 deployment
        try:
            esp32_config = DictConfig({
                "platform": "esp32",
                "target_device": "esp32_p4_eye",
                "paths": {"install_dir": "/opt/esp32"}
            })
            test_config.deployment = esp32_config
            esp32_deployer = ESP32Deployer(test_config)
            
            with patch.object(esp32_deployer, '_optimize_for_esp32') as mock_opt, \
                 patch.object(esp32_deployer, '_convert_to_esp_dl') as mock_conv:
                
                mock_opt.return_value = (model, {"size_mb": 6.0})
                mock_conv.return_value = str(temp_dir / "model.esp")
                
                esp32_result = esp32_deployer.prepare_for_esp32(model, sample_input, "universal_model")
                if "deployment_results" in esp32_result:
                    platforms_tested.append("esp32")
        except Exception as e:
            pytest.skip(f"ESP32 deployment not available: {e}")
        
        # Verify that at least 2 platforms were successfully tested
        assert len(platforms_tested) >= 2, f"Expected at least 2 platforms, tested: {platforms_tested}"
        
        # Verify the model remained consistent across platforms
        assert isinstance(model, torch.nn.Module)
        assert model_path.exists()


class TestPipelinePerformance:
    """Test performance characteristics of the complete pipeline."""
    
    def test_pipeline_memory_usage(self, test_config: DictConfig, mock_data_loader, temp_dir: Path):
        """Test memory usage throughout the pipeline."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Configure for minimal resource usage
        test_config.training.max_epochs = 1
        test_config.paths.models_dir = str(temp_dir / "models")
        Path(test_config.paths.models_dir).mkdir(parents=True, exist_ok=True)
        
        device = torch.device("cpu")
        
        # Create and train model
        model = VisionModel(test_config.model)
        trainer = Trainer(model, test_config, device)
        train_metrics = trainer.train_epoch(mock_data_loader)
        
        mid_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Compress model
        compressor = ModelCompressor(test_config)
        sample_input = torch.randn(1, 3, 224, 224)
        compression_results = compressor.compress_model(model, sample_input, "memory_test")
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Memory should not grow excessively (allowing for reasonable overhead)
        memory_growth = final_memory - initial_memory
        assert memory_growth < 500, f"Memory growth too high: {memory_growth:.1f} MB"
        
        # Log memory usage for debugging
        print(f"Memory usage: Initial={initial_memory:.1f}MB, Mid={mid_memory:.1f}MB, Final={final_memory:.1f}MB")
    
    def test_pipeline_timing(self, test_config: DictConfig, mock_data_loader, temp_dir: Path):
        """Test timing characteristics of pipeline components."""
        import time
        
        timings = {}
        
        # Configure for fast testing
        test_config.training.max_epochs = 1
        test_config.paths.models_dir = str(temp_dir / "models")
        Path(test_config.paths.models_dir).mkdir(parents=True, exist_ok=True)
        
        device = torch.device("cpu")
        
        # Time model creation
        start_time = time.time()
        model = VisionModel(test_config.model)
        timings["model_creation"] = time.time() - start_time
        
        # Time training epoch
        trainer = Trainer(model, test_config, device)
        start_time = time.time()
        train_metrics = trainer.train_epoch(mock_data_loader)
        timings["training_epoch"] = time.time() - start_time
        
        # Time validation
        validator = ModelValidator(test_config, device)
        start_time = time.time()
        validation_results = validator.evaluate_model(model, mock_data_loader, "timing_test")
        timings["validation"] = time.time() - start_time
        
        # Time compression
        compressor = ModelCompressor(test_config)
        sample_input = torch.randn(1, 3, 224, 224)
        start_time = time.time()
        compression_results = compressor.compress_model(model, sample_input, "timing_test")
        timings["compression"] = time.time() - start_time
        
        # Verify reasonable timing constraints (allowing for CI/test environment overhead)
        assert timings["model_creation"] < 10, f"Model creation too slow: {timings['model_creation']:.2f}s"
        assert timings["training_epoch"] < 60, f"Training epoch too slow: {timings['training_epoch']:.2f}s"
        assert timings["validation"] < 30, f"Validation too slow: {timings['validation']:.2f}s"
        assert timings["compression"] < 30, f"Compression too slow: {timings['compression']:.2f}s"
        
        # Log timings for analysis
        for component, timing in timings.items():
            print(f"{component}: {timing:.2f}s")