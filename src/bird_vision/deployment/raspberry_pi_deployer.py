"""Raspberry Pi deployment module for Bird Vision models.

This module handles deployment of optimized bird identification models
to Raspberry Pi devices with camera integration.
"""

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from ..compression.model_compressor import ModelCompressor
from ..utils.model_utils import ModelProfiler


logger = logging.getLogger(__name__)


class RaspberryPiDeployer:
    """Deploy optimized models to Raspberry Pi devices."""
    
    def __init__(self, config: DictConfig):
        """Initialize the Raspberry Pi deployer.
        
        Args:
            config: Deployment configuration
        """
        self.config = config
        self.deployment_config = config.deployment
        self.target_device = self.deployment_config.target_device
        self.profiler = ModelProfiler()
        
        # Set up paths
        self.install_dir = Path(self.deployment_config.paths.install_dir)
        self.model_dir = Path(self.deployment_config.paths.model_dir)
        self.config_dir = Path(self.deployment_config.paths.config_dir)
        
        logger.info(f"Initialized Raspberry Pi deployer for {self.target_device}")
    
    def deploy_model(
        self,
        model: nn.Module,
        model_path: Union[str, Path],
        output_dir: Union[str, Path],
        **kwargs
    ) -> Dict[str, Any]:
        """Deploy a model to Raspberry Pi.
        
        Args:
            model: PyTorch model to deploy
            model_path: Path to the model file
            output_dir: Output directory for deployment artifacts
            **kwargs: Additional deployment options
            
        Returns:
            Deployment result dictionary
        """
        logger.info("Starting Raspberry Pi model deployment...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        deployment_result = {
            "platform": "raspberry_pi",
            "target_device": self.target_device,
            "timestamp": time.time(),
            "success": False,
            "artifacts": {},
            "performance": {},
            "errors": []
        }
        
        try:
            # Step 1: Optimize model for ARM architecture
            logger.info("Optimizing model for ARM architecture...")
            optimized_model, optimization_stats = self._optimize_for_arm(
                model, model_path, output_dir
            )
            deployment_result["optimization_stats"] = optimization_stats
            
            # Step 2: Export to ONNX for runtime optimization
            logger.info("Exporting to ONNX format...")
            onnx_path = self._export_to_onnx(optimized_model, output_dir)
            deployment_result["artifacts"]["onnx_model"] = str(onnx_path)
            
            # Step 3: Generate inference script
            logger.info("Generating inference script...")
            inference_script = self._generate_inference_script(output_dir)
            deployment_result["artifacts"]["inference_script"] = str(inference_script)
            
            # Step 4: Generate camera integration
            logger.info("Generating camera integration...")
            camera_script = self._generate_camera_integration(output_dir)
            deployment_result["artifacts"]["camera_script"] = str(camera_script)
            
            # Step 5: Create installation package
            logger.info("Creating installation package...")
            install_package = self._create_installation_package(output_dir)
            deployment_result["artifacts"]["install_package"] = str(install_package)
            
            # Step 6: Generate systemd service
            logger.info("Generating systemd service...")
            service_file = self._generate_systemd_service(output_dir)
            deployment_result["artifacts"]["service_file"] = str(service_file)
            
            # Step 7: Performance validation
            logger.info("Validating performance...")
            performance_stats = self._validate_performance(optimized_model)
            deployment_result["performance"] = performance_stats
            
            # Step 8: Generate deployment documentation
            logger.info("Generating deployment documentation...")
            docs = self._generate_deployment_docs(output_dir, deployment_result)
            deployment_result["artifacts"]["documentation"] = str(docs)
            
            deployment_result["success"] = True
            logger.info("Raspberry Pi deployment completed successfully!")
            
        except Exception as e:
            error_msg = f"Deployment failed: {str(e)}"
            logger.error(error_msg)
            deployment_result["errors"].append(error_msg)
            deployment_result["success"] = False
        
        # Save deployment result
        result_file = output_dir / "deployment_result.json"
        with open(result_file, 'w') as f:
            json.dump(deployment_result, f, indent=2)
        
        return deployment_result
    
    def _optimize_for_arm(
        self,
        model: nn.Module,
        model_path: Union[str, Path],
        output_dir: Path
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Optimize model for ARM architecture."""
        # Use model compressor for optimization
        compressor = ModelCompressor(self.config)
        
        # Configure ARM-specific optimizations
        compression_config = {
            "quantization": {
                "enabled": True,
                "method": "dynamic",
                "dtype": "int8",
                "backend": "fbgemm"  # Good for ARM
            },
            "pruning": {
                "enabled": self.deployment_config.model_optimization.pruning.enabled,
                "sparsity": self.deployment_config.model_optimization.pruning.sparsity,
                "structured": True  # Better for ARM
            },
            "optimization": {
                "optimize_for_inference": True,
                "use_onnx_runtime": True
            }
        }
        
        # Apply optimizations
        optimized_model = compressor.compress_model(
            model, compression_config, output_dir / "optimized_model.pth"
        )
        
        # Get optimization statistics
        original_size = self._get_model_size(model_path)
        optimized_size = self._get_model_size(output_dir / "optimized_model.pth")
        
        optimization_stats = {
            "original_size_mb": original_size / (1024 * 1024),
            "optimized_size_mb": optimized_size / (1024 * 1024),
            "compression_ratio": original_size / optimized_size,
            "quantization_applied": True,
            "pruning_applied": compression_config["pruning"]["enabled"]
        }
        
        return optimized_model, optimization_stats
    
    def _export_to_onnx(self, model: nn.Module, output_dir: Path) -> Path:
        """Export model to ONNX format for ARM optimization."""
        onnx_path = output_dir / "model_optimized.onnx"
        
        # Create dummy input for tracing
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,  # Good compatibility
            do_constant_folding=True,  # Optimize constant operations
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        logger.info(f"Model exported to ONNX: {onnx_path}")
        return onnx_path
    
    def _generate_inference_script(self, output_dir: Path) -> Path:
        """Generate optimized inference script for Raspberry Pi."""
        script_path = output_dir / "bird_vision_inference.py"
        
        script_content = f'''#!/usr/bin/env python3
"""
Bird Vision Inference Script for Raspberry Pi
Generated automatically by Bird Vision deployment system.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import onnxruntime as ort
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BirdVisionInference:
    """Optimized bird vision inference for Raspberry Pi."""
    
    def __init__(self, model_path: str, config_path: str):
        """Initialize the inference engine.
        
        Args:
            model_path: Path to ONNX model
            config_path: Path to configuration file
        """
        self.model_path = model_path
        self.config_path = config_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize ONNX Runtime session
        self._init_onnx_session()
        
        # Performance tracking
        self.inference_times = []
        
    def _init_onnx_session(self):
        """Initialize ONNX Runtime session with ARM optimizations."""
        # Configure providers for ARM
        providers = ['CPUExecutionProvider']
        
        # Session options for ARM optimization
        session_options = ort.SessionOptions()
        session_options.inter_op_num_threads = {self.deployment_config.runtime.inter_op_num_threads}
        session_options.intra_op_num_threads = {self.deployment_config.runtime.intra_op_num_threads}
        session_options.enable_mem_pattern = True
        session_options.enable_cpu_mem_arena = True
        
        # Create session
        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=session_options,
            providers=providers
        )
        
        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        logger.info(f"ONNX session initialized for {{self.input_shape}}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for inference."""
        # Resize to model input size
        height, width = self.input_shape[2], self.input_shape[3]
        image = Image.fromarray(image).resize((width, height), Image.BILINEAR)
        
        # Convert to numpy array
        image_array = np.array(image).astype(np.float32)
        
        # Normalize
        mean = np.array({self.deployment_config.preprocessing.normalization.mean})
        std = np.array({self.deployment_config.preprocessing.normalization.std})
        image_array = (image_array / 255.0 - mean) / std
        
        # Add batch dimension and transpose to NCHW
        image_array = np.transpose(image_array, (2, 0, 1))
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    def predict(self, image: np.ndarray) -> Dict:
        """Run inference on image."""
        start_time = time.time()
        
        # Preprocess
        input_tensor = self.preprocess_image(image)
        
        # Run inference
        outputs = self.session.run([self.output_name], {{self.input_name: input_tensor}})
        predictions = outputs[0][0]  # Remove batch dimension
        
        # Post-process
        probabilities = self._softmax(predictions)
        
        # Get top predictions
        top_k = {self.deployment_config.output.top_k_predictions}
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # Format results
        results = {{
            "predictions": [
                {{
                    "class_id": int(idx),
                    "confidence": float(probabilities[idx]),
                    "class_name": f"bird_species_{{idx}}"  # Map to actual names
                }}
                for idx in top_indices
                if probabilities[idx] > {self.deployment_config.output.confidence_threshold}
            ],
            "inference_time_ms": inference_time * 1000,
            "timestamp": time.time()
        }}
        
        return results
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax to logits."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        if not self.inference_times:
            return {{"message": "No inference performed yet"}}
        
        times_ms = [t * 1000 for t in self.inference_times]
        return {{
            "total_inferences": len(times_ms),
            "avg_inference_time_ms": np.mean(times_ms),
            "min_inference_time_ms": np.min(times_ms),
            "max_inference_time_ms": np.max(times_ms),
            "std_inference_time_ms": np.std(times_ms)
        }}


if __name__ == "__main__":
    # Example usage
    model_path = "/opt/bird_vision/models/model_optimized.onnx"
    config_path = "/opt/bird_vision/configs/config.json"
    
    # Initialize inference engine
    engine = BirdVisionInference(model_path, config_path)
    
    # Test with dummy image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = engine.predict(dummy_image)
    
    print(json.dumps(result, indent=2))
    print(json.dumps(engine.get_performance_stats(), indent=2))
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        return script_path
    
    def _generate_camera_integration(self, output_dir: Path) -> Path:
        """Generate camera integration script."""
        script_path = output_dir / "bird_vision_camera.py"
        
        script_content = f'''#!/usr/bin/env python3
"""
Bird Vision Camera Integration for Raspberry Pi
Handles camera capture and real-time bird identification.
"""

import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Callable

import cv2
import numpy as np

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    logging.warning("picamera2 not available, falling back to OpenCV")

from bird_vision_inference import BirdVisionInference

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BirdVisionCamera:
    """Camera integration for real-time bird detection."""
    
    def __init__(self, config_path: str, model_path: str):
        """Initialize camera system.
        
        Args:
            config_path: Path to configuration file
            model_path: Path to ONNX model
        """
        self.config_path = config_path
        self.model_path = model_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.camera_config = self.config.get("camera", {{}})
        
        # Initialize inference engine
        self.inference_engine = BirdVisionInference(model_path, config_path)
        
        # Camera setup
        self.camera = None
        self.running = False
        self.capture_thread = None
        
        # Detection callback
        self.detection_callback: Optional[Callable] = None
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
    def initialize_camera(self) -> bool:
        """Initialize camera interface."""
        try:
            if PICAMERA2_AVAILABLE:
                logger.info("Using picamera2 interface")
                return self._init_picamera2()
            else:
                logger.info("Using OpenCV interface")
                return self._init_opencv()
        except Exception as e:
            logger.error(f"Failed to initialize camera: {{e}}")
            return False
    
    def _init_picamera2(self) -> bool:
        """Initialize picamera2 interface."""
        self.camera = Picamera2()
        
        # Configure camera
        config = self.camera.create_still_configuration(
            main={{
                "size": ({self.deployment_config.camera.resolution.width}, 
                        {self.deployment_config.camera.resolution.height}),
                "format": "RGB888"
            }}
        )
        self.camera.configure(config)
        
        # Set camera controls
        controls = {{}}
        if self.camera_config.get("auto_exposure", True):
            controls["AeEnable"] = True
        if self.camera_config.get("auto_white_balance", True):
            controls["AwbEnable"] = True
            
        self.camera.set_controls(controls)
        self.camera.start()
        
        logger.info("picamera2 initialized successfully")
        return True
    
    def _init_opencv(self) -> bool:
        """Initialize OpenCV camera interface."""
        self.camera = cv2.VideoCapture(0)
        
        if not self.camera.isOpened():
            logger.error("Failed to open camera with OpenCV")
            return False
        
        # Set camera properties
        width = {self.deployment_config.camera.resolution.width}
        height = {self.deployment_config.camera.resolution.height}
        fps = {self.deployment_config.camera.framerate}
        
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.camera.set(cv2.CAP_PROP_FPS, fps)
        
        logger.info("OpenCV camera initialized successfully")
        return True
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from camera."""
        if not self.camera:
            logger.error("Camera not initialized")
            return None
        
        try:
            if PICAMERA2_AVAILABLE and isinstance(self.camera, Picamera2):
                # picamera2 capture
                frame = self.camera.capture_array()
                return frame
            else:
                # OpenCV capture
                ret, frame = self.camera.read()
                if ret:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    return frame
                else:
                    logger.warning("Failed to capture frame")
                    return None
                    
        except Exception as e:
            logger.error(f"Error capturing frame: {{e}}")
            return None
    
    def start_continuous_detection(self, callback: Optional[Callable] = None):
        """Start continuous bird detection."""
        self.detection_callback = callback
        self.running = True
        self.start_time = time.time()
        self.frame_count = 0
        
        self.capture_thread = threading.Thread(target=self._detection_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        logger.info("Started continuous detection")
    
    def stop_continuous_detection(self):
        """Stop continuous detection."""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=5)
        
        logger.info("Stopped continuous detection")
    
    def _detection_loop(self):
        """Main detection loop."""
        while self.running:
            try:
                # Capture frame
                frame = self.capture_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                # Run inference
                result = self.inference_engine.predict(frame)
                self.frame_count += 1
                
                # Add frame info
                result["frame_number"] = self.frame_count
                result["fps"] = self.frame_count / (time.time() - self.start_time)
                
                # Call detection callback if provided
                if self.detection_callback:
                    self.detection_callback(result, frame)
                
                # Log significant detections
                predictions = result.get("predictions", [])
                if predictions:
                    best_prediction = predictions[0]
                    if best_prediction["confidence"] > {self.deployment_config.output.confidence_threshold}:
                        logger.info(f"Bird detected: {{best_prediction['class_name']}} "
                                  f"({{best_prediction['confidence']:.2f}})")
                
                # Control frame rate
                time.sleep(1.0 / {self.deployment_config.camera.framerate})
                
            except Exception as e:
                logger.error(f"Error in detection loop: {{e}}")
                time.sleep(1)
    
    def cleanup(self):
        """Cleanup camera resources."""
        self.stop_continuous_detection()
        
        if self.camera:
            if PICAMERA2_AVAILABLE and isinstance(self.camera, Picamera2):
                self.camera.stop()
                self.camera.close()
            else:
                self.camera.release()
        
        logger.info("Camera cleanup completed")
    
    def get_system_stats(self) -> Dict:
        """Get system performance statistics."""
        return {{
            "total_frames": self.frame_count,
            "runtime_seconds": time.time() - self.start_time,
            "average_fps": self.frame_count / (time.time() - self.start_time) if self.frame_count > 0 else 0,
            "inference_stats": self.inference_engine.get_performance_stats()
        }}


# Example detection callback
def detection_callback(result: Dict, frame: np.ndarray):
    """Example callback for handling detections."""
    predictions = result.get("predictions", [])
    if predictions:
        best = predictions[0]
        print(f"Detection: {{best['class_name']}} ({{best['confidence']:.2f}}) "
              f"at {{result['inference_time_ms']:.1f}}ms")


if __name__ == "__main__":
    import signal
    import sys
    
    # Configuration paths
    config_path = "/opt/bird_vision/configs/config.json"
    model_path = "/opt/bird_vision/models/model_optimized.onnx"
    
    # Initialize camera system
    camera_system = BirdVisionCamera(config_path, model_path)
    
    # Initialize camera
    if not camera_system.initialize_camera():
        print("Failed to initialize camera")
        sys.exit(1)
    
    # Signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\\nShutting down...")
        camera_system.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start detection
        camera_system.start_continuous_detection(detection_callback)
        
        # Keep running
        while True:
            time.sleep(1)
            stats = camera_system.get_system_stats()
            if stats["total_frames"] > 0:
                print(f"FPS: {{stats['average_fps']:.1f}}, "
                      f"Frames: {{stats['total_frames']}}, "
                      f"Avg inference: {{stats['inference_stats'].get('avg_inference_time_ms', 0):.1f}}ms")
    
    except KeyboardInterrupt:
        print("\\nShutting down...")
    finally:
        camera_system.cleanup()
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        return script_path
    
    def _create_installation_package(self, output_dir: Path) -> Path:
        """Create installation package for Raspberry Pi."""
        package_dir = output_dir / "raspberry_pi_package"
        package_dir.mkdir(exist_ok=True)
        
        # Create installation script
        install_script = package_dir / "install.sh"
        
        install_content = f'''#!/bin/bash
# Bird Vision Raspberry Pi Installation Script
set -e

echo "Installing Bird Vision on Raspberry Pi..."

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    echo "Warning: This doesn't appear to be a Raspberry Pi"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for Python 3.9+
python3 --version | grep -E "3\\.(9|10|11|12)" || {{
    echo "Error: Python 3.9+ required"
    exit 1
}}

# Update system
echo "Updating system packages..."
sudo apt-get update

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get install -y \\
    python3-pip \\
    python3-venv \\
    python3-dev \\
    libcamera-apps \\
    python3-opencv \\
    python3-numpy \\
    python3-picamera2 \\
    libatlas-base-dev \\
    libopenblas-dev

# Create installation directory
echo "Creating installation directories..."
sudo mkdir -p {self.deployment_config.paths.install_dir}
sudo mkdir -p {self.deployment_config.paths.model_dir}
sudo mkdir -p {self.deployment_config.paths.config_dir}
sudo mkdir -p {self.deployment_config.paths.log_dir}
mkdir -p {self.deployment_config.paths.data_dir}

# Set permissions
sudo chown -R pi:pi {self.deployment_config.paths.install_dir}
sudo chown -R pi:pi {self.deployment_config.paths.log_dir}

# Create virtual environment
echo "Creating virtual environment..."
cd {self.deployment_config.paths.install_dir}
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install onnxruntime opencv-python pillow numpy

# Copy files
echo "Installing Bird Vision files..."
cp bird_vision_inference.py {self.deployment_config.paths.install_dir}/
cp bird_vision_camera.py {self.deployment_config.paths.install_dir}/
cp model_optimized.onnx {self.deployment_config.paths.model_dir}/
cp config.json {self.deployment_config.paths.config_dir}/

# Make scripts executable
chmod +x {self.deployment_config.paths.install_dir}/bird_vision_inference.py
chmod +x {self.deployment_config.paths.install_dir}/bird_vision_camera.py

# Install systemd service
if [ -f "bird-vision.service" ]; then
    echo "Installing systemd service..."
    sudo cp bird-vision.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable bird-vision.service
fi

# Test installation
echo "Testing installation..."
cd {self.deployment_config.paths.install_dir}
source venv/bin/activate
python3 bird_vision_inference.py

echo "Bird Vision installation completed successfully!"
echo "Start the service with: sudo systemctl start bird-vision"
echo "View logs with: sudo journalctl -u bird-vision -f"
'''
        
        with open(install_script, 'w') as f:
            f.write(install_content)
        
        # Make executable
        os.chmod(install_script, 0o755)
        
        # Create config file
        config_file = package_dir / "config.json"
        config_dict = OmegaConf.to_container(self.config, resolve=True)
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        return package_dir
    
    def _generate_systemd_service(self, output_dir: Path) -> Path:
        """Generate systemd service file."""
        service_file = output_dir / "raspberry_pi_package" / "bird-vision.service"
        
        service_content = f'''[Unit]
Description=Bird Vision Real-time Detection Service
After=network.target
Wants=network.target

[Service]
Type=simple
User={self.deployment_config.service.user}
Group={self.deployment_config.service.group}
WorkingDirectory={self.deployment_config.paths.install_dir}
Environment=PATH={self.deployment_config.paths.install_dir}/venv/bin:/usr/local/bin:/usr/bin:/bin
ExecStart={self.deployment_config.paths.install_dir}/venv/bin/python3 {self.deployment_config.paths.install_dir}/bird_vision_camera.py
Restart={self.deployment_config.service.restart_policy}
RestartSec=10
StandardOutput=journal
StandardError=journal

# Resource limits
MemoryMax=1G
CPUQuota=80%

# Security settings
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths={self.deployment_config.paths.data_dir} {self.deployment_config.paths.log_dir}

[Install]
WantedBy=multi-user.target
'''
        
        with open(service_file, 'w') as f:
            f.write(service_content)
        
        return service_file
    
    def _validate_performance(self, model: nn.Module) -> Dict[str, Any]:
        """Validate model performance against targets."""
        target_config = self.deployment_config.performance_targets[self.target_device]
        
        # Profile model
        dummy_input = torch.randn(1, 3, 224, 224)
        profile_result = self.profiler.profile_model(model, dummy_input)
        
        # Check against targets
        performance_check = {
            "target_device": self.target_device,
            "model_size_mb": profile_result["model_size_mb"],
            "estimated_inference_time_ms": profile_result["forward_time_ms"] * 2,  # Conservative estimate for Pi
            "meets_size_target": profile_result["model_size_mb"] <= target_config["model_size_mb"],
            "meets_time_target": profile_result["forward_time_ms"] * 2 <= target_config["inference_time_ms"],
            "targets": target_config,
            "profile_results": profile_result
        }
        
        return performance_check
    
    def _generate_deployment_docs(self, output_dir: Path, deployment_result: Dict) -> Path:
        """Generate deployment documentation."""
        docs_file = output_dir / "RASPBERRY_PI_DEPLOYMENT.md"
        
        docs_content = f'''# Bird Vision Raspberry Pi Deployment

## Deployment Summary

- **Target Device**: {self.target_device}
- **Deployment Date**: {time.strftime("%Y-%m-%d %H:%M:%S")}
- **Status**: {"✅ Success" if deployment_result["success"] else "❌ Failed"}

## Performance Metrics

{self._format_performance_table(deployment_result.get("performance", {{}}))}

## Installation Instructions

### 1. Prepare Raspberry Pi

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Enable camera
sudo raspi-config
# Navigate to: Interface Options > Camera > Enable
```

### 2. Install Bird Vision

```bash
# Transfer deployment package to Pi
scp -r raspberry_pi_package pi@your-pi-ip:~/

# Run installation
cd ~/raspberry_pi_package
chmod +x install.sh
./install.sh
```

### 3. Start Service

```bash
# Start service manually
sudo systemctl start bird-vision

# Enable auto-start
sudo systemctl enable bird-vision

# Check status
sudo systemctl status bird-vision
```

### 4. View Logs

```bash
# Real-time logs
sudo journalctl -u bird-vision -f

# Recent logs
sudo journalctl -u bird-vision --since "1 hour ago"
```

## Configuration

Configuration file: `{self.deployment_config.paths.config_dir}/config.json`

Key settings:
- Camera resolution: {self.deployment_config.camera.resolution.width}x{self.deployment_config.camera.resolution.height}
- Frame rate: {self.deployment_config.camera.framerate} FPS
- Confidence threshold: {self.deployment_config.output.confidence_threshold}

## Troubleshooting

### Common Issues

1. **Camera not detected**
   ```bash
   # Check camera connection
   libcamera-hello --list-cameras
   
   # Test camera capture
   libcamera-still -o test.jpg
   ```

2. **Service fails to start**
   ```bash
   # Check service logs
   sudo journalctl -u bird-vision --no-pager
   
   # Test manual execution
   cd {self.deployment_config.paths.install_dir}
   source venv/bin/activate
   python3 bird_vision_camera.py
   ```

3. **Poor performance**
   - Ensure adequate cooling
   - Check CPU/memory usage: `htop`
   - Consider overclocking (with proper cooling)

### Performance Optimization

1. **GPU Memory Split**
   ```bash
   # Allocate more memory to GPU
   sudo raspi-config
   # Advanced Options > Memory Split > 128
   ```

2. **CPU Governor**
   ```bash
   # Set performance mode
   echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
   ```

## API Usage

The deployment includes a REST API server on port {self.deployment_config.integration.api_server.port}:

```bash
# Health check
curl http://localhost:{self.deployment_config.integration.api_server.port}/health

# Get detection from camera
curl http://localhost:{self.deployment_config.integration.api_server.port}/detect

# Get system stats
curl http://localhost:{self.deployment_config.integration.api_server.port}/stats
```

## Files Included

{self._format_artifacts_list(deployment_result.get("artifacts", {{}}))}

## Support

For issues or questions:
1. Check logs: `sudo journalctl -u bird-vision -f`
2. Test manual execution
3. Verify camera connectivity
4. Check system resources
'''
        
        with open(docs_file, 'w') as f:
            f.write(docs_content)
        
        return docs_file
    
    def _format_performance_table(self, performance: Dict) -> str:
        """Format performance metrics as markdown table."""
        if not performance:
            return "No performance data available."
        
        target_config = self.deployment_config.performance_targets[self.target_device]
        
        return f'''
| Metric | Actual | Target | Status |
|--------|--------|--------|--------|
| Model Size | {performance.get("model_size_mb", "N/A"):.1f} MB | {target_config["model_size_mb"]} MB | {"✅" if performance.get("meets_size_target", False) else "❌"} |
| Inference Time | {performance.get("estimated_inference_time_ms", "N/A"):.1f} ms | {target_config["inference_time_ms"]} ms | {"✅" if performance.get("meets_time_target", False) else "❌"} |
'''
    
    def _format_artifacts_list(self, artifacts: Dict) -> str:
        """Format artifacts list as markdown."""
        if not artifacts:
            return "No artifacts generated."
        
        artifact_list = []
        for name, path in artifacts.items():
            artifact_list.append(f"- **{name.replace('_', ' ').title()}**: `{path}`")
        
        return "\n".join(artifact_list)
    
    def _get_model_size(self, model_path: Union[str, Path]) -> int:
        """Get model file size in bytes."""
        return Path(model_path).stat().st_size
    
    @staticmethod
    def detect_raspberry_pi() -> Optional[str]:
        """Detect Raspberry Pi model."""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
            
            if 'Raspberry Pi' in cpuinfo:
                if 'BCM2711' in cpuinfo:  # Pi 4
                    return 'rpi4'
                elif 'BCM2712' in cpuinfo:  # Pi 5
                    return 'rpi5'
                elif 'BCM2837' in cpuinfo:  # Pi 3/Zero 2W
                    return 'rpi_zero2w'
                else:
                    return 'rpi_unknown'
            return None
        except FileNotFoundError:
            return None
    
    def validate_deployment_environment(self) -> Dict[str, Any]:
        """Validate the deployment environment."""
        validation_result = {
            "platform_detected": self.detect_raspberry_pi(),
            "python_version": sys.version,
            "checks": {},
            "warnings": [],
            "errors": []
        }
        
        # Check Python version
        python_version = sys.version_info
        if python_version >= (3, 9):
            validation_result["checks"]["python_version"] = "✅ Pass"
        else:
            validation_result["checks"]["python_version"] = "❌ Fail"
            validation_result["errors"].append("Python 3.9+ required")
        
        # Check for camera
        try:
            import subprocess
            result = subprocess.run(['libcamera-hello', '--list-cameras'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                validation_result["checks"]["camera"] = "✅ Pass"
            else:
                validation_result["checks"]["camera"] = "❌ Fail"
                validation_result["errors"].append("Camera not detected")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            validation_result["checks"]["camera"] = "⚠️ Unknown"
            validation_result["warnings"].append("Could not test camera")
        
        return validation_result