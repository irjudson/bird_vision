# Bird Vision Installation Guide

## Python Version Requirements

Bird Vision is optimized for **Python 3.10+** and has been tested with **Python 3.13**.

## Quick Installation

### Basic Installation
```bash
# Clone the repository
git clone <repository-url>
cd bird_vision

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e ".[dev]"
```

### Platform-Specific Installations

#### Development Environment
```bash
pip install -r requirements-dev.txt
```

#### Raspberry Pi Deployment
```bash
# On Raspberry Pi OS
sudo apt update
sudo apt install -y python3-picamera2 libcamera-apps python3-dev

# Install with Pi wheels for better performance
pip install --extra-index-url https://www.piwheels.org/simple/ -r requirements-raspberry-pi.txt
```

#### ESP32 Development
```bash
pip install -r requirements-esp32.txt

# Install ESP-IDF separately (follow Espressif documentation)
```

#### Audio Processing
```bash
# Install system dependencies first
# Ubuntu/Debian: sudo apt install libsndfile1 ffmpeg
# macOS: brew install libsndfile ffmpeg

pip install -r requirements-audio.txt
```

#### TensorFlow Lite / LiteRT
```bash
# Modern LiteRT (recommended for mobile deployment)
pip install -r requirements-tflite.txt

# Alternative: Full TensorFlow (includes tf.lite)
pip install tensorflow
```

## Using pyproject.toml (Recommended)

For a complete installation with all optional dependencies:

```bash
# Basic installation
pip install -e .

# Development installation
pip install -e ".[dev]"

# Raspberry Pi installation
pip install -e ".[raspberry_pi]"

# ESP32 development
pip install -e ".[esp32]"

# Audio processing
pip install -e ".[audio]"

# TensorFlow Lite / LiteRT
pip install -e ".[tflite]"

# All features
pip install -e ".[dev,raspberry_pi,esp32,audio,tflite]"
```

## Key Dependencies

### Core Framework
- **PyTorch 2.4+**: Deep learning framework with Python 3.13 support
- **Torchvision 0.19+**: Computer vision models and utilities
- **TimM 1.0+**: Pre-trained vision models

### Computer Vision
- **OpenCV 4.10+**: Image processing and computer vision
- **Albumentations 1.4+**: Advanced data augmentation

### Model Deployment
- **ONNX 1.16+**: Cross-platform model format
- **ONNXRuntime 1.19+**: High-performance inference
- **CoreML Tools 8.0+**: iOS deployment
- **OpenVINO 2024.4+**: Intel optimization
- **LiteRT (ai-edge-litert) 1.0+**: Mobile/edge deployment (formerly TensorFlow Lite)

### Configuration & Tracking
- **Hydra 1.3.2+**: Configuration management
- **MLflow 2.16+**: Experiment tracking
- **Weights & Biases 0.18+**: Advanced experiment tracking

## Verification

Test your installation:

```bash
# Check core imports
python -c "import torch, torchvision, timm; print('Core ML libraries OK')"

# Check computer vision
python -c "import cv2, albumentations; print('CV libraries OK')"

# Check deployment tools
python -c "import onnx, onnxruntime; print('Deployment tools OK')"

# Run the CLI
bird-vision --help
```

## Troubleshooting

### Common Issues

1. **NumPy Compatibility**: We pin NumPy to `>=1.26.0,<2.1.0` for compatibility
2. **PyTorch Version**: Ensure you have PyTorch 2.4+ for Python 3.13 support
3. **ARM Devices**: Use piwheels on Raspberry Pi for pre-compiled wheels
4. **TensorFlow Lite**: Use `ai-edge-litert` instead of deprecated `tensorflow-lite` package

### Platform-Specific Notes

#### macOS Apple Silicon
```bash
# Use conda for better ARM64 support if needed
conda install pytorch torchvision torchaudio -c pytorch
pip install -r requirements.txt
```

#### Windows
- Consider using conda or virtualenv
- Some dependencies may require Visual Studio Build Tools

#### Raspberry Pi
- Always use piwheels: `--extra-index-url https://www.piwheels.org/simple/`
- Enable camera: `sudo raspi-config` → Interface Options → Camera

## Development Setup

For contributors:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linting
ruff check .
black --check .
mypy .
```

## Version Compatibility Matrix

| Python | PyTorch | Status |
|--------|---------|--------|
| 3.10   | 2.4+    | ✅ Supported |
| 3.11   | 2.4+    | ✅ Supported |
| 3.12   | 2.4+    | ✅ Supported |
| 3.13   | 2.4+    | ✅ Fully Tested |
| 3.9    | 2.4+    | ⚠️ Legacy (not recommended) |