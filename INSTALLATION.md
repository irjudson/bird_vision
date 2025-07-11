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

#### Mobile Deployment (ONNX Runtime - Python 3.13 Compatible)
```bash
# ONNX Runtime for mobile deployment (Python 3.13 compatible)
pip install -r requirements-tflite.txt

# Note: LiteRT (ai-edge-litert) doesn't support Python 3.13 yet
# Alternative for Python ≤3.12: Full TensorFlow (includes tf.lite)
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

# Mobile Deployment (ONNX Runtime)
pip install -e ".[tflite]"

# All features
pip install -e ".[dev,raspberry_pi,esp32,audio,tflite]"
```

## Key Dependencies (Latest Versions - January 2025)

### Core Framework
- **PyTorch 2.7.1+**: Latest deep learning framework with full Python 3.13 support
- **Torchvision 0.20.1+**: Computer vision models and utilities
- **TimM 1.0.12+**: Latest pre-trained vision models

### Data Science Stack
- **NumPy 2.2.1+**: Latest with Python 3.13 support and performance improvements
- **Pandas 2.3.0+**: Modern data manipulation with Python 3.13 compatibility
- **Scikit-learn 1.7.0+**: Latest ML algorithms with optimizations

### Computer Vision
- **OpenCV 4.12.0+**: Latest image processing and computer vision
- **Albumentations 1.4.23+**: Advanced data augmentation
- **Pillow 11.1.0+**: Modern image processing library

### Model Deployment
- **ONNX 1.18.0+**: Latest cross-platform model format
- **ONNXRuntime 1.21.0+**: High-performance inference engine
- **CoreML Tools 8.2.0+**: Latest iOS deployment tools
- **OpenVINO 2024.6.0+**: Intel optimization suite
- **ONNX Runtime 1.21.0+**: Cross-platform mobile/edge deployment (Python 3.13 compatible)
- **LiteRT (ai-edge-litert)**: Google's mobile deployment (Python ≤3.12 only)

### Configuration & Tracking
- **Hydra 1.3.2+**: Configuration management
- **MLflow 2.21.0+**: Latest experiment tracking
- **Weights & Biases 0.21.0+**: Advanced experiment tracking
- **Pydantic 2.10.2+**: Modern data validation

### Development Tools
- **Ruff 0.10.0+**: Ultra-fast Python linter and formatter
- **Black 24.12.0+**: Latest code formatter
- **MyPy 1.14.0+**: Advanced type checking

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

1. **NumPy 2.x**: Latest NumPy 2.2.1+ with major performance improvements and Python 3.13 support
2. **PyTorch Version**: Ensure you have PyTorch 2.7.1+ for full Python 3.13 support
3. **TensorFlow**: Limited Python 3.13 support - use Python 3.12 if TensorFlow is critical
4. **ARM Devices**: Use piwheels on Raspberry Pi for pre-compiled wheels
5. **Mobile Deployment**: Use ONNX Runtime 1.21.0+ for Python 3.13 compatible mobile deployment
6. **LiteRT**: ai-edge-litert only supports Python ≤3.12 currently

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

## Version Compatibility Matrix (January 2025)

| Python | PyTorch | TensorFlow | NumPy | Status |
|--------|---------|------------|-------|--------|
| 3.10   | 2.7.1+  | 2.18.0+    | 2.2.1+ | ✅ Fully Supported |
| 3.11   | 2.7.1+  | 2.18.0+    | 2.2.1+ | ✅ Fully Supported |
| 3.12   | 2.7.1+  | 2.18.0+    | 2.2.1+ | ✅ Fully Supported |
| 3.13   | 2.7.1+  | ❌ No Support | 2.2.1+ | ✅ **Recommended** (ONNX Runtime for mobile) |
| 3.9    | 2.7.1+  | 2.18.0+    | 2.2.1+ | ⚠️ Legacy (EOL soon) |

**Key Updates:**
- **Python 3.13**: Full support except TensorFlow and LiteRT
- **Mobile Deployment**: Use ONNX Runtime for Python 3.13 compatibility
- **NumPy 2.x**: Major performance improvements, breaking changes from 1.x
- **PyTorch 2.7.1**: Latest with comprehensive Python 3.13 support
- **TensorFlow & LiteRT**: Use Python ≤3.12 for compatibility