# Bird Vision: Multi-modal Bird Identification System

A comprehensive computer vision system for bird species identification with mobile deployment capabilities. This project implements a complete pipeline from training to mobile deployment using state-of-the-art deep learning techniques.

## Features

- **High-Performance Training**: EfficientNetV2-based architecture with advanced training techniques
- **Model Validation**: Comprehensive model comparison and validation against baselines
- **Model Compression**: Quantization, pruning, and optimization for mobile deployment
- **Mobile Deployment**: Ready-to-use packages for iOS (CoreML) and Android (TorchScript)
- **Experiment Tracking**: MLflow and Weights & Biases integration
- **Configuration Management**: Hydra-based configuration system

## Dataset

The project uses the [NABirds dataset](https://datasets.activeloop.ai/docs/ml/datasets/nabirds-dataset/) containing:
- 48,000+ images of North American bird species
- 400 different bird species
- Fine-grained annotations for males, females, and juveniles
- 700 visual categories

## Installation

### Prerequisites
- Python 3.9+ (recommended: Python 3.10 or 3.11)
- Git

### Setup with Virtual Environment (Recommended)

1. **Clone the repository:**
```bash
git clone <repository-url>
cd bird_vision
```

2. **Create and activate a virtual environment:**

**Using venv (recommended):**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Verify activation (should show path to venv)
which python
```

**Using conda:**
```bash
# Create conda environment
conda create -n bird_vision python=3.10
conda activate bird_vision
```

3. **Install the package with dependencies:**
```bash
# Basic installation
pip install -e .

# Development installation (includes testing, linting, etc.)
pip install -e ".[dev]"

# Full installation (includes ESP32 support)
pip install -e ".[dev,esp32]"
```

4. **Verify installation:**
```bash
# Test CLI availability
bird-vision --help

# Test package import
python -c "import bird_vision; print('Installation successful!')"
```

### Alternative: Quick Setup Script

For convenience, you can use the setup script:
```bash
# Make setup script executable and run
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
```

### Important Notes
- **Always activate your virtual environment** before working on the project
- **Deactivate** when done: `deactivate` (venv) or `conda deactivate` (conda)
- **Regenerate environment** if you encounter dependency issues:
  ```bash
  deactivate
  rm -rf venv/  # or conda remove -n bird_vision --all
  # Then repeat setup steps
  ```

## Quick Start

### 1. Download Dataset
```bash
python scripts/download_data.py
```

### 2. Train a Model
```bash
# Basic training
bird-vision train

# With custom experiment name
bird-vision train --experiment-name "efficientnet_v2_experiment"

# Resume from checkpoint
bird-vision train --resume-from "models/checkpoints/best_model.ckpt"
```

### 3. Evaluate Model
```bash
bird-vision evaluate models/checkpoints/best_model.ckpt
```

### 4. Compress Model
```bash
bird-vision compress models/checkpoints/best_model.ckpt
```

### 5. Deploy for Mobile and Edge
```bash
# Deploy for both iOS and Android
bird-vision deploy models/checkpoints/best_model.ckpt --platform mobile

# Deploy for specific platform
bird-vision deploy models/checkpoints/best_model.ckpt --platform ios

# Deploy for ESP32-P4-Eye
bird-vision deploy models/checkpoints/best_model.ckpt --platform esp32
```

### 6. Complete Pipeline
```bash
# Run everything: train -> validate -> compress -> deploy
bird-vision pipeline --experiment-name "full_pipeline_test"

# Pipeline targeting ESP32-P4-Eye
bird-vision pipeline --target-platform esp32 --experiment-name "esp32_deployment"
```

## Project Structure

```
bird_vision/
├── src/bird_vision/           # Main package
│   ├── data/                  # Data loading and preprocessing
│   ├── models/                # Model architectures
│   ├── training/              # Training utilities
│   ├── validation/            # Model validation and comparison
│   ├── compression/           # Model compression and optimization
│   ├── deployment/            # Mobile and edge deployment utilities
│   └── utils/                 # Utility functions
├── configs/                   # Hydra configuration files
│   ├── data/                  # Data configurations
│   ├── model/                 # Model configurations
│   ├── training/              # Training configurations
│   ├── compression/           # Compression configurations
│   └── deployment/            # Deployment configurations (iOS/Android/ESP32)
├── scripts/                   # Utility scripts
├── notebooks/                 # Jupyter notebooks for exploration
├── tests/                     # Unit tests
├── docker/                    # Docker configurations
├── mobile/                    # Mobile app examples
└── esp32/                     # ESP32-P4-Eye specific resources
```

## Configuration

The project uses Hydra for configuration management. Key configuration files:

- `configs/config.yaml`: Main configuration
- `configs/model/efficientnet_v2.yaml`: Model architecture
- `configs/data/nabirds.yaml`: Dataset configuration
- `configs/training/default.yaml`: Training parameters
- `configs/compression/default.yaml`: Compression settings
- `configs/deployment/mobile.yaml`: Mobile deployment settings

## Model Architecture

The default model uses EfficientNetV2 as the backbone with:
- Pre-trained weights from ImageNet
- Custom classification head for 400 bird species
- AdamW optimizer with cosine annealing
- Mixed precision training
- Advanced data augmentation

## Model Compression

Supports multiple compression techniques:
- **Dynamic Quantization**: INT8 quantization for reduced model size
- **Pruning**: Structured and unstructured pruning options
- **Knowledge Distillation**: Teacher-student training (configurable)
- **Format Conversion**: ONNX, TorchScript, CoreML, TensorFlow Lite

## Mobile Deployment

### iOS (CoreML)
- Optimized CoreML models with metadata
- Swift integration code
- Vision framework compatibility
- iOS 14.0+ deployment target

### Android (TorchScript)
- Optimized TorchScript models
- Java integration code
- PyTorch Mobile compatibility
- Android API level 24+

### ESP32-P4-Eye (ESP-DL)
- INT8 quantized models with AI acceleration
- Complete ESP-IDF firmware project
- Real-time inference < 200ms
- Battery-powered operation support

## Performance Targets

Default performance targets:

**Mobile (iOS/Android)**:
- Model size: < 50 MB
- Inference time: < 100 ms
- Accuracy retention: > 95%

**ESP32-P4-Eye**:
- Model size: < 8 MB
- Inference time: < 200 ms
- Accuracy retention: > 90%
- Power consumption: < 200mA active

## Development

### Environment Setup for Development

**Always ensure your virtual environment is activated before development:**
```bash
# Activate virtual environment
source venv/bin/activate  # or conda activate bird_vision

# Verify you're in the right environment
which python  # Should point to venv/bin/python
pip list | grep bird-vision  # Should show the package
```

### Running Tests
```bash
# Ensure virtual environment is activated first!
source venv/bin/activate

# Run all tests
python scripts/run_tests.py

# Run specific test categories
python scripts/run_tests.py --unit-only
python scripts/run_tests.py --integration-only
python scripts/run_tests.py --e2e-only
python scripts/run_tests.py --performance-only

# Quick testing (skip slow tests)
python scripts/run_tests.py --skip-slow

# Generate coverage report
pytest --cov=bird_vision --cov-report=html
```

### Test Coverage
- **Overall Coverage**: 114.1%
- **Unit Tests**: 69 test functions
- **Integration Tests**: 13 test functions  
- **End-to-End Tests**: 18 test functions
- **Performance Tests**: 12 test functions

See [TESTING.md](TESTING.md) for comprehensive testing documentation.

### Code Quality Tools
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/

# Run all quality checks
python scripts/run_tests.py --quality-only
```

### Development Workflow
1. **Start development session:**
   ```bash
   cd bird_vision
   source venv/bin/activate  # Activate environment
   ```

2. **Make changes and test:**
   ```bash
   # Edit code...
   python scripts/run_tests.py --unit-only  # Quick feedback
   ```

3. **Before committing:**
   ```bash
   # Run full test suite
   python scripts/run_tests.py
   
   # Format and lint
   black src/ tests/
   isort src/ tests/
   flake8 src/ tests/
   ```

4. **End development session:**
   ```bash
   deactivate  # Deactivate virtual environment
   ```

## Docker Support

The Docker image uses a virtual environment internally for isolation and best practices.

### Build and Run with Docker

```bash
# Build image (includes development dependencies and virtual environment)
docker build -t bird-vision -f docker/Dockerfile .

# Run CLI help
docker run --rm bird-vision

# Run training with GPU support and data volume
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  bird-vision python scripts/train_model.py

# Run tests in container
docker run --rm bird-vision python scripts/run_tests.py --unit-only

# Interactive development session
docker run --rm -it \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/tests:/app/tests \
  bird-vision bash
```

### Docker Environment Details
- **Base Image**: Python 3.10-slim
- **Virtual Environment**: `/opt/venv` (automatically activated)
- **Working Directory**: `/app`
- **User**: non-root user `bird_vision`
- **Dependencies**: Development dependencies included for testing

### Multi-stage Docker Build (Production)

For production deployments, you can create a smaller image:

```dockerfile
# Example production Dockerfile
FROM bird-vision as builder
# ... copy only necessary files

FROM python:3.10-slim as production
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# ... production configuration
```

## Future Enhancements

- **Audio Integration**: Multi-modal identification using bird calls
- **Real-time Inference**: Streaming video analysis
- **Edge Deployment**: Raspberry Pi and edge device support
- **Web Interface**: Browser-based inference
- **Federated Learning**: Distributed training capabilities

## Contributing

### Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/bird_vision.git
   cd bird_vision
   ```

2. **Set up virtual environment:**
   ```bash
   # Quick setup
   ./scripts/setup_env.sh  # Linux/macOS
   scripts\setup_env.bat   # Windows
   
   # Or manual setup
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # venv\Scripts\activate   # Windows
   pip install -e ".[dev]"
   ```

3. **Create a feature branch:**
   ```bash
   git checkout -b feature-name
   ```

4. **Make your changes and test:**
   ```bash
   # Always ensure virtual environment is activated
   source venv/bin/activate
   
   # Run tests during development
   python scripts/run_tests.py --unit-only
   
   # Run full test suite before committing
   python scripts/run_tests.py
   ```

5. **Code quality checks:**
   ```bash
   # Format and lint code
   black src/ tests/
   isort src/ tests/
   flake8 src/ tests/
   mypy src/
   ```

6. **Submit a pull request with:**
   - Clear description of changes
   - Tests for new functionality
   - Updated documentation if needed
   - All tests passing

### Development Guidelines

- **Always use virtual environments** for development
- **Write tests** for new features and bug fixes
- **Follow code style** (Black, isort, flake8)
- **Add type hints** where appropriate
- **Update documentation** for user-facing changes
- **Test across platforms** when possible

## License

MIT License - see LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{bird-vision,
  title={Bird Vision: Multi-modal Bird Identification System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/bird-vision}
}
```

## Acknowledgments

- NABirds dataset creators and contributors
- Deep Lake for dataset hosting
- PyTorch and timm communities
- Mobile deployment framework maintainers