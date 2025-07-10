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

1. Clone the repository:
```bash
git clone <repository-url>
cd bird_vision
```

2. Install in development mode:
```bash
pip install -e ".[dev,audio]"
```

3. Install pre-commit hooks:
```bash
pre-commit install
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

### Running Tests
```bash
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

### Code Formatting
```bash
black src/ tests/
isort src/ tests/
```

### Type Checking
```bash
mypy src/
```

### Pre-commit Hooks
```bash
pre-commit run --all-files
```

## Docker Support

Build and run with Docker:
```bash
# Build image
docker build -t bird-vision -f docker/Dockerfile .

# Run training
docker run --gpus all -v $(pwd)/data:/app/data bird-vision python scripts/train_model.py
```

## Future Enhancements

- **Audio Integration**: Multi-modal identification using bird calls
- **Real-time Inference**: Streaming video analysis
- **Edge Deployment**: Raspberry Pi and edge device support
- **Web Interface**: Browser-based inference
- **Federated Learning**: Distributed training capabilities

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

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