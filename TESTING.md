# Testing Guide for Bird Vision

This document provides comprehensive information about testing the Bird Vision system, including end-to-end testing capabilities.

## Table of Contents

- [Testing Overview](#testing-overview)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Test Categories](#test-categories)
- [End-to-End Testing](#end-to-end-testing)
- [Performance Benchmarking](#performance-benchmarking)
- [CI/CD Pipeline](#cicd-pipeline)
- [Writing Tests](#writing-tests)
- [Troubleshooting](#troubleshooting)

## Testing Overview

The Bird Vision project includes a comprehensive testing framework that covers:

- **Unit Tests**: Individual component testing
- **Integration Tests**: Cross-component interaction testing
- **End-to-End Tests**: Complete pipeline testing from data to deployment
- **Performance Benchmarks**: Speed, memory, and quality metrics
- **Platform Tests**: iOS, Android, and ESP32-P4-Eye deployment testing

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── unit/                    # Unit tests for individual components
│   ├── test_data.py         # Data loading and preprocessing
│   ├── test_models.py       # Model architectures
│   ├── test_training.py     # Training components
│   ├── test_compression.py  # Model compression
│   └── test_deployment.py   # Deployment utilities
├── integration/             # Integration tests
│   └── test_platform_integration.py  # Platform-specific integration
├── e2e/                     # End-to-end tests
│   └── test_full_pipeline.py         # Complete pipeline testing
├── benchmarks/              # Performance benchmarks
│   └── test_performance.py           # Speed and memory benchmarks
└── fixtures/                # Test data and utilities
    └── mock_data.py         # Mock data generators
```

## Running Tests

### Quick Start

```bash
# Run all tests
python scripts/run_tests.py

# Run specific test categories
python scripts/run_tests.py --unit-only
python scripts/run_tests.py --integration-only
python scripts/run_tests.py --e2e-only
python scripts/run_tests.py --performance-only

# Run with verbose output
python scripts/run_tests.py --verbose

# Skip slow tests for quick feedback
python scripts/run_tests.py --skip-slow
```

### Using pytest directly

```bash
# Run all tests
pytest

# Run specific test files
pytest tests/unit/test_models.py
pytest tests/e2e/test_full_pipeline.py

# Run with coverage
pytest --cov=bird_vision --cov-report=html

# Run performance benchmarks
pytest tests/benchmarks/ --benchmark-only

# Run with specific markers
pytest -m "not slow"
pytest -m "unit"
pytest -m "integration"
```

### Docker Testing

```bash
# Build test image
docker build -f docker/Dockerfile -t bird-vision-test .

# Run tests in container
docker run --rm bird-vision-test python scripts/run_tests.py
```

## Test Categories

### Unit Tests

Test individual components in isolation:

```bash
# Data pipeline tests
pytest tests/unit/test_data.py

# Model architecture tests
pytest tests/unit/test_models.py

# Training component tests
pytest tests/unit/test_training.py

# Compression algorithm tests
pytest tests/unit/test_compression.py

# Deployment utility tests
pytest tests/unit/test_deployment.py
```

**Coverage**: Data loading, model forward/backward passes, training loops, compression algorithms, deployment utilities.

### Integration Tests

Test component interactions and platform-specific functionality:

```bash
# Platform integration tests
pytest tests/integration/test_platform_integration.py
```

**Coverage**: iOS/Android/ESP32 deployment pipelines, cross-platform compatibility, model format conversions.

### End-to-End Tests

Test complete workflows from data to deployment:

```bash
# Full pipeline tests
pytest tests/e2e/test_full_pipeline.py
```

**Coverage**: 
- Data loading → Model training → Validation → Compression → Deployment
- CLI command integration
- Error handling and recovery
- Memory management

### Performance Benchmarks

Measure system performance across different scenarios:

```bash
# Run performance benchmarks
pytest tests/benchmarks/test_performance.py --benchmark-only

# Generate benchmark report
pytest tests/benchmarks/ --benchmark-json=benchmark.json
```

**Metrics Tested**:
- Inference speed (batch sizes 1-16)
- Memory usage patterns
- Model compression ratios
- Export/conversion times
- Platform-specific performance targets

## End-to-End Testing

### Complete Pipeline Testing

The E2E tests validate the entire Bird Vision pipeline:

```python
# Example: Test complete training to deployment pipeline
def test_full_pipeline():
    # 1. Data Loading
    data_module = create_mock_data_module()
    
    # 2. Model Training
    model = train_model(data_module, epochs=2)
    
    # 3. Model Validation
    metrics = validate_model(model, test_data)
    assert metrics["accuracy"] > 0.7
    
    # 4. Model Compression
    compressed_model = compress_model(model)
    assert compressed_model.size_mb < original_model.size_mb
    
    # 5. Platform Deployment
    ios_package = deploy_to_ios(compressed_model)
    android_package = deploy_to_android(compressed_model)
    esp32_package = deploy_to_esp32(compressed_model)
    
    # 6. Validation
    assert all(package.success for package in [ios_package, android_package, esp32_package])
```

### Platform-Specific E2E Testing

#### iOS Deployment Pipeline
```bash
# Test iOS-specific pipeline
pytest tests/integration/test_platform_integration.py::TestiOSIntegration -v
```

Tests:
- CoreML model conversion
- Swift integration code generation
- iOS-specific optimizations
- Vision framework compatibility

#### Android Deployment Pipeline
```bash
# Test Android-specific pipeline
pytest tests/integration/test_platform_integration.py::TestAndroidIntegration -v
```

Tests:
- TorchScript optimization for mobile
- Java integration code generation
- PyTorch Mobile compatibility
- Android-specific optimizations

#### ESP32-P4-Eye Deployment Pipeline
```bash
# Test ESP32-specific pipeline
pytest tests/integration/test_platform_integration.py::TestESP32Integration -v
```

Tests:
- ESP-DL model conversion
- Firmware generation (ESP-IDF)
- AI accelerator optimization
- Camera interface configuration
- Build script generation

### Data Integrity Testing

Validates data flow through the entire pipeline:

```python
def test_data_flow_integrity():
    """Test data maintains consistency through pipeline."""
    # Verify tensor shapes remain consistent
    # Check gradient flow in training
    # Validate inference consistency
    # Test device compatibility
```

### Error Handling and Recovery

Tests system resilience:

```python
def test_error_handling():
    """Test error handling in end-to-end scenarios."""
    # Invalid input handling
    # Missing file handling
    # Device mismatch handling
    # Memory management under stress
```

## Performance Benchmarking

### Model Performance Tests

```bash
# Inference speed benchmarks
pytest tests/benchmarks/test_performance.py::TestModelPerformance::test_inference_speed_benchmark

# Memory usage benchmarks
pytest tests/benchmarks/test_performance.py::TestModelPerformance::test_memory_usage_benchmark

# Batch scaling tests
pytest tests/benchmarks/test_performance.py::TestModelPerformance::test_batch_scaling_performance
```

### Compression Performance Tests

```bash
# Quantization benchmarks
pytest tests/benchmarks/test_performance.py::TestCompressionPerformance::test_quantization_performance

# Pruning benchmarks
pytest tests/benchmarks/test_performance.py::TestCompressionPerformance::test_pruning_performance
```

### Platform-Specific Performance Tests

```bash
# Mobile performance targets
pytest tests/benchmarks/test_performance.py::TestPlatformSpecificPerformance::test_mobile_performance_targets

# ESP32 performance targets
pytest tests/benchmarks/test_performance.py::TestPlatformSpecificPerformance::test_esp32_performance_targets
```

**Performance Targets Tested**:

| Platform | Model Size | Inference Time | Accuracy Retention |
|----------|------------|----------------|-------------------|
| Mobile (iOS/Android) | < 50 MB | < 100 ms | > 95% |
| ESP32-P4-Eye | < 8 MB | < 200 ms | > 90% |

### Stress Testing

```bash
# Continuous inference stress test
pytest tests/benchmarks/test_performance.py::TestStressTest::test_continuous_inference_stress

# Memory leak detection
pytest tests/benchmarks/test_performance.py::TestStressTest::test_memory_leak_detection
```

## CI/CD Pipeline

### GitHub Actions Workflow

The CI pipeline (`.github/workflows/ci.yml`) includes:

1. **Code Quality**: Black, isort, flake8, mypy
2. **Unit Tests**: Cross-platform (Ubuntu, Windows, macOS)
3. **Integration Tests**: Platform deployment testing
4. **Performance Tests**: Nightly benchmarks
5. **Security Scans**: Safety, bandit, semgrep
6. **Build Tests**: Package building and Docker
7. **Deployment Tests**: End-to-end deployment validation

### Running CI Locally

```bash
# Install act (GitHub Actions runner)
# https://github.com/nektos/act

# Run the entire CI pipeline locally
act

# Run specific jobs
act -j unit-tests
act -j integration-tests
act -j performance-tests
```

### Test Triggers

- **Push/PR**: All tests except performance
- **Schedule (nightly)**: Complete test suite including benchmarks
- **Manual**: Triggered with `[benchmark]` or `[deploy-test]` in commit message

## Writing Tests

### Test Structure Guidelines

```python
class TestNewFeature:
    """Test new feature functionality."""
    
    def test_basic_functionality(self, fixture_name):
        """Test basic feature operation."""
        # Arrange
        setup_data = create_test_data()
        
        # Act
        result = new_feature.process(setup_data)
        
        # Assert
        assert result.success
        assert result.output_shape == expected_shape
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test invalid inputs
        with pytest.raises(ValueError):
            new_feature.process(invalid_data)
    
    def test_performance(self, benchmark):
        """Test performance characteristics."""
        result = benchmark(new_feature.process, test_data)
        assert result < performance_threshold
```

### Mock Data Usage

```python
from tests.fixtures.mock_data import (
    MockBirdDataGenerator,
    create_temporary_mock_dataset
)

def test_with_mock_data():
    """Test using generated mock data."""
    temp_dir, metadata = create_temporary_mock_dataset(
        num_classes=5,
        samples_per_class=10
    )
    
    # Use temp_dir and metadata for testing
    assert metadata["num_classes"] == 5
    assert len(metadata["image_paths"]) == 50
```

### Platform-Specific Testing

```python
@pytest.mark.skipif(platform.system() != "Darwin", reason="iOS testing requires macOS")
def test_ios_specific_feature():
    """Test iOS-specific functionality."""
    pass

@pytest.mark.requires_gpu
def test_gpu_accelerated_feature():
    """Test GPU-specific functionality."""
    pass
```

### Performance Testing

```python
def test_performance_target(benchmark):
    """Test meets performance targets."""
    result = benchmark.pedantic(
        target_function,
        args=(test_input,),
        iterations=10,
        rounds=5
    )
    
    # Assert performance targets
    assert result.stats.mean < 0.1  # Under 100ms
    assert result.stats.stddev < 0.01  # Low variance
```

## Troubleshooting

### Common Test Issues

#### Import Errors
```bash
# Ensure package is installed in development mode
pip install -e .

# Check Python path
python -c "import bird_vision; print(bird_vision.__file__)"
```

#### Missing Dependencies
```bash
# Install all test dependencies
pip install -e ".[dev]"

# Install platform-specific dependencies
pip install -e ".[esp32]"  # For ESP32 tests
```

#### GPU Tests Failing
```bash
# Skip GPU tests on CPU-only systems
pytest -m "not requires_gpu"

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

#### Slow Tests Timing Out
```bash
# Increase timeout
pytest --timeout=300

# Skip slow tests
pytest -m "not slow"

# Run specific fast tests only
python scripts/run_tests.py --skip-slow
```

#### Memory Issues
```bash
# Run tests with memory monitoring
pytest --memray

# Reduce batch sizes in tests
# Check conftest.py for memory-optimized fixtures
```

### Test Data Issues

#### Mock Data Generation Fails
```bash
# Check disk space
df -h

# Verify write permissions
ls -la tests/fixtures/

# Clear temporary files
rm -rf /tmp/bird_vision_test_*
```

#### Real Data Tests Fail
```bash
# Skip data download tests in CI
pytest -m "not requires_internet"

# Use cached data
export BIRD_VISION_USE_CACHE=1
```

### Platform-Specific Issues

#### iOS Tests (macOS required)
```bash
# Install Xcode command line tools
xcode-select --install

# Check CoreML availability
python -c "import coremltools; print('CoreML available')"
```

#### ESP32 Tests
```bash
# Install ESP-IDF dependencies
# Note: ESP32 tests use mocked ESP-DL conversion

# Check ESP32 dependencies
python -c "from bird_vision.deployment.esp32_deployer import ESP32Deployer; print('ESP32 deployer available')"
```

### Debug Mode

Enable debug mode for detailed test output:

```bash
# Enable debug logging
export BIRD_VISION_LOG_LEVEL=DEBUG

# Run with maximum verbosity
pytest -vvv --tb=long --show-capture=all

# Save debug output
pytest --tb=short > test_debug.log 2>&1
```

### Performance Debug

```bash
# Profile test execution
pytest --profile

# Memory profiling
pytest --memray --memray-bin-path=memray_output/

# Benchmark comparison
pytest-benchmark compare benchmark_results.json
```

## Test Coverage

Target coverage levels:
- **Overall**: > 80%
- **Core modules**: > 90%
- **CLI interface**: > 85%
- **Platform deployment**: > 75%

Generate coverage reports:

```bash
# HTML coverage report
pytest --cov=bird_vision --cov-report=html
open htmlcov/index.html

# Terminal coverage report
pytest --cov=bird_vision --cov-report=term-missing

# XML coverage for CI
pytest --cov=bird_vision --cov-report=xml
```

## Contributing Tests

When contributing new features:

1. **Write tests first** (TDD approach)
2. **Include unit tests** for all new functions/classes
3. **Add integration tests** for cross-component features
4. **Update E2E tests** if pipeline changes
5. **Add performance tests** for computationally intensive features
6. **Update documentation** including this testing guide

Test quality checklist:
- [ ] Tests are deterministic (no random failures)
- [ ] Tests are isolated (no dependencies between tests)
- [ ] Tests are fast (unit tests < 1s, integration < 10s)
- [ ] Tests cover edge cases and error conditions
- [ ] Tests include performance assertions where relevant
- [ ] Tests work across all supported platforms