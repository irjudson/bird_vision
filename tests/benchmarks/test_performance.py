"""Performance benchmarking tests for Bird Vision system."""

import pytest
import torch
import time
import psutil
import gc
from pathlib import Path
from typing import Dict, List, Tuple
from omegaconf import DictConfig

from bird_vision.models.vision_model import VisionModel
from bird_vision.utils.model_utils import ModelProfiler


class TestModelPerformance:
    """Test model performance benchmarks."""
    
    def test_inference_speed_benchmark(self, sample_model: VisionModel, benchmark_config: Dict):
        """Benchmark model inference speed."""
        sample_model.eval()
        device = torch.device("cpu")
        sample_model.to(device)
        
        results = {}
        
        for input_size in benchmark_config["input_sizes"]:
            batch_size, channels, height, width = input_size
            
            # Create input tensor
            input_tensor = torch.randn(batch_size, channels, height, width).to(device)
            
            # Warmup runs
            with torch.no_grad():
                for _ in range(benchmark_config["num_warmup_runs"]):
                    _ = sample_model(input_tensor)
            
            # Benchmark runs
            torch.cuda.synchronize() if device.type == "cuda" else None
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(benchmark_config["num_benchmark_runs"]):
                    output = sample_model(input_tensor)
            
            torch.cuda.synchronize() if device.type == "cuda" else None
            end_time = time.time()
            
            # Calculate metrics
            total_time = end_time - start_time
            avg_time_per_inference = total_time / benchmark_config["num_benchmark_runs"]
            throughput = batch_size / avg_time_per_inference  # samples per second
            
            results[f"{input_size}"] = {
                "avg_inference_time_ms": avg_time_per_inference * 1000,
                "throughput_samples_per_sec": throughput,
                "total_time_sec": total_time,
            }
            
            # Assert reasonable performance
            assert avg_time_per_inference < 1.0  # Should be under 1 second
            assert output.shape[0] == batch_size
        
        return results
    
    def test_memory_usage_benchmark(self, sample_model: VisionModel, benchmark_config: Dict):
        """Benchmark model memory usage."""
        device = torch.device("cpu")
        sample_model.to(device)
        
        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        results = {}
        
        for batch_size in benchmark_config["batch_sizes"]:
            # Clear cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            input_tensor = torch.randn(batch_size, 3, 224, 224).to(device)
            
            # Measure memory before inference
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Run inference
            sample_model.eval()
            with torch.no_grad():
                output = sample_model(input_tensor)
            
            # Measure memory after inference
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            
            memory_used = memory_after - memory_before
            memory_per_sample = memory_used / batch_size if batch_size > 0 else 0
            
            results[f"batch_{batch_size}"] = {
                "memory_used_mb": memory_used,
                "memory_per_sample_mb": memory_per_sample,
                "total_memory_mb": memory_after,
            }
            
            # Assert memory usage is reasonable
            assert memory_after < benchmark_config["memory_threshold_mb"]
            assert output.shape[0] == batch_size
        
        return results
    
    def test_model_size_benchmark(self, sample_model: VisionModel):
        """Benchmark model size and parameter count."""
        profiler = ModelProfiler()
        
        # Create dummy input for profiling
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Profile model
        stats = profiler.profile_model(sample_model, dummy_input)
        
        # Extract metrics
        results = {
            "model_size_mb": stats["size_mb"],
            "total_parameters": stats["parameters"],
            "trainable_parameters": stats["trainable_parameters"],
            "flops": stats["flops"],
            "inference_time_ms": stats["avg_inference_time_ms"],
        }
        
        # Assert reasonable model size
        assert results["model_size_mb"] < 100  # Should be under 100MB for test model
        assert results["total_parameters"] > 1000  # Should have reasonable number of parameters
        assert results["trainable_parameters"] <= results["total_parameters"]
        assert results["inference_time_ms"] > 0
        
        return results
    
    def test_batch_scaling_performance(self, sample_model: VisionModel):
        """Test how performance scales with batch size."""
        sample_model.eval()
        device = torch.device("cpu")
        sample_model.to(device)
        
        batch_sizes = [1, 2, 4, 8, 16]
        results = {}
        
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, 3, 224, 224).to(device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    _ = sample_model(input_tensor)
            
            # Benchmark
            start_time = time.time()
            with torch.no_grad():
                for _ in range(20):
                    output = sample_model(input_tensor)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 20
            throughput = batch_size / avg_time
            time_per_sample = avg_time / batch_size
            
            results[batch_size] = {
                "total_time_ms": avg_time * 1000,
                "time_per_sample_ms": time_per_sample * 1000,
                "throughput_samples_per_sec": throughput,
            }
            
            assert output.shape[0] == batch_size
        
        # Check that larger batches are more efficient per sample
        if len(results) >= 2:
            batch_1_time_per_sample = results[1]["time_per_sample_ms"]
            batch_max_time_per_sample = results[max(batch_sizes)]["time_per_sample_ms"]
            
            # Larger batches should be more efficient (lower time per sample)
            efficiency_ratio = batch_1_time_per_sample / batch_max_time_per_sample
            assert efficiency_ratio > 1.0  # Should be more efficient
        
        return results


class TestCompressionPerformance:
    """Test compression algorithm performance."""
    
    def test_quantization_performance(self, sample_model: VisionModel, sample_input: torch.Tensor):
        """Benchmark quantization performance and accuracy."""
        sample_model.eval()
        
        # Get original output
        with torch.no_grad():
            original_output = sample_model(sample_input)
        
        # Measure quantization time
        start_time = time.time()
        quantized_model = torch.quantization.quantize_dynamic(
            sample_model,
            qconfig_spec={torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8,
        )
        quantization_time = time.time() - start_time
        
        # Test quantized model
        with torch.no_grad():
            quantized_output = quantized_model(sample_input)
        
        # Calculate metrics
        output_diff = torch.abs(original_output - quantized_output)
        max_diff = output_diff.max().item()
        mean_diff = output_diff.mean().item()
        
        # Size comparison
        from bird_vision.utils.model_utils import get_model_size_mb
        original_size = get_model_size_mb(sample_model)
        quantized_size = get_model_size_mb(quantized_model)
        size_reduction = (original_size - quantized_size) / original_size * 100
        
        results = {
            "quantization_time_sec": quantization_time,
            "original_size_mb": original_size,
            "quantized_size_mb": quantized_size,
            "size_reduction_percent": size_reduction,
            "max_output_diff": max_diff,
            "mean_output_diff": mean_diff,
        }
        
        # Assertions
        assert quantization_time < 60  # Should complete within 1 minute
        assert size_reduction > 0  # Should reduce size
        assert max_diff < 1.0  # Output shouldn't change too much
        
        return results
    
    def test_pruning_performance(self, sample_model: VisionModel, sample_input: torch.Tensor):
        """Benchmark pruning performance."""
        import torch.nn.utils.prune as prune
        
        sample_model.eval()
        
        # Get original metrics
        with torch.no_grad():
            original_output = sample_model(sample_input)
        
        original_params = sum(p.numel() for p in sample_model.parameters())
        
        # Apply pruning
        start_time = time.time()
        
        # Find layers to prune
        modules_to_prune = []
        for name, module in sample_model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                modules_to_prune.append((module, 'weight'))
        
        # Apply global unstructured pruning
        if modules_to_prune:
            prune.global_unstructured(
                modules_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=0.3,
            )
            
            # Make pruning permanent
            for module, param_name in modules_to_prune:
                prune.remove(module, param_name)
        
        pruning_time = time.time() - start_time
        
        # Test pruned model
        with torch.no_grad():
            pruned_output = sample_model(sample_input)
        
        # Calculate sparsity
        pruned_params = sum(p.numel() for p in sample_model.parameters())
        zero_params = sum((p == 0).sum().item() for p in sample_model.parameters())
        sparsity = zero_params / pruned_params * 100
        
        # Output difference
        output_diff = torch.abs(original_output - pruned_output)
        max_diff = output_diff.max().item()
        mean_diff = output_diff.mean().item()
        
        results = {
            "pruning_time_sec": pruning_time,
            "original_parameters": original_params,
            "pruned_parameters": pruned_params,
            "sparsity_percent": sparsity,
            "max_output_diff": max_diff,
            "mean_output_diff": mean_diff,
        }
        
        # Assertions
        assert pruning_time < 30  # Should complete within 30 seconds
        assert sparsity > 0  # Should achieve some sparsity
        assert pruned_params == original_params  # Parameter count shouldn't change
        
        return results


class TestDeploymentPerformance:
    """Test deployment performance across platforms."""
    
    def test_torchscript_export_performance(self, sample_model: VisionModel, sample_input: torch.Tensor, temp_dir: Path):
        """Benchmark TorchScript export performance."""
        sample_model.eval()
        
        # Measure tracing time
        start_time = time.time()
        traced_model = torch.jit.trace(sample_model, sample_input)
        tracing_time = time.time() - start_time
        
        # Save model
        model_path = temp_dir / "traced_model.pt"
        start_save_time = time.time()
        traced_model.save(model_path)
        save_time = time.time() - start_save_time
        
        # Load model
        start_load_time = time.time()
        loaded_model = torch.jit.load(model_path)
        load_time = time.time() - start_load_time
        
        # Test performance
        with torch.no_grad():
            original_output = sample_model(sample_input)
            traced_output = traced_model(sample_input)
            loaded_output = loaded_model(sample_input)
        
        # File size
        file_size_mb = model_path.stat().st_size / 1024 / 1024
        
        results = {
            "tracing_time_sec": tracing_time,
            "save_time_sec": save_time,
            "load_time_sec": load_time,
            "file_size_mb": file_size_mb,
            "output_consistency": torch.allclose(original_output, traced_output, atol=1e-5),
            "load_consistency": torch.allclose(traced_output, loaded_output, atol=1e-5),
        }
        
        # Assertions
        assert tracing_time < 30  # Should trace within 30 seconds
        assert save_time < 10  # Should save within 10 seconds
        assert load_time < 5  # Should load within 5 seconds
        assert results["output_consistency"]
        assert results["load_consistency"]
        
        return results
    
    def test_onnx_export_performance(self, sample_model: VisionModel, sample_input: torch.Tensor, temp_dir: Path):
        """Benchmark ONNX export performance."""
        sample_model.eval()
        
        onnx_path = temp_dir / "model.onnx"
        
        # Measure export time
        start_time = time.time()
        torch.onnx.export(
            sample_model,
            sample_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            input_names=["input"],
            output_names=["output"],
        )
        export_time = time.time() - start_time
        
        # File size
        file_size_mb = onnx_path.stat().st_size / 1024 / 1024
        
        results = {
            "export_time_sec": export_time,
            "file_size_mb": file_size_mb,
            "export_success": onnx_path.exists(),
        }
        
        # Assertions
        assert export_time < 60  # Should export within 1 minute
        assert results["export_success"]
        assert file_size_mb > 0
        
        return results


class TestPlatformSpecificPerformance:
    """Test platform-specific performance requirements."""
    
    def test_mobile_performance_targets(self, sample_model: VisionModel, performance_thresholds: Dict):
        """Test mobile performance targets."""
        sample_model.eval()
        device = torch.device("cpu")
        sample_model.to(device)
        
        # Mobile input size
        mobile_input = torch.randn(1, 3, 224, 224).to(device)
        
        # Benchmark inference time
        warmup_runs = 10
        benchmark_runs = 50
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = sample_model(mobile_input)
        
        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(benchmark_runs):
                output = sample_model(mobile_input)
        end_time = time.time()
        
        avg_inference_time_ms = (end_time - start_time) / benchmark_runs * 1000
        
        # Model size
        from bird_vision.utils.model_utils import get_model_size_mb
        model_size_mb = get_model_size_mb(sample_model)
        
        # Get thresholds
        mobile_thresholds = performance_thresholds["mobile"]
        
        results = {
            "inference_time_ms": avg_inference_time_ms,
            "model_size_mb": model_size_mb,
            "meets_time_target": avg_inference_time_ms <= mobile_thresholds["max_inference_time_ms"],
            "meets_size_target": model_size_mb <= mobile_thresholds["max_model_size_mb"],
        }
        
        # Log results for visibility
        print(f"Mobile Performance Results: {results}")
        
        return results
    
    def test_esp32_performance_targets(self, sample_model: VisionModel, performance_thresholds: Dict):
        """Test ESP32-P4 performance targets."""
        sample_model.eval()
        device = torch.device("cpu")
        sample_model.to(device)
        
        # ESP32 input size
        esp32_input = torch.randn(1, 3, 224, 224).to(device)
        
        # Simulate ESP32 constraints (single threaded, limited computation)
        torch.set_num_threads(1)
        
        try:
            # Benchmark inference time
            warmup_runs = 5
            benchmark_runs = 20
            
            # Warmup
            with torch.no_grad():
                for _ in range(warmup_runs):
                    _ = sample_model(esp32_input)
            
            # Benchmark
            start_time = time.time()
            with torch.no_grad():
                for _ in range(benchmark_runs):
                    output = sample_model(esp32_input)
            end_time = time.time()
            
            avg_inference_time_ms = (end_time - start_time) / benchmark_runs * 1000
            
            # Model size
            from bird_vision.utils.model_utils import get_model_size_mb
            model_size_mb = get_model_size_mb(sample_model)
            
            # Get thresholds
            esp32_thresholds = performance_thresholds["esp32"]
            
            results = {
                "inference_time_ms": avg_inference_time_ms,
                "model_size_mb": model_size_mb,
                "meets_time_target": avg_inference_time_ms <= esp32_thresholds["max_inference_time_ms"],
                "meets_size_target": model_size_mb <= esp32_thresholds["max_model_size_mb"],
            }
            
            # Log results for visibility
            print(f"ESP32 Performance Results: {results}")
            
            return results
        
        finally:
            # Reset thread count
            torch.set_num_threads(0)


class TestStressTest:
    """Stress tests for system reliability."""
    
    def test_continuous_inference_stress(self, sample_model: VisionModel):
        """Test continuous inference over extended period."""
        sample_model.eval()
        device = torch.device("cpu")
        sample_model.to(device)
        
        input_tensor = torch.randn(1, 3, 224, 224).to(device)
        
        # Run continuous inference for specified duration
        duration_seconds = 30  # Reduced for testing
        start_time = time.time()
        inference_count = 0
        errors = 0
        
        while time.time() - start_time < duration_seconds:
            try:
                with torch.no_grad():
                    output = sample_model(input_tensor)
                    assert output.shape == (1, 10)
                    assert not torch.isnan(output).any()
                    assert not torch.isinf(output).any()
                inference_count += 1
            except Exception as e:
                errors += 1
                if errors > 5:  # Too many errors
                    raise e
        
        elapsed_time = time.time() - start_time
        avg_fps = inference_count / elapsed_time
        
        results = {
            "duration_sec": elapsed_time,
            "total_inferences": inference_count,
            "errors": errors,
            "avg_fps": avg_fps,
            "error_rate": errors / inference_count if inference_count > 0 else 0,
        }
        
        # Assertions
        assert errors == 0  # Should have no errors
        assert inference_count > 0  # Should complete some inferences
        assert avg_fps > 1  # Should achieve reasonable throughput
        
        return results
    
    def test_memory_leak_detection(self, sample_model: VisionModel):
        """Test for memory leaks during repeated inference."""
        sample_model.eval()
        device = torch.device("cpu")
        sample_model.to(device)
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run multiple inference cycles
        for cycle in range(10):
            input_tensor = torch.randn(4, 3, 224, 224).to(device)
            
            with torch.no_grad():
                for _ in range(10):
                    output = sample_model(input_tensor)
            
            # Clear references
            del input_tensor, output
            gc.collect()
            
            # Check memory every few cycles
            if cycle % 3 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_increase = current_memory - initial_memory
                
                # Memory shouldn't increase too much
                assert memory_increase < 100  # Less than 100MB increase
        
        final_memory = process.memory_info().rss / 1024 / 1024
        total_memory_increase = final_memory - initial_memory
        
        results = {
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "memory_increase_mb": total_memory_increase,
        }
        
        # Should not have significant memory leak
        assert total_memory_increase < 50  # Less than 50MB total increase
        
        return results