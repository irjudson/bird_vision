"""Model utility functions."""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from thop import profile, clever_format


class ModelProfiler:
    """Profile model performance and statistics."""
    
    def profile_model(self, model: nn.Module, sample_input: torch.Tensor) -> Dict[str, Any]:
        """Profile model with comprehensive statistics."""
        model.eval()
        
        # Basic model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Model size in MB (assuming float32)
        model_size_mb = total_params * 4 / (1024 * 1024)
        
        # FLOPs calculation
        try:
            macs, params = profile(model, inputs=(sample_input,), verbose=False)
            flops = 2 * macs  # Multiply-accumulate operations -> FLOPs
            flops_str, params_str = clever_format([flops, params], "%.3f")
        except Exception:
            flops = 0
            flops_str = "N/A"
            params_str = f"{total_params:.3f}"
        
        # Inference time (rough estimate)
        inference_times = []
        model.eval()
        with torch.no_grad():
            # Warm up
            for _ in range(10):
                _ = model(sample_input)
            
            # Measure
            import time
            for _ in range(100):
                start_time = time.time()
                _ = model(sample_input)
                inference_times.append((time.time() - start_time) * 1000)  # ms
        
        avg_inference_time = sum(inference_times) / len(inference_times)
        
        return {
            "parameters": total_params,
            "trainable_parameters": trainable_params,
            "size_mb": model_size_mb,
            "flops": flops,
            "flops_str": flops_str,
            "params_str": params_str,
            "avg_inference_time_ms": avg_inference_time,
            "input_shape": list(sample_input.shape),
        }


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in megabytes."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    return (param_size + buffer_size) / (1024 * 1024)


def freeze_layers(model: nn.Module, layer_names: list) -> None:
    """Freeze specified layers."""
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = False


def unfreeze_layers(model: nn.Module, layer_names: list) -> None:
    """Unfreeze specified layers."""
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = True