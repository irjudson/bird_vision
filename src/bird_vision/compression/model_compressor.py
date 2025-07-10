"""Model compression and optimization utilities."""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic, prepare_qat, convert
import torch.nn.utils.prune as prune
import onnx
from onnxruntime.quantization import quantize_dynamic as onnx_quantize_dynamic
from onnxruntime.quantization.quantize import QuantType
import coremltools as ct
import tensorflow as tf
from omegaconf import DictConfig
from loguru import logger

from bird_vision.utils.model_utils import ModelProfiler


class ModelCompressor:
    """Model compression and optimization for mobile deployment."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.compression_cfg = cfg.compression
        self.output_dir = Path(cfg.paths.models_dir) / "compressed"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.profiler = ModelProfiler()
    
    def compress_model(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        model_name: str = "model",
    ) -> Dict[str, Any]:
        """Apply all configured compression techniques."""
        logger.info(f"Starting model compression for {model_name}")
        
        compressed_models = {}
        compression_stats = {}
        
        # Original model stats
        original_stats = self.profiler.profile_model(model, sample_input)
        compression_stats["original"] = original_stats
        
        # Apply quantization
        if self.compression_cfg.quantization.enabled:
            quantized_model, quant_stats = self._apply_quantization(
                model, sample_input, model_name
            )
            compressed_models["quantized"] = quantized_model
            compression_stats["quantized"] = quant_stats
        
        # Apply pruning
        if self.compression_cfg.pruning.enabled:
            pruned_model, prune_stats = self._apply_pruning(
                model, sample_input, model_name
            )
            compressed_models["pruned"] = pruned_model
            compression_stats["pruned"] = prune_stats
        
        # Knowledge distillation (if teacher model provided)
        if self.compression_cfg.distillation.enabled:
            logger.info("Knowledge distillation requires separate training - skipping in compression step")
        
        # Export to different formats
        export_results = self._export_models(
            compressed_models if compressed_models else {"original": model},
            sample_input,
            model_name,
        )
        
        results = {
            "compressed_models": compressed_models,
            "compression_stats": compression_stats,
            "export_results": export_results,
            "compression_summary": self._generate_compression_summary(compression_stats),
        }
        
        logger.info("Model compression completed")
        return results
    
    def _apply_quantization(
        self, model: nn.Module, sample_input: torch.Tensor, model_name: str
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply dynamic quantization to the model."""
        logger.info("Applying quantization...")
        
        # Prepare model for quantization
        model_copy = type(model)(model.cfg).load_state_dict(model.state_dict())
        model_copy.eval()
        
        # Dynamic quantization (post-training)
        quantized_model = quantize_dynamic(
            model_copy,
            qconfig_spec={nn.Linear, nn.Conv2d},
            dtype=torch.qint8,
        )
        
        # Profile quantized model
        quant_stats = self.profiler.profile_model(quantized_model, sample_input)
        
        # Save quantized model
        quant_path = self.output_dir / f"{model_name}_quantized.pth"
        torch.save(quantized_model.state_dict(), quant_path)
        
        logger.info(f"Quantization completed. Size reduction: {quant_stats['size_mb']:.2f}MB")
        return quantized_model, quant_stats
    
    def _apply_pruning(
        self, model: nn.Module, sample_input: torch.Tensor, model_name: str
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply pruning to the model."""
        logger.info("Applying pruning...")
        
        # Create copy of model
        model_copy = type(model)(model.cfg)
        model_copy.load_state_dict(model.state_dict())
        model_copy.eval()
        
        # Apply pruning to linear and conv layers
        sparsity = self.compression_cfg.pruning.sparsity
        structured = self.compression_cfg.pruning.structured
        
        modules_to_prune = []
        for name, module in model_copy.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                modules_to_prune.append((module, 'weight'))
        
        if structured:
            # Structured pruning
            for module, param_name in modules_to_prune:
                prune.ln_structured(module, name=param_name, amount=sparsity, n=2, dim=0)
        else:
            # Unstructured pruning
            prune.global_unstructured(
                modules_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=sparsity,
            )
        
        # Remove pruning reparameterization to make pruning permanent
        for module, param_name in modules_to_prune:
            prune.remove(module, param_name)
        
        # Profile pruned model
        prune_stats = self.profiler.profile_model(model_copy, sample_input)
        
        # Save pruned model
        prune_path = self.output_dir / f"{model_name}_pruned.pth"
        torch.save(model_copy.state_dict(), prune_path)
        
        logger.info(f"Pruning completed. Sparsity: {sparsity}, Size: {prune_stats['size_mb']:.2f}MB")
        return model_copy, prune_stats
    
    def _export_models(
        self,
        models: Dict[str, nn.Module],
        sample_input: torch.Tensor,
        model_name: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Export models to various formats."""
        export_results = {}
        
        for variant_name, model in models.items():
            variant_results = {}
            
            # ONNX Export
            if self.compression_cfg.onnx_export.enabled:
                onnx_result = self._export_to_onnx(model, sample_input, model_name, variant_name)
                variant_results["onnx"] = onnx_result
            
            # Mobile formats
            if self.compression_cfg.mobile_export.torchscript:
                ts_result = self._export_to_torchscript(model, sample_input, model_name, variant_name)
                variant_results["torchscript"] = ts_result
            
            if self.compression_cfg.mobile_export.coreml:
                coreml_result = self._export_to_coreml(model, sample_input, model_name, variant_name)
                variant_results["coreml"] = coreml_result
            
            if self.compression_cfg.mobile_export.tflite:
                tflite_result = self._export_to_tflite(model, sample_input, model_name, variant_name)
                variant_results["tflite"] = tflite_result
            
            export_results[variant_name] = variant_results
        
        return export_results
    
    def _export_to_onnx(
        self, model: nn.Module, sample_input: torch.Tensor, model_name: str, variant: str
    ) -> Dict[str, Any]:
        """Export model to ONNX format."""
        try:
            onnx_path = self.output_dir / f"{model_name}_{variant}.onnx"
            
            # Export to ONNX
            torch.onnx.export(
                model,
                sample_input,
                onnx_path,
                export_params=True,
                opset_version=self.compression_cfg.onnx_export.opset_version,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=self.compression_cfg.onnx_export.dynamic_axes,
            )
            
            # Verify ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            # Quantize ONNX model if requested
            if self.compression_cfg.quantization.enabled:
                quantized_onnx_path = self.output_dir / f"{model_name}_{variant}_quantized.onnx"
                onnx_quantize_dynamic(
                    str(onnx_path),
                    str(quantized_onnx_path),
                    weight_type=QuantType.QUInt8,
                )
            
            file_size = onnx_path.stat().st_size / (1024 * 1024)  # MB
            
            return {
                "success": True,
                "path": str(onnx_path),
                "size_mb": file_size,
                "format": "onnx",
            }
        
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _export_to_torchscript(
        self, model: nn.Module, sample_input: torch.Tensor, model_name: str, variant: str
    ) -> Dict[str, Any]:
        """Export model to TorchScript format."""
        try:
            ts_path = self.output_dir / f"{model_name}_{variant}.pt"
            
            # Trace the model
            traced_model = torch.jit.trace(model, sample_input)
            traced_model.save(ts_path)
            
            file_size = ts_path.stat().st_size / (1024 * 1024)  # MB
            
            return {
                "success": True,
                "path": str(ts_path),
                "size_mb": file_size,
                "format": "torchscript",
            }
        
        except Exception as e:
            logger.error(f"TorchScript export failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _export_to_coreml(
        self, model: nn.Module, sample_input: torch.Tensor, model_name: str, variant: str
    ) -> Dict[str, Any]:
        """Export model to CoreML format."""
        try:
            coreml_path = self.output_dir / f"{model_name}_{variant}.mlmodel"
            
            # Trace the model first
            traced_model = torch.jit.trace(model, sample_input)
            
            # Convert to CoreML
            coreml_model = ct.convert(
                traced_model,
                inputs=[ct.ImageType(shape=sample_input.shape)],
                classifier_config=ct.ClassifierConfig(
                    class_labels=[f"class_{i}" for i in range(400)]  # NABirds classes
                ),
            )
            
            coreml_model.save(coreml_path)
            
            file_size = coreml_path.stat().st_size / (1024 * 1024)  # MB
            
            return {
                "success": True,
                "path": str(coreml_path),
                "size_mb": file_size,
                "format": "coreml",
            }
        
        except Exception as e:
            logger.error(f"CoreML export failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _export_to_tflite(
        self, model: nn.Module, sample_input: torch.Tensor, model_name: str, variant: str
    ) -> Dict[str, Any]:
        """Export model to TensorFlow Lite format."""
        try:
            tflite_path = self.output_dir / f"{model_name}_{variant}.tflite"
            
            # First convert to ONNX, then to TFLite via TensorFlow
            onnx_path = self.output_dir / f"temp_{model_name}_{variant}.onnx"
            
            # Export to ONNX first
            torch.onnx.export(
                model, sample_input, onnx_path,
                export_params=True, opset_version=11,
                input_names=["input"], output_names=["output"]
            )
            
            # Note: Direct ONNX to TFLite conversion requires additional tools
            # This is a placeholder for the actual conversion process
            logger.warning("TFLite export requires onnx-tf converter - placeholder implementation")
            
            return {
                "success": False,
                "error": "TFLite export requires additional conversion tools",
                "note": "Use onnx-tf converter for full implementation",
            }
        
        except Exception as e:
            logger.error(f"TFLite export failed: {e}")
            return {"success": False, "error": str(e)}
        
        finally:
            # Cleanup temporary ONNX file
            if 'onnx_path' in locals() and Path(onnx_path).exists():
                os.remove(onnx_path)
    
    def _export_to_esp_dl(
        self, model: nn.Module, sample_input: torch.Tensor, model_name: str, variant: str
    ) -> Dict[str, Any]:
        """Export model to ESP-DL format for ESP32-P4."""
        try:
            from bird_vision.deployment.esp32_deployer import ESP32Deployer
            
            # Create ESP32 deployer with current config
            esp32_config = self.cfg.copy()
            esp32_config.deployment.target_platform = "esp32_p4_eye"
            
            esp32_deployer = ESP32Deployer(esp32_config)
            
            # Convert to ESP-DL format
            esp_dl_result = esp32_deployer._convert_to_esp_dl(
                model, sample_input, f"{model_name}_{variant}", self.output_dir
            )
            
            if esp_dl_result.get("success"):
                return {
                    "success": True,
                    "esp_dl_files": esp_dl_result["esp_dl_files"],
                    "format": "esp_dl",
                    "note": "ESP-DL format for ESP32-P4 AI acceleration",
                }
            else:
                return esp_dl_result
        
        except Exception as e:
            logger.error(f"ESP-DL export failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_compression_summary(
        self, compression_stats: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate compression summary statistics."""
        if "original" not in compression_stats:
            return {}
        
        original = compression_stats["original"]
        summary = {
            "original_size_mb": original["size_mb"],
            "original_params": original["parameters"],
        }
        
        for variant, stats in compression_stats.items():
            if variant == "original":
                continue
            
            size_reduction = (original["size_mb"] - stats["size_mb"]) / original["size_mb"] * 100
            param_reduction = (original["parameters"] - stats["parameters"]) / original["parameters"] * 100
            
            summary[f"{variant}_size_reduction_percent"] = size_reduction
            summary[f"{variant}_param_reduction_percent"] = param_reduction
            summary[f"{variant}_final_size_mb"] = stats["size_mb"]
        
        return summary