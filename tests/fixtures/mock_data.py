"""Mock data generators for testing."""

import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Tuple
import tempfile


class MockBirdDataGenerator:
    """Generate mock bird data for testing."""
    
    def __init__(self, num_classes: int = 10, samples_per_class: int = 20):
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.class_names = [f"Bird_Species_{i:02d}" for i in range(num_classes)]
    
    def generate_mock_images(self, output_dir: Path) -> Dict[str, Any]:
        """Generate mock bird images."""
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        image_paths = []
        labels = []
        
        for class_id in range(self.num_classes):
            class_dir = images_dir / f"class_{class_id:02d}"
            class_dir.mkdir(exist_ok=True)
            
            for sample_id in range(self.samples_per_class):
                # Generate realistic bird-like image
                image = self._generate_bird_like_image(class_id, sample_id)
                
                image_path = class_dir / f"sample_{sample_id:03d}.jpg"
                image.save(image_path, quality=95)
                
                image_paths.append(str(image_path.relative_to(output_dir)))
                labels.append(class_id)
        
        metadata = {
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "num_samples": len(image_paths),
            "samples_per_class": self.samples_per_class,
            "image_paths": image_paths,
            "labels": labels,
            "image_size": [224, 224],
            "format": "RGB",
        }
        
        # Save metadata
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata
    
    def _generate_bird_like_image(self, class_id: int, sample_id: int) -> Image.Image:
        """Generate a bird-like synthetic image."""
        # Set seed for reproducibility
        np.random.seed(class_id * 1000 + sample_id)
        
        # Create base image
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Generate background (sky-like)
        sky_blue = [135 + np.random.randint(-20, 20) for _ in range(3)]
        image[:, :] = sky_blue
        
        # Add some texture/noise
        noise = np.random.randint(-30, 30, (224, 224, 3))
        image = np.clip(image.astype(int) + noise, 0, 255).astype(np.uint8)
        
        # Add bird-like shape (simple ellipse)
        center_x, center_y = 112 + np.random.randint(-30, 30), 112 + np.random.randint(-30, 30)
        
        # Bird body colors (class-dependent)
        bird_colors = [
            [139, 69, 19],   # Brown
            [255, 0, 0],     # Red
            [0, 0, 255],     # Blue
            [255, 255, 0],   # Yellow
            [0, 255, 0],     # Green
            [255, 165, 0],   # Orange
            [128, 0, 128],   # Purple
            [255, 192, 203], # Pink
            [0, 255, 255],   # Cyan
            [255, 20, 147],  # Deep Pink
        ]
        
        bird_color = bird_colors[class_id % len(bird_colors)]
        bird_color = [c + np.random.randint(-30, 30) for c in bird_color]
        bird_color = [max(0, min(255, c)) for c in bird_color]
        
        # Draw bird body (ellipse)
        y, x = np.ogrid[:224, :224]
        body_mask = ((x - center_x) / 40) ** 2 + ((y - center_y) / 25) ** 2 <= 1
        
        # Draw bird head (smaller circle)
        head_x, head_y = center_x - 20, center_y - 20
        head_mask = ((x - head_x) / 15) ** 2 + ((y - head_y) / 15) ** 2 <= 1
        
        # Draw wing (elongated ellipse)
        wing_x, wing_y = center_x + 15, center_y
        wing_mask = ((x - wing_x) / 30) ** 2 + ((y - wing_y) / 10) ** 2 <= 1
        
        # Apply bird parts
        combined_mask = body_mask | head_mask | wing_mask
        image[combined_mask] = bird_color
        
        # Add some random features (spots, stripes) based on class
        if class_id % 3 == 0:  # Add spots
            for _ in range(np.random.randint(3, 8)):
                spot_x = center_x + np.random.randint(-30, 30)
                spot_y = center_y + np.random.randint(-20, 20)
                spot_mask = ((x - spot_x) / 3) ** 2 + ((y - spot_y) / 3) ** 2 <= 1
                if np.any(spot_mask):
                    darker_color = [max(0, c - 50) for c in bird_color]
                    image[spot_mask] = darker_color
        
        return Image.fromarray(image)
    
    def generate_mock_dataset_splits(self, metadata: Dict[str, Any]) -> Dict[str, List[int]]:
        """Generate train/val/test splits."""
        total_samples = metadata["num_samples"]
        indices = list(range(total_samples))
        
        # Stratified split to ensure each class is represented
        splits = {"train": [], "val": [], "test": []}
        
        for class_id in range(self.num_classes):
            class_start = class_id * self.samples_per_class
            class_indices = indices[class_start:class_start + self.samples_per_class]
            
            # 70% train, 20% val, 10% test
            train_end = int(0.7 * len(class_indices))
            val_end = int(0.9 * len(class_indices))
            
            splits["train"].extend(class_indices[:train_end])
            splits["val"].extend(class_indices[train_end:val_end])
            splits["test"].extend(class_indices[val_end:])
        
        return splits


class MockModelStateGenerator:
    """Generate mock model states for testing."""
    
    @staticmethod
    def generate_mock_checkpoint(
        model_config: Dict[str, Any],
        epoch: int = 10,
        val_accuracy: float = 0.85
    ) -> Dict[str, Any]:
        """Generate a mock model checkpoint."""
        # Generate fake state dict
        state_dict = {}
        
        # Add some realistic layer names and shapes
        layer_configs = [
            ("backbone.features.0.0.weight", (32, 3, 3, 3)),
            ("backbone.features.0.1.weight", (32,)),
            ("backbone.features.0.1.bias", (32,)),
            ("head.0.weight", (512, 1280)),
            ("head.0.bias", (512,)),
            ("head.2.weight", (model_config.get("num_classes", 10), 512)),
            ("head.2.bias", (model_config.get("num_classes", 10),)),
        ]
        
        for name, shape in layer_configs:
            state_dict[name] = torch.randn(shape)
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": state_dict,
            "optimizer_state_dict": {
                "state": {},
                "param_groups": [{"lr": 0.001, "weight_decay": 0.01}],
            },
            "scheduler_state_dict": {"last_epoch": epoch},
            "best_val_metric": val_accuracy,
            "val_metrics": {
                "accuracy": val_accuracy,
                "f1_macro": val_accuracy - 0.05,
                "loss": 0.5,
            },
            "train_metrics": {
                "accuracy": val_accuracy + 0.05,
                "f1_macro": val_accuracy,
                "loss": 0.3,
            },
            "config": model_config,
        }
        
        return checkpoint
    
    @staticmethod
    def generate_mock_model_comparison_data() -> List[Dict[str, Any]]:
        """Generate mock model comparison data."""
        models = [
            {
                "model_name": "baseline_efficientnet",
                "overall_metrics": {
                    "accuracy": 0.82,
                    "f1_macro": 0.80,
                    "precision_macro": 0.81,
                    "recall_macro": 0.82,
                },
                "detailed_metrics": {
                    "num_classes": 10,
                    "total_samples": 200,
                    "top_k_accuracies": {"top_1_accuracy": 0.82, "top_5_accuracy": 0.95},
                },
            },
            {
                "model_name": "optimized_efficientnet",
                "overall_metrics": {
                    "accuracy": 0.87,
                    "f1_macro": 0.85,
                    "precision_macro": 0.86,
                    "recall_macro": 0.87,
                },
                "detailed_metrics": {
                    "num_classes": 10,
                    "total_samples": 200,
                    "top_k_accuracies": {"top_1_accuracy": 0.87, "top_5_accuracy": 0.97},
                },
            },
        ]
        
        return models


class MockDeploymentDataGenerator:
    """Generate mock deployment data for testing."""
    
    @staticmethod
    def generate_mock_compression_results() -> Dict[str, Any]:
        """Generate mock compression results."""
        return {
            "compressed_models": {
                "quantized": "mock_quantized_model",
                "pruned": "mock_pruned_model",
            },
            "compression_stats": {
                "original": {"size_mb": 25.0, "parameters": 5000000},
                "quantized": {"size_mb": 6.25, "parameters": 5000000},
                "pruned": {"size_mb": 17.5, "parameters": 3500000},
            },
            "export_results": {
                "original": {
                    "onnx": {"success": True, "path": "/mock/path/model.onnx", "size_mb": 25.0},
                    "torchscript": {"success": True, "path": "/mock/path/model.pt", "size_mb": 25.0},
                },
                "quantized": {
                    "onnx": {"success": True, "path": "/mock/path/model_quantized.onnx", "size_mb": 6.25},
                    "torchscript": {"success": True, "path": "/mock/path/model_quantized.pt", "size_mb": 6.25},
                },
            },
            "compression_summary": {
                "original_size_mb": 25.0,
                "original_params": 5000000,
                "quantized_size_reduction_percent": 75.0,
                "quantized_final_size_mb": 6.25,
                "pruned_size_reduction_percent": 30.0,
                "pruned_param_reduction_percent": 30.0,
                "pruned_final_size_mb": 17.5,
            },
        }
    
    @staticmethod
    def generate_mock_mobile_deployment_results() -> Dict[str, Any]:
        """Generate mock mobile deployment results."""
        return {
            "compression_results": MockDeploymentDataGenerator.generate_mock_compression_results(),
            "deployment_results": {
                "ios": {
                    "success": True,
                    "model_path": "/mock/path/ios/BirdClassifier.mlmodel",
                    "size_mb": 6.25,
                    "format": "coreml",
                    "integration_files": ["/mock/path/ios/BirdClassifierClassifier.swift"],
                },
                "android": {
                    "success": True,
                    "model_path": "/mock/path/android/bird_classifier.pt",
                    "size_mb": 6.25,
                    "format": "torchscript",
                    "integration_files": ["/mock/path/android/BirdClassifierClassifier.java"],
                },
            },
            "metadata": {
                "model_info": {
                    "name": "bird_classifier",
                    "version": "1.0.0",
                    "description": "Mobile bird species classifier",
                },
                "model_stats": {
                    "size_mb": 6.25,
                    "parameters": 5000000,
                    "avg_inference_time_ms": 45.2,
                },
                "preprocessing": {
                    "input_size": [224, 224],
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                    "format": "RGB",
                },
                "deployment_platforms": ["ios", "android"],
            },
            "packages": {
                "ios": "/mock/path/bird_classifier_ios_package",
                "android": "/mock/path/bird_classifier_android_package",
            },
        }
    
    @staticmethod
    def generate_mock_esp32_deployment_results() -> Dict[str, Any]:
        """Generate mock ESP32 deployment results."""
        return {
            "esp_dl_model": {
                "success": True,
                "onnx_path": "/mock/path/esp32/bird_classifier.onnx",
                "esp_dl_files": {
                    "coefficients": "/mock/path/esp32/bird_classifier_coefficients.hpp",
                    "header": "/mock/path/esp32/bird_classifier_model.hpp",
                    "config": "/mock/path/esp32/bird_classifier_config.json",
                },
                "quantization_info": {
                    "quantization_scheme": "per_tensor_symmetric",
                    "dtype": "int8",
                    "ai_accelerator_compatible": True,
                },
            },
            "firmware": {
                "success": True,
                "firmware_dir": "/mock/path/esp32/firmware",
                "files": {
                    "main": {
                        "main_cpp": "/mock/path/esp32/firmware/main/main.cpp",
                        "camera_header": "/mock/path/esp32/firmware/main/camera_interface.h",
                    },
                    "cmake": {
                        "main": "/mock/path/esp32/firmware/CMakeLists.txt",
                        "component": "/mock/path/esp32/firmware/main/CMakeLists.txt",
                    },
                    "scripts": {
                        "build": "/mock/path/esp32/firmware/build.sh",
                        "flash": "/mock/path/esp32/firmware/flash.sh",
                    },
                },
            },
            "package": {
                "success": True,
                "package_dir": "/mock/path/esp32/bird_classifier_esp32_package",
                "readme": "/mock/path/esp32/bird_classifier_esp32_package/README.md",
                "deployment_info": {
                    "model_name": "bird_classifier_esp32",
                    "target_device": "ESP32-P4-EYE",
                    "esp_idf_version": "v5.1",
                    "model_format": "ESP-DL",
                },
            },
            "deployment_info": {
                "target_device": "ESP32-P4-EYE",
                "model_stats": {
                    "size_mb": 4.2,
                    "parameters": 3500000,
                    "avg_inference_time_ms": 156.8,
                },
                "constraints": {
                    "max_size_mb": 8,
                    "max_inference_ms": 200,
                    "min_accuracy": 0.90,
                },
                "optimization_features": {
                    "quantization": "int8",
                    "ai_accelerator": True,
                    "memory_optimization": True,
                    "layer_fusion": True,
                },
                "camera_specs": {
                    "resolution": [640, 480],
                    "format": "RGB565",
                    "fps": 15,
                },
            },
        }


def create_temporary_mock_dataset(num_classes: int = 5, samples_per_class: int = 10) -> Tuple[Path, Dict[str, Any]]:
    """Create a temporary mock dataset for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    
    generator = MockBirdDataGenerator(num_classes, samples_per_class)
    metadata = generator.generate_mock_images(temp_dir)
    splits = generator.generate_mock_dataset_splits(metadata)
    
    # Save splits
    splits_file = temp_dir / "splits.json"
    with open(splits_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    return temp_dir, metadata