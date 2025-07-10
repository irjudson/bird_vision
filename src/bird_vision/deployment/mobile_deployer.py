"""Mobile deployment utilities for iOS and Android."""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
import coremltools as ct
from omegaconf import DictConfig
from loguru import logger

from bird_vision.compression.model_compressor import ModelCompressor
from bird_vision.utils.model_utils import ModelProfiler


class MobileDeployer:
    """Mobile deployment utility for iOS and Android platforms."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.deployment_cfg = cfg.deployment
        self.output_dir = Path(cfg.paths.models_dir) / "mobile_deployment"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.compressor = ModelCompressor(cfg)
        self.profiler = ModelProfiler()
    
    def prepare_for_mobile(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        model_name: str = "bird_classifier",
        class_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Prepare model for mobile deployment."""
        logger.info(f"Preparing {model_name} for mobile deployment")
        
        deployment_results = {}
        
        # Compress model first
        compression_results = self.compressor.compress_model(model, sample_input, model_name)
        
        # Select best compressed model based on target platform requirements
        selected_model = self._select_optimal_model(compression_results, model)
        
        # Create iOS deployment package
        if "ios" in self.deployment_cfg.target_platform or self.deployment_cfg.target_platform == "mobile":
            ios_result = self._prepare_ios_deployment(
                selected_model, sample_input, model_name, class_labels
            )
            deployment_results["ios"] = ios_result
        
        # Create Android deployment package
        if "android" in self.deployment_cfg.target_platform or self.deployment_cfg.target_platform == "mobile":
            android_result = self._prepare_android_deployment(
                selected_model, sample_input, model_name, class_labels
            )
            deployment_results["android"] = android_result
        
        # Create ESP32 deployment package
        if "esp32" in self.deployment_cfg.target_platform or self.deployment_cfg.target_platform == "esp32_p4_eye":
            esp32_result = self._prepare_esp32_deployment(
                selected_model, sample_input, model_name, class_labels
            )
            deployment_results["esp32"] = esp32_result
        
        # Generate deployment metadata
        metadata = self._generate_deployment_metadata(
            selected_model, sample_input, model_name, deployment_results
        )
        
        # Create deployment packages
        packages = self._create_deployment_packages(deployment_results, metadata, model_name)
        
        return {
            "compression_results": compression_results,
            "deployment_results": deployment_results,
            "metadata": metadata,
            "packages": packages,
        }
    
    def _select_optimal_model(
        self, compression_results: Dict[str, Any], original_model: nn.Module
    ) -> nn.Module:
        """Select the best compressed model based on target requirements."""
        compressed_models = compression_results.get("compressed_models", {})
        compression_stats = compression_results.get("compression_stats", {})
        
        # Check if any compressed model meets size requirements
        max_size_mb = self.cfg.compression.performance_targets.max_model_size_mb
        min_accuracy = self.cfg.compression.performance_targets.min_accuracy_retention
        
        # For now, prefer quantized model if available and within size limits
        if "quantized" in compressed_models:
            quant_stats = compression_stats.get("quantized", {})
            if quant_stats.get("size_mb", float('inf')) <= max_size_mb:
                logger.info("Selected quantized model for mobile deployment")
                return compressed_models["quantized"]
        
        if "pruned" in compressed_models:
            prune_stats = compression_stats.get("pruned", {})
            if prune_stats.get("size_mb", float('inf')) <= max_size_mb:
                logger.info("Selected pruned model for mobile deployment")
                return compressed_models["pruned"]
        
        logger.info("Using original model for mobile deployment")
        return original_model
    
    def _prepare_ios_deployment(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        model_name: str,
        class_labels: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Prepare model for iOS deployment using CoreML."""
        logger.info("Preparing iOS deployment package")
        
        ios_dir = self.output_dir / "ios"
        ios_dir.mkdir(exist_ok=True)
        
        try:
            # Trace the model
            model.eval()
            traced_model = torch.jit.trace(model, sample_input)
            
            # Convert to CoreML
            if class_labels is None:
                class_labels = [f"bird_species_{i}" for i in range(400)]  # NABirds default
            
            # Create CoreML model with proper configuration
            input_shape = sample_input.shape
            coreml_model = ct.convert(
                traced_model,
                inputs=[ct.ImageType(
                    name="image",
                    shape=input_shape,
                    bias=self.deployment_cfg.preprocessing.mean,
                    scale=1.0 / (torch.tensor(self.deployment_cfg.preprocessing.std) * 255.0),
                )],
                outputs=[ct.TensorType(name="probabilities")],
                classifier_config=ct.ClassifierConfig(class_labels),
                minimum_deployment_target=ct.target.iOS(self.deployment_cfg.ios.deployment_target),
            )
            
            # Add metadata
            coreml_model.short_description = self.deployment_cfg.metadata.description
            coreml_model.author = self.deployment_cfg.metadata.author
            coreml_model.license = self.deployment_cfg.metadata.license
            coreml_model.version = self.deployment_cfg.metadata.model_version
            
            # Save CoreML model
            coreml_path = ios_dir / f"{model_name}.mlmodel"
            coreml_model.save(coreml_path)
            
            # Create iOS integration code
            self._create_ios_integration_code(ios_dir, model_name, class_labels)
            
            file_size = coreml_path.stat().st_size / (1024 * 1024)
            
            return {
                "success": True,
                "model_path": str(coreml_path),
                "size_mb": file_size,
                "format": "coreml",
                "integration_files": list(ios_dir.glob("*.swift")),
            }
        
        except Exception as e:
            logger.error(f"iOS deployment preparation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _prepare_android_deployment(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        model_name: str,
        class_labels: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Prepare model for Android deployment."""
        logger.info("Preparing Android deployment package")
        
        android_dir = self.output_dir / "android"
        android_dir.mkdir(exist_ok=True)
        
        try:
            # For Android, we'll use TorchScript as it's more straightforward
            model.eval()
            traced_model = torch.jit.trace(model, sample_input)
            
            # Optimize for mobile
            optimized_model = torch.jit.optimize_for_inference(traced_model)
            
            # Save TorchScript model
            torchscript_path = android_dir / f"{model_name}.pt"
            optimized_model.save(torchscript_path)
            
            # Create Android integration code
            self._create_android_integration_code(android_dir, model_name, class_labels)
            
            file_size = torchscript_path.stat().st_size / (1024 * 1024)
            
            return {
                "success": True,
                "model_path": str(torchscript_path),
                "size_mb": file_size,
                "format": "torchscript",
                "integration_files": list(android_dir.glob("*.java")),
            }
        
        except Exception as e:
            logger.error(f"Android deployment preparation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _prepare_esp32_deployment(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        model_name: str,
        class_labels: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Prepare model for ESP32-P4-Eye deployment."""
        logger.info("Preparing ESP32-P4-Eye deployment package")
        
        try:
            from bird_vision.deployment.esp32_deployer import ESP32Deployer
            
            # Create ESP32-specific config
            esp32_config = self.cfg.copy()
            esp32_config.deployment.target_platform = "esp32_p4_eye"
            
            # Use ESP32 deployer
            esp32_deployer = ESP32Deployer(esp32_config)
            esp32_result = esp32_deployer.prepare_for_esp32(
                model, sample_input, model_name, class_labels
            )
            
            if esp32_result.get("esp_dl_model", {}).get("success"):
                return {
                    "success": True,
                    "deployment_result": esp32_result,
                    "format": "esp_dl",
                    "package_path": esp32_result.get("package", {}).get("package_dir"),
                    "size_info": esp32_result.get("deployment_info", {}),
                }
            else:
                return {
                    "success": False,
                    "error": "ESP32 deployment failed",
                    "details": esp32_result,
                }
        
        except Exception as e:
            logger.error(f"ESP32 deployment preparation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _create_ios_integration_code(
        self, ios_dir: Path, model_name: str, class_labels: List[str]
    ) -> None:
        """Create iOS Swift integration code."""
        swift_code = f'''
import CoreML
import UIKit
import Vision

class {model_name.title()}Classifier {{
    
    private var model: VNCoreMLModel?
    private let classLabels = {json.dumps(class_labels, indent=8)}
    
    init() {{
        setupModel()
    }}
    
    private func setupModel() {{
        guard let modelURL = Bundle.main.url(forResource: "{model_name}", withExtension: "mlmodel"),
              let coreMLModel = try? MLModel(contentsOf: modelURL),
              let visionModel = try? VNCoreMLModel(for: coreMLModel) else {{
            print("Failed to load Core ML model")
            return
        }}
        self.model = visionModel
    }}
    
    func classifyImage(_ image: UIImage, completion: @escaping (String?, Float?) -> Void) {{
        guard let model = model,
              let ciImage = CIImage(image: image) else {{
            completion(nil, nil)
            return
        }}
        
        let request = VNCoreMLRequest(model: model) {{ request, error in
            guard let results = request.results as? [VNClassificationObservation],
                  let topResult = results.first else {{
                completion(nil, nil)
                return
            }}
            
            completion(topResult.identifier, topResult.confidence)
        }}
        
        let handler = VNImageRequestHandler(ciImage: ciImage)
        try? handler.perform([request])
    }}
    
    func getTopPredictions(_ image: UIImage, topK: Int = 5, completion: @escaping ([(String, Float)]) -> Void) {{
        guard let model = model,
              let ciImage = CIImage(image: image) else {{
            completion([])
            return
        }}
        
        let request = VNCoreMLRequest(model: model) {{ request, error in
            guard let results = request.results as? [VNClassificationObservation] else {{
                completion([])
                return
            }}
            
            let topResults = Array(results.prefix(topK)).map {{ result in
                (result.identifier, result.confidence)
            }}
            completion(topResults)
        }}
        
        let handler = VNImageRequestHandler(ciImage: ciImage)
        try? handler.perform([request])
    }}
}}
'''
        
        swift_file = ios_dir / f"{model_name.title()}Classifier.swift"
        swift_file.write_text(swift_code)
    
    def _create_android_integration_code(
        self, android_dir: Path, model_name: str, class_labels: List[str]
    ) -> None:
        """Create Android Java integration code."""
        java_code = f'''
package com.example.birdvision;

import android.content.Context;
import android.graphics.Bitmap;
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class {model_name.title()}Classifier {{
    
    private Module module;
    private String[] classLabels = {{{", ".join(f'"{label}"' for label in class_labels)}}};
    
    // Preprocessing constants
    private static final float[] NORM_MEAN = {{0.485f, 0.456f, 0.406f}};
    private static final float[] NORM_STD = {{0.229f, 0.224f, 0.225f}};
    
    public {model_name.title()}Classifier(Context context) {{
        try {{
            module = Module.load(assetFilePath(context, "{model_name}.pt"));
        }} catch (IOException e) {{
            throw new RuntimeException("Error loading model", e);
        }}
    }}
    
    public String classifyImage(Bitmap bitmap) {{
        Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
            bitmap, NORM_MEAN, NORM_STD
        );
        
        IValue inputIValue = IValue.from(inputTensor);
        Tensor outputTensor = module.forward(inputIValue).toTensor();
        
        float[] scores = outputTensor.getDataAsFloatArray();
        int maxIndex = argmax(scores);
        
        return classLabels[maxIndex];
    }}
    
    public List<ClassificationResult> getTopPredictions(Bitmap bitmap, int topK) {{
        Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
            bitmap, NORM_MEAN, NORM_STD
        );
        
        IValue inputIValue = IValue.from(inputTensor);
        Tensor outputTensor = module.forward(inputIValue).toTensor();
        
        float[] scores = outputTensor.getDataAsFloatArray();
        
        List<ClassificationResult> results = new ArrayList<>();
        int[] topIndices = argtopk(scores, topK);
        
        for (int i = 0; i < topIndices.length; i++) {{
            int index = topIndices[i];
            results.add(new ClassificationResult(
                classLabels[index], 
                scores[index]
            ));
        }}
        
        return results;
    }}
    
    private int argmax(float[] array) {{
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {{
            if (array[i] > array[maxIndex]) {{
                maxIndex = i;
            }}
        }}
        return maxIndex;
    }}
    
    private int[] argtopk(float[] array, int k) {{
        // Simple implementation - sort indices by scores
        Integer[] indices = new Integer[array.length];
        for (int i = 0; i < array.length; i++) {{
            indices[i] = i;
        }}
        
        java.util.Arrays.sort(indices, (a, b) -> Float.compare(array[b], array[a]));
        
        int[] result = new int[Math.min(k, indices.length)];
        for (int i = 0; i < result.length; i++) {{
            result[i] = indices[i];
        }}
        
        return result;
    }}
    
    private String assetFilePath(Context context, String assetName) throws IOException {{
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {{
            return file.getAbsolutePath();
        }}
        
        try (InputStream is = context.getAssets().open(assetName);
             FileOutputStream os = new FileOutputStream(file)) {{
            byte[] buffer = new byte[4 * 1024];
            int read;
            while ((read = is.read(buffer)) != -1) {{
                os.write(buffer, 0, read);
            }}
            os.flush();
        }}
        return file.getAbsolutePath();
    }}
    
    public static class ClassificationResult {{
        public final String className;
        public final float confidence;
        
        public ClassificationResult(String className, float confidence) {{
            this.className = className;
            this.confidence = confidence;
        }}
    }}
}}
'''
        
        java_file = android_dir / f"{model_name.title()}Classifier.java"
        java_file.write_text(java_code)
    
    def _generate_deployment_metadata(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        model_name: str,
        deployment_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate comprehensive deployment metadata."""
        model_stats = self.profiler.profile_model(model, sample_input)
        
        metadata = {
            "model_info": {
                "name": model_name,
                "version": self.deployment_cfg.metadata.model_version,
                "description": self.deployment_cfg.metadata.description,
                "author": self.deployment_cfg.metadata.author,
                "license": self.deployment_cfg.metadata.license,
            },
            "model_stats": model_stats,
            "preprocessing": {
                "input_size": self.deployment_cfg.preprocessing.input_size,
                "mean": self.deployment_cfg.preprocessing.mean,
                "std": self.deployment_cfg.preprocessing.std,
                "format": self.deployment_cfg.preprocessing.format,
            },
            "postprocessing": {
                "output_format": self.deployment_cfg.postprocessing.output_format,
                "top_k": self.deployment_cfg.postprocessing.top_k,
                "confidence_threshold": self.deployment_cfg.postprocessing.confidence_threshold,
            },
            "performance_targets": self.cfg.compression.performance_targets,
            "deployment_platforms": list(deployment_results.keys()),
            "deployment_timestamp": str(torch.tensor([]).new_empty(0).storage().data_ptr()),  # Placeholder
        }
        
        # Save metadata
        metadata_file = self.output_dir / f"{model_name}_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        return metadata
    
    def _create_deployment_packages(
        self,
        deployment_results: Dict[str, Any],
        metadata: Dict[str, Any],
        model_name: str,
    ) -> Dict[str, str]:
        """Create deployment packages for each platform."""
        packages = {}
        
        for platform, result in deployment_results.items():
            if not result.get("success", False):
                continue
            
            package_dir = self.output_dir / f"{model_name}_{platform}_package"
            package_dir.mkdir(exist_ok=True)
            
            # Copy model file
            model_src = Path(result["model_path"])
            model_dst = package_dir / model_src.name
            shutil.copy2(model_src, model_dst)
            
            # Copy integration files
            platform_dir = self.output_dir / platform
            for file_path in platform_dir.glob("*"):
                if file_path.is_file():
                    shutil.copy2(file_path, package_dir / file_path.name)
            
            # Add README
            readme_content = self._generate_platform_readme(platform, result, metadata)
            (package_dir / "README.md").write_text(readme_content)
            
            packages[platform] = str(package_dir)
        
        return packages
    
    def _generate_platform_readme(
        self, platform: str, result: Dict[str, Any], metadata: Dict[str, Any]
    ) -> str:
        """Generate platform-specific README."""
        model_name = metadata["model_info"]["name"]
        
        readme = f"""# {model_name.title()} - {platform.title()} Deployment Package

## Model Information
- **Name**: {metadata["model_info"]["name"]}
- **Version**: {metadata["model_info"]["version"]}
- **Description**: {metadata["model_info"]["description"]}
- **Model Size**: {result["size_mb"]:.2f} MB
- **Format**: {result["format"]}

## Preprocessing Requirements
- **Input Size**: {metadata["preprocessing"]["input_size"]}
- **Normalization Mean**: {metadata["preprocessing"]["mean"]}
- **Normalization Std**: {metadata["preprocessing"]["std"]}
- **Input Format**: {metadata["preprocessing"]["format"]}

## Usage
"""
        
        if platform == "ios":
            readme += """
### iOS Integration

1. Add the `.mlmodel` file to your Xcode project
2. Include the `*Classifier.swift` file in your project
3. Use the classifier in your code:

```swift
let classifier = BirdClassifier()
classifier.classifyImage(yourImage) { className, confidence in
    print("Predicted: \\(className ?? "Unknown") with confidence: \\(confidence ?? 0)")
}
```
"""
        
        elif platform == "android":
            readme += """
### Android Integration

1. Add the `.pt` model file to your `assets` folder
2. Include the PyTorch Android library in your `build.gradle`:
   ```gradle
   implementation 'org.pytorch:pytorch_android:1.12.2'
   implementation 'org.pytorch:pytorch_android_torchvision:1.12.2'
   ```
3. Include the `*Classifier.java` file in your project
4. Use the classifier in your code:

```java
BirdClassifier classifier = new BirdClassifier(context);
String prediction = classifier.classifyImage(bitmap);
```
"""
        
        readme += f"""
## Performance Targets
- **Max Model Size**: {metadata["performance_targets"]["max_model_size_mb"]} MB
- **Max Inference Time**: {metadata["performance_targets"]["max_inference_time_ms"]} ms
- **Min Accuracy Retention**: {metadata["performance_targets"]["min_accuracy_retention"]}

## License
{metadata["model_info"]["license"]}
"""
        
        return readme