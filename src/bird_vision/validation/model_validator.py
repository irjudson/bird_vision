"""Model validation and comparison utilities."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig
from loguru import logger

from bird_vision.utils.metrics import MetricsCalculator


class ModelValidator:
    """Model validation and comparison system."""
    
    def __init__(self, cfg: DictConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.metrics_calculator = MetricsCalculator(cfg.model.metrics)
    
    def evaluate_model(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        model_name: str = "model",
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        logger.info(f"Evaluating model: {model_name}")
        
        model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        batch_metrics = []
        
        with torch.no_grad():
            for images, targets in test_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                # Collect results
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Calculate batch metrics
                batch_metrics.append(self.metrics_calculator.calculate(outputs, targets))
        
        # Calculate overall metrics
        overall_metrics = self.metrics_calculator.average_metrics(batch_metrics)
        
        # Additional detailed metrics
        detailed_results = self._calculate_detailed_metrics(
            all_targets, all_predictions, all_probabilities
        )
        
        # Combine results
        results = {
            "model_name": model_name,
            "overall_metrics": overall_metrics,
            "detailed_metrics": detailed_results,
            "predictions": all_predictions,
            "targets": all_targets,
            "probabilities": all_probabilities,
        }
        
        if save_results:
            self._save_evaluation_results(results, model_name)
        
        return results
    
    def _calculate_detailed_metrics(
        self,
        targets: List[int],
        predictions: List[int],
        probabilities: List[List[float]],
    ) -> Dict[str, Any]:
        """Calculate detailed classification metrics."""
        # Classification report
        class_report = classification_report(
            targets, predictions, output_dict=True, zero_division=0
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(targets, predictions)
        
        # Per-class accuracy
        per_class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
        per_class_accuracy = np.nan_to_num(per_class_accuracy)
        
        # Top-k accuracy
        probs_array = np.array(probabilities)
        top_k_accuracies = {}
        for k in [1, 3, 5]:
            if k <= probs_array.shape[1]:
                top_k_acc = self._calculate_top_k_accuracy(targets, probs_array, k)
                top_k_accuracies[f"top_{k}_accuracy"] = top_k_acc
        
        return {
            "classification_report": class_report,
            "confusion_matrix": conf_matrix.tolist(),
            "per_class_accuracy": per_class_accuracy.tolist(),
            "top_k_accuracies": top_k_accuracies,
            "num_classes": len(np.unique(targets)),
            "total_samples": len(targets),
        }
    
    def _calculate_top_k_accuracy(
        self, targets: List[int], probabilities: np.ndarray, k: int
    ) -> float:
        """Calculate top-k accuracy."""
        top_k_pred = np.argsort(probabilities, axis=1)[:, -k:]
        correct = 0
        for i, target in enumerate(targets):
            if target in top_k_pred[i]:
                correct += 1
        return correct / len(targets)
    
    def compare_models(
        self,
        models_results: List[Dict[str, Any]],
        comparison_metrics: List[str] = None,
    ) -> Dict[str, Any]:
        """Compare multiple model evaluation results."""
        if comparison_metrics is None:
            comparison_metrics = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
        
        comparison = {
            "models": [],
            "metrics_comparison": {},
            "best_model": None,
            "improvement_analysis": {},
        }
        
        # Extract metrics for comparison
        for result in models_results:
            model_name = result["model_name"]
            comparison["models"].append(model_name)
            
            for metric in comparison_metrics:
                if metric not in comparison["metrics_comparison"]:
                    comparison["metrics_comparison"][metric] = {}
                
                metric_value = result["overall_metrics"].get(metric, 0.0)
                comparison["metrics_comparison"][metric][model_name] = metric_value
        
        # Determine best model (based on accuracy by default)
        best_metric = "accuracy"
        best_score = 0
        best_model = None
        
        for model_name in comparison["models"]:
            score = comparison["metrics_comparison"][best_metric].get(model_name, 0)
            if score > best_score:
                best_score = score
                best_model = model_name
        
        comparison["best_model"] = {
            "name": best_model,
            "score": best_score,
            "metric": best_metric,
        }
        
        # Calculate improvements
        if len(models_results) >= 2:
            baseline = models_results[0]
            for i, current in enumerate(models_results[1:], 1):
                improvement = self._calculate_improvement(
                    baseline["overall_metrics"],
                    current["overall_metrics"],
                    comparison_metrics,
                )
                comparison["improvement_analysis"][current["model_name"]] = improvement
        
        return comparison
    
    def _calculate_improvement(
        self,
        baseline_metrics: Dict[str, float],
        current_metrics: Dict[str, float],
        metrics: List[str],
    ) -> Dict[str, float]:
        """Calculate improvement over baseline."""
        improvements = {}
        for metric in metrics:
            baseline_val = baseline_metrics.get(metric, 0.0)
            current_val = current_metrics.get(metric, 0.0)
            
            if baseline_val > 0:
                improvement = ((current_val - baseline_val) / baseline_val) * 100
            else:
                improvement = 0.0
            
            improvements[f"{metric}_improvement_percent"] = improvement
            improvements[f"{metric}_absolute_improvement"] = current_val - baseline_val
        
        return improvements
    
    def _save_evaluation_results(
        self, results: Dict[str, Any], model_name: str
    ) -> None:
        """Save evaluation results to files."""
        results_dir = Path(self.cfg.paths.artifacts_dir) / "evaluation_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics as JSON (excluding large arrays)
        metrics_data = {
            "model_name": results["model_name"],
            "overall_metrics": results["overall_metrics"],
            "detailed_metrics": {
                k: v for k, v in results["detailed_metrics"].items()
                if k not in ["confusion_matrix", "per_class_accuracy"]
            },
        }
        
        metrics_file = results_dir / f"{model_name}_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics_data, f, indent=2)
        
        # Save confusion matrix plot
        self._plot_confusion_matrix(
            results["detailed_metrics"]["confusion_matrix"],
            results_dir / f"{model_name}_confusion_matrix.png",
        )
        
        logger.info(f"Evaluation results saved to {results_dir}")
    
    def _plot_confusion_matrix(
        self, confusion_matrix: List[List[int]], save_path: Path
    ) -> None:
        """Plot and save confusion matrix."""
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            confusion_matrix,
            annot=False,
            fmt="d",
            cmap="Blues",
            square=True,
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    
    def validate_against_baseline(
        self,
        current_model: nn.Module,
        baseline_model: nn.Module,
        test_loader: DataLoader,
        threshold: float = 0.02,
    ) -> Dict[str, Any]:
        """Validate current model against baseline with improvement threshold."""
        # Evaluate both models
        current_results = self.evaluate_model(
            current_model, test_loader, "current_model", save_results=False
        )
        baseline_results = self.evaluate_model(
            baseline_model, test_loader, "baseline_model", save_results=False
        )
        
        # Compare models
        comparison = self.compare_models([baseline_results, current_results])
        
        # Check if improvement meets threshold
        current_accuracy = current_results["overall_metrics"].get("accuracy", 0.0)
        baseline_accuracy = baseline_results["overall_metrics"].get("accuracy", 0.0)
        
        improvement = current_accuracy - baseline_accuracy
        meets_threshold = improvement >= threshold
        
        validation_result = {
            "meets_threshold": meets_threshold,
            "required_improvement": threshold,
            "actual_improvement": improvement,
            "current_accuracy": current_accuracy,
            "baseline_accuracy": baseline_accuracy,
            "comparison": comparison,
            "recommendation": (
                "Deploy current model" if meets_threshold
                else "Keep baseline model"
            ),
        }
        
        logger.info(
            f"Validation result: {validation_result['recommendation']} "
            f"(improvement: {improvement:.4f}, threshold: {threshold})"
        )
        
        return validation_result