"""Metrics calculation utilities."""

from typing import Dict, List, Any
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


class MetricsCalculator:
    """Calculate various classification metrics."""
    
    def __init__(self, metric_names: List[str]):
        self.metric_names = metric_names
        self.metric_functions = {
            "accuracy": self._accuracy,
            "top_5_accuracy": self._top_k_accuracy,
            "f1_macro": self._f1_macro,
            "precision_macro": self._precision_macro,
            "recall_macro": self._recall_macro,
        }
    
    def calculate(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Calculate metrics for a batch."""
        metrics = {}
        
        # Convert to probabilities and predictions
        probabilities = F.softmax(outputs, dim=1)
        predictions = torch.argmax(outputs, dim=1)
        
        # Convert to numpy for sklearn metrics
        targets_np = targets.cpu().numpy()
        predictions_np = predictions.cpu().numpy()
        probabilities_np = probabilities.cpu().numpy()
        
        for metric_name in self.metric_names:
            if metric_name in self.metric_functions:
                if "top_" in metric_name:
                    k = int(metric_name.split("_")[1])
                    metrics[metric_name] = self.metric_functions[metric_name](
                        targets_np, probabilities_np, k
                    )
                else:
                    metrics[metric_name] = self.metric_functions[metric_name](
                        targets_np, predictions_np
                    )
        
        return metrics
    
    def average_metrics(self, batch_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Average metrics across batches."""
        if not batch_metrics:
            return {}
        
        averaged = {}
        for metric_name in self.metric_names:
            values = [batch[metric_name] for batch in batch_metrics if metric_name in batch]
            if values:
                averaged[metric_name] = np.mean(values)
        
        return averaged
    
    def _accuracy(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate accuracy."""
        return accuracy_score(targets, predictions)
    
    def _top_k_accuracy(self, targets: np.ndarray, probabilities: np.ndarray, k: int) -> float:
        """Calculate top-k accuracy."""
        top_k_pred = np.argsort(probabilities, axis=1)[:, -k:]
        correct = 0
        for i, target in enumerate(targets):
            if target in top_k_pred[i]:
                correct += 1
        return correct / len(targets)
    
    def _f1_macro(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate macro F1 score."""
        return f1_score(targets, predictions, average="macro", zero_division=0)
    
    def _precision_macro(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate macro precision."""
        return precision_score(targets, predictions, average="macro", zero_division=0)
    
    def _recall_macro(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate macro recall."""
        return recall_score(targets, predictions, average="macro", zero_division=0)