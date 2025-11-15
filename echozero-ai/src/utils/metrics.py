"""
Evaluation metrics
"""
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict


class MetricsCalculator:
    """
    Calculate classification metrics
    """

    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Reset accumulated predictions"""
        self.all_preds = []
        self.all_labels = []

    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        """
        Update with new predictions

        Args:
            preds: Predicted logits or class indices [batch, num_classes] or [batch]
            labels: Ground truth labels [batch]
        """
        # Convert logits to class predictions if needed
        if preds.dim() > 1:
            preds = preds.argmax(dim=-1)

        # Move to CPU and convert to numpy
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        self.all_preds.extend(preds.tolist())
        self.all_labels.extend(labels.tolist())

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics

        Returns:
            Dictionary of metrics
        """
        if len(self.all_preds) == 0:
            return {}

        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)

        # Accuracy
        accuracy = accuracy_score(labels, preds)

        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='macro', zero_division=0
        )

        # Confusion matrix
        cm = confusion_matrix(labels, preds)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

        return metrics

    def compute_and_reset(self) -> Dict[str, float]:
        """Compute metrics and reset"""
        metrics = self.compute()
        self.reset()
        return metrics
