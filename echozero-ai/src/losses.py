"""
Custom loss functions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EchoZeroLoss(nn.Module):
    """
    Combined loss for EchoZero-Hybrid model:
    - Classification loss (cross-entropy)
    - Retention regularization (encourage efficient information filtering)
    - Diversity loss (encourage diverse reservoir states)
    """

    def __init__(
        self,
        num_classes: int = 2,
        retention_weight: float = 0.01,
        diversity_weight: float = 0.001,
        target_retention: float = 0.7,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.retention_weight = retention_weight
        self.diversity_weight = diversity_weight
        self.target_retention = target_retention

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        retained_mass: float,
        features: torch.Tensor = None,
    ):
        """
        Args:
            logits: Model predictions [batch, num_classes]
            labels: Ground truth labels [batch]
            retained_mass: Fraction of information retained by purpose gate
            features: Optional feature representations for diversity loss [batch, dim]

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual loss components
        """
        # Classification loss
        cls_loss = self.ce_loss(logits, labels)

        # Retention regularization
        # Penalize deviation from target retention rate
        retention_loss = torch.tensor(
            (retained_mass - self.target_retention) ** 2,
            device=logits.device,
        )

        # Diversity loss (if features provided)
        diversity_loss = torch.tensor(0.0, device=logits.device)
        if features is not None:
            # Encourage diverse feature representations
            # Penalize high correlation between samples
            features_norm = F.normalize(features, p=2, dim=1)
            correlation_matrix = torch.matmul(features_norm, features_norm.t())
            # Exclude diagonal (self-correlation)
            mask = ~torch.eye(features.size(0), dtype=torch.bool, device=features.device)
            diversity_loss = correlation_matrix[mask].abs().mean()

        # Total loss
        total_loss = (
            cls_loss +
            self.retention_weight * retention_loss +
            self.diversity_weight * diversity_loss
        )

        loss_dict = {
            'total': total_loss.item(),
            'classification': cls_loss.item(),
            'retention': retention_loss.item(),
            'diversity': diversity_loss.item(),
        }

        return total_loss, loss_dict
