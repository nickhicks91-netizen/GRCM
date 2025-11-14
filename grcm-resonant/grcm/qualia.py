"""Qualia simulation: Phenomenal state distributions."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class QualiaModule(nn.Module):
    """
    Simulates qualia as softmax distribution over phenomenal states.

    Default 4-dimensional qualia:
        [0] calm - low arousal, stable
        [1] alert - high arousal, focused
        [2] curious - exploratory, wide bandwidth
        [3] conflicted - dissonant, ethical halt trigger

    Formula: qualia = softmax(linear(freq))
    Ethical: If qualia[3] > 0.6 → halt (dissonance protection)

    Args:
        freq_dim: Dimension of frequency input
        qualia_dim: Number of qualia dimensions (default 4)
    """

    def __init__(self, freq_dim: int, qualia_dim: int = 4):
        super().__init__()
        self.qualia_dim = qualia_dim
        self.projection = nn.Linear(freq_dim, qualia_dim)

    def forward(self, freq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            freq: (batch, freq_dim) Frequency embeddings

        Returns:
            qualia: (batch, qualia_dim) Probability distribution over states
        """
        logits = self.projection(freq)
        qualia = F.softmax(logits, dim=-1)
        return qualia

    def check_dissonance(self, qualia: torch.Tensor, threshold: float = 0.6) -> bool:
        """
        Check if conflicted qualia exceeds threshold (ethical halt).

        Args:
            qualia: (batch, qualia_dim) Qualia distribution
            threshold: Dissonance threshold (default 0.6)

        Returns:
            halt: True if any sample exceeds dissonance threshold
        """
        conflicted = qualia[:, 3]  # Last dimension = conflicted
        return (conflicted > threshold).any().item()
