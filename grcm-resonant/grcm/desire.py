"""Desire module for goal-directed agency and bandwidth modulation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DesireModule(nn.Module):
    """
    Implements desire-driven agency via learned desire vectors.

    Aligns frequency embeddings with current desire (cosine similarity).
    Gates memory updates when alignment > 0.5 (resonant with goal).
    Modulates bandwidth: higher alignment → wider bandwidth (seeking behavior).

    Args:
        freq_dim: Dimension of frequency space
        num_desires: Number of distinct desire states (default 4)
            Example: [curiosity, safety, social, exploration]
    """

    def __init__(self, freq_dim: int, num_desires: int = 4):
        super().__init__()
        self.desire_vectors = nn.Parameter(torch.randn(num_desires, freq_dim))
        self.current_desire_idx = 0
        self.num_desires = num_desires

    def set_desire(self, idx: int) -> None:
        """Switch to a different desire state."""
        assert 0 <= idx < self.num_desires, f"Desire index must be in [0, {self.num_desires})"
        self.current_desire_idx = idx

    def forward(self, freq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            freq: (batch, freq_dim) Frequency embeddings

        Returns:
            alignment: (batch, 1) Cosine similarity with current desire
            bandwidth_bias: (batch, 1) Bandwidth modulation (0.2 * alignment)
        """
        desire_vec = self.desire_vectors[self.current_desire_idx]

        # Cosine similarity: 1 = aligned, -1 = opposed
        alignment = F.cosine_similarity(
            freq,
            desire_vec.unsqueeze(0),
            dim=-1
        ).unsqueeze(-1)

        # Bandwidth bias: positive alignment widens bandwidth (seeking)
        bandwidth_bias = 0.2 * alignment

        return alignment, bandwidth_bias
