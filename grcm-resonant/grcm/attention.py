"""Resonant attention mechanism for frequency-based coherence gating."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


class ResonantAttention(nn.Module):
    """
    Computes coherence as ReLU(1 - |freq_input - node_freq| / bandwidth).
    Gates memory updates when coherence > threshold (default 0.7).

    Formula: coherence = max(0, 1 - |f - n| / b)
    where f=freq_input, n=node_freq, b=bandwidth (adaptive)

    Args:
        freq_dim: Dimension of frequency space
        base_bandwidth: Initial bandwidth (default 0.5)
        min_bandwidth: Minimum allowed bandwidth (default 0.1)
        max_bandwidth: Maximum allowed bandwidth (default 1.0)
    """

    def __init__(
        self,
        freq_dim: int,
        base_bandwidth: float = 0.5,
        min_bandwidth: float = 0.1,
        max_bandwidth: float = 1.0
    ):
        super().__init__()
        self.node_freq = nn.Parameter(torch.randn(freq_dim))
        self.base_bw = base_bandwidth
        self.bandwidth = nn.Parameter(torch.tensor(base_bandwidth))
        self.min_bw = min_bandwidth
        self.max_bw = max_bandwidth

    def forward(
        self,
        freq_input: torch.Tensor,
        bw_bias: Union[float, torch.Tensor] = 0.0
    ) -> torch.Tensor:
        """
        Args:
            freq_input: (batch, freq_dim) Frequency embeddings
            bw_bias: Bandwidth bias from desire alignment (widens for seeking)

        Returns:
            coherence: (batch, 1) Coherence scores [0, 1]
        """
        # Handle bias as scalar or tensor
        bias = bw_bias.mean() if hasattr(bw_bias, 'mean') else bw_bias

        # Adaptive bandwidth with clamping
        temp_bw = self.bandwidth + bias
        temp_bw = torch.clamp(temp_bw, min=self.min_bw, max=self.max_bw)

        # Compute frequency distance
        delta = torch.abs(freq_input - self.node_freq)

        # Coherence: 1 when freq_input == node_freq, 0 when delta >= temp_bw
        coherence = F.relu(1 - delta / temp_bw)

        # Mean over frequency dimension
        return coherence.mean(dim=-1, keepdim=True)
