"""Harmonic frequency embedding with memory modulation."""
import torch
import torch.nn as nn
from typing import Optional


class HarmonicEmbedding(nn.Module):
    """
    Maps grounded input to frequency space with optional memory modulation.

    The embedding creates a 'frequency representation' that can be modulated
    by the episodic identity token for context-aware processing.

    Args:
        input_dim: Dimension of grounded input
        freq_dim: Dimension of frequency space
        memory_size: Size of memory/identity token (default 32)
    """

    def __init__(self, input_dim: int, freq_dim: int, memory_size: int = 32):
        super().__init__()
        self.fc = nn.Linear(input_dim, freq_dim)
        self.modulator = nn.Linear(memory_size, freq_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        memory_context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) Grounded input
            memory_context: (memory_size,) Optional identity token for modulation

        Returns:
            freq: (batch, freq_dim) Frequency-embedded representation
        """
        base = torch.tanh(self.fc(x))

        if memory_context is not None:
            batch_size = x.size(0)
            # Broadcast memory context to batch
            mod_ctx = memory_context.unsqueeze(0).repeat(batch_size, 1)
            mod = torch.sigmoid(self.modulator(mod_ctx))
            base = base * mod  # Multiplicative modulation

        return base
