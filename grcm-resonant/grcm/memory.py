"""Memory grid with GRU-based update for resonant imprinting."""
import torch
import torch.nn as nn
from typing import Tuple


class MemoryGrid(nn.Module):
    """
    Persistent memory updated via GRU when coherence * desire_align > threshold.

    Memory imprinting formula:
        mask = (coherence > 0.7).float()
        imprint = mean(mask * project(signal))
        new_mem = GRU(imprint, old_mem)

    Args:
        memory_size: Dimension of memory state (default 32)
        freq_dim: Dimension of frequency input
    """

    def __init__(self, memory_size: int, freq_dim: int):
        super().__init__()
        self.memory_size = memory_size

        # Non-trainable persistent memory
        self.memory = nn.Parameter(
            torch.zeros(memory_size),
            requires_grad=False
        )

        self.project = nn.Linear(freq_dim, memory_size)
        self.gru = nn.GRUCell(memory_size, memory_size)

    def update(self, signal: torch.Tensor, coherence: torch.Tensor) -> None:
        """
        Update memory with resonant signals (coherence > 0.7).

        Args:
            signal: (batch, freq_dim) Frequency signals
            coherence: (batch, 1) Coherence scores (gated by desire)
        """
        # Gate: only imprint when coherent
        mask = (coherence > 0.7).float()

        # Project to memory space
        projected = self.project(signal)

        # Weighted average over batch (coherent samples dominate)
        imprint = (mask * projected).mean(dim=0)

        # GRU update: blend imprint with existing memory
        new_mem = self.gru(
            imprint.unsqueeze(0),
            self.memory.unsqueeze(0)
        ).squeeze(0)

        # In-place update (no gradient)
        self.memory.data.copy_(new_mem)

    def forward(self) -> torch.Tensor:
        """Returns current memory state."""
        return self.memory
