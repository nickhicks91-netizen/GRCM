"""Reflection head for memory-frequency alignment."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReflectionHead(nn.Module):
    """
    Measures alignment between current frequency state and memory.

    Reflects memory back to frequency space and computes cosine similarity.
    High alignment → current state resonates with past experiences.

    Args:
        freq_dim: Dimension of frequency space
        memory_size: Dimension of memory state
    """

    def __init__(self, freq_dim: int, memory_size: int):
        super().__init__()
        self.reflect = nn.Linear(memory_size, freq_dim)

    def forward(self, freq: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            freq: (batch, freq_dim) Current frequency state
            memory: (memory_size,) Memory state

        Returns:
            alignment: (batch, 1) Cosine similarity with reflected memory
        """
        batch_size = freq.size(0)

        # Broadcast memory to batch
        memory_batched = memory.unsqueeze(0).repeat(batch_size, 1)

        # Project memory to frequency space
        reflected = torch.tanh(self.reflect(memory_batched))

        # Cosine similarity between current freq and reflected memory
        alignment = F.cosine_similarity(freq, reflected, dim=-1).unsqueeze(-1)

        return alignment
