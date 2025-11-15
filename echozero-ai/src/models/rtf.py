"""
RTF: Retention Transformer Features module
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RTFBlock(nn.Module):
    """
    Retention Transformer Features (RTF) block combining
    retention mechanisms with transformer architecture.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        retention_factor: float = 0.9,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.retention_factor = retention_factor

        # Multi-head retention attention
        self.norm1 = nn.LayerNorm(dim)
        self.retention_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward network
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )

        # Retention gating
        self.retention_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, retention_state: torch.Tensor = None):
        """
        Args:
            x: Input [batch, seq_len, dim] or [batch, dim]
            retention_state: Previous retention state

        Returns:
            output: Processed output
            new_retention_state: Updated retention state
        """
        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(1)

        batch_size, seq_len, _ = x.shape

        # Initialize retention state
        if retention_state is None:
            retention_state = torch.zeros_like(x)

        # Retention-enhanced attention
        normed = self.norm1(x)
        attn_output, _ = self.retention_attn(normed, normed, normed)

        # Apply retention gate
        gate = self.retention_gate(x)
        retained = gate * attn_output + (1 - gate) * x
        x = x + retained

        # Update retention state
        new_retention_state = (
            self.retention_factor * retention_state +
            (1 - self.retention_factor) * x
        )

        # Feed-forward
        x = x + self.mlp(self.norm2(x))

        if is_2d:
            x = x.squeeze(1)
            new_retention_state = new_retention_state.squeeze(1)

        return x, new_retention_state


class RTF(nn.Module):
    """
    Stack of RTF blocks for deep retention-transformer processing
    """

    def __init__(
        self,
        dim: int,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList([
            RTFBlock(dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, retention_states: list = None):
        """
        Args:
            x: Input [batch, seq_len, dim] or [batch, dim]
            retention_states: List of previous retention states for each block

        Returns:
            output: Final output
            new_retention_states: Updated retention states
        """
        if retention_states is None:
            retention_states = [None] * self.depth

        new_retention_states = []

        for block, state in zip(self.blocks, retention_states):
            x, new_state = block(x, state)
            new_retention_states.append(new_state)

        x = self.norm(x)

        return x, new_retention_states
