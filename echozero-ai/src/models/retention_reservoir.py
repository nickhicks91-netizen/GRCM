"""
Retention Reservoir: Memory-efficient state retention mechanism
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RetentionReservoir(nn.Module):
    """
    Implements a retention-based reservoir that maintains
    compressed memory state with controlled capacity.
    """

    def __init__(
        self,
        dim: int,
        reservoir_size: int = 512,
        retention_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.reservoir_size = reservoir_size
        self.retention_heads = retention_heads
        self.head_dim = dim // retention_heads

        # Retention projections
        self.q_ret = nn.Linear(dim, dim)
        self.k_ret = nn.Linear(dim, dim)
        self.v_ret = nn.Linear(dim, dim)

        # Reservoir compression
        self.compress = nn.Linear(dim, reservoir_size)
        self.expand = nn.Linear(reservoir_size, dim)

        # Decay parameters (learnable retention weights)
        self.gamma = nn.Parameter(torch.ones(retention_heads))

        # Output projection
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Layer norm
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, prev_state: torch.Tensor = None):
        """
        Args:
            x: Input [batch, seq_len, dim] or [batch, dim]
            prev_state: Previous reservoir state [batch, reservoir_size]

        Returns:
            output: Processed output
            new_state: Updated reservoir state
            retention_stats: Statistics dictionary
        """
        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(1)

        batch_size, seq_len, _ = x.shape

        # Initialize state if needed
        if prev_state is None:
            prev_state = torch.zeros(
                batch_size, self.reservoir_size,
                device=x.device, dtype=x.dtype
            )

        # Retention mechanism
        q = self.q_ret(x).view(batch_size, seq_len, self.retention_heads, self.head_dim)
        k = self.k_ret(x).view(batch_size, seq_len, self.retention_heads, self.head_dim)
        v = self.v_ret(x).view(batch_size, seq_len, self.retention_heads, self.head_dim)

        # Compute retention scores with decay
        q = q.transpose(1, 2)  # [batch, heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Position-aware decay
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        decay = self.gamma.view(1, -1, 1, 1) ** positions.view(1, 1, -1, 1)

        # Retention attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores * decay
        retention_weights = F.softmax(scores, dim=-1)
        retention_weights = self.dropout(retention_weights)

        # Apply retention
        retained = torch.matmul(retention_weights, v)
        retained = retained.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)

        # Update reservoir state
        compressed = self.compress(retained.mean(dim=1))  # Pool sequence
        new_state = 0.9 * prev_state + 0.1 * compressed  # Exponential moving average

        # Expand state and combine with retention
        expanded_state = self.expand(new_state).unsqueeze(1)
        output = self.norm(retained + expanded_state)
        output = self.out_proj(output)

        # Statistics
        retention_stats = {
            'retention_weights': retention_weights.detach().mean().item(),
            'state_norm': new_state.norm().item(),
            'gamma_mean': self.gamma.mean().item(),
        }

        if is_2d:
            output = output.squeeze(1)

        return output, new_state, retention_stats
