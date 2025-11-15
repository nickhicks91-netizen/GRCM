"""
TEA: Token-Enhanced Attention module
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TokenEnhancedAttention(nn.Module):
    """
    Token-Enhanced Attention (TEA) with relative position encoding
    and dynamic token importance weighting.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        # QKV projections
        self.qkv = nn.Linear(dim, dim * 3)

        # Token enhancement network
        self.token_enhance = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        # Relative position bias
        self.rel_pos_bias = nn.Embedding(2 * max_seq_len - 1, num_heads)

        # Output projection
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: Input [batch, seq_len, dim] or [batch, dim]
            mask: Optional attention mask

        Returns:
            output: Attended output
            attn_weights: Attention weights for analysis
        """
        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(1)

        batch_size, seq_len, _ = x.shape

        # Token enhancement
        enhanced = self.token_enhance(x)
        x = x + enhanced

        # QKV projection
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Add relative position bias
        positions = torch.arange(seq_len, device=x.device)
        rel_pos = positions.unsqueeze(1) - positions.unsqueeze(0)
        rel_pos = rel_pos + seq_len - 1  # Shift to positive indices
        rel_pos = rel_pos.clamp(0, 2 * seq_len - 2)
        rel_pos_bias = self.rel_pos_bias(rel_pos).permute(2, 0, 1)  # [heads, seq, seq]
        attn = attn + rel_pos_bias.unsqueeze(0)

        # Apply mask if provided
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = (attn_weights @ v).transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        output = self.out_proj(output)

        if is_2d:
            output = output.squeeze(1)

        return output, attn_weights
