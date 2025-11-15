"""
Purpose Gate: Adaptive information filtering mechanism
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PurposeGate(nn.Module):
    """
    Learnable gate that filters input based on purpose/importance.
    Uses attention-like mechanism to determine information relevance.
    """

    def __init__(self, dim: int, threshold: float = 0.5, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.threshold = threshold
        self.num_heads = num_heads

        # Query/Key/Value projections for gating
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # Gate prediction network
        self.gate_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )

        # Learnable threshold
        self.learned_threshold = nn.Parameter(torch.tensor(threshold))

    def forward(self, x: torch.Tensor, return_stats: bool = False):
        """
        Args:
            x: Input tensor [batch, seq_len, dim] or [batch, dim]
            return_stats: Whether to return gating statistics

        Returns:
            gated_output: Filtered output
            stats: Optional statistics dict
        """
        # Handle both 2D and 3D inputs
        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(1)  # [batch, 1, dim]

        batch_size, seq_len, _ = x.shape

        # Compute attention-based importance
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)

        # Compute gate values
        gate_logits = self.gate_net(context)  # [batch, seq_len, 1]

        # Apply learned threshold
        gates = (gate_logits > self.learned_threshold).float()

        # Soft gating during training, hard during eval
        if self.training:
            gates = gate_logits

        # Apply gates
        gated_output = x * gates

        # Compute statistics
        stats = {}
        if return_stats:
            stats = {
                'gate_values': gate_logits.detach(),
                'retention_rate': gates.mean().item(),
                'num_retained': gates.sum().item(),
                'total_tokens': gates.numel(),
            }

        if is_2d:
            gated_output = gated_output.squeeze(1)

        return gated_output, stats
