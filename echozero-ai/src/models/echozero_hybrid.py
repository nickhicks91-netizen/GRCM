"""
EchoZero-Hybrid: Main model combining all components
"""
import torch
import torch.nn as nn
from typing import Dict, Optional

from .purpose_gate import PurposeGate
from .retention_reservoir import RetentionReservoir
from .tea import TokenEnhancedAttention
from .rtf import RTF
from .echo_core import EchoCore


class EchoZeroHybrid(nn.Module):
    """
    EchoZero-Hybrid: Purpose-Gated Retention Reservoir Architecture

    Architecture flow:
    Input → Purpose Gate → Echo Core → Retention Reservoir → TEA → RTF → Output
    """

    def __init__(
        self,
        dim: int = 256,
        num_classes: int = 2,
        input_channels: int = 1,
        reservoir_dim: int = 512,
        reservoir_size: int = 256,
        num_heads: int = 8,
        rtf_depth: int = 4,
        dropout: float = 0.1,
        use_purpose_gate: bool = True,
        use_echo_core: bool = True,
        use_retention: bool = True,
        use_tea: bool = True,
        use_rtf: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.use_purpose_gate = use_purpose_gate
        self.use_echo_core = use_echo_core
        self.use_retention = use_retention
        self.use_tea = use_tea
        self.use_rtf = use_rtf

        # Input embedding
        self.input_proj = nn.Linear(input_channels, dim)

        # Core components
        if use_purpose_gate:
            self.purpose_gate = PurposeGate(dim=dim, num_heads=num_heads)

        if use_echo_core:
            self.echo_core = EchoCore(
                input_dim=dim,
                reservoir_dim=reservoir_dim,
                spectral_radius=0.9,
                sparsity=0.1,
            )
            # Project echo output back to dim
            self.echo_proj = nn.Linear(dim, dim)

        if use_retention:
            self.retention_reservoir = RetentionReservoir(
                dim=dim,
                reservoir_size=reservoir_size,
                retention_heads=num_heads,
                dropout=dropout,
            )

        if use_tea:
            self.tea = TokenEnhancedAttention(
                dim=dim,
                num_heads=num_heads,
                dropout=dropout,
            )

        if use_rtf:
            self.rtf = RTF(
                dim=dim,
                depth=rtf_depth,
                num_heads=num_heads,
                dropout=dropout,
            )

        # Output heads
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)

        # Pooling for sequence aggregation
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(
        self,
        x: torch.Tensor,
        echo_state: Optional[torch.Tensor] = None,
        retention_state: Optional[torch.Tensor] = None,
        rtf_states: Optional[list] = None,
        return_all_stats: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input tensor [batch, seq_len] or [batch, seq_len, channels]
            echo_state: Previous echo state
            retention_state: Previous retention state
            rtf_states: Previous RTF states
            return_all_stats: Whether to return detailed statistics

        Returns:
            Dictionary containing:
                - logits: Classification logits [batch, num_classes]
                - features: Final feature representation [batch, dim]
                - retained_mass: Fraction of information retained
                - states: Updated states for recurrent processing
                - stats: Optional detailed statistics
        """
        # Handle input dimensions
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [batch, seq_len, 1]

        batch_size, seq_len, _ = x.shape

        # Project to model dimension
        x = self.input_proj(x)  # [batch, seq_len, dim]

        stats = {}

        # Purpose Gate
        if self.use_purpose_gate:
            x, gate_stats = self.purpose_gate(x, return_stats=True)
            stats['purpose_gate'] = gate_stats
            retained_mass = gate_stats['retention_rate']
        else:
            retained_mass = 1.0

        # Echo Core
        new_echo_state = None
        if self.use_echo_core:
            echo_out, new_echo_state, echo_stats = self.echo_core(x, echo_state)
            x = x + self.echo_proj(echo_out)  # Residual connection
            stats['echo_core'] = echo_stats

        # Retention Reservoir
        new_retention_state = None
        if self.use_retention:
            x, new_retention_state, ret_stats = self.retention_reservoir(x, retention_state)
            stats['retention'] = ret_stats

        # Token-Enhanced Attention
        if self.use_tea:
            tea_out, attn_weights = self.tea(x)
            x = x + tea_out  # Residual
            stats['tea_attention_mean'] = attn_weights.mean().item()

        # RTF Blocks
        new_rtf_states = None
        if self.use_rtf:
            x, new_rtf_states = self.rtf(x, rtf_states)

        # Normalize
        x = self.norm(x)

        # Pool sequence dimension
        if x.dim() == 3:
            # Global average pooling
            pooled = x.mean(dim=1)  # [batch, dim]
        else:
            pooled = x

        # Classification
        logits = self.classifier(pooled)

        # Prepare output
        output = {
            'logits': logits,
            'features': pooled,
            'retained_mass': retained_mass,
            'states': {
                'echo': new_echo_state,
                'retention': new_retention_state,
                'rtf': new_rtf_states,
            }
        }

        if return_all_stats:
            output['stats'] = stats

        return output

    def reset_states(self):
        """Reset all recurrent states"""
        # This is handled by passing None to forward()
        pass
